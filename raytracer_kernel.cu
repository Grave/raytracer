/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef _RAYTRACER_KERNEL_H_
#define _RAYTRACER_KERNEL_H_

// Utilities and system includes
#include <helper_cuda.h>

#include <vector_types.h>
#include <vector_functions.h>
#include <math_functions.h>
#include <cutil_math.h>

#include "sphere.h"

texture<float4, 1, cudaReadModeElementType> g_inTex;

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
	r = clamp(r, 0.0f, 255.0f);
	g = clamp(g, 0.0f, 255.0f);
	b = clamp(b, 0.0f, 255.0f);
	return (int(b)<<16) | (int(g)<<8) | int(r);
}

// Color storage to handle reflexions
struct BounceColor
{
	float3 color;
	float reflect_coef;
};

// Ray structure
struct Ray
{	
	__device__ Ray(){};
	__device__ Ray(const float3 &o,const float3 &d)
	{
		ori = o;
		dir = d;
		dir = normalize(dir);
		inv_dir = make_float3(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);
	}
	
	float3 ori;
	float3 dir;
	float3 inv_dir;
};

struct HitRecord
{
	__device__ HitRecord() {t = UINT_MAX;hit_index = -1; color = make_float3(0,0,0);}
	__device__ void resetT(){t = UINT_MAX; hit_index = -1;}
	
	float t;
	float3 color;
	float3 normal;
	int hit_index; 
	
};

// intersection code
__device__ int RayBoxIntersection(const float3 &BBMin, const float3 &BBMax, const float3 &RayOrg, const float3 &RayDirInv, float &tmin, float &tmax)
{
	float l1   = (BBMin.x - RayOrg.x) * RayDirInv.x;
	float l2   = (BBMax.x - RayOrg.x) * RayDirInv.x;
	tmin = fminf(l1,l2);
	tmax = fmaxf(l1,l2);

	l1   = (BBMin.y - RayOrg.y) * RayDirInv.y;
	l2   = (BBMax.y - RayOrg.y) * RayDirInv.y;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	l1   = (BBMin.z - RayOrg.z) * RayDirInv.z;
	l2   = (BBMax.z - RayOrg.z) * RayDirInv.z;
	tmin = fmaxf(fminf(l1,l2), tmin);
	tmax = fminf(fmaxf(l1,l2), tmax);

	return ((tmax >= tmin) && (tmax >= 0.0f));
}

__device__ int RaySphereIntersection(const Ray  &ray, const float3 sphere_center, const float sphere_radius, float &t)
{
	float b, c, d;

	float3 sr = ray.ori - sphere_center;
	b =  dot(sr, ray.dir);
	c = dot(sr, sr) - (sphere_radius * sphere_radius);
	d = b*b - c;
	
	if (d > 0) 
	{
		float e = sqrt(d);
		float t0 = -b - e;
		
		if(t0 < 0)
			t = -b + e;
		else
			t = min(-b-e,-b+e);
		return 1;
	}
	
	return 0;
}

__device__ int RayGroundIntersection(const Ray  &ray, const float3 ground_normal, float &t)
{
	float d = dot(ground_normal, ray.dir);
	
	if (d != 0)
	{
		float b = -( dot(ray.ori, ground_normal) + 1) / d;
		
		if (b > 0)
		{
			t = b;
			return 1;
		}
		else
		{
			t = 0;
			return 0;
		}
	}
	
	return 0;
}

// main raytracing function
__global__ void raytrace( unsigned int *out_data,
						   const int w,
						   const int h,
						   const int number_of_spheres,
						   const float3 a, const float3 b, const float3 c, 
						   const float3 campos,
						   const float3 light_pos,
						   const float3 light_color,
						   const Sphere *spheres,
						   const float4 *materials,
						   const float3 scene_aabb_min, 
						   const float3 scene_aabb_max)
{

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// a few constant values that we assume
	// could be stored in the futures in a struct
	const bool antialias = true;
	const int max_ray_depth = 4;
	
	// or as part of the material
	const float amb_coef = 0.4f;
	const float diffuse_coef = 0.7f;
	const float specular_coef = 0.7f;
	const float spec_hardness = 15.f;
	const float att_coef = 1.0f;
	
	const float3 ground_normal = make_float3(0.f, 1.f, 0.f);
	const float3 backg_color = make_float3(1.f, 1.f, 1.f);
	
	int pixelsTraced = 1;
	
	if (antialias)
		pixelsTraced = 4;
	
	int ray_depth = 0;
	bool continue_path = true;

	BounceColor bounce_array[max_ray_depth]; // Use this as stack for reflected rays
	
	float3 finalColor = make_float3(0, 0, 0);
	
	while(pixelsTraced--) 
	{
		float xx = x;
		float yy = y;

		if (antialias)
		{
			// nudge in a cross pattern around the pixel center
			xx += 0.25f - .5f*(pixelsTraced&1);
			yy += 0.25f - .5f*((pixelsTraced&2)>>1);
		}
		
		float xf = (xx-0.5)/((float)w);
		float yf = (yy-0.5)/((float)h);
		
		float3 t1 = c+(a*xf);
		float3 t2 = b*yf;
		float3 image_pos = t1 + t2;
		Ray r(image_pos,image_pos-campos);
		HitRecord hit_r;
		
		float t_min, t_max;
		continue_path = RayBoxIntersection(scene_aabb_min, scene_aabb_max, r.ori, r.inv_dir,t_min, t_max);
		
		ray_depth = 0;
			
		while(continue_path && ray_depth < max_ray_depth)
		{
			hit_r.resetT();
			bounce_array[ray_depth].color = backg_color; //default color for this ray
			
			float t = 0;
			bool hit = false;
			bool hit_ground = false;
			
			// search through the spheres and find the nearest hit point
			for(int i = 0; i < number_of_spheres; i++)
			{
				float4 s = tex1Dfetch(g_inTex,i);
				
				hit = RaySphereIntersection(r, make_float3(s.x, s.y, s.z), s.w, t);
	
				// account for floating point imprecision
				if(hit && t < hit_r.t && t > 0.001)
				{
					hit_r.t = t; 
					hit_r.hit_index = i;
				}
			}
	
			hit = RaySphereIntersection(r, light_pos, 0.4f, t);
			// stop if it hits light source
			if (hit && t < hit_r.t && t > 0.001)
			{
				bounce_array[ray_depth].color = light_color;
				continue_path = false;
				++ray_depth;
				break;
			}

			// check if it hits the ground first as well
			hit = RayGroundIntersection(r, ground_normal, t);
			if (hit && t < hit_r.t && t > 0.001)
			{
				hit_r.hit_index = 0;
				hit_r.t = t;
				hit_ground = true;
			}
			
			// we have hit (either with sphere or ground)
			if(hit_r.hit_index >= 0)
			{
				// get material data (assume 0 for ground)
				int mat_index = 0;
				
				if (!hit_ground)
					mat_index = spheres[hit_r.hit_index].mat_id;
				
				const float4 mat = materials[ mat_index ];
				const float3 mat_color = make_float3(mat.x, mat.y, mat.z);
				const float reflection_coef = mat.w;
				
				bounce_array[ray_depth].reflect_coef = reflection_coef;
				
				// calculate hitpoint
				float3 hitpoint = r.ori + r.dir *hit_r.t;
				
				// create the normal (different for ground and sphere)
				if (!hit_ground)
				{
					float4 s = tex1Dfetch(g_inTex, hit_r.hit_index);
					hit_r.normal = hitpoint - make_float3(s.x, s.y, s.z);
					hit_r.normal = normalize(hit_r.normal);
				}
				else
				{
					hit_r.normal = ground_normal;
				}
				
				// calculate reflected ray
				float3 reflec_v = reflect(r.dir, hit_r.normal);
				
				// for more than 1 light add loop starting here
				
				// Add light container iteration
				float3 L = light_pos - hitpoint;
				L = normalize(L);
				
				// create a shadow ray
				Ray shadow_ray(hitpoint, L);
				bool in_shadow = false;
				for(int i = 0; i < number_of_spheres; i++) // Find if any object blocks the light
				{
					// ignore self
					if (i == hit_r.hit_index)
						continue;
					
					float4 b = tex1Dfetch(g_inTex,i);

					float t = 0;
					in_shadow = RaySphereIntersection(shadow_ray, make_float3(b.x, b.y, b.z), b.w, t);

					// found a blocker
					if(in_shadow && t > 0.001)
					{
						//att_coef = 0.0f; // full attenuation as a shaddow effect
						break;
					}
				}
				
				float3 amb_color = make_float3(1.0f, 1.0f, 1.0f);
				
				float3 pixel_color;
				
				/*
				 * Classic Phong model would be:
				 * 
				 * ambient = ambient color * ambient coefficient * diffuse color
				 * diffuse = diffuse coefficient  * diffuse color ( surface normal . light vector )
				 * specular = specular coefficient * specular color ( reflection vector . view vector ) ^ roughness/shine
				 * 
				 * final = ambient + attenuation coefficient * light color [ diffuse + specular ]
				 */
				
				// Use blinn-Phong
				pixel_color = amb_coef * amb_color * mat_color; // ambient color
				
				if (!in_shadow)
				{
					float3 diffuse_specular = max(dot(hit_r.normal, L), 0.f) * mat_color * light_color * diffuse_coef;
					
					float3 H = L + (-r.dir);
					H = normalize(H);
					
					float specular_intensity = powf( max(dot(hit_r.normal, H), 0.f), spec_hardness);
					
					diffuse_specular += att_coef * specular_coef * specular_intensity * mat_color;
					
					pixel_color += att_coef * diffuse_specular;
				}
				
				bounce_array[ray_depth].color = pixel_color;
				bounce_array[ray_depth].color *= (1.0f - bounce_array[ray_depth].reflect_coef);
				
				// reflection
				if(reflection_coef > 0.001f)
				{
					r = Ray(hitpoint, reflec_v);
				}
				else
				{
					continue_path = false;
				}
			}
			else
			{
				continue_path = false;
			}
			
			++ray_depth;
		}
	
		if (ray_depth > 0)
		{
			// last ray was a hit
			--ray_depth;
			
			// for the remaining process normally
			for (int i = ray_depth-1; i >= 0; --i)
			{
				float r_coef = bounce_array[i].reflect_coef;
				bounce_array[i].color = bounce_array[i+1].color * r_coef + bounce_array[i].color * (1.0f - r_coef);
			}
		}
		else
		{
			//background coloring
			//bounce_array[0].color -= (yf - 0.5f);
		}
		
		finalColor += bounce_array[0].color;
	}
	
	if (antialias)
		finalColor /= 4.f;
	
	// photon exposure
	float exposure = -2.f;
	finalColor.x = 1.f - expf(finalColor.x * exposure);
	finalColor.y = 1.f - expf(finalColor.y * exposure);
	finalColor.z = 1.f - expf(finalColor.z * exposure);
	
	int val = rgbToInt( min(finalColor.x, 1.f) * 255, 
						min(finalColor.y, 1.f) * 255, 
						min(finalColor.z, 1.f) * 255);
	
	out_data[y * w + x] = val;
}

extern "C" 
{
	void rayTraceImage(unsigned int *pbo_out, int w, int h, int number_of_spheres,
		               float3 a, float3 b, float3 c, 
		               float3 campos,
					   float3 light_pos,
					   float3 light_color,
					   Sphere *spheres,
					   float4 *materials,
					   float3 scene_aabbox_min, float3 scene_aabbox_max)
	{

		dim3 block(8,8,1);
		dim3 grid(w/block.x,h/block.y, 1);
		raytrace<<<grid, block>>>(pbo_out,w,h,number_of_spheres,a,b,c,campos,light_pos,light_color, spheres, materials,scene_aabbox_min,scene_aabbox_max);

	}

	void bindSpheres(float *dev_sphere_p, unsigned int number_of_spheres)
	{
		g_inTex.normalized = false;                      // access with normalized texture coordinates
		g_inTex.filterMode = cudaFilterModePoint;        // Point mode, so no 
		g_inTex.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

		size_t size = sizeof(float4) * number_of_spheres;       
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTexture(0, g_inTex, dev_sphere_p, channelDesc, size));
	}
}

#endif // #ifndef _RAYTRACER_KERNEL_H_

