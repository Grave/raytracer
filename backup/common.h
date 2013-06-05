#ifndef COMMON_H
#define COMMON_H

struct Camera
{
    float3 a;
    float3 b;
    float3 c;
    
    float3 position;
    float rotation;
    float distance;
    float height;
    
    Camera()
    {
        rotation = 0;
        height = 0.1f;
        distance = 3.f;
    }
};

struct Sphere 
{
	unsigned int vert_idx; // index in vertices array
	unsigned int mat_id; // material ID
};

struct Material
{
    float3 color;
    float reflection_coef;
    float refraction_coef;
    
    Material()
    {
        color = make_float3(0, 0, 0);
        reflection_coef = refraction_coef = 0;
    }
    
    Material(float r, float g, float b,
             float reflect_coef, float refract_coef = 0)
    {
        color = make_float3(r, g, b);
        reflection_coef = reflect_coef;
        refraction_coef = refract_coef;
    }
};

#endif // COMMON_H
