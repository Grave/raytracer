
// OpenGL Graphics includes
#include <GL/glew.h>
#include <GLUT/glut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include <rendercheck_gl.h>

// Shared Library Test Functions
#include <shrQATest.h>

#include <vector>
#include <string>
#include <fstream>
#include <cutil_math.h>

#include "sphere.h"

using namespace std;

#define MAX_EPSILON   10
#define REFRESH_DELAY     10 //ms

const char *sSDKname = "raytracer";

unsigned int g_TotalErrors = 0;

// CheckFBO/BackBuffer class objects
CheckRender *g_CheckRender = NULL;

////////////////////////////////////////////////////////////////////////////////
// structs
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

////////////////////////////////////////////////////////////////////////////////
// constants / global variables
unsigned int window_width = 512;
unsigned int window_height = 512;
unsigned int image_width = 512;
unsigned int image_height = 512;
int iGLUTWindowHandle = 0;          // handle to the GLUT window

// pbo variables
GLuint pbo_dest;
struct cudaGraphicsResource *cuda_pbo_dest_resource;

unsigned int size_tex_data;
unsigned int num_texels;
unsigned int num_values;

// global vectors containing information about the world objects
vector<float4> g_vMaterials;
vector<float4> g_vVerticies;
vector<Sphere> g_vSpheres;

// pointers that the cuda device uses to access information
Sphere *g_pCuda_spheres;
float4 *g_pCuda_materials;
float *g_pCuda_vert;

// camera parameters
Camera g_camera;

// scene bounding box
float3 g_scene_aabbox_min;
float3 g_scene_aabbox_max;

// light
const float g_fLightx = 4;
const float g_fLighty = 4;
const float g_fLightz = 3;
const float g_lightColor[3] = {1.f, 1.f, 1.f};

GLuint g_tex_cudaResult;  // where we will copy the CUDA result

bool g_bQAReadback   = false;
bool g_bGLVerify     = false;
bool g_bQATest       = false;
bool g_bAnimate      = true;
int  g_iIndex         = 0;

int   *pArgc = NULL;
char **pArgv = NULL;

// FPS counter
static int g_iFpsCount = 0;
static int g_iFpsLimit = 1;
StopWatchInterface *g_pTimer = NULL;

bool IsOpenGLAvailable(const char *appName)
{
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// the interface between C++ and CUDA -----------
// the implementation of RayTraceImage is in the 
// "raytracer_kernel.cu" file
extern "C" void rayTraceImage(unsigned int *pbo_out, int w, int h, int number_of_spheres,
							  float3 a, float3 b, float3 c, 
							  float3 campos,
							  float3 light_pos,
							  float3 g_lightColor,
							  Sphere *spheres,
							  float4 *materials,
							  float3 g_scene_aabbox_min , float3 g_scene_aabbox_max);

// a method for binding the loaded spheres to a Cuda texture
// the implementation of bindSpheres is in the 
// "raytracer_kernel.cu" file
extern "C" void bindSpheres(float *bindSpheres, unsigned int number_of_triangles);

// Forward declarations
void runStdProgram(int argc, char **argv);
void FreeResource();
void Cleanup(int iExitCode);
void CleanupNoPrompt(int iExitCode);

// GL functionality
bool initCUDA(int argc, char **argv, bool bUseGL);
bool initGL(int *argc, char **argv);

void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource);
void deletePBO(GLuint *pbo);

void createTextureDst(GLuint *g_tex_cudaResult, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint *tex);

void display();
void renderScene();
void keyboard(unsigned char key, int x, int y);

void raytrace();
void displayTexture();
void loadObjects();
void updateCamera();

void displayImage(GLuint texture);

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void raytrace()
{
    cudaArray *in_array;
    unsigned int *out_data;
    
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_dest_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&out_data, &num_bytes,
                                                         cuda_pbo_dest_resource));
    //printf("CUDA mapped pointer of pbo_out: May access %ld bytes, expected %d\n", num_bytes, size_tex_data);
    
    // perform raytrace
    rayTraceImage(out_data, image_width, image_height, g_vSpheres.size(),
		g_camera.a, g_camera.b, g_camera.c, 
		g_camera.position, 
		make_float3(g_fLightx, g_fLighty, g_fLightz), 
		make_float3(g_lightColor[0], g_lightColor[1], g_lightColor[2]),
		g_pCuda_spheres,
		g_pCuda_materials,
		g_scene_aabbox_min , g_scene_aabbox_max);
    
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_dest_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_resource)
{
    // set up vertex data parameter
    num_texels = image_width * image_height;
    num_values = num_texels * 4;
    size_tex_data = sizeof(GLubyte) * num_values;
    void *data = malloc(size_tex_data);
    
    // create buffer object
    glGenBuffers(1, pbo);
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, data, GL_DYNAMIC_DRAW);
    free(data);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(pbo_resource, *pbo, cudaGraphicsMapFlagsNone));

    SDK_CHECK_ERROR_GL();
}

void deletePBO(GLuint *pbo)
{
    glDeleteBuffers(1, pbo);
    SDK_CHECK_ERROR_GL();
    *pbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! render 3D scene
////////////////////////////////////////////////////////////////////////////////

void updateCamera()
{
    float tx = cos(g_camera.rotation) * g_camera.distance;
    float tz = -sin(g_camera.rotation) * g_camera.distance;
    
	float3 campos = make_float3(tx, g_camera.height, tz);
	float3 cam_dir = -campos;
	cam_dir = normalize(cam_dir);
	float3 cam_up  = make_float3(0, 1, 0);
	float3 cam_right = cross(cam_dir,cam_up);
	cam_right = normalize(cam_right);

	cam_up = cross(cam_dir, cam_right);
	cam_up = -cam_up;
	cam_up = normalize(cam_up);
	
	float FOV = 60.0f;
	float theta = (FOV * 3.1415 * 0.5) / 180.0f;
	float half_width = tanf(theta);
	float aspect = (float)image_width / (float)image_height;

	float u0 = -half_width * aspect;
	float v0 = -half_width;
	float u1 =  half_width * aspect;
	float v1 =  half_width;
	float dist_to_image = 1;

    g_camera.position = campos;
	g_camera.a = (u1-u0) * cam_right;
	g_camera.b = (v1-v0) * cam_up;
	g_camera.c = campos + u0*cam_right + v0 * cam_up + dist_to_image * cam_dir;
}

// display image to the screen as textured quad
void displayTexture()
{
	// render a screen sized quad
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode( GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, window_width, window_height);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glDisable(GL_TEXTURE_2D);
	SDK_CHECK_ERROR_GL();
}

void renderScene()
{
    // run the Cuda kernel for raytracing
    raytrace();

    // CUDA generated data in cuda memory or in a mapped PBO made of BGRA 8 bits
    // - use glTexSubImage2D(), there is the potential to loose performance in possible hidden conversion

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_dest);

    glBindTexture(GL_TEXTURE_2D, g_tex_cudaResult);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    image_width, image_height,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
    glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    SDK_CHECK_ERROR_GL();
}

// display image to the screen as textured quad
void displayImage(GLuint texture)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, window_width, window_height);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&g_pTimer);

    updateCamera();

    renderScene();
        
    // processImage();
    //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    displayImage(g_tex_cudaResult);
    
    cudaDeviceSynchronize();
    sdkStopTimer(&g_pTimer);

    // flip backbuffer
    glutSwapBuffers();

    // Update fps counter, fps/title display and log
    if (++g_iFpsCount == g_iFpsLimit)
    {
        char cTitle[256];
        float fps = 1000.0f / sdkGetAverageTimerValue(&g_pTimer);
        sprintf(cTitle, "CUDA GL Raytracing (%d x %d): %.1f fps", window_width, window_height, fps);
        glutSetWindowTitle(cTitle);
        printf("%s\n", cTitle);
        g_iFpsCount = 0;
        g_iFpsLimit = (int)((fps > 1.0f) ? fps : 1.0f);
        sdkResetTimer(&g_pTimer);
    }
			
}

void timerEvent(int value)
{
    if (g_bAnimate)
    {
        // for next iteration of the raytracer demo
    }

    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            g_bQATest = true;
            CleanupNoPrompt(EXIT_SUCCESS);
            break;
            
        case '+':
            g_camera.rotation += 0.08;
            break;

        case '-':
            g_camera.rotation -= 0.08;
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Cuda texture handlers
////////////////////////////////////////////////////////////////////////////////
void createTextureDst(GLuint *g_tex_cudaResult, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, g_tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *g_tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    SDK_CHECK_ERROR_GL();
}

void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    shrQAStart(argc, argv);

    printf("%s Starting...\n\n", argv[0]);
    g_bQATest = (checkCmdLineFlag(argc, (const char **)argv,  "qatest") ||
               checkCmdLineFlag(argc, (const char **)argv, "glverify"));

    pArgc = &argc;
    pArgv = argv;

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        printf("[%s]\n", argv[0]);
        printf("   Does not explicitly support -device=n\n");
        printf("   This sample requires OpenGL.  Only -qatest and -glverify are supported\n");
        printf("exiting...\n");
        shrQAFinishExit(argc, (const char **)argv, QA_WAIVED);
    }

    if (g_bQATest)
    {
        printf("(Test with OpenGL verification)\n");
        g_bGLVerify     = true;
        g_bAnimate         = false;
        g_iIndex         = 0;

        runStdProgram(argc, argv);
    }
    else
    {
        printf("(Interactive OpenGL Demo)\n");
        g_bGLVerify     = false;
        g_bAnimate         = true;
        g_iIndex         = 0;

        runStdProgram(argc, argv);
    }

    shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
}

////////////////////////////////////////////////////////////////////////////////
//! Cleanup procedures
////////////////////////////////////////////////////////////////////////////////
void FreeResource()
{
    if (!g_bQAReadback)
    {
        sdkDeleteTimer(&g_pTimer);

        // unregister this buffer object with CUDA
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_dest_resource));
        deletePBO(&pbo_dest);
        
        deleteTexture(&g_tex_cudaResult);
    }

    cudaDeviceReset();

    if (iGLUTWindowHandle)
    {
        glutDestroyWindow(iGLUTWindowHandle);
    }

    // finalize logs and leave
    printf("postProcessGL.exe Exiting...\n");
}

void Cleanup(int iExitCode)
{
    FreeResource();
    shrQAFinishExit(*pArgc, (const char **)pArgv, (iExitCode == EXIT_SUCCESS) ? QA_PASSED : QA_FAILED);
}

void CleanupNoPrompt(int iExitCode)
{
    FreeResource();
    printf("%s\n", (iExitCode == EXIT_SUCCESS) ? "PASSED" : "FAILED");
    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Allocate Buffers
////////////////////////////////////////////////////////////////////////////////
void initCUDABuffers()
{
	size_t size = g_vVerticies.size() * sizeof(float4);
    if(size > 0)
	{
		checkCudaErrors(cudaMalloc((void **)&g_pCuda_vert, size));
		cudaMemcpy(g_pCuda_vert, &g_vVerticies[0], size, cudaMemcpyHostToDevice);
		
		size = g_vSpheres.size() * sizeof(Sphere);
		checkCudaErrors(cudaMalloc((void **)&g_pCuda_spheres, size));
		cudaMemcpy(g_pCuda_spheres, &g_vSpheres[0], size, cudaMemcpyHostToDevice);
		
		size = g_vMaterials.size() * sizeof(float4);
		checkCudaErrors(cudaMalloc((void **)&g_pCuda_materials, size));
		cudaMemcpy(g_pCuda_materials, &g_vMaterials[0], size, cudaMemcpyHostToDevice);
		
		bindSpheres(g_pCuda_vert, size);
	}
}

void initGLBuffers()
{
    // create pbo
    createPBO(&pbo_dest, &cuda_pbo_dest_resource);
    
    // create texture that will receive the result of CUDA
    createTextureDst(&g_tex_cudaResult, image_width, image_height);
    
    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Run standard demo loop with or without GL verification
////////////////////////////////////////////////////////////////////////////////
void runStdProgram(int argc, char **argv)
{
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return;
    }

    // Now initialize CUDA context (GL context has been created already)
    initCUDA(argc, argv, true);

    sdkCreateTimer(&g_pTimer);
    sdkResetTimer(&g_pTimer);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    
    loadObjects();

    initGLBuffers();
    initCUDABuffers();

    // Creating the Auto-Validation Code
    if (g_bGLVerify)
    {
        g_CheckRender = new CheckBackBuffer(window_width, window_height, 4);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);
    }

    printf("\n"
           "\tControls\n"
           "\t[=] : Rotate camera right\n"
           "\t[-] : Rotate camera left\n"
           "\t[esc] - Quit\n\n"
          );

    // start rendering mainloop
    glutMainLoop();

    // Normally unused return path
    if (!g_bQAReadback)
    {
        CleanupNoPrompt(EXIT_SUCCESS);
    }
    else
    {
        Cleanup(EXIT_SUCCESS);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize CUDA context
////////////////////////////////////////////////////////////////////////////////
bool initCUDA(int argc, char **argv, bool bUseGL)
{
    if (bUseGL)
    {
        findCudaGLDevice(argc, (const char **)argv);
    }
    else
    {
        findCudaDevice(argc, (const char **)argv);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    if (IsOpenGLAvailable(sSDKname))
    {
        fprintf(stderr, "   OpenGL device is Available\n");
    }
    else
    {
        fprintf(stderr, "   OpenGL device is NOT Available, [%s] exiting...\n", sSDKname);
        shrQAFinishExit(*argc, (const char **)argv, QA_WAIVED);
        return false;
    }

    // Create GL context
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("CUDA raytracing");

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported(
            "GL_VERSION_2_0 "
            "GL_ARB_pixel_buffer_object "
            "GL_EXT_framebuffer_object "
        ))
    {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);
    
    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Add objects to the scene
////////////////////////////////////////////////////////////////////////////////
void loadObjects()
{
    // Materials - (r, g, b, reflection_coeficient)
    g_vMaterials.push_back(make_float4(1.0f, 1.0f, 1.0f, 0.3f));
    g_vMaterials.push_back(make_float4(0.0f, 0.0f, 0.0f, 0.9f));
    g_vMaterials.push_back(make_float4(0.0f, 1.0f, 1.0f, 0.5f));
    g_vMaterials.push_back(make_float4(1.0f, 0.0f, 1.0f, 0.1f));
    
    g_vSpheres.resize(3);
    g_vVerticies.resize(3);
    
    // Verticies - sphere center( x, y, z ) + radii (w)
    g_vVerticies[0] = make_float4(0, 0.5f, 0, 0.5f);
    g_vSpheres[0].vert_idx = 0;
    g_vSpheres[0].mat_id = 1;
    
    g_vVerticies[1] = make_float4(0, -0.5f, -0.5f, 0.5f);
    g_vSpheres[1].vert_idx = 1;
    g_vSpheres[1].mat_id = 2;
    
    g_vVerticies[2] = make_float4(0, -0.5f, 0.5f, 0.5f);
    g_vSpheres[2].vert_idx = 2;
    g_vSpheres[2].mat_id = 3;
    
    g_scene_aabbox_min.x = -5;
	g_scene_aabbox_min.y = -5;
	g_scene_aabbox_min.z = -5;

	g_scene_aabbox_max.x = 5;
	g_scene_aabbox_max.y = 5;
	g_scene_aabbox_max.z = 5;
}
