//
//  VolumeManager.h
//
//  Created by Nuttiiya Seekhao on 9/07/17.
//  Copyright (c) 2017 Nuttiiya Seekhao. All rights reserved.
//

#ifndef Volume_Manager_h
#define Volume_Manager_h

#include "common_vis.h"
#include "draw_util.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

//typedef unsigned char VolumeType;
typedef float VolumeType;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void bufferECMmap(cudaMemcpy3DParms copyParams);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize, cudaMemcpy3DParms &copyParams);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale);
extern "C" void render_kernel_dim(dim3 gridSize, dim3 blockSize, uint *d_output, uint nx, uint ny, uint nz,
															    uint imageW, uint imageH,
                                  float density, float brightness, float transferOffset, float transferScale);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);



class VolumeManager
{
public:
	VolumeManager();

private:

};


#endif
