/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "convolutionFFT_common.h"
#include "convolutionFFT.cuh"


extern "C" void extractCenter(
	    float *d_Dst,
	    float *d_Src,
	    int dataH,
	    int dataW,
	    int windowH,
	    int windowW
)
{
    assert(d_Src != d_Dst);
    assert(windowH <= dataH);
    assert(windowW <= dataW);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(dataW, threads.x), iDivUp(dataH, threads.y));

    SET_FLOAT_BASE;
    extractCenter_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        dataH,
        dataW,
        windowH,
        windowW
    );
    getLastCudaError("extractWindow_kernel<<<>>> execution failed\n");
}

extern "C" void unpadResult(
	    float *d_Dst,
	    float *d_Src,
	    int dataH,
	    int dataW,
	    int fftH,
	    int fftW
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(dataW, threads.x), iDivUp(dataH, threads.y));

    SET_FLOAT_BASE;
    unpadResult_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        dataH,
        dataW,
        fftH,
        fftW
    );
    getLastCudaError("unpadResult_kernel<<<>>> execution failed\n");
}
/*
extern "C" void computeCoverage3D(
	    float *d_Wind,
	    float *d_Kern,
	    int dataD,
	    int dataH,
	    int dataW,
	    int windowD,
	    int windowH,
	    int windowW
)
{
    assert(d_Src != d_Dst);
    assert(windowD <= dataD);
    assert(windowH <= dataH);
    assert(windowW <= dataW);
    dim3 threads(8, 8, 4);
    dim3 grid(iDivUp(dataW, threads.x), iDivUp(dataH, threads.y), iDivUp(dataD, threads.z));

    SET_FLOAT_BASE;
    conputeCoverage3D_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        dataD,
        dataH,
        dataW,
        windowD,
        windowH,
        windowW
    );
    getLastCudaError("computeCoverage3D_kernel<<<>>> execution failed\n");
}
*/
extern "C" void extractCenter3D(
	    float *d_Dst,
	    float *d_Src,
	    int dataD,
	    int dataH,
	    int dataW,
	    int windowD,
	    int windowH,
	    int windowW
)
{
    assert(d_Src != d_Dst);
    assert(windowD <= dataD);
    assert(windowH <= dataH);
    assert(windowW <= dataW);
    dim3 threads(8, 8, 4);
    dim3 grid(iDivUp(dataW, threads.x), iDivUp(dataH, threads.y), iDivUp(dataD, threads.z));

    SET_FLOAT_BASE;
    extractCenter3D_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        dataD,
        dataH,
        dataW,
        windowD,
        windowH,
        windowW
    );
    getLastCudaError("extractWindow3D_kernel<<<>>> execution failed\n");
}

extern "C" void unpadResult3D(
	    float *d_Dst,
	    float *d_Src,
	    int dataD,
	    int dataH,
	    int dataW,
	    int fftD,
	    int fftH,
	    int fftW
)
{
    assert(d_Src != d_Dst);
    dim3 threads(8, 8, 4);
    dim3 grid(iDivUp(dataW, threads.x), iDivUp(dataH, threads.y), iDivUp(dataD, threads.z));

    SET_FLOAT_BASE;
    unpadResult3D_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        dataD,
        dataH,
        dataW,
        fftD,
        fftH,
        fftW
    );
    getLastCudaError("unpadResult3D_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y));

    SET_FLOAT_BASE;
    padKernel_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );
    getLastCudaError("padKernel_kernel<<<>>> execution failed\n");
}


////////////////////////////////////////////////////////////////////////////////
// Prepare data for "base line chem padding" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataConstantVF(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelH,
    int kernelW,
    int kernelY,
    int kernelX,
    int epiBoundary,
    float  baseChem
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

    SET_FLOAT_BASE;
    padDataConstantVF_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX,
        epiBoundary,
        baseChem
    );
    getLastCudaError("padDataConstantVF_kernel<<<>>> execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "right wall" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataRightWall(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

    SET_FLOAT_BASE;
    padDataRightWall_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );
    getLastCudaError("padDataRightWall_kernel<<<>>> execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "mirror pad" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataMirror(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

    SET_FLOAT_BASE;
    padDataMirror_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );
    getLastCudaError("padDataMirror_kernel<<<>>> execution failed\n");
}


////////////////////////////////////////////////////////////////////////////////
/// Position 3D convolution kernel center at (0, 0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
extern "C" void padKernel3D(
		float *d_Dst,
		float *d_Src,
		int fftD,
		int fftH,
		int fftW,
		int kernelD,
		int kernelH,
		int kernelW,
		int kernelZ,
		int kernelY,
		int kernelX
)
{
	assert(d_Src != d_Dst);
	dim3 threads(8, 8, 4);
	dim3 grid(iDivUp(kernelW, threads.x), iDivUp(kernelH, threads.y), iDivUp(kernelD, threads.z));

	SET_FLOAT_BASE;
	padKernel3D_kernel<<<grid, threads>>>(
			d_Dst,
			d_Src,
			fftD,
			fftH,
			fftW,
			kernelD,
			kernelH,
			kernelW,
			kernelZ,
			kernelY,
			kernelX
	);
	getLastCudaError("padKernel3D_kernel<<<>>> execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Prepare 3D data for "base line chem padding" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataConstantVF3D(
    float *d_Dst,
    float *d_Src,
    int fftD,
    int fftH,
    int fftW,
    int dataD,
    int dataH,
    int dataW,
    int kernelD,
    int kernelH,
    int kernelW,
    int kernelZ,
    int kernelY,
    int kernelX,
    int epiBoundary,
    float  baseChem
)
{
	assert(d_Src != d_Dst);
	dim3 threads(8, 8, 4);
	dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y), iDivUp(fftD, threads.z));

    SET_FLOAT_BASE;
    padDataConstantVF3D_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
  			fftD,
  			fftH,
  			fftW,
  			dataD,
  			dataH,
  			dataW,
  			kernelD,
  			kernelH,
  			kernelW,
  			kernelZ,
  			kernelY,
  			kernelX,
        epiBoundary,
        baseChem
    );
    getLastCudaError("padDataConstantVF3D_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder(
    float *d_Dst,
    float *d_Src,
    int fftH,
    int fftW,
    int dataH,
    int dataW,
    int kernelW,
    int kernelH,
    int kernelY,
    int kernelX
)
{
    assert(d_Src != d_Dst);
    dim3 threads(32, 8);
    dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y));

    SET_FLOAT_BASE;
    padDataClampToBorder_kernel<<<grid, threads>>>(
        d_Dst,
        d_Src,
        fftH,
        fftW,
        dataH,
        dataW,
        kernelH,
        kernelW,
        kernelY,
        kernelX
    );
    getLastCudaError("padDataClampToBorder_kernel<<<>>> execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Prepare 3D data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
extern "C" void padDataClampToBorder3D(
	    float *d_Dst,
	    float *d_Src,
	    int fftD,
	    int fftH,
	    int fftW,
	    int dataD,
	    int dataH,
	    int dataW,
	    int kernelD,
	    int kernelH,
	    int kernelW,
	    int kernelZ,
	    int kernelY,
	    int kernelX
)
{
	assert(d_Src != d_Dst);
	dim3 threads(8, 8, 4);
	dim3 grid(iDivUp(fftW, threads.x), iDivUp(fftH, threads.y), iDivUp(fftD, threads.z));

	SET_FLOAT_BASE;
	padDataClampToBorder3D_kernel<<<grid, threads>>>(
			d_Dst,
			d_Src,
			fftD,
			fftH,
			fftW,
			dataD,
			dataH,
			dataW,
			kernelD,
			kernelH,
			kernelW,
			kernelZ,
			kernelY,
			kernelX
	);
	getLastCudaError("padDataClampToBorder3D_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize(
    fComplex *d_Dst,
    fComplex *d_Src,
    int fftH,
    int fftW,
    int padding
)
{
    assert(fftW % 2 == 0);
    const int dataSize = fftH * (fftW / 2 + padding);

    modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
        d_Dst,
        d_Src,
        dataSize,
        1.0f / (float)(fftW *fftH)
    );
    getLastCudaError("modulateAndNormalize() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Compute dst^n in-place, where dst is complex vector
////////////////////////////////////////////////////////////////////////////////
extern "C" void complexPower(
    fComplex *d_Dst,
	int fftD,
	int fftH,
	int fftW,
    int padding,
    int n
)
{
	assert(fftW % 2 == 0);
	const int dataSize = fftD * fftH * (fftW / 2 + padding);

	complexPower_kernel<<<iDivUp(dataSize, 256), 256>>>(
			d_Dst,
			dataSize,
			n
	);
	getLastCudaError("vecCpow() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded 3D kernel
// WITHOUT normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulate3D(
		fComplex *d_Dst,
		fComplex *d_Src,
		int fftD,
		int fftH,
		int fftW,
		int padding
)
{
	assert(fftW % 2 == 0);
	const int dataSize = fftD * fftH * (fftW / 2 + padding);

	modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
			d_Dst,
			d_Src,
			dataSize,
			1.0f
	);
	getLastCudaError("modulate3D() execution failed\n");
}


////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded 3D kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
extern "C" void modulateAndNormalize3D(
		fComplex *d_Dst,
		fComplex *d_Src,
		int fftD,
		int fftH,
		int fftW,
		int padding
)
{
	assert(fftW % 2 == 0);
	const int dataSize = fftD * fftH * (fftW / 2 + padding);

	modulateAndNormalize_kernel<<<iDivUp(dataSize, 256), 256>>>(
			d_Dst,
			d_Src,
			dataSize,
			1.0f / (float)(fftW*fftH*fftD)
	);
	getLastCudaError("modulateAndNormalize3D() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
static const double PI = 3.1415926535897932384626433832795;
static const uint BLOCKDIM = 256;

extern "C" void spPostprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
)
{
    assert(d_Src != d_Dst);
    assert(DX % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = DY * (DX / 2);
    const double phaseBase = dir * PI / (double)DX;

    SET_FCOMPLEX_BASE;
    spPostprocess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
        (fComplex *)d_Dst,
        (fComplex *)d_Src,
        DY, DX, threadCount, padding,
        (float)phaseBase
    );
    getLastCudaError("spPostprocess2D_kernel<<<>>> execution failed\n");
}

extern "C" void spPreprocess2D(
    void *d_Dst,
    void *d_Src,
    uint DY,
    uint DX,
    uint padding,
    int dir
)
{
    assert(d_Src != d_Dst);
    assert(DX % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = DY * (DX / 2);
    const double phaseBase = -dir * PI / (double)DX;

    SET_FCOMPLEX_BASE;
    spPreprocess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
        (fComplex *)d_Dst,
        (fComplex *)d_Src,
        DY, DX, threadCount, padding,
        (float)phaseBase
    );
    getLastCudaError("spPreprocess2D_kernel<<<>>> execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess2D + modulateAndNormalize + spPreprocess2D
////////////////////////////////////////////////////////////////////////////////
extern "C" void spProcess2D(
    void *d_Dst,
    void *d_SrcA,
    void *d_SrcB,
    uint DY,
    uint DX,
    int dir
)
{
    assert(DY % 2 == 0);

#if(POWER_OF_TWO)
    uint log2DX, log2DY;
    uint factorizationRemX = factorRadix2(log2DX, DX);
    uint factorizationRemY = factorRadix2(log2DY, DY);
    assert(factorizationRemX == 1 && factorizationRemY == 1);
#endif

    const uint threadCount = (DY / 2) * DX;
    const double phaseBase = dir * PI / (double)DX;

    SET_FCOMPLEX_BASE_A;
    SET_FCOMPLEX_BASE_B;
    spProcess2D_kernel<<<iDivUp(threadCount, BLOCKDIM), BLOCKDIM>>>(
        (fComplex *)d_Dst,
        (fComplex *)d_SrcA,
        (fComplex *)d_SrcB,
        DY, DX, threadCount,
        (float)phaseBase,
        0.5f / (float)(DY *DX)
    );
    getLastCudaError("spProcess2D_kernel<<<>>> execution failed\n");
}
