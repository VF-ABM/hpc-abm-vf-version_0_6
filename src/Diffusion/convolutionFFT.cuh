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

#include <cuComplex.h>
#include <math.h>

#define  USE_TEXTURE 0//1
#define POWER_OF_TWO 1


#if(USE_TEXTURE)
texture<float, 1, cudaReadModeElementType> texFloat;
#define   LOAD_FLOAT(i) tex1Dfetch(texFloat, i)
#define  SET_FLOAT_BASE checkCudaErrors( cudaBindTexture(0, texFloat, d_Src) )
#else
#define  LOAD_FLOAT(i) d_Src[i]
#define SET_FLOAT_BASE
#endif


__global__ void extractCenter_kernel(
		float *d_Dst,
		float *d_Src,
		int dataH,
		int dataW,
		int windowH,
		int windowW
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	const int borderH = (dataH - windowH) / 2;
	const int borderW = (dataW - windowW) / 2;

	const int windowTop = borderH;
	const int windowBot = borderH + windowH;
	const int windowLef = borderW;
	const int windowRig = borderW + windowW;

	if (y >= windowTop && y < windowBot  && x >= windowLef && x < windowRig)
	{
		int dy = y - borderH;
		int dx = x - borderW;
		d_Dst[dy * windowW + dx] = LOAD_FLOAT(y * dataW + x);
	}
}

__global__ void unpadResult_kernel(
		float *d_Dst,
		float *d_Src,
		int dataH,
		int dataW,
		int fftH,
		int fftW
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < dataH && x < dataW)
	{
		d_Dst[y * dataW + x] = LOAD_FLOAT(y * fftW + x);
	}
}

__global__ void extractCenter3D_kernel(
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
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	const int borderD = (dataD - windowD) / 2;
	const int borderH = (dataH - windowH) / 2;
	const int borderW = (dataW - windowW) / 2;

	const int windowFro = borderD;
	const int windowBac = borderD + windowD;
	const int windowTop = borderH;
	const int windowBot = borderH + windowH;
	const int windowLef = borderW;
	const int windowRig = borderW + windowW;

	if ( z >= windowFro && z < windowBac &&
			y >= windowTop && y < windowBot  &&
			x >= windowLef && x < windowRig)
	{
		int dz = z - borderD;
		int dy = y - borderH;
		int dx = x - borderW;
		d_Dst[dz * windowH * windowW + dy * windowW + dx] = LOAD_FLOAT(z * dataH * dataW + y * dataW + x);
	}
}

__global__ void computeCoverage3D_kernel(
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
	/*    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int x = blockDim.x * blockIdx.x + threadIdx.x;

    const int borderD = (dataD - windowD) / 2;
    const int borderH = (dataH - windowH) / 2;
    const int borderW = (dataW - windowW) / 2;

    const int windowFro = borderD;
    const int windowBac = borderD + windowD;
    const int windowTop = borderH;
    const int windowBot = borderH + windowH;
    const int windowLef = borderW;
    const int windowRig = borderW + windowW;

    if ( z >= windowFro && z < windowBac &&
    		y >= windowTop && y < windowBot  &&
    		x >= windowLef && x < windowRig)
    {
    	int dz = z - borderD;
    	int dy = y - borderH;
    	int dx = x - borderW;
        d_Dst[dz * windowH * windowW + dy * windowW + dx] = LOAD_FLOAT(z * dataH * dataW + y * dataW + x);
    }
	 */
}

__global__ void unpadResult3D_kernel(
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
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (z < dataD && y < dataH && x < dataW)
	{
		d_Dst[z * dataH * dataW + y * dataW + x] = LOAD_FLOAT(z * fftH * fftW + y * fftW + x);
	}
}




////////////////////////////////////////////////////////////////////////////////
/// Position convolution kernel center at (0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel_kernel(
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
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (y < kernelH && x < kernelW)
	{
		int ky = y - kernelY;

		if (ky < 0)
		{
			ky += fftH;
		}

		int kx = x - kernelX;

		if (kx < 0)
		{
			kx += fftW;
		}

		d_Dst[ky * fftW + kx] = LOAD_FLOAT(y * kernelW + x);
	}
}


////////////////////////////////////////////////////////////////////////////////
// Prepare data for "base line chem padding" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataConstantVF_kernel(
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
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderY = dataH - 1;
	const int borderX = epiBoundary;

	if (y < fftH && x < fftW)
	{

		int dy, dx;

		if (y < borderY && y > 0 && x < borderX && x > 0) {
			dy = y;
			dx = x;
		} else {
			dy = -1;
			dx = -1;
		}


		if (dx < 0 || dy < 0)
			d_Dst[y * fftW + x] = baseChem;												// Constant padding
			else
				d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataRightWall_kernel(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		int kernelY,
		int kernelX
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderH = dataH + kernelY;
	const int borderW = dataW + kernelX;

	if (y < fftH && x < fftW)
	{
		int dy, dx;

		do
		{
			if (y < dataH)
			{
				dy = y;
			}

			if (x < dataW)
			{
				dx = x;

				if (y >= dataH && y < borderH)
				{
					dy = -1;							// Constant Padding
					break;
				}
			}

			if (y >= dataH && y < borderH)
			{
				dy = dataH - (y - dataH) - 1;

			}

			if (x >= dataW && x < borderW)
			{
				dx = dataW - (x - dataW) - 1;

				if (y >= borderH)						// Mirror Padding
				{
					dy = fftH - y - 1;
					break;
				}
			}

			if (y >= borderH)
			{
				dy = -1;

			}

			if (x >= borderW)
			{
				dx = -1;
			}
		} while(0);

		// TODO: load chemical baseline to GPU and use here
		if (dx < 0 || dy < 0)
			d_Dst[y * fftW + x] = 0.0001;								// Constant padding
		else
			d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);		// Mirror padding
	}
}



////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataMirror_kernel(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		int kernelY,
		int kernelX
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderH = dataH + kernelY;
	const int borderW = dataW + kernelX;

	if (y < fftH && x < fftW)
	{
		int dy, dx;

		if (y < dataH)
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - (y - dataH) - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - (x - dataW) - 1;
		}

		if (y >= borderH)
		{
			dy = fftH - y - 1;
		}

		if (x >= borderW)
		{
			dx = fftW - x - 1;
		}

		d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
	}
}

////////////////////////////////////////////////////////////////////////////////
/// Position 3D convolution kernel center at (0, 0, 0) in the image
////////////////////////////////////////////////////////////////////////////////
__global__ void padKernel3D_kernel(
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
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (z < kernelD && y < kernelH && x < kernelW)
	{
		int kz = z - kernelZ;

		if (kz < 0)
		{
			kz += fftD;
		}

		int ky = y - kernelY;

		if (ky < 0)
		{
			ky += fftH;
		}

		int kx = x - kernelX;

		if (kx < 0)
		{
			kx += fftW;
		}

		d_Dst[kz * fftH * fftW + ky * fftW + kx] = LOAD_FLOAT(z * kernelH * kernelW + y * kernelW + x);
	}
}


////////////////////////////////////////////////////////////////////////////////
// Prepare data for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder_kernel(
		float *d_Dst,
		float *d_Src,
		int fftH,
		int fftW,
		int dataH,
		int dataW,
		int kernelH,
		int kernelW,
		int kernelY,
		int kernelX
)
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderH = dataH + kernelY;
	const int borderW = dataW + kernelX;

	if (y < fftH && x < fftW)
	{
		int dy, dx;

		if (y < dataH)
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - 1;
		}

		if (y >= borderH)
		{
			dy = 0;
		}

		if (x >= borderW)
		{
			dx = 0;
		}

		d_Dst[y * fftW + x] = LOAD_FLOAT(dy * dataW + dx);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Prepare 3Dvdata for "pad to border" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataClampToBorder3D_kernel(
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
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderD = dataD + kernelZ;
	const int borderH = dataH + kernelY;
	const int borderW = dataW + kernelX;

	if (z < fftD && y < fftH && x < fftW)
	{
		int dz, dy, dx;

		if (z < dataD)
		{
			dz = z;
		}

		if (y < dataH)
		{
			dy = y;
		}

		if (x < dataW)
		{
			dx = x;
		}

		if (z >= dataD && z < borderD)
		{
			dz = dataD - 1;
		}

		if (y >= dataH && y < borderH)
		{
			dy = dataH - 1;
		}

		if (x >= dataW && x < borderW)
		{
			dx = dataW - 1;
		}

		if (z >= borderD)
		{
			dz = 0;
		}

		if (y >= borderH)
		{
			dy = 0;
		}

		if (x >= borderW)
		{
			dx = 0;
		}

		d_Dst[z * fftH * fftW + y * fftW + x] = LOAD_FLOAT(dz * dataH * dataW + dy * dataW + dx);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Prepare 3D data for "base line chem padding" addressing mode
////////////////////////////////////////////////////////////////////////////////
__global__ void padDataConstantVF3D_kernel(
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

	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int borderZ = dataD - 1;
	const int borderY = dataH - 1;
	const int borderX = epiBoundary;

	if (z < fftD && y < fftH && x < fftW)
	{

		int dy, dx, dz;

		if (z < borderZ && z > 0 && y < borderY && y > 0 && x < borderX && x > 0) {
			dy = y;
			dx = x;
			dz = z;
		} else {
			dy = -1;
			dx = -1;
			dz = -1;
		}


		if (dx < 0 || dy < 0 || dz < 0)
			d_Dst[z * fftW * fftH + y * fftW + x] = baseChem;												// Constant padding
		else
			d_Dst[z * fftW * fftH + y * fftW + x] = LOAD_FLOAT(dz * dataW * dataH + dy * dataW + dx);
	}
}



////////////////////////////////////////////////////////////////////////////////
// Compute z^n in-place, where z is complex
////////////////////////////////////////////////////////////////////////////////
inline __device__ void cpow(fComplex &z, const int &n) {
	//	fComplex t = {(powf(cuCabsf(z), n) * cos(n*atan2f(z.x, z.y))),
	//			(powf(cuCabsf(z), n) * sin(n*atan2f(z.x, z.y)))};

	// (a+i*b)^n
	//  Real component = (a^2+b^2)^(n/2)*cos(n*arctan(b/a))
	//  Imaginary component =(a^2+b^2)^(n/2)*sin(n*arctan(b/a))

	float a = z.x, b = z.y;
	float ab2 = powf(a, 2.0) + powf(b, 2.0f);
	float nf = __int2float_rn(n);
	float nf_2 = nf/2.0f;

	float re  = pow(ab2, nf_2) * cos(nf*atanf(b/a));
	float im  = pow(ab2, nf_2) * sin(nf*atanf(b/a));

	fComplex t = {re, im};

	z = t;
}

__global__ void complexPower_kernel(
		fComplex *d_Dst,
		int dataSize,
		int n
)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
	{
		return;
	}

	fComplex a = d_Dst[i];

	cpow(a, n);

	d_Dst[i] = a;
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
inline __device__ void mulAndScale(fComplex &a, const fComplex &b, const float &c)
{
	fComplex t = {c *(a.x * b.x - a.y * b.y), c *(a.y * b.x + a.x * b.y)};
	a = t;
}

__global__ void modulateAndNormalize_kernel(
		fComplex *d_Dst,
		fComplex *d_Src,
		int dataSize,
		float c
)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= dataSize)
	{
		return;
	}

	fComplex a = d_Src[i];
	fComplex b = d_Dst[i];

	mulAndScale(a, b, c);

	d_Dst[i] = a;
}



////////////////////////////////////////////////////////////////////////////////
// 2D R2C / C2R post/preprocessing kernels
////////////////////////////////////////////////////////////////////////////////
#if(USE_TEXTURE)
texture<fComplex, 1, cudaReadModeElementType> texComplexA;
texture<fComplex, 1, cudaReadModeElementType> texComplexB;
#define    LOAD_FCOMPLEX(i) tex1Dfetch(texComplexA, i)
#define  LOAD_FCOMPLEX_A(i) tex1Dfetch(texComplexA, i)
#define  LOAD_FCOMPLEX_B(i) tex1Dfetch(texComplexB, i)

#define   SET_FCOMPLEX_BASE checkCudaErrors( cudaBindTexture(0, texComplexA,  d_Src) )
#define SET_FCOMPLEX_BASE_A checkCudaErrors( cudaBindTexture(0, texComplexA, d_SrcA) )
#define SET_FCOMPLEX_BASE_B checkCudaErrors( cudaBindTexture(0, texComplexB, d_SrcB) )
#else
#define    LOAD_FCOMPLEX(i)  d_Src[i]
#define  LOAD_FCOMPLEX_A(i) d_SrcA[i]
#define  LOAD_FCOMPLEX_B(i) d_SrcB[i]

#define   SET_FCOMPLEX_BASE
#define SET_FCOMPLEX_BASE_A
#define SET_FCOMPLEX_BASE_B
#endif

inline __device__ void spPostprocessC2C(fComplex &D1, fComplex &D2, const fComplex &twiddle)
{
	float A1 = 0.5f * (D1.x + D2.x);
	float B1 = 0.5f * (D1.y - D2.y);
	float A2 = 0.5f * (D1.y + D2.y);
	float B2 = 0.5f * (D1.x - D2.x);

	D1.x = A1 + (A2 * twiddle.x + B2 * twiddle.y);
	D1.y = (A2 * twiddle.y - B2 * twiddle.x) + B1;
	D2.x = A1 - (A2 * twiddle.x + B2 * twiddle.y);
	D2.y = (A2 * twiddle.y - B2 * twiddle.x) - B1;
}

//Premultiply by 2 to account for 1.0 / (DZ * DY * DX) normalization
inline __device__ void spPreprocessC2C(fComplex &D1, fComplex &D2, const fComplex &twiddle)
{
	float A1 = /* 0.5f * */ (D1.x + D2.x);
	float B1 = /* 0.5f * */ (D1.y - D2.y);
	float A2 = /* 0.5f * */ (D1.y + D2.y);
	float B2 = /* 0.5f * */ (D1.x - D2.x);

	D1.x = A1 - (A2 * twiddle.x - B2 * twiddle.y);
	D1.y = (B2 * twiddle.x + A2 * twiddle.y) + B1;
	D2.x = A1 + (A2 * twiddle.x - B2 * twiddle.y);
	D2.y = (B2 * twiddle.x + A2 * twiddle.y) - B1;
}

inline __device__ void getTwiddle(fComplex &twiddle, float phase)
{
	__sincosf(phase, &twiddle.y, &twiddle.x);
}

inline __device__ uint mod(uint a, uint DA)
{
	//(DA - a) % DA, assuming a <= DA
	return a ? (DA - a) : a;
}

static inline uint factorRadix2(uint &log2N, uint n)
{
	if (!n)
	{
		log2N = 0;
		return 0;
	}
	else
	{
		for (log2N = 0; n % 2 == 0; n /= 2, log2N++);

		return n;
	}
}

inline __device__ void udivmod(uint &dividend, uint divisor, uint &rem)
{
#if(!POWER_OF_TWO)
	rem = dividend % divisor;
	dividend /= divisor;
#else
	rem = dividend & (divisor - 1);
	dividend >>= (__ffs(divisor) - 1);
#endif
}

__global__ void spPostprocess2D_kernel(
		fComplex *d_Dst,
		fComplex *d_Src,
		uint DY,
		uint DX,
		uint threadCount,
		uint padding,
		float phaseBase
)
{
	const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId >= threadCount)
	{
		return;
	}

	uint x, y, i = threadId;
	udivmod(i, DX / 2, x);
	udivmod(i, DY, y);

	//Avoid overwrites in columns DX / 2 by different threads
	if ((x == 0) && (y > DY / 2))
	{
		return;
	}

	const uint srcOffset = i * DY * DX;
	const uint dstOffset = i * DY * (DX + padding);

	//Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
	{
		const uint  loadPos1 = srcOffset +          y * DX +          x;
		const uint  loadPos2 = srcOffset + mod(y, DY) * DX + mod(x, DX);
		const uint storePos1 = dstOffset +          y * (DX + padding) +        x;
		const uint storePos2 = dstOffset + mod(y, DY) * (DX + padding) + (DX - x);

		fComplex D1 = LOAD_FCOMPLEX(loadPos1);
		fComplex D2 = LOAD_FCOMPLEX(loadPos2);

		fComplex twiddle;
		getTwiddle(twiddle, phaseBase * (float)x);
		spPostprocessC2C(D1, D2, twiddle);

		d_Dst[storePos1] = D1;
		d_Dst[storePos2] = D2;
	}

	//Process x = DX / 2
	if (x == 0)
	{
		const uint  loadPos1 = srcOffset +          y * DX + DX / 2;
		const uint  loadPos2 = srcOffset + mod(y, DY) * DX + DX / 2;
		const uint storePos1 = dstOffset +          y * (DX + padding) + DX / 2;
		const uint storePos2 = dstOffset + mod(y, DY) * (DX + padding) + DX / 2;

		fComplex D1 = LOAD_FCOMPLEX(loadPos1);
		fComplex D2 = LOAD_FCOMPLEX(loadPos2);

		//twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(dir * j * PI / 2)
		fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
		spPostprocessC2C(D1, D2, twiddle);

		d_Dst[storePos1] = D1;
		d_Dst[storePos2] = D2;
	}
}

__global__ void spPreprocess2D_kernel(
		fComplex *d_Dst,
		fComplex *d_Src,
		uint DY,
		uint DX,
		uint threadCount,
		uint padding,
		float phaseBase
)
{
	const uint threadId = blockIdx.x *  blockDim.x + threadIdx.x;

	if (threadId >= threadCount)
	{
		return;
	}

	uint x, y, i = threadId;
	udivmod(i, DX / 2, x);
	udivmod(i, DY, y);

	//Avoid overwrites in columns 0 and DX / 2 by different threads (lower and upper halves)
	if ((x == 0) && (y > DY / 2))
	{
		return;
	}

	const uint srcOffset = i * DY * (DX + padding);
	const uint dstOffset = i * DY * DX;

	//Process x = [0 .. DX / 2 - 1] U [DX / 2 + 1 .. DX]
	{
		const uint  loadPos1 = srcOffset +          y * (DX + padding) +        x;
		const uint  loadPos2 = srcOffset + mod(y, DY) * (DX + padding) + (DX - x);
		const uint storePos1 = dstOffset +          y * DX +          x;
		const uint storePos2 = dstOffset + mod(y, DY) * DX + mod(x, DX);

		fComplex D1 = LOAD_FCOMPLEX(loadPos1);
		fComplex D2 = LOAD_FCOMPLEX(loadPos2);

		fComplex twiddle;
		getTwiddle(twiddle, phaseBase * (float)x);
		spPreprocessC2C(D1, D2, twiddle);

		d_Dst[storePos1] = D1;
		d_Dst[storePos2] = D2;
	}

	//Process x = DX / 2
	if (x == 0)
	{
		const uint  loadPos1 = srcOffset +          y * (DX + padding) + DX / 2;
		const uint  loadPos2 = srcOffset + mod(y, DY) * (DX + padding) + DX / 2;
		const uint storePos1 = dstOffset +          y * DX + DX / 2;
		const uint storePos2 = dstOffset + mod(y, DY) * DX + DX / 2;

		fComplex D1 = LOAD_FCOMPLEX(loadPos1);
		fComplex D2 = LOAD_FCOMPLEX(loadPos2);

		//twiddle = getTwiddle(phaseBase * (DX / 2)) = exp(-dir * j * PI / 2)
		fComplex twiddle = {0, (phaseBase > 0) ? 1.0f : -1.0f};
		spPreprocessC2C(D1, D2, twiddle);

		d_Dst[storePos1] = D1;
		d_Dst[storePos2] = D2;
	}
}



////////////////////////////////////////////////////////////////////////////////
// Combined spPostprocess2D + modulateAndNormalize + spPreprocess2D
////////////////////////////////////////////////////////////////////////////////
__global__ void spProcess2D_kernel(
		fComplex *d_Dst,
		fComplex *d_SrcA,
		fComplex *d_SrcB,
		uint DY,
		uint DX,
		uint threadCount,
		float phaseBase,
		float c
)
{
	const uint threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId >= threadCount)
	{
		return;
	}

	uint x, y, i = threadId;
	udivmod(i, DX, x);
	udivmod(i, DY / 2, y);

	const uint offset = i * DY * DX;

	//Avoid overwrites in rows 0 and DY / 2 by different threads (left and right halves)
	//Otherwise correctness for in-place transformations is affected
	if ((y == 0) && (x > DX / 2))
	{
		return;
	}

	fComplex twiddle;

	//Process y = [0 .. DY / 2 - 1] U [DY - (DY / 2) + 1 .. DY - 1]
	{
		const uint pos1 = offset +          y * DX +          x;
		const uint pos2 = offset + mod(y, DY) * DX + mod(x, DX);

		fComplex D1 = LOAD_FCOMPLEX_A(pos1);
		fComplex D2 = LOAD_FCOMPLEX_A(pos2);
		fComplex K1 = LOAD_FCOMPLEX_B(pos1);
		fComplex K2 = LOAD_FCOMPLEX_B(pos2);
		getTwiddle(twiddle, phaseBase * (float)x);

		spPostprocessC2C(D1, D2, twiddle);
		spPostprocessC2C(K1, K2, twiddle);
		mulAndScale(D1, K1, c);
		mulAndScale(D2, K2, c);
		spPreprocessC2C(D1, D2, twiddle);

		d_Dst[pos1] = D1;
		d_Dst[pos2] = D2;
	}

	if (y == 0)
	{
		const uint pos1 = offset + (DY / 2) * DX +          x;
		const uint pos2 = offset + (DY / 2) * DX + mod(x, DX);

		fComplex D1 = LOAD_FCOMPLEX_A(pos1);
		fComplex D2 = LOAD_FCOMPLEX_A(pos2);
		fComplex K1 = LOAD_FCOMPLEX_B(pos1);
		fComplex K2 = LOAD_FCOMPLEX_B(pos2);

		spPostprocessC2C(D1, D2, twiddle);
		spPostprocessC2C(K1, K2, twiddle);
		mulAndScale(D1, K1, c);
		mulAndScale(D2, K2, c);
		spPreprocessC2C(D1, D2, twiddle);

		d_Dst[pos1] = D1;
		d_Dst[pos2] = D2;
	}
}
