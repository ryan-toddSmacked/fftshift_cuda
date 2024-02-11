#pragma once
#ifndef __FFTSHIFT_H__
#define __FFTSHIFT_H__

#include <cuComplex.h>
#include <driver_types.h>

// Enumerations for dimensionality of the FFT 2D shift
enum FFT2DShiftDim
{
    FFT_BOTH = 0,   // Shift along both dimensions
    FFT_WIDTH = 1,  // Shift only along the width, x-axis
    FFT_HEIGHT = 2  // Shift only along the height, y-axis
};

template <typename T>
cudaError_t fftshift(T *d_dst, const T *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim = FFT_BOTH, cudaStream_t stream = 0);

extern template cudaError_t fftshift<float>(float *d_dst, const float *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);
extern template cudaError_t fftshift<double>(double *d_dst, const double *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);
extern template cudaError_t fftshift<cuFloatComplex>(cuFloatComplex *d_dst, const cuFloatComplex *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);
extern template cudaError_t fftshift<cuDoubleComplex>(cuDoubleComplex *d_dst, const cuDoubleComplex *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);

#endif // __FFTSHIFT_H__
