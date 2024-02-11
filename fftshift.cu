#include <cuda_runtime.h>
#include "fftshift.cuh"

template <typename T>
__global__ void fftshift1D_kernel_even(T * d_dst, const T * d_src, unsigned int length)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = (length >> 1);
    if (i < half)
    {
        d_dst[i] = d_src[i + half];
        d_dst[i + half] = d_src[i];
    }
}

template <typename T>
__global__ void fftshift1D_kernel_odd(T * d_dst, const T * d_src, unsigned int length)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = (length >> 1);
    if (i < half)
    {
        d_dst[i] = d_src[i + half + 1];
        d_dst[i + half] = d_src[i];
    }
}

template <typename T>
__global__ void fftshift1D_kernel_inplace_even(T * d_data, unsigned int length)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = (length >> 1);
    if (i < half)
    {
        T tmp = d_data[i];
        d_data[i] = d_data[i + half];
        d_data[i + half] = tmp;
    }
}

template <typename T>
__global__ void fftshift1D_kernel_inplace_odd(T * d_data, unsigned int length)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int half = (length >> 1);
    if (i < half)
    {
        T tmp = d_data[i];
        d_data[i] = d_data[i + half + 1];
        d_data[i + half] = tmp;
    }
}

template <typename T>
__global__ void fftshift2D_kernel_even_even(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int half_width = (width >> 1);
    const unsigned int half_height = (height >> 1);

    if (i >= width)
        return;

    if (j < half_height)
    {
        unsigned int src_y = j;
        unsigned int dst_y = j + half_height;
        unsigned int src_x = i;
        unsigned int dst_x = (i + half_width) % width;

        unsigned int src_idx = src_y * width + src_x;
        unsigned int dst_idx = dst_y * width + dst_x;

        d_dst[dst_idx] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx];
    }
}

template <typename T>
__global__ void fftshift2D_kernel_even_odd(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int half_width = (width >> 1);
    const unsigned int half_height = (height >> 1);

    if (i >= width)
        return;

    if (j < half_height)
    {
        unsigned int src_y = j;
        unsigned int dst_y_to = j + half_height;
        unsigned int dst_y_from = j + half_height + 1;
        unsigned int src_x = i;
        unsigned int dst_x = (i + half_width) % width;

        unsigned int src_idx = src_y * width + src_x;
        unsigned int dst_idx_to = dst_y_to * width + dst_x;
        unsigned int dst_idx_from = dst_y_from * width + dst_x;
        
        d_dst[dst_idx_to] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx_from];
    }
}

template <typename T>
__global__ void fftshift2D_kernel_odd_even(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int half_width = (width >> 1);
    const unsigned int half_height = (height >> 1);

    if (j >= height)
        return;

    if (i < half_width)
    {
        unsigned int src_y = j;
        unsigned int dst_y = (j + half_height) % height;
        unsigned int src_x = i;
        unsigned int dst_x_to = i + half_width;
        unsigned int dst_x_from = i + half_width + 1;

        unsigned int src_idx = src_y * width + src_x;
        unsigned int dst_idx_to = dst_y * width + dst_x_to;
        unsigned int dst_idx_from = dst_y * width + dst_x_from;

        d_dst[dst_idx_to] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx_from];
    }
}

template <typename T>
__global__ void fftshift2D_kernel_odd_odd(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int half_width = (width >> 1);
    const unsigned int half_height = (height >> 1);

    if (i < half_width && j < half_height)
    {
        {
            // Top left to bottom right
            unsigned int src_y = j;
            unsigned int dst_y = j + half_height;
            unsigned int src_x = i;
            unsigned int dst_x = i + half_width;

            unsigned int src_idx = src_y * width + src_x;
            unsigned int dst_idx = dst_y * width + dst_x;

            d_dst[dst_idx] = d_src[src_idx];

            // Bottom right to top left
            src_y = j + half_height + 1;
            dst_y = j;
            src_x = i + half_width + 1;
            dst_x = i;

            src_idx = src_y * width + src_x;
            dst_idx = dst_y * width + dst_x;

            d_dst[dst_idx] = d_src[src_idx];
        }

        {
            // Top right to bottom left
            unsigned int src_y = j;
            unsigned int dst_y = j + half_height;
            unsigned int src_x = i + half_width + 1;
            unsigned int dst_x = i;

            unsigned int src_idx = src_y * width + src_x;
            unsigned int dst_idx = dst_y * width + dst_x;

            d_dst[dst_idx] = d_src[src_idx];

            // Bottom left to top right
            src_y = j + half_height + 1;
            dst_y = j;
            src_x = i;
            dst_x = i + half_width;

            src_idx = src_y * width + src_x;
            dst_idx = dst_y * width + dst_x;

            d_dst[dst_idx] = d_src[src_idx];
        }
    }
}

template <typename T>
__global__ void fftshift2D_width_kernel_even(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int half_width = (width >> 1);
    //const unsigned int half_height = (height >> 1);

    if (i >= half_width)
        return;

    if (j < height)
    {
        unsigned int src_x = i;
        unsigned int dst_x = i + half_width;
        unsigned int src_y = j;
        unsigned int dst_y = j;

        unsigned int src_idx = src_y * width + src_x;
        unsigned int dst_idx = dst_y * width + dst_x;

        d_dst[dst_idx] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx];
    }
}

template <typename T>
__global__ void fftshift2D_width_kernel_odd(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int half_width = (width >> 1);
    //const unsigned int half_height = (height >> 1);

    if (i >= half_width)
        return;

    if (j < height)
    {
        unsigned int src_x = i;
        unsigned int dst_x_to = i + half_width;
        unsigned int dst_x_from = i + half_width + 1;
        unsigned int src_y = j;
        unsigned int dst_y = j;

        unsigned int src_idx = src_y * width + src_x;
        unsigned int dst_idx_to = dst_y * width + dst_x_to;
        unsigned int dst_idx_from = dst_y * width + dst_x_from;

        d_dst[dst_idx_to] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx_from];
    }
}

template <typename T>
__global__ void fftshift2D_height_kernel_even(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    //const unsigned int half_width = (width >> 1);
    const unsigned int half_height = (height >> 1);

    if (j >= half_height)
        return;

    if (i < width)
    {
        unsigned int src_x = i;
        unsigned int dst_x = i;
        unsigned int src_y = j;
        unsigned int dst_y = j + half_height;

        unsigned int src_idx = src_y * width + src_x;
        unsigned int dst_idx = dst_y * width + dst_x;

        d_dst[dst_idx] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx];
    }
}

template <typename T>
__global__ void fftshift2D_height_kernel_odd(T * d_dst, const T * d_src, unsigned int width, unsigned int height)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    //const unsigned int half_width = (width >> 1);
    const unsigned int half_height = (height >> 1);

    if (j >= half_height)
        return;

    if (i < width)
    {
        //unsigned int src_x = i;
        //unsigned int dst_x = i;
        unsigned int src_y_to = j + half_height;
        unsigned int src_y_from = j + half_height + 1;

        unsigned int src_idx = j * width + i;
        unsigned int dst_idx_to = src_y_to * width + i;
        unsigned int dst_idx_from = src_y_from * width + i;

        d_dst[dst_idx_to] = d_src[src_idx];
        d_dst[src_idx] = d_src[dst_idx_from];
    }
}


template <typename T>
cudaError_t fftshift(T *d_dst, const T *d_src, unsigned int width, unsigned int height, FFT2DShiftDim siftDim, cudaStream_t stream)
{
    if (width == 0 || height == 0)
    {
        return cudaErrorInvalidValue;
    }

    if (width == 1 || height == 1)
    {
        const unsigned int length = height * width;
        const unsigned int half = length >> 1;
        constexpr dim3 block = {256,1,1};
        dim3 grid;
        if (d_dst != d_src)
        {
            if (length % 2 == 0)
            {
                grid = {(half + block.x - 1) / block.x, 1, 1};
                fftshift1D_kernel_even<T><<<grid, block, 0, stream>>>(d_dst, d_src, length);
            }
            else
            {
                T tmp;
                grid = {(half + 1 + block.x - 1) / block.x, 1, 1};
                cudaMemcpy(&tmp, d_src + half, sizeof(T), cudaMemcpyDeviceToHost);
                fftshift1D_kernel_odd<T><<<grid, block, 0, stream>>>(d_dst, d_src, length);
                cudaMemcpyAsync(d_dst + length - 1, &tmp, sizeof(T), cudaMemcpyHostToDevice, stream);
            }
        }
        else
        {
            if (length % 2 == 0)
            {
                grid = {(half + block.x - 1) / block.x, 1, 1};
                fftshift1D_kernel_inplace_even<T><<<grid, block, 0, stream>>>(d_dst, length);
            }
            else
            {
                T tmp;
                grid = {(half + 1 + block.x - 1) / block.x, 1, 1};
                cudaMemcpy(&tmp, d_dst + half, sizeof(T), cudaMemcpyDeviceToHost);
                fftshift1D_kernel_inplace_odd<T><<<grid, block, 0, stream>>>(d_dst, length);
                cudaMemcpyAsync(d_dst + length - 1, &tmp, sizeof(T), cudaMemcpyHostToDevice, stream);
            }
        }
    }
    else
    {
        const unsigned int half_width = width >> 1;
        const unsigned int half_height = height >> 1;
        dim3 block;
        dim3 grid;
        switch (siftDim)
        {
            case FFT2DShiftDim::FFT_BOTH:
                if (d_dst != d_src)
                {
                    if (width % 2 == 0 && height % 2 == 0)
                    {
                        block = {64, 4, 1};
                        grid = {(width + block.x - 1) / block.x, (half_height + block.y - 1) / block.y, 1};
                        fftshift2D_kernel_even_even<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);
                    }
                    else if (width % 2 == 0 && height % 2 != 0)
                    {
                        T *d_tmp_middle_row;
                        block = {64, 4, 1};
                        grid = {(width + block.x - 1) / block.x, (half_height + 1 + block.y - 1) / block.y, 1};

                        cudaMalloc(&d_tmp_middle_row, width * sizeof(T));

                        fftshift(d_tmp_middle_row, d_src + half_height * width, width, 1, FFT2DShiftDim::FFT_WIDTH, stream);
                        fftshift2D_kernel_even_odd<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);

                        cudaMemcpyAsync(d_dst + (height-1) * width, d_tmp_middle_row, width * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                        cudaStreamSynchronize(stream);

                        cudaFree(d_tmp_middle_row);
                    }
                    else if (width % 2 != 0 && height % 2 == 0)
                    {
                        T *d_tmp_middle_col;
                        block = {64, 4, 1};
                        grid = {(half_width + 1 + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1};

                        cudaMalloc(&d_tmp_middle_col, height * sizeof(T));
                        cudaMemcpy2D(d_tmp_middle_col, sizeof(T), d_src + half_width, width * sizeof(T), sizeof(T), height, cudaMemcpyDeviceToDevice);

                        fftshift(d_tmp_middle_col, d_tmp_middle_col, 1, height, FFT2DShiftDim::FFT_HEIGHT, stream);
                        fftshift2D_kernel_odd_even<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);

                        cudaMemcpy2DAsync(d_dst + width - 1, width * sizeof(T), d_tmp_middle_col, sizeof(T), sizeof(T), height, cudaMemcpyDeviceToDevice, stream);
                        cudaStreamSynchronize(stream);

                        cudaFree(d_tmp_middle_col);
                    }
                    else
                    {
                        T *d_tmp_middle_row;
                        T *d_tmp_middle_col;
                        block = {64, 4, 1};
                        grid = {(half_width + 1 + block.x - 1) / block.x, (half_height + 1 + block.y - 1) / block.y, 1};

                        cudaMalloc(&d_tmp_middle_row, width * sizeof(T));
                        cudaMalloc(&d_tmp_middle_col, height * sizeof(T));

                        fftshift(d_tmp_middle_row, d_src + half_height * width, width, 1, FFT2DShiftDim::FFT_WIDTH, stream);
                        cudaMemcpy2DAsync(d_tmp_middle_col, sizeof(T), d_src + half_width, width * sizeof(T), sizeof(T), height, cudaMemcpyDeviceToDevice, stream);
                        fftshift2D_kernel_odd_odd<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);
                        cudaStreamSynchronize(stream);

                        fftshift(d_tmp_middle_col, d_tmp_middle_col, 1, height, FFT2DShiftDim::FFT_HEIGHT, stream);
                        cudaMemcpy2DAsync(d_dst + width - 1, width * sizeof(T), d_tmp_middle_col, sizeof(T), sizeof(T), height, cudaMemcpyDeviceToDevice, stream);
                        cudaMemcpyAsync(d_dst + (height-1) * width, d_tmp_middle_row, width * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                        cudaStreamSynchronize(stream);
                        
                        cudaFree(d_tmp_middle_row);
                        cudaFree(d_tmp_middle_col);
                    }
                }
                else
                {
                    // Inplace, allocate temporary memory
                    T *d_tmp;
                    cudaMalloc(&d_tmp, width * height * sizeof(T));
                    cudaMemcpy(d_tmp, d_dst, width * height * sizeof(T), cudaMemcpyDeviceToDevice);
                    fftshift(d_dst, d_tmp, width, height, siftDim, stream);
                    cudaStreamSynchronize(stream);
                    cudaFree(d_tmp);
                }
                break;
            
            case FFT2DShiftDim::FFT_WIDTH:
                if (d_dst != d_src)
                {
                    if (width % 2 == 0)
                    {
                        block = {64, 4, 1};
                        grid = {(half_width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1};
                        fftshift2D_width_kernel_even<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);
                    }
                    else
                    {
                        block = {64, 4, 1};
                        grid = {(half_width + 1 + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1};
                        cudaMemcpy2DAsync(d_dst + width - 1, width * sizeof(T), d_src + half_width, width * sizeof(T), sizeof(T), height, cudaMemcpyDeviceToDevice, stream);
                        fftshift2D_width_kernel_odd<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);
                    }
                }
                else
                {
                    // Inplace, allocate temporary memory
                    T *d_tmp;
                    cudaMalloc(&d_tmp, width * height * sizeof(T));
                    cudaMemcpy(d_tmp, d_dst, width * height * sizeof(T), cudaMemcpyDeviceToDevice);
                    fftshift(d_dst, d_tmp, width, height, siftDim, stream);
                    cudaStreamSynchronize(stream);
                    cudaFree(d_tmp);
                }
                break;
            
            case FFT2DShiftDim::FFT_HEIGHT:
                if (d_dst != d_src)
                {
                    if (height % 2 == 0)
                    {
                        block = {64, 4, 1};
                        grid = {(width + block.x - 1) / block.x, (half_height + block.y - 1) / block.y, 1};
                        fftshift2D_height_kernel_even<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);
                    }
                    else
                    {
                        block = {64, 4, 1};
                        grid = {(width + block.x - 1) / block.x, (half_height + 1 + block.y - 1) / block.y, 1};
                        cudaMemcpyAsync(d_dst + (height-1) * width, d_src + half_height * width, width * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                        fftshift2D_height_kernel_odd<T><<<grid, block, 0, stream>>>(d_dst, d_src, width, height);
                    }
                }
                break;

            default:
                return cudaErrorInvalidValue;
        }
    }

    return cudaPeekAtLastError();
}



template cudaError_t fftshift<float>(float *d_dst, const float *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);
template cudaError_t fftshift<double>(double *d_dst, const double *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);
template cudaError_t fftshift<cuFloatComplex>(cuFloatComplex *d_dst, const cuFloatComplex *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);
template cudaError_t fftshift<cuDoubleComplex>(cuDoubleComplex *d_dst, const cuDoubleComplex *d_src, unsigned int width, unsigned int height, FFT2DShiftDim shiftDim, cudaStream_t stream);

