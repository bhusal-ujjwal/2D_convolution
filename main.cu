#include <iostream>
#include <png.h>
#include "utils/pngio.h"
#include "cuda_runtime.h"

#define CUDA_CHECK_RETURN(value) \
    { cudaError_t err = value; \
      if (err != cudaSuccess) { \
        fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(1); \
      } }

#define FILTER_SIZE 3
#define BLOCK_SIZE 16
#define TAIL_SIZE (BLOCK_SIZE - (FILTER_SIZE / 2) * 2)
#define SHARED_MEM_SIZE ((BLOCK_SIZE + FILTER_SIZE - 1) * (BLOCK_SIZE + FILTER_SIZE - 1))

__constant__ float CONVOLUTION_MASKS[3][FILTER_SIZE * FILTER_SIZE] = {
    {-1, -1, -1, -1, 8, -1, -1, -1, -1},             // Ridge
    {1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9}, // Normalized Box Blur
    {0, -1, 0, -1, 5, -1, 0, -1, 0}                   // Sharpen
};

__global__ void applyConvolution(unsigned char *output, const unsigned char *input, int width, int height, int maskIndex) {
    __shared__ unsigned char sharedMem[BLOCK_SIZE + FILTER_SIZE - 1][BLOCK_SIZE + FILTER_SIZE - 1];

    int x = threadIdx.x + blockIdx.x * TAIL_SIZE;
    int y = threadIdx.y + blockIdx.y * TAIL_SIZE;
    int sharedX = threadIdx.x + FILTER_SIZE / 2;
    int sharedY = threadIdx.y + FILTER_SIZE / 2;

    if (x < width && y < height) {
        int offset = FILTER_SIZE / 2;
        float result[3] = {0.0f, 0.0f, 0.0f};

        for (int c = 0; c < 3; ++c) {
            for (int i = -offset; i <= offset; ++i)
                for (int j = -offset; j <= offset; ++j)
                    sharedMem[sharedY + j][sharedX + i] = input[(((y + j) * width) + (x + i)) * 3 + c];

            __syncthreads();

            for (int i = -offset; i <= offset; ++i)
                for (int j = -offset; j <= offset; ++j)
                    result[c] += sharedMem[sharedY + j][sharedX + i] * CONVOLUTION_MASKS[maskIndex][j + offset + (i + offset) * FILTER_SIZE];

            __syncthreads();
        }

        if (x < width && y < height) {
            for (int c = 0; c < 3; ++c) {
                output[((y * width + x) * 3) + c] = static_cast<unsigned char>(result[c]);
            }
        }
    }
}

int main() {
    png::image<png::rgb_pixel> img("../lenna.png");
    unsigned int width = img.get_width();
    unsigned int height = img.get_height();
    unsigned int size = width * height * 3 * sizeof(unsigned char);

    unsigned char *h_data = new unsigned char[size];
    pvg::pngToRgb(h_data, img);

    unsigned char *d_input, *d_output;

    CUDA_CHECK_RETURN(cudaMalloc(&d_input, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, size));

    cudaMemcpy(d_input, h_data, size, cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((width + TAIL_SIZE - 1) / TAIL_SIZE, (height + TAIL_SIZE - 1) / TAIL_SIZE);

    for (int i = 0; i < 3; ++i) {
        applyConvolution<<<grid_dim, block_dim>>>(d_output, d_input, width, height, i);
        cudaDeviceSynchronize();
        CUDA_CHECK_RETURN(cudaGetLastError());

        cudaMemcpy(h_data, d_output, size, cudaMemcpyDeviceToHost);
        pvg::rgbToPng(img, h_data);
        img.write("../lenna_new_" + std::to_string(i) + ".png");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_data;

    return 0;
}
