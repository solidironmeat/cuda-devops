#include "operations/operations.h"

__global__ void add_kernel(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_elements) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

__global__ void sub_kernel(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements) {
        d_c[i] = d_a[i] - d_b[i];
    }
}

__global__ void mul_kernel(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements) {
        d_c[i] = d_a[i] * d_b[i];
    }
}

__global__ void mod_kernel(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements) {
        d_c[i] = fmodf(d_a[i], d_b[i]);
    }
}

__global__ void div_kernel(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements) {
        d_c[i] = d_a[i] / d_b[i];
    }
}

__global__ void pow_kernel(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_elements) {
        d_c[i] = pow(d_a[i], d_b[i]);
    }
}

// Wrapper function to launch the kernel
void add(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    add_kernel << <1, num_elements >> > (d_a, d_b, d_c, num_elements);
    cudaDeviceSynchronize();
}

void sub(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    sub_kernel << <1, num_elements >> > (d_a, d_b, d_c, num_elements);
    cudaDeviceSynchronize();
}

void mul(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    mul_kernel << <1, num_elements >> > (d_a, d_b, d_c, num_elements);
    cudaDeviceSynchronize();
}

void mod(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    mod_kernel << <1, num_elements >> > (d_a, d_b, d_c, num_elements);
    cudaDeviceSynchronize();
}

void div(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    div_kernel << <1, num_elements >> > (d_a, d_b, d_c, num_elements);
    cudaDeviceSynchronize();
}

void pow(const float* d_a, const float* d_b, float* d_c, size_t num_elements) {
    pow_kernel << <1, num_elements >> > (d_a, d_b, d_c, num_elements);
    cudaDeviceSynchronize();
}