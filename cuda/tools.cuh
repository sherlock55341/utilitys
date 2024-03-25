#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace cuda::kernels {
template <typename T, typename U>
__global__ void fill(T *arr, U val, size_t N) {
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        arr[idx] = val;
    }
}
} // namespace cuda::kernels
namespace cuda {

template <typename T>
__forceinline__ __device__ __host__ void swap(T &x, T &y) {
    T t(x);
    x = y;
    y = t;
}

template <typename T, typename U>
void fill(T *arr, U val, size_t N, int THREADS_PER_BLOCK) {
    kernels::fill<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK>>>(arr, val, N);
}
} // namespace cuda

UTILS_END
