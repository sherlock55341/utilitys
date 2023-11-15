#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace cuda {

template <typename T> 
__forceinline__ __device__ __host__ void swap(T &x, T &y) {
    T t(x);
    x = y;
    y = t;
}

__forceinline__ __device__ size_t naiveIdx(){
    return threadIdx.x + blockIdx.x * blockDim.x;
}
} // namespace cuda

UTILS_END
