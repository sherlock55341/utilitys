/**
 * @file api.cu
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-05-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "api.hpp"

namespace utils::cuda {
void warmUpDevice(int deviceIdx){
    cudaSetDevice(deviceIdx);
    cudaFree(0);
}

void setDevice(int deviceIdx) { cudaSetDevice(deviceIdx); }

void mallocDevice(void **ptr, size_t size) { cudaMalloc(ptr, size); }

void freeDevice(void *ptr) { cudaFree(ptr); }

void sync() { cudaDeviceSynchronize(); }

void h2d(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void d2h(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void d2d(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

void err(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "[CUDA ERROR] : FILE [" << file << "] LINE[" << line
                  << "] " << cudaGetErrorString(err) << std::endl;
        assert(err == cudaSuccess);
        exit(1);
    }
}
} // namespace utils::cuda
