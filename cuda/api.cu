#include "api.hpp"

UTILS_BEGIN

void cuda::mallocDevice(void **ptr, size_t size){
    cudaMalloc(ptr, size);
}

void cuda::freeDevice(void *ptr){
    cudaFree(ptr);
}

void cuda::sync(){
    cudaDeviceSynchronize();
}

void cuda::h2d(void *dst, void *src, size_t size){
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda::d2h(void *dst, void *src, size_t size){
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void cuda::err(){
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        assert(err == cudaSuccess);
        exit(1);
    }
}

UTILS_END