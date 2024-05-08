/**
 * @file data.hpp
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief
 * @version 0.1
 * @date 2024-05-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "api.hpp"

UTILS_BEGIN

namespace cuda::Transfer {
/**
 * @brief copy num of elements to device
 *
 * @tparam T
 * @param ptr pointer on (host-before, device-after)
 * @param num number of elements
 */
template <typename T> void cuda(T *&ptr, size_t num) {
    T *tmpPtr = nullptr;
    mallocDevice((void **)&tmpPtr, sizeof(T) * num);
    h2d(tmpPtr, ptr, sizeof(T) * num);
    free(ptr);
    ptr = tmpPtr;
}
/**
 * @brief copy num of elements to host
 *
 * @tparam T
 * @param ptr pointer on (device-before, host-after)
 * @param num number of elements
 */
template <typename T> void cpu(T *&ptr, size_t num) {
    T *tmpPtr = nullptr;
    tmpPtr = (T *)malloc(sizeof(T) * num);
    d2h(tmpPtr, ptr, sizeof(T) * num);
    freeDevice(ptr);
    ptr = tmpPtr;
}
/**
 * @brief copy num of elements from src to dst
 *
 * @tparam T
 * @param dst pointer on device
 * @param src pointer on host
 * @param num number of elements
 */
template <typename T> void cuda(T *&dst, T *src, size_t num) {
    mallocDevice((void **)&dst, sizeof(T) * num);
    h2d(dst, src, sizeof(T) * num);
}
/**
 * @brief copy num of elements from src to dst
 * 
 * @tparam T 
 * @param dst pointer on host
 * @param src pointer on dst
 * @param num number of elements
 */
template <typename T> void cpu(T *&dst, T *src, size_t num) {
    dst = (T *)malloc(sizeof(T) * num);
    d2h(dst, src, sizeof(T) * num);
}

} // namespace cuda::Transfer

UTILS_END