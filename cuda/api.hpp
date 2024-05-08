/**
 * @file api.hpp
 * @author Chunyuan Zhao (zhaochunyuan@stu.pku.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2024-05-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace cuda {
/**
 * @brief Warm up the deviceIdx-th device 
 * 
 * @param deviceIdx 
 */
void warmUpDevice(int deviceIdx);
/**
 * @brief Set the device to use
 * 
 * @param deviceIdx 
 */
void setDevice(int deviceIdx);
/**
 * @brief Allocates size bytes of uninitialized storage on device
 * 
 * @param ptr pointer
 * @param size 
 */
void mallocDevice(void** ptr, size_t size);
/**
 * @brief Deallocates the space on device
 * 
 * @param ptr 
 */
void freeDevice(void* ptr);
/**
 * @brief Blocks until the device has completedall preceding requested tasks
 * 
 */
void sync();
/**
 * @brief Copies size bytes from src to dst, host to device
 * 
 * @param dst 
 * @param src 
 * @param size 
 */
void h2d(void* dst, void* src, size_t size);
/**
 * @brief Copies size bytes from src to dst, device to host
 * 
 * @param dst 
 * @param src 
 * @param size 
 */
void d2h(void* dst, void* src, size_t size);
/**
 * @brief Copies size bytes from src to dst, host to host
 * 
 * @param dst 
 * @param src 
 * @param size 
 */
void d2d(void *dst, void *src, size_t size);
/**
 * @brief Check if there is some error and report filename and line
 * 
 * @param file filename
 * @param line line
 */
void err(const char* file, int line);
}  // namespace cuda

UTILS_END