#pragma once

#include "../global.hpp"

UTILS_BEGIN

namespace cuda {

void mallocDevice(void** ptr, size_t size);

void freeDevice(void* ptr);

void sync();

void h2d(void* dst, void* src, size_t size);

void d2h(void* dst, void* src, size_t size);

void err();
}  // namespace cuda

UTILS_END