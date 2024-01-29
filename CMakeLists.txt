cmake_minimum_required(VERSION 3.15)

project(utils_lib C CXX)

set(CMAKE_CXX_STANDARD 17)

add_definitions(-Wno-format-security -Wno-long-long)

add_subdirectory(cpp)

if (CUDA_FOUND)

add_subdirectory(cuda)

endif(CUDA_FOUND)

if(enable_main)

add_executable(main main.cpp)

target_link_libraries(main utils_lib_cxx utils_lib_cuda)

endif(enable_main)