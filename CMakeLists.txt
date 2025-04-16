cmake_minimum_required(VERSION 3.10)

project(aligned_vs_unaligned_memory_access LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sum main.cpp)

if (MSVC)
    target_compile_options(sum PRIVATE /arch:AVX)
else()
    target_compile_options(sum PRIVATE -mavx)
endif()