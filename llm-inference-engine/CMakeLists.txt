cmake_minimum_required(VERSION 3.15)
project(llm_inference CXX)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# 检测AVX2支持
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    add_definitions(-D__AVX2__)
    message(STATUS "AVX2 support enabled")
else()
    message(STATUS "AVX2 not supported")
endif()

# 包含目录
include_directories(${PROJECT_SOURCE_DIR}/cpp/include)

# 源文件
set(SOURCES
    cpp/src/kv_cache.cpp
    cpp/src/quantization.cpp
    cpp/src/utils.cpp
)

# 创建静态库
add_library(llm_inference_static STATIC ${SOURCES})

# 创建共享库
add_library(llm_inference_shared SHARED ${SOURCES})

# 设置输出名称
set_target_properties(llm_inference_static PROPERTIES OUTPUT_NAME llm_inference)
set_target_properties(llm_inference_shared PROPERTIES OUTPUT_NAME llm_inference)

# ============================================================
# 测试程序
# ============================================================

# KV Cache测试
add_executable(test_kv_cache cpp/tests/test_kv_cache.cpp)
target_link_libraries(test_kv_cache llm_inference_static)

# 量化测试
add_executable(test_quantization cpp/tests/test_quantization.cpp)
target_link_libraries(test_quantization llm_inference_static)

# 性能基准测试
add_executable(benchmark cpp/tests/benchmark.cpp)
target_link_libraries(benchmark llm_inference_static)

# ============================================================
# Python绑定（pybind11）
# ============================================================

# 查找pybind11
find_package(pybind11 CONFIG)

if(pybind11_FOUND)
    message(STATUS "pybind11 found, building Python bindings")

    pybind11_add_module(llm_inference_py cpp/src/bindings.cpp ${SOURCES})

    # 设置输出目录
    set_target_properties(llm_inference_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python
    )
else()
    message(STATUS "pybind11 not found, skipping Python bindings")
    message(STATUS "Install with: pip install pybind11")
endif()

# ============================================================
# 安装
# ============================================================

install(TARGETS llm_inference_static llm_inference_shared
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY cpp/include/
    DESTINATION include/llm_inference
    FILES_MATCHING PATTERN "*.h"
)

# ============================================================
# 打印配置信息
# ============================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "LLM Inference Engine Configuration")
message(STATUS "========================================")
message(STATUS "C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags:       ${CMAKE_CXX_FLAGS}")
message(STATUS "AVX2:            ${COMPILER_SUPPORTS_AVX2}")
message(STATUS "pybind11:        ${pybind11_FOUND}")
message(STATUS "========================================")
message(STATUS "")
