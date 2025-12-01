/**
 * utils.cpp
 *
 * 工具函数实现
 */

#include "utils.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace llm_inference {

// ============================================================
// 内存管理
// ============================================================

namespace memory {

double get_memory_usage_mb() {
#ifdef __linux__
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (pages * page_size) / (1024.0 * 1024.0);
#else
    return 0.0;  // 其他平台暂不支持
#endif
}

void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
#endif
    return ptr;
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

std::string format_memory_size(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
}

} // namespace memory

// ============================================================
// 数学函数
// ============================================================

namespace math {

void softmax(const float* input, int size, float* output) {
    // 找到最大值（数值稳定性）
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    // 计算exp和sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // 归一化
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }
}

void relu(const float* input, int size, float* output) {
    for (int i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void gelu(const float* input, int size, float* output) {
    // GELU(x) = x * Φ(x)
    // 近似：GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float coeff = 0.044715f;

    for (int i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void layernorm(
    const float* input,
    int size,
    const float* gamma,
    const float* beta,
    float* output,
    float eps
) {
    // 计算均值
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += input[i];
    }
    mean /= size;

    // 计算方差
    float variance = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    // 归一化
    float std_dev = std::sqrt(variance + eps);
    for (int i = 0; i < size; ++i) {
        float normalized = (input[i] - mean) / std_dev;
        output[i] = gamma[i] * normalized + beta[i];
    }
}

void transpose(const float* input, int rows, int cols, float* output) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void vector_add(const float* a, const float* b, int n, float* output) {
    for (int i = 0; i < n; ++i) {
        output[i] = a[i] + b[i];
    }
}

void scalar_multiply(const float* input, float scalar, int n, float* output) {
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * scalar;
    }
}

} // namespace math

// ============================================================
// 字符串工具
// ============================================================

namespace string_utils {

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";

    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

void print_progress_bar(int current, int total, int bar_width) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(bar_width * progress);

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% "
              << current << "/" << total << "\r";
    std::cout.flush();

    if (current == total) {
        std::cout << std::endl;
    }
}

} // namespace string_utils

} // namespace llm_inference
