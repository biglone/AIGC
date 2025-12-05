/**
 * utils.h
 *
 * 通用工具函数
 */

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace llm_inference {

/**
 * 计时器 - 性能测试工具
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    // 重置计时器
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    // 获取经过的时间（毫秒）
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    // 获取经过的时间（秒）
    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }

    // 打印经过的时间
    void print(const std::string& label = "") const {
        if (!label.empty()) {
            std::cout << label << ": ";
        }
        std::cout << elapsed_ms() << " ms" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * 性能统计器
 */
class PerformanceStats {
public:
    void add_sample(double value) {
        samples_.push_back(value);
        sum_ += value;
        count_++;
    }

    double mean() const {
        return count_ > 0 ? sum_ / count_ : 0.0;
    }

    double min() const {
        if (samples_.empty()) return 0.0;
        return *std::min_element(samples_.begin(), samples_.end());
    }

    double max() const {
        if (samples_.empty()) return 0.0;
        return *std::max_element(samples_.begin(), samples_.end());
    }

    void print(const std::string& name) const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << name << " Statistics:\n";
        std::cout << "  Count: " << count_ << "\n";
        std::cout << "  Mean:  " << mean() << " ms\n";
        std::cout << "  Min:   " << min() << " ms\n";
        std::cout << "  Max:   " << max() << " ms\n";
    }

    void reset() {
        samples_.clear();
        sum_ = 0.0;
        count_ = 0;
    }

private:
    std::vector<double> samples_;
    double sum_ = 0.0;
    int count_ = 0;
};

/**
 * 内存管理工具
 */
namespace memory {

// 获取当前进程内存占用（MB）
double get_memory_usage_mb();

// 对齐内存分配（用于SIMD）
void* aligned_malloc(size_t size, size_t alignment = 64);
void aligned_free(void* ptr);

// 格式化内存大小
std::string format_memory_size(size_t bytes);

} // namespace memory

/**
 * 数学工具函数
 */
namespace math {

// Softmax
void softmax(const float* input, int size, float* output);

// ReLU
void relu(const float* input, int size, float* output);

// GELU (Gaussian Error Linear Unit)
void gelu(const float* input, int size, float* output);

// LayerNorm
void layernorm(
    const float* input,
    int size,
    const float* gamma,
    const float* beta,
    float* output,
    float eps = 1e-5f
);

// 矩阵转置
void transpose(
    const float* input,
    int rows, int cols,
    float* output
);

// 向量点积
float dot_product(const float* a, const float* b, int n);

// 向量加法
void vector_add(const float* a, const float* b, int n, float* output);

// 标量乘法
void scalar_multiply(const float* input, float scalar, int n, float* output);

} // namespace math

/**
 * 字符串工具
 */
namespace string_utils {

// 分割字符串
std::vector<std::string> split(const std::string& s, char delimiter);

// 去除首尾空格
std::string trim(const std::string& s);

// 格式化输出进度条
void print_progress_bar(int current, int total, int bar_width = 50);

} // namespace string_utils

/**
 * 日志工具
 */
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { level_ = level; }

    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }

private:
    Logger() : level_(LogLevel::INFO) {}

    void log(LogLevel level, const std::string& msg) {
        if (level < level_) return;

        const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
        std::cout << "[" << level_str[static_cast<int>(level)] << "] "
                  << msg << std::endl;
    }

    LogLevel level_;
};

// 便捷的日志宏
#define LOG_DEBUG(msg) Logger::instance().debug(msg)
#define LOG_INFO(msg) Logger::instance().info(msg)
#define LOG_WARNING(msg) Logger::instance().warning(msg)
#define LOG_ERROR(msg) Logger::instance().error(msg)

/**
 * 断言工具
 */
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << #condition << "\n" \
                      << "Message: " << message << "\n" \
                      << "File: " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort(); \
        } \
    } while (false)

/**
 * 性能分析宏
 */
#define PROFILE_SCOPE(name) \
    Timer _timer_##__LINE__; \
    struct _ProfileScope##__LINE__ { \
        Timer& timer; \
        const char* name; \
        _ProfileScope##__LINE__(Timer& t, const char* n) : timer(t), name(n) {} \
        ~_ProfileScope##__LINE__() { timer.print(name); } \
    } _profile_##__LINE__(_timer_##__LINE__, name)

} // namespace llm_inference

#endif // UTILS_H
