/**
 * quantization.h
 *
 * 量化技术实现 - 减少内存和加速计算
 *
 * 支持的量化方法：
 * - INT8量化（8位整数）：75%内存节省
 * - INT4量化（4位整数）：87.5%内存节省
 */

#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>

namespace llm_inference {

/**
 * INT8量化器
 *
 * 量化公式：
 *   q = round(x / scale)
 *   x_dequant = q * scale
 *
 * 其中 scale = max(|x|) / 127
 */
class INT8Quantizer {
public:
    /**
     * 量化FP32权重到INT8
     *
     * @param weights FP32权重
     * @param size 权重数量
     * @return 量化后的权重和scale
     */
    struct QuantizedWeights {
        std::vector<int8_t> data;
        float scale;
    };

    static QuantizedWeights quantize(const float* weights, int size);

    /**
     * 反量化INT8到FP32
     *
     * @param quantized 量化的权重
     * @param output 输出的FP32权重
     */
    static void dequantize(const QuantizedWeights& quantized, float* output);

    /**
     * INT8矩阵乘法（核心优化）
     *
     * C = A @ B
     * A: [m, k] FP32
     * B: [k, n] INT8（量化的权重）
     * C: [m, n] FP32
     *
     * @param A 输入矩阵（FP32）
     * @param B_quantized 量化的权重矩阵（INT8）
     * @param m, k, n 矩阵维度
     * @param output 输出矩阵（FP32）
     */
    static void matmul_int8(
        const float* A,
        const QuantizedWeights& B_quantized,
        int m, int k, int n,
        float* output
    );

    /**
     * 对称量化（用于激活值）
     *
     * @param input 输入FP32数组
     * @param size 数组大小
     * @param output 输出INT8数组
     * @return scale因子
     */
    static float quantize_symmetric(const float* input, int size, int8_t* output);

    /**
     * 对称反量化
     */
    static void dequantize_symmetric(const int8_t* input, int size, float scale, float* output);

private:
    // 计算最优scale
    static float compute_scale(const float* data, int size);
};

/**
 * INT4量化器（更激进的量化）
 *
 * 量化范围：[-8, 7]
 * 存储：2个INT4打包成1个INT8
 */
class INT4Quantizer {
public:
    struct QuantizedWeights {
        std::vector<uint8_t> data;  // 打包的INT4数据
        float scale;
        int size;  // 原始元素数量
    };

    /**
     * 量化到INT4
     */
    static QuantizedWeights quantize(const float* weights, int size);

    /**
     * 反量化
     */
    static void dequantize(const QuantizedWeights& quantized, float* output);

    /**
     * INT4矩阵乘法
     */
    static void matmul_int4(
        const float* A,
        const QuantizedWeights& B_quantized,
        int m, int k, int n,
        float* output
    );

private:
    // 打包2个INT4到1个UINT8
    static uint8_t pack_int4(int8_t a, int8_t b);

    // 解包UINT8到2个INT4
    static void unpack_int4(uint8_t packed, int8_t& a, int8_t& b);

    static float compute_scale(const float* data, int size);
};

/**
 * SIMD优化的INT8矩阵乘法
 *
 * 使用AVX2指令集加速（8路并行）
 * 要求：编译时添加 -mavx2
 */
#ifdef __AVX2__
namespace simd {

/**
 * AVX2优化的INT8 GEMM
 *
 * 性能提升：2-4x（相比标量实现）
 */
void matmul_int8_avx2(
    const float* A,
    const INT8Quantizer::QuantizedWeights& B_quantized,
    int m, int k, int n,
    float* output
);

/**
 * AVX2优化的向量点积
 */
float dot_product_avx2(const float* a, const float* b, int n);

} // namespace simd
#endif // __AVX2__

/**
 * 分组量化（Group Quantization）
 *
 * 将权重分组，每组使用独立的scale
 * 优势：更高的量化精度
 */
class GroupQuantizer {
public:
    static constexpr int GROUP_SIZE = 128;  // 每组的元素数

    struct GroupQuantizedWeights {
        std::vector<int8_t> data;
        std::vector<float> scales;  // 每组一个scale
        int size;
        int group_size;
    };

    /**
     * 分组量化
     */
    static GroupQuantizedWeights quantize(
        const float* weights,
        int size,
        int group_size = GROUP_SIZE
    );

    /**
     * 分组反量化
     */
    static void dequantize(const GroupQuantizedWeights& quantized, float* output);

    /**
     * 分组量化的矩阵乘法
     */
    static void matmul_grouped(
        const float* A,
        const GroupQuantizedWeights& B_quantized,
        int m, int k, int n,
        float* output
    );
};

/**
 * 量化工具函数
 */
namespace quantization_utils {

/**
 * 计算量化误差（MSE）
 */
float compute_quantization_error(
    const float* original,
    const float* quantized,
    int size
);

/**
 * 打印量化统计信息
 */
void print_quantization_stats(
    const float* original,
    const float* quantized,
    int size
);

/**
 * 内存节省计算
 */
struct MemorySavings {
    size_t original_bytes;
    size_t quantized_bytes;
    float compression_ratio;
};

MemorySavings compute_memory_savings(
    int size,
    const std::string& quant_type  // "int8" or "int4"
);

} // namespace quantization_utils

} // namespace llm_inference

#endif // QUANTIZATION_H
