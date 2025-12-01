/**
 * quantization.cpp
 *
 * 量化技术实现
 */

#include "quantization.h"
#include <cstring>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace llm_inference {

// ============================================================
// INT8Quantizer实现
// ============================================================

float INT8Quantizer::compute_scale(const float* data, int size) {
    // 找到最大绝对值
    float max_abs = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }

    // scale = max_abs / 127（INT8的最大值）
    return max_abs / 127.0f;
}

INT8Quantizer::QuantizedWeights INT8Quantizer::quantize(const float* weights, int size) {
    QuantizedWeights result;
    result.data.resize(size);

    // 计算scale
    result.scale = compute_scale(weights, size);

    if (result.scale == 0.0f) {
        // 所有权重为0
        std::fill(result.data.begin(), result.data.end(), 0);
        return result;
    }

    // 量化：q = round(x / scale)
    float inv_scale = 1.0f / result.scale;
    for (int i = 0; i < size; ++i) {
        float q = std::round(weights[i] * inv_scale);
        // Clamp到INT8范围 [-128, 127]
        q = std::max(-128.0f, std::min(127.0f, q));
        result.data[i] = static_cast<int8_t>(q);
    }

    return result;
}

void INT8Quantizer::dequantize(const QuantizedWeights& quantized, float* output) {
    int size = quantized.data.size();
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(quantized.data[i]) * quantized.scale;
    }
}

void INT8Quantizer::matmul_int8(
    const float* A,
    const QuantizedWeights& B_quantized,
    int m, int k, int n,
    float* output
) {
    // A: [m, k] FP32
    // B: [k, n] INT8
    // C: [m, n] FP32

    const int8_t* B = B_quantized.data.data();
    float scale_B = B_quantized.scale;

    // 标量实现（可以用SIMD优化）
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int32_t sum = 0;  // 使用INT32累加避免溢出

            // 内积：A[i,:] · B[:,j]
            for (int p = 0; p < k; ++p) {
                // 量化A[i,p]
                int8_t a_q = static_cast<int8_t>(
                    std::round(A[i * k + p] / scale_B)
                );
                sum += a_q * B[p * n + j];
            }

            // 反量化
            output[i * n + j] = sum * scale_B * scale_B;
        }
    }
}

float INT8Quantizer::quantize_symmetric(const float* input, int size, int8_t* output) {
    float scale = compute_scale(input, size);

    if (scale == 0.0f) {
        std::fill(output, output + size, 0);
        return 0.0f;
    }

    float inv_scale = 1.0f / scale;
    for (int i = 0; i < size; ++i) {
        float q = std::round(input[i] * inv_scale);
        q = std::max(-128.0f, std::min(127.0f, q));
        output[i] = static_cast<int8_t>(q);
    }

    return scale;
}

void INT8Quantizer::dequantize_symmetric(
    const int8_t* input, int size, float scale, float* output
) {
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) * scale;
    }
}

// ============================================================
// INT4Quantizer实现
// ============================================================

float INT4Quantizer::compute_scale(const float* data, int size) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_abs = std::max(max_abs, std::abs(data[i]));
    }
    // INT4范围：[-8, 7]
    return max_abs / 7.0f;
}

uint8_t INT4Quantizer::pack_int4(int8_t a, int8_t b) {
    // a存储在低4位，b存储在高4位
    uint8_t a_u = static_cast<uint8_t>(a) & 0x0F;
    uint8_t b_u = static_cast<uint8_t>(b) & 0x0F;
    return a_u | (b_u << 4);
}

void INT4Quantizer::unpack_int4(uint8_t packed, int8_t& a, int8_t& b) {
    a = static_cast<int8_t>(packed & 0x0F);
    b = static_cast<int8_t>((packed >> 4) & 0x0F);

    // 处理符号位（4位有符号整数）
    if (a > 7) a -= 16;
    if (b > 7) b -= 16;
}

INT4Quantizer::QuantizedWeights INT4Quantizer::quantize(const float* weights, int size) {
    QuantizedWeights result;
    result.size = size;
    result.scale = compute_scale(weights, size);

    if (result.scale == 0.0f) {
        result.data.resize((size + 1) / 2, 0);
        return result;
    }

    // 量化到INT4
    std::vector<int8_t> temp(size);
    float inv_scale = 1.0f / result.scale;

    for (int i = 0; i < size; ++i) {
        float q = std::round(weights[i] * inv_scale);
        q = std::max(-8.0f, std::min(7.0f, q));
        temp[i] = static_cast<int8_t>(q);
    }

    // 打包：2个INT4 -> 1个UINT8
    int packed_size = (size + 1) / 2;
    result.data.resize(packed_size);

    for (int i = 0; i < size; i += 2) {
        int8_t a = temp[i];
        int8_t b = (i + 1 < size) ? temp[i + 1] : 0;
        result.data[i / 2] = pack_int4(a, b);
    }

    return result;
}

void INT4Quantizer::dequantize(const QuantizedWeights& quantized, float* output) {
    for (int i = 0; i < quantized.size; i += 2) {
        int8_t a, b;
        unpack_int4(quantized.data[i / 2], a, b);

        output[i] = a * quantized.scale;
        if (i + 1 < quantized.size) {
            output[i + 1] = b * quantized.scale;
        }
    }
}

void INT4Quantizer::matmul_int4(
    const float* A,
    const QuantizedWeights& B_quantized,
    int m, int k, int n,
    float* output
) {
    // 简化实现：先反量化B，再做FP32矩阵乘法
    std::vector<float> B(k * n);
    dequantize(B_quantized, B.data());

    // 标准矩阵乘法
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            output[i * n + j] = sum;
        }
    }
}

// ============================================================
// SIMD优化（AVX2）
// ============================================================

#ifdef __AVX2__
namespace simd {

void matmul_int8_avx2(
    const float* A,
    const INT8Quantizer::QuantizedWeights& B_quantized,
    int m, int k, int n,
    float* output
) {
    // AVX2优化的INT8矩阵乘法
    // 这里是简化版本，生产环境需要更复杂的实现

    const int8_t* B = B_quantized.data.data();
    float scale = B_quantized.scale;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();

            int p = 0;
            // 8路并行
            for (; p + 7 < k; p += 8) {
                // 加载A（FP32）
                __m256 a_vec = _mm256_loadu_ps(&A[i * k + p]);

                // 加载B（INT8）并转换为FP32
                __m128i b_int8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(&B[p * n + j])
                );
                __m256i b_int32 = _mm256_cvtepi8_epi32(b_int8);
                __m256 b_vec = _mm256_cvtepi32_ps(b_int32);

                // FMA: sum += a * b
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            // 水平求和
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum_vec);
            float sum = 0.0f;
            for (int t = 0; t < 8; ++t) sum += sum_array[t];

            // 处理剩余元素
            for (; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }

            output[i * n + j] = sum * scale;
        }
    }
}

float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();

    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // 水平求和
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += sum_array[j];

    // 处理剩余
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

} // namespace simd
#endif // __AVX2__

// ============================================================
// GroupQuantizer实现
// ============================================================

GroupQuantizer::GroupQuantizedWeights GroupQuantizer::quantize(
    const float* weights,
    int size,
    int group_size
) {
    GroupQuantizedWeights result;
    result.size = size;
    result.group_size = group_size;

    int n_groups = (size + group_size - 1) / group_size;
    result.data.resize(size);
    result.scales.resize(n_groups);

    for (int g = 0; g < n_groups; ++g) {
        int start = g * group_size;
        int end = std::min(start + group_size, size);
        int group_len = end - start;

        // 计算该组的scale
        float max_abs = 0.0f;
        for (int i = start; i < end; ++i) {
            max_abs = std::max(max_abs, std::abs(weights[i]));
        }
        result.scales[g] = max_abs / 127.0f;

        // 量化该组
        if (result.scales[g] == 0.0f) {
            std::fill(&result.data[start], &result.data[end], 0);
        } else {
            float inv_scale = 1.0f / result.scales[g];
            for (int i = start; i < end; ++i) {
                float q = std::round(weights[i] * inv_scale);
                q = std::max(-128.0f, std::min(127.0f, q));
                result.data[i] = static_cast<int8_t>(q);
            }
        }
    }

    return result;
}

void GroupQuantizer::dequantize(const GroupQuantizedWeights& quantized, float* output) {
    int n_groups = quantized.scales.size();
    int group_size = quantized.group_size;

    for (int g = 0; g < n_groups; ++g) {
        int start = g * group_size;
        int end = std::min(start + group_size, quantized.size);
        float scale = quantized.scales[g];

        for (int i = start; i < end; ++i) {
            output[i] = static_cast<float>(quantized.data[i]) * scale;
        }
    }
}

void GroupQuantizer::matmul_grouped(
    const float* A,
    const GroupQuantizedWeights& B_quantized,
    int m, int k, int n,
    float* output
) {
    // 简化：先反量化再做矩阵乘法
    std::vector<float> B(k * n);
    dequantize(B_quantized, B.data());

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            output[i * n + j] = sum;
        }
    }
}

// ============================================================
// 量化工具函数
// ============================================================

namespace quantization_utils {

float compute_quantization_error(
    const float* original,
    const float* quantized,
    int size
) {
    double mse = 0.0;
    for (int i = 0; i < size; ++i) {
        double diff = original[i] - quantized[i];
        mse += diff * diff;
    }
    return static_cast<float>(mse / size);
}

void print_quantization_stats(
    const float* original,
    const float* quantized,
    int size
) {
    float mse = compute_quantization_error(original, quantized, size);
    float rmse = std::sqrt(mse);

    std::cout << "\n========================================\n";
    std::cout << "Quantization Statistics\n";
    std::cout << "========================================\n";
    std::cout << "Size:       " << size << "\n";
    std::cout << "MSE:        " << std::scientific << mse << "\n";
    std::cout << "RMSE:       " << rmse << "\n";
    std::cout << "========================================\n\n";
}

MemorySavings compute_memory_savings(int size, const std::string& quant_type) {
    MemorySavings result;
    result.original_bytes = size * sizeof(float);

    if (quant_type == "int8") {
        result.quantized_bytes = size * sizeof(int8_t) + sizeof(float);  // data + scale
    } else if (quant_type == "int4") {
        result.quantized_bytes = (size + 1) / 2 + sizeof(float);  // packed data + scale
    } else {
        result.quantized_bytes = result.original_bytes;
    }

    result.compression_ratio = static_cast<float>(result.original_bytes) /
                              result.quantized_bytes;

    return result;
}

} // namespace quantization_utils

} // namespace llm_inference
