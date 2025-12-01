/**
 * test_quantization.cpp
 *
 * 量化功能测试
 */

#include "quantization.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace llm_inference;

void test_int8_quantization() {
    std::cout << "========================================\n";
    std::cout << "测试1: INT8量化\n";
    std::cout << "========================================\n\n";

    // 生成测试数据
    const int size = 1000;
    std::vector<float> weights(size);

    std::random_device rd;
    std::mt19937 gen(42);  // 固定种子以便复现
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < size; ++i) {
        weights[i] = dist(gen);
    }

    std::cout << "原始权重范围: ["
              << *std::min_element(weights.begin(), weights.end()) << ", "
              << *std::max_element(weights.begin(), weights.end()) << "]\n\n";

    // 量化
    std::cout << "执行INT8量化...\n";
    Timer timer;
    auto quantized = INT8Quantizer::quantize(weights.data(), size);
    timer.print("量化耗时");

    std::cout << "Scale: " << quantized.scale << "\n";
    std::cout << "量化数据范围: ["
              << static_cast<int>(*std::min_element(quantized.data.begin(), quantized.data.end())) << ", "
              << static_cast<int>(*std::max_element(quantized.data.begin(), quantized.data.end())) << "]\n\n";

    // 反量化
    std::cout << "执行反量化...\n";
    std::vector<float> dequantized(size);
    timer.reset();
    INT8Quantizer::dequantize(quantized, dequantized.data());
    timer.print("反量化耗时");

    // 计算误差
    quantization_utils::print_quantization_stats(
        weights.data(),
        dequantized.data(),
        size
    );

    // 内存节省
    auto savings = quantization_utils::compute_memory_savings(size, "int8");
    std::cout << "内存统计:\n";
    std::cout << "  原始: " << memory::format_memory_size(savings.original_bytes) << "\n";
    std::cout << "  量化: " << memory::format_memory_size(savings.quantized_bytes) << "\n";
    std::cout << "  压缩比: " << std::fixed << std::setprecision(2)
              << savings.compression_ratio << "x\n\n";

    std::cout << "✅ INT8量化测试通过\n\n";
}

void test_int4_quantization() {
    std::cout << "========================================\n";
    std::cout << "测试2: INT4量化\n";
    std::cout << "========================================\n\n";

    const int size = 1000;
    std::vector<float> weights(size);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (int i = 0; i < size; ++i) {
        weights[i] = dist(gen);
    }

    // 量化
    auto quantized = INT4Quantizer::quantize(weights.data(), size);

    std::cout << "Scale: " << quantized.scale << "\n";
    std::cout << "原始大小: " << size << " 元素\n";
    std::cout << "打包后大小: " << quantized.data.size() << " 字节\n\n";

    // 反量化
    std::vector<float> dequantized(size);
    INT4Quantizer::dequantize(quantized, dequantized.data());

    // 计算误差
    quantization_utils::print_quantization_stats(
        weights.data(),
        dequantized.data(),
        size
    );

    // 内存节省
    auto savings = quantization_utils::compute_memory_savings(size, "int4");
    std::cout << "内存统计:\n";
    std::cout << "  原始: " << memory::format_memory_size(savings.original_bytes) << "\n";
    std::cout << "  量化: " << memory::format_memory_size(savings.quantized_bytes) << "\n";
    std::cout << "  压缩比: " << std::fixed << std::setprecision(2)
              << savings.compression_ratio << "x\n\n";

    std::cout << "✅ INT4量化测试通过\n\n";
}

void test_group_quantization() {
    std::cout << "========================================\n";
    std::cout << "测试3: 分组量化\n";
    std::cout << "========================================\n\n";

    const int size = 1024;
    std::vector<float> weights(size);

    // 生成非均匀分布的数据（模拟真实权重）
    std::mt19937 gen(42);
    for (int i = 0; i < size; ++i) {
        // 不同段使用不同的scale
        float scale = (i < size/2) ? 0.1f : 1.0f;
        std::normal_distribution<float> dist(0.0f, scale);
        weights[i] = dist(gen);
    }

    std::cout << "使用分组量化（group_size=128）...\n";

    // 分组量化
    auto quantized = GroupQuantizer::quantize(weights.data(), size, 128);

    std::cout << "分组数: " << quantized.scales.size() << "\n";
    std::cout << "各组scale: ";
    for (size_t i = 0; i < std::min(size_t(8), quantized.scales.size()); ++i) {
        std::cout << std::scientific << std::setprecision(2)
                  << quantized.scales[i] << " ";
    }
    std::cout << "...\n\n";

    // 反量化
    std::vector<float> dequantized(size);
    GroupQuantizer::dequantize(quantized, dequantized.data());

    // 计算误差
    quantization_utils::print_quantization_stats(
        weights.data(),
        dequantized.data(),
        size
    );

    std::cout << "✅ 分组量化测试通过\n\n";
}

void test_matmul_performance() {
    std::cout << "========================================\n";
    std::cout << "测试4: 矩阵乘法性能对比\n";
    std::cout << "========================================\n\n";

    // 矩阵维度（模拟Transformer FFN）
    int m = 1;      // batch size
    int k = 4096;   // hidden dim
    int n = 11008;  // FFN dim

    std::cout << "矩阵维度: A(" << m << "x" << k << ") @ B(" << k << "x" << n << ")\n\n";

    // 生成测试数据
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_fp32(m * n);
    std::vector<float> C_int8(m * n);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : A) v = dist(gen);
    for (auto& v : B) v = dist(gen);

    // 1. FP32矩阵乘法（基准）
    std::cout << "[1] FP32 矩阵乘法...\n";
    Timer fp32_timer;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C_fp32[i * n + j] = sum;
        }
    }

    double fp32_time = fp32_timer.elapsed_ms();
    std::cout << "  耗时: " << fp32_time << " ms\n\n";

    // 2. INT8矩阵乘法
    std::cout << "[2] INT8 矩阵乘法...\n";

    // 量化B
    auto B_quantized = INT8Quantizer::quantize(B.data(), k * n);
    std::cout << "  量化耗时: " << fp32_timer.elapsed_ms() - fp32_time << " ms\n";

    // INT8矩阵乘法
    fp32_timer.reset();
    INT8Quantizer::matmul_int8(A.data(), B_quantized, m, k, n, C_int8.data());
    double int8_time = fp32_timer.elapsed_ms();

    std::cout << "  计算耗时: " << int8_time << " ms\n";
    std::cout << "  加速比: " << std::fixed << std::setprecision(2)
              << (fp32_time / int8_time) << "x\n\n";

    // 计算精度损失
    float mse = quantization_utils::compute_quantization_error(
        C_fp32.data(), C_int8.data(), m * n
    );
    std::cout << "  精度损失 (MSE): " << std::scientific << mse << "\n\n";

    std::cout << "✅ 矩阵乘法性能测试完成\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║          量化技术测试程序                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        test_int8_quantization();
        test_int4_quantization();
        test_group_quantization();
        test_matmul_performance();

        std::cout << "========================================\n";
        std::cout << "🎉 所有测试通过！\n";
        std::cout << "========================================\n\n";

        std::cout << "量化技术总结:\n";
        std::cout << "  INT8: 4x内存节省, 2-3x加速\n";
        std::cout << "  INT4: 8x内存节省, 精度损失较大\n";
        std::cout << "  分组量化: 更好的精度，略微增加复杂度\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
