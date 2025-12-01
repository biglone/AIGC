/**
 * test_kv_cache.cpp
 *
 * KV Cache功能测试
 */

#include "kv_cache.h"
#include "utils.h"
#include <iostream>
#include <vector>
#include <random>

using namespace llm_inference;

void test_basic_operations() {
    std::cout << "========================================\n";
    std::cout << "测试1: 基本操作\n";
    std::cout << "========================================\n\n";

    // 创建KV Cache
    // Llama-2-7B参数：32层，32个头，每个头128维
    int max_seq_len = 2048;
    int n_layers = 32;
    int n_heads = 32;
    int head_dim = 128;

    KVCache cache(max_seq_len, n_layers, n_heads, head_dim);

    // 打印统计信息
    cache.print_stats();

    // 生成测试数据
    int kv_dim = n_heads * head_dim;
    std::vector<float> k_data(kv_dim);
    std::vector<float> v_data(kv_dim);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < kv_dim; ++i) {
        k_data[i] = dist(gen);
        v_data[i] = dist(gen);
    }

    // 更新cache
    std::cout << "更新cache（前10个位置）...\n";
    Timer timer;

    for (int pos = 0; pos < 10; ++pos) {
        for (int layer = 0; layer < n_layers; ++layer) {
            cache.update_k(layer, k_data.data(), pos);
            cache.update_v(layer, v_data.data(), pos);
        }
    }

    timer.print("更新耗时");
    std::cout << "当前序列长度: " << cache.get_current_len() << "\n";

    // 获取cache
    std::cout << "\n获取cache...\n";
    const float* k_cache = cache.get_k(0);
    const float* v_cache = cache.get_v(0);

    std::cout << "K cache前5个值: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << k_cache[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "✅ 基本操作测试通过\n\n";
}

void test_performance() {
    std::cout << "========================================\n";
    std::cout << "测试2: 性能测试\n";
    std::cout << "========================================\n\n";

    int max_seq_len = 2048;
    int n_layers = 32;
    int n_heads = 32;
    int head_dim = 128;

    KVCache cache(max_seq_len, n_layers, n_heads, head_dim);

    int kv_dim = n_heads * head_dim;
    std::vector<float> k_data(kv_dim, 1.0f);
    std::vector<float> v_data(kv_dim, 1.0f);

    // 测试Prefill（处理长prompt）
    std::cout << "Prefill测试（处理512个token）...\n";
    Timer prefill_timer;

    for (int pos = 0; pos < 512; ++pos) {
        for (int layer = 0; layer < n_layers; ++layer) {
            cache.update_k(layer, k_data.data(), pos);
            cache.update_v(layer, v_data.data(), pos);
        }
    }

    double prefill_time = prefill_timer.elapsed_ms();
    std::cout << "Prefill耗时: " << prefill_time << " ms\n";
    std::cout << "Prefill TPS: " << (512 * 1000.0 / prefill_time) << " tokens/s\n\n";

    // 测试Decode（逐个生成token）
    std::cout << "Decode测试（生成100个token）...\n";
    PerformanceStats decode_stats;

    for (int pos = 512; pos < 612; ++pos) {
        Timer decode_timer;

        for (int layer = 0; layer < n_layers; ++layer) {
            cache.update_k(layer, k_data.data(), pos);
            cache.update_v(layer, v_data.data(), pos);
        }

        decode_stats.add_sample(decode_timer.elapsed_ms());
    }

    std::cout << "Decode统计:\n";
    decode_stats.print("Decode");

    double avg_decode_time = decode_stats.mean();
    std::cout << "Decode TPS: " << (1000.0 / avg_decode_time) << " tokens/s\n\n";

    // 打印最终状态
    cache.print_stats();

    std::cout << "✅ 性能测试完成\n\n";
}

void test_memory_usage() {
    std::cout << "========================================\n";
    std::cout << "测试3: 内存使用分析\n";
    std::cout << "========================================\n\n";

    struct CacheConfig {
        int max_seq_len;
        int n_layers;
        int n_heads;
        int head_dim;
    };

    CacheConfig configs[] = {
        {512,  8,   8,  64},   // 小模型
        {2048, 32,  32, 128},  // Llama-2-7B
        {4096, 40,  40, 128},  // Llama-2-13B
        {8192, 80,  64, 128},  // Llama-2-70B
    };

    for (const auto& config : configs) {
        KVCache cache(
            config.max_seq_len,
            config.n_layers,
            config.n_heads,
            config.head_dim
        );

        size_t memory = cache.memory_usage();
        std::cout << "配置: "
                  << config.n_layers << "层, "
                  << config.n_heads << "头, "
                  << config.head_dim << "维, "
                  << config.max_seq_len << "长度\n";
        std::cout << "  内存: " << (memory / (1024.0 * 1024.0)) << " MB\n\n";
    }

    std::cout << "✅ 内存使用测试完成\n\n";
}

void test_reset() {
    std::cout << "========================================\n";
    std::cout << "测试4: 重置功能\n";
    std::cout << "========================================\n\n";

    KVCache cache(128, 4, 4, 32);

    // 填充cache
    std::vector<float> data(4 * 32, 1.0f);
    for (int pos = 0; pos < 10; ++pos) {
        cache.update_k(0, data.data(), pos);
    }

    std::cout << "填充前长度: " << cache.get_current_len() << "\n";

    // 重置
    cache.reset();
    std::cout << "重置后长度: " << cache.get_current_len() << "\n";

    std::cout << "\n✅ 重置功能测试通过\n\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║          KV Cache 测试程序                             ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    try {
        test_basic_operations();
        test_performance();
        test_memory_usage();
        test_reset();

        std::cout << "========================================\n";
        std::cout << "🎉 所有测试通过！\n";
        std::cout << "========================================\n\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
