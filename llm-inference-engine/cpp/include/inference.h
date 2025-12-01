/**
 * inference.h
 *
 * LLM推理引擎 - 主接口
 *
 * 集成所有优化技术：
 * - KV Cache
 * - INT8量化
 * - SIMD加速
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <vector>
#include <memory>
#include "kv_cache.h"
#include "quantization.h"
#include "utils.h"

namespace llm_inference {

/**
 * 模型配置
 */
struct ModelConfig {
    // 模型架构
    int vocab_size;      // 词表大小
    int n_layers;        // 层数
    int n_heads;         // 注意力头数
    int d_model;         // 模型维度
    int d_ff;            // FFN中间层维度
    int max_seq_len;     // 最大序列长度

    // 派生参数
    int head_dim() const { return d_model / n_heads; }

    // 默认配置（Llama-2-7B）
    static ModelConfig llama_7b() {
        ModelConfig config;
        config.vocab_size = 32000;
        config.n_layers = 32;
        config.n_heads = 32;
        config.d_model = 4096;
        config.d_ff = 11008;
        config.max_seq_len = 2048;
        return config;
    }
};

/**
 * 生成配置
 */
struct GenerationConfig {
    int max_new_tokens = 100;    // 生成的最大token数
    float temperature = 1.0f;    // 温度（控制随机性）
    int top_k = 50;              // Top-K采样
    float top_p = 0.9f;          // Top-P（nucleus）采样
    bool do_sample = true;       // 是否采样（否则贪心）
    int seed = -1;               // 随机种子（-1表示随机）
};

/**
 * 推理统计信息
 */
struct InferenceStats {
    double prefill_time_ms;      // Prefill阶段耗时
    double decode_time_ms;       // Decode阶段耗时
    int tokens_generated;        // 生成的token数
    double tokens_per_second;    // 吞吐量（tokens/s）

    void print() const;
};

/**
 * LLM推理引擎
 *
 * 支持两种模式：
 * 1. Prefill: 处理输入prompt，填充KV Cache
 * 2. Decode: 逐个生成新token
 */
class LLMInference {
public:
    /**
     * 构造函数
     *
     * @param config 模型配置
     * @param use_kv_cache 是否使用KV Cache
     * @param use_quantization 是否使用量化
     */
    LLMInference(
        const ModelConfig& config,
        bool use_kv_cache = true,
        bool use_quantization = false
    );

    /**
     * 析构函数
     */
    ~LLMInference();

    /**
     * 加载模型权重
     *
     * @param model_path 模型文件路径（.gguf格式）
     */
    void load_model(const std::string& model_path);

    /**
     * 文本生成（高层接口）
     *
     * @param prompt 输入文本
     * @param gen_config 生成配置
     * @return 生成的文本
     */
    std::string generate(
        const std::string& prompt,
        const GenerationConfig& gen_config = GenerationConfig()
    );

    /**
     * 流式生成（支持实时输出）
     *
     * @param prompt 输入文本
     * @param callback 每生成一个token的回调函数
     * @param gen_config 生成配置
     */
    void generate_stream(
        const std::string& prompt,
        std::function<void(const std::string&)> callback,
        const GenerationConfig& gen_config = GenerationConfig()
    );

    /**
     * Prefill阶段（处理输入prompt）
     *
     * @param tokens 输入token序列
     * @return 最后一个token的logits
     */
    std::vector<float> prefill(const std::vector<int>& tokens);

    /**
     * Decode阶段（生成单个token）
     *
     * @param token 当前token
     * @return 下一个token的logits
     */
    std::vector<float> decode(int token);

    /**
     * 重置状态（清空KV Cache）
     */
    void reset();

    /**
     * 获取推理统计信息
     */
    const InferenceStats& get_stats() const { return stats_; }

    /**
     * 打印模型信息
     */
    void print_model_info() const;

private:
    // 配置
    ModelConfig config_;
    bool use_kv_cache_;
    bool use_quantization_;

    // KV Cache
    std::unique_ptr<KVCache> kv_cache_;

    // 模型权重（简化版本，实际需要更复杂的结构）
    struct Weights {
        // Token Embedding
        std::vector<float> token_emb;  // [vocab_size, d_model]

        // 每层的权重
        struct LayerWeights {
            // Attention
            std::vector<float> wq;  // Query权重
            std::vector<float> wk;  // Key权重
            std::vector<float> wv;  // Value权重
            std::vector<float> wo;  // Output权重

            // FFN
            std::vector<float> w1;  // Gate权重
            std::vector<float> w2;  // Down权重
            std::vector<float> w3;  // Up权重

            // LayerNorm
            std::vector<float> ln1_gamma;
            std::vector<float> ln1_beta;
            std::vector<float> ln2_gamma;
            std::vector<float> ln2_beta;
        };
        std::vector<LayerWeights> layers;

        // 输出层
        std::vector<float> output_norm_gamma;
        std::vector<float> output_norm_beta;
        std::vector<float> lm_head;  // [d_model, vocab_size]
    };
    Weights weights_;

    // 量化的权重
    std::vector<INT8Quantizer::QuantizedWeights> quantized_weights_;

    // 统计信息
    InferenceStats stats_;

    // 内部函数
    std::vector<float> forward_layer(
        int layer_idx,
        const std::vector<float>& input,
        int seq_pos
    );

    std::vector<float> attention(
        int layer_idx,
        const std::vector<float>& input,
        int seq_pos
    );

    std::vector<float> ffn(
        int layer_idx,
        const std::vector<float>& input
    );

    // Token采样
    int sample_token(const std::vector<float>& logits, const GenerationConfig& config);
    int greedy_sample(const std::vector<float>& logits);
    int top_k_sample(const std::vector<float>& logits, int k);
    int top_p_sample(const std::vector<float>& logits, float p);

    // Tokenizer（简化版本）
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& tokens);
    std::string detokenize_single(int token);
};

/**
 * 批量推理（支持多个请求并行处理）
 */
class BatchInference {
public:
    BatchInference(const ModelConfig& config, int max_batch_size = 8);

    /**
     * 批量生成
     *
     * @param prompts 多个输入prompt
     * @param gen_config 生成配置
     * @return 生成的文本列表
     */
    std::vector<std::string> generate_batch(
        const std::vector<std::string>& prompts,
        const GenerationConfig& gen_config = GenerationConfig()
    );

private:
    ModelConfig config_;
    int max_batch_size_;
    std::vector<std::unique_ptr<LLMInference>> engines_;
};

/**
 * 模型加载工具
 */
namespace model_loader {

/**
 * 从GGUF文件加载模型
 *
 * GGUF是llama.cpp的模型格式
 */
bool load_gguf(const std::string& path, ModelConfig& config, LLMInference::Weights& weights);

/**
 * 打印GGUF模型信息
 */
void print_gguf_info(const std::string& path);

} // namespace model_loader

} // namespace llm_inference

#endif // INFERENCE_H
