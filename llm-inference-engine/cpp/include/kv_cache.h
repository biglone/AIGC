/**
 * kv_cache.h
 *
 * KV Cache实现 - LLM推理加速的核心技术
 *
 * 原理：
 * - 生成式模型每次生成token时，需要计算所有历史token的K和V
 * - KV Cache缓存已计算的K和V，避免重复计算
 * - 加速效果：50x+（对于长序列）
 */

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <vector>
#include <memory>
#include <stdexcept>

namespace llm_inference {

/**
 * KV Cache类
 *
 * 存储结构：
 * - k_cache_: [n_layers][max_seq_len][n_heads * head_dim]
 * - v_cache_: [n_layers][max_seq_len][n_heads * head_dim]
 */
class KVCache {
public:
    /**
     * 构造函数
     *
     * @param max_seq_len 最大序列长度
     * @param n_layers 模型层数
     * @param n_heads 注意力头数
     * @param head_dim 每个头的维度
     */
    KVCache(int max_seq_len, int n_layers, int n_heads, int head_dim);

    /**
     * 析构函数
     */
    ~KVCache();

    /**
     * 更新指定层的K cache
     *
     * @param layer 层索引
     * @param k 新的K值 [n_heads * head_dim]
     * @param seq_pos 序列位置
     */
    void update_k(int layer, const float* k, int seq_pos);

    /**
     * 更新指定层的V cache
     *
     * @param layer 层索引
     * @param v 新的V值 [n_heads * head_dim]
     * @param seq_pos 序列位置
     */
    void update_v(int layer, const float* v, int seq_pos);

    /**
     * 获取指定层的完整K cache
     *
     * @param layer 层索引
     * @return K cache指针
     */
    const float* get_k(int layer) const;

    /**
     * 获取指定层的完整V cache
     *
     * @param layer 层索引
     * @return V cache指针
     */
    const float* get_v(int layer) const;

    /**
     * 获取当前序列长度
     */
    int get_current_len() const { return current_len_; }

    /**
     * 重置cache（清空所有缓存）
     */
    void reset();

    /**
     * 获取cache占用的内存大小（字节）
     */
    size_t memory_usage() const;

    /**
     * 打印cache统计信息
     */
    void print_stats() const;

private:
    int max_seq_len_;   // 最大序列长度
    int n_layers_;      // 层数
    int n_heads_;       // 注意力头数
    int head_dim_;      // 每个头的维度
    int current_len_;   // 当前序列长度

    // K和V缓存
    // 使用连续内存提高缓存命中率
    std::vector<std::vector<float>> k_cache_;  // [n_layers][max_seq_len * n_heads * head_dim]
    std::vector<std::vector<float>> v_cache_;  // [n_layers][max_seq_len * n_heads * head_dim]

    // 辅助函数
    int get_kv_size() const { return n_heads_ * head_dim_; }
    void check_layer_bounds(int layer) const;
    void check_seq_bounds(int seq_pos) const;
};

/**
 * PagedKVCache - 分页KV Cache（高级版本）
 *
 * 灵感来自vLLM的PagedAttention
 * 优势：
 * - 内存利用率更高（按需分配）
 * - 支持动态batch
 */
class PagedKVCache {
public:
    static constexpr int PAGE_SIZE = 16;  // 每页16个token

    PagedKVCache(int n_layers, int n_heads, int head_dim, int max_pages = 1024);

    // 分配新页
    int allocate_page();

    // 释放页
    void free_page(int page_id);

    // 获取指定位置的K/V
    float* get_k_at(int layer, int page_id, int offset);
    float* get_v_at(int layer, int page_id, int offset);

private:
    int n_layers_;
    int n_heads_;
    int head_dim_;
    int max_pages_;

    std::vector<bool> page_used_;  // 页是否被使用
    std::vector<std::vector<float>> k_pages_;  // K的分页存储
    std::vector<std::vector<float>> v_pages_;  // V的分页存储
};

} // namespace llm_inference

#endif // KV_CACHE_H
