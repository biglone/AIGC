/**
 * kv_cache.cpp
 *
 * KV Cache实现
 */

#include "kv_cache.h"
#include <cstring>
#include <iostream>
#include <iomanip>

namespace llm_inference {

KVCache::KVCache(int max_seq_len, int n_layers, int n_heads, int head_dim)
    : max_seq_len_(max_seq_len),
      n_layers_(n_layers),
      n_heads_(n_heads),
      head_dim_(head_dim),
      current_len_(0) {

    // 分配内存
    int kv_size = max_seq_len_ * n_heads_ * head_dim_;

    k_cache_.resize(n_layers_);
    v_cache_.resize(n_layers_);

    for (int i = 0; i < n_layers_; ++i) {
        k_cache_[i].resize(kv_size, 0.0f);
        v_cache_[i].resize(kv_size, 0.0f);
    }

    std::cout << "KVCache initialized: "
              << n_layers_ << " layers, "
              << max_seq_len_ << " max seq len, "
              << n_heads_ << " heads, "
              << head_dim_ << " head dim" << std::endl;
    std::cout << "Memory usage: " << memory_usage() / (1024.0 * 1024.0)
              << " MB" << std::endl;
}

KVCache::~KVCache() {
    // std::vector自动释放内存
}

void KVCache::update_k(int layer, const float* k, int seq_pos) {
    check_layer_bounds(layer);
    check_seq_bounds(seq_pos);

    int kv_dim = get_kv_size();
    int offset = seq_pos * kv_dim;

    // 复制K到cache
    std::memcpy(&k_cache_[layer][offset], k, kv_dim * sizeof(float));

    // 更新当前长度
    if (seq_pos >= current_len_) {
        current_len_ = seq_pos + 1;
    }
}

void KVCache::update_v(int layer, const float* v, int seq_pos) {
    check_layer_bounds(layer);
    check_seq_bounds(seq_pos);

    int kv_dim = get_kv_size();
    int offset = seq_pos * kv_dim;

    // 复制V到cache
    std::memcpy(&v_cache_[layer][offset], v, kv_dim * sizeof(float));
}

const float* KVCache::get_k(int layer) const {
    check_layer_bounds(layer);
    return k_cache_[layer].data();
}

const float* KVCache::get_v(int layer) const {
    check_layer_bounds(layer);
    return v_cache_[layer].data();
}

void KVCache::reset() {
    current_len_ = 0;

    // 可选：清零内存（性能考虑可以不做）
    // for (auto& k : k_cache_) {
    //     std::fill(k.begin(), k.end(), 0.0f);
    // }
    // for (auto& v : v_cache_) {
    //     std::fill(v.begin(), v.end(), 0.0f);
    // }
}

size_t KVCache::memory_usage() const {
    size_t kv_size = max_seq_len_ * n_heads_ * head_dim_;
    return 2 * n_layers_ * kv_size * sizeof(float);  // K和V
}

void KVCache::print_stats() const {
    std::cout << "\n========================================\n";
    std::cout << "KV Cache Statistics\n";
    std::cout << "========================================\n";
    std::cout << "Layers:         " << n_layers_ << "\n";
    std::cout << "Max Seq Len:    " << max_seq_len_ << "\n";
    std::cout << "Current Len:    " << current_len_ << "\n";
    std::cout << "Heads:          " << n_heads_ << "\n";
    std::cout << "Head Dim:       " << head_dim_ << "\n";
    std::cout << "Memory Usage:   "
              << std::fixed << std::setprecision(2)
              << memory_usage() / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Utilization:    "
              << std::fixed << std::setprecision(1)
              << (100.0 * current_len_ / max_seq_len_) << "%\n";
    std::cout << "========================================\n\n";
}

void KVCache::check_layer_bounds(int layer) const {
    if (layer < 0 || layer >= n_layers_) {
        throw std::out_of_range(
            "Layer index out of range: " + std::to_string(layer) +
            " (valid: 0-" + std::to_string(n_layers_ - 1) + ")"
        );
    }
}

void KVCache::check_seq_bounds(int seq_pos) const {
    if (seq_pos < 0 || seq_pos >= max_seq_len_) {
        throw std::out_of_range(
            "Sequence position out of range: " + std::to_string(seq_pos) +
            " (valid: 0-" + std::to_string(max_seq_len_ - 1) + ")"
        );
    }
}

// ============================================================
// PagedKVCache实现
// ============================================================

PagedKVCache::PagedKVCache(int n_layers, int n_heads, int head_dim, int max_pages)
    : n_layers_(n_layers),
      n_heads_(n_heads),
      head_dim_(head_dim),
      max_pages_(max_pages) {

    int page_kv_size = PAGE_SIZE * n_heads * head_dim;

    page_used_.resize(max_pages_, false);
    k_pages_.resize(max_pages_);
    v_pages_.resize(max_pages_);

    for (int i = 0; i < max_pages_; ++i) {
        k_pages_[i].resize(n_layers_ * page_kv_size, 0.0f);
        v_pages_[i].resize(n_layers_ * page_kv_size, 0.0f);
    }

    std::cout << "PagedKVCache initialized: "
              << max_pages_ << " pages, "
              << PAGE_SIZE << " tokens/page" << std::endl;
}

int PagedKVCache::allocate_page() {
    for (int i = 0; i < max_pages_; ++i) {
        if (!page_used_[i]) {
            page_used_[i] = true;
            return i;
        }
    }
    throw std::runtime_error("No free pages available");
}

void PagedKVCache::free_page(int page_id) {
    if (page_id >= 0 && page_id < max_pages_) {
        page_used_[page_id] = false;
    }
}

float* PagedKVCache::get_k_at(int layer, int page_id, int offset) {
    if (page_id < 0 || page_id >= max_pages_) {
        throw std::out_of_range("Page ID out of range");
    }
    if (offset < 0 || offset >= PAGE_SIZE) {
        throw std::out_of_range("Offset out of range");
    }

    int kv_dim = n_heads_ * head_dim_;
    int layer_offset = layer * PAGE_SIZE * kv_dim;
    int token_offset = offset * kv_dim;

    return &k_pages_[page_id][layer_offset + token_offset];
}

float* PagedKVCache::get_v_at(int layer, int page_id, int offset) {
    if (page_id < 0 || page_id >= max_pages_) {
        throw std::out_of_range("Page ID out of range");
    }
    if (offset < 0 || offset >= PAGE_SIZE) {
        throw std::out_of_range("Offset out of range");
    }

    int kv_dim = n_heads_ * head_dim_;
    int layer_offset = layer * PAGE_SIZE * kv_dim;
    int token_offset = offset * kv_dim;

    return &v_pages_[page_id][layer_offset + token_offset];
}

} // namespace llm_inference
