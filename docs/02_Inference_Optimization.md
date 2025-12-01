# LLM推理优化理论

## 目录
1. [推理性能指标](#推理性能指标)
2. [KV Cache优化](#kv-cache优化)
3. [量化技术](#量化技术)
4. [SIMD向量化](#simd向量化)
5. [其他优化技术](#其他优化技术)

---

## 推理性能指标

### 核心指标

**1. TTFT (Time To First Token)**
- 从输入到第一个token输出的时间
- 反映Prefill阶段性能
- 用户感知延迟的关键

**2. TPS (Tokens Per Second)**
- 每秒生成的token数量
- 反映Decode阶段吞吐量
- 影响总体生成速度

**3. 延迟 (Latency)**
- 生成每个token的平均时间
- E2E延迟 = TTFT + (生成长度 / TPS)

**4. 吞吐量 (Throughput)**
- 系统每秒处理的总token数
- 批处理场景的关键指标

### 性能瓶颈分析

**内存带宽限制（Memory-Bound）**

Decode阶段的主要瓶颈：
```
计算强度 = FLOPs / Bytes

LLM推理计算强度很低！
理论: 2 FLOPs/Byte
实际: 需要从内存读取大量权重

结论: GPU利用率低（5-20%），大部分时间在等待内存
```

**计算限制（Compute-Bound）**

Prefill阶段的特点：
```
- 大批量矩阵乘法
- GPU利用率高（60-80%）
- 相对较快
```

### 优化目标

**不同场景的优化重点：**

| 场景 | 关键指标 | 优化方向 |
|-----|---------|---------|
| 在线聊天 | TTFT | 减少Prefill时间 |
| 代码生成 | TPS | 提升Decode速度 |
| 批量处理 | Throughput | 增加批大小 |
| 嵌入式设备 | Memory | 量化、剪枝 |

---

## KV Cache优化

### 问题：重复计算

**没有Cache的情况：**

```python
# 生成第1个token
t=0: Q₀K₀ᵀ         # 计算1次

# 生成第2个token（重新计算所有！）
t=1: Q₁K₀ᵀ, Q₁K₁ᵀ  # K₀重复计算

# 生成第3个token
t=2: Q₂K₀ᵀ, Q₂K₁ᵀ, Q₂K₂ᵀ  # K₀,K₁重复计算

总计算次数: 1 + 2 + 3 + ... + n = O(n²)
```

### 解决方案：缓存K和V

**核心思想：** 已计算的K、V矩阵不会改变，缓存起来复用！

```python
class KVCache:
    def __init__(self, max_seq_len, n_layers, n_heads, head_dim):
        # 预分配cache空间
        self.k_cache = zeros(n_layers, max_seq_len, n_heads, head_dim)
        self.v_cache = zeros(n_layers, max_seq_len, n_heads, head_dim)
        self.seq_len = 0

    def update(self, layer, k, v):
        # O(1) 更新
        self.k_cache[layer, self.seq_len] = k
        self.v_cache[layer, self.seq_len] = v
        self.seq_len += 1

    def get(self, layer):
        # 返回所有已缓存的K、V
        return self.k_cache[layer, :self.seq_len], \
               self.v_cache[layer, :self.seq_len]
```

### 数学推导

**标准Attention（无Cache）：**
```
对于序列长度n，生成第t个token时：

Q_t = X_t W_Q              # (1, d) @ (d, d) = (1, d)
K_{0:t} = X_{0:t} W_K      # (t, d) @ (d, d) = (t, d)  ← 重复计算！
V_{0:t} = X_{0:t} W_V      # (t, d) @ (d, d) = (t, d)  ← 重复计算！

scores = Q_t @ K_{0:t}^T / √d_k  # (1, d) @ (d, t) = (1, t)
output = softmax(scores) @ V_{0:t}  # (1, t) @ (t, d) = (1, d)
```

**使用KV Cache：**
```
第一次（t=0，Prefill）：
计算并缓存所有: K_{0:n}, V_{0:n}

后续（t>0，Decode）：
只计算新的: k_t = x_t W_K, v_t = x_t W_V
追加到cache: cache.append(k_t, v_t)
使用完整cache: attention(Q_t, cache.K, cache.V)
```

### 复杂度分析

**时间复杂度：**

| 阶段 | 无Cache | 有Cache | 加速比 |
|-----|---------|---------|--------|
| Prefill | O(n²d) | O(n²d) | 1x |
| Decode (每token) | O(n²d) | O(nd) | **n倍** |
| 生成m个token | O(mn²d) | O(n²d + mnd) | **~n倍** |

**空间复杂度：**

```
每层KV Cache大小:
2 × seq_len × n_heads × head_dim × sizeof(float)

Llama-2-7B示例 (32层):
2 × 2048 × 32 × 128 × 2 bytes = 32 MB/层
总计: 32 × 32 MB = 1 GB

占模型权重的 1GB / 14GB ≈ 7%
```

### PagedAttention优化

**问题：** 静态KV Cache浪费内存

传统方法：
```
为每个请求预分配 max_seq_len 大小的cache
实际使用可能只有 20-30%
```

**vLLM的解决方案：**

分页管理，类似操作系统虚拟内存：
```python
# 将cache分成固定大小的block
block_size = 16  # 每个block存16个token

# 动态分配block
class PagedKVCache:
    def __init__(self, n_blocks):
        self.blocks = [Block(block_size) for _ in range(n_blocks)]
        self.free_blocks = list(range(n_blocks))

    def allocate(self, n_tokens):
        n_needed = ceil(n_tokens / block_size)
        allocated = self.free_blocks[:n_needed]
        self.free_blocks = self.free_blocks[n_needed:]
        return allocated

    def free(self, block_ids):
        self.free_blocks.extend(block_ids)
```

**优势：**
- 内存利用率提升 20-30% → 90%+
- 支持更大的批处理
- 减少内存碎片

---

## 量化技术

### 为什么需要量化？

**内存占用：**
```
Llama-2-7B:
FP32: 7B × 4 = 28 GB  ← 消费级显卡装不下！
FP16: 7B × 2 = 14 GB
INT8: 7B × 1 = 7 GB   ← 可以放入
INT4: 7B × 0.5 = 3.5 GB
```

**内存带宽：**
```
读取权重时间 ∝ 数据量
INT8比FP16快 2x
INT4比FP16快 4x
```

### 量化数学原理

**对称量化（Symmetric Quantization）：**

```
量化过程:
scale = max(|W|) / (2^(bits-1) - 1)
W_quantized = round(W / scale)

反量化:
W_dequantized = W_quantized × scale
```

**示例（INT8）：**
```python
import numpy as np

# 原始权重
W = np.array([0.5, -0.3, 0.8, -0.9])

# 量化
scale = np.max(np.abs(W)) / 127
W_int8 = np.round(W / scale).astype(np.int8)

print(f"Scale: {scale:.4f}")
print(f"INT8: {W_int8}")
print(f"Max value: {np.max(np.abs(W_int8))}")  # ≤ 127

# 反量化
W_dequant = W_int8 * scale
print(f"Error: {np.mean((W - W_dequant)**2):.6f}")
```

输出：
```
Scale: 0.0071
INT8: [ 70 -42 113 -127]
Max value: 127
Error: 0.000012
```

### 非对称量化（Asymmetric）

适用于非零中心分布：
```
scale = (max(W) - min(W)) / 255
zero_point = round(-min(W) / scale)

W_quantized = round(W / scale) + zero_point
W_dequantized = (W_quantized - zero_point) × scale
```

### 分组量化（Group Quantization）

**问题：** 权重分布不均匀

```
某些行的权重范围: [-0.1, 0.1]
其他行的权重范围: [-2.0, 2.0]

用同一个scale会导致：
- 小权重量化误差大
- 大权重浪费表示范围
```

**解决：** 每组使用独立的scale

```python
def group_quantize(W, group_size=128):
    n = len(W)
    n_groups = ceil(n / group_size)

    W_quant = []
    scales = []

    for i in range(n_groups):
        start = i * group_size
        end = min(start + group_size, n)
        group = W[start:end]

        # 每组独立量化
        scale = np.max(np.abs(group)) / 127
        W_group_quant = np.round(group / scale)

        W_quant.extend(W_group_quant)
        scales.append(scale)

    return np.array(W_quant), np.array(scales)
```

**效果：**
```
普通INT8: SQNR ≈ 35 dB
分组INT8 (g=128): SQNR ≈ 42 dB  ← 提升7dB！
```

### INT4量化

更激进的压缩：

```python
def pack_int4(values):
    # 两个INT4打包成一个INT8
    packed = []
    for i in range(0, len(values), 2):
        low = values[i] & 0x0F      # 低4位
        high = values[i+1] & 0x0F   # 高4位
        packed.append((high << 4) | low)
    return np.array(packed, dtype=np.int8)

def unpack_int4(packed):
    values = []
    for byte in packed:
        low = byte & 0x0F           # 提取低4位
        high = (byte >> 4) & 0x0F   # 提取高4位
        values.extend([low, high])
    return np.array(values, dtype=np.int8) - 8  # 转换为有符号
```

**精度 vs 压缩：**

| 方法 | 压缩比 | 精度损失 | 适用场景 |
|-----|--------|---------|---------|
| FP16 | 1x | 基线 | 训练、高精度推理 |
| INT8 | 2x | <1% | 推荐，性价比高 |
| INT4 | 4x | 1-3% | 资源受限设备 |
| INT4 + Group | 4x | <1% | 最佳实践 |

### 量化感知训练 vs 训练后量化

**PTQ (Post-Training Quantization)：**
- 直接量化已训练模型
- 无需重新训练
- 精度略低

**QAT (Quantization-Aware Training)：**
- 训练时模拟量化
- 模型学习适应量化误差
- 精度更高，但成本高

本项目使用PTQ！

---

## SIMD向量化

### SIMD基础

**SIMD = Single Instruction, Multiple Data**

一条指令同时处理多个数据：

```
标量加法 (scalar):
for i in range(8):
    c[i] = a[i] + b[i]  # 8条指令

向量加法 (SIMD):
c_vec = a_vec + b_vec   # 1条指令，处理8个float
```

### AVX2指令集

**寄存器：**
- 256位宽度
- 可存储 8个float (32位) 或 4个double (64位)

**核心指令：**

```cpp
// 加载数据
__m256 a = _mm256_loadu_ps(ptr);  // 加载8个float

// 基本运算
__m256 c = _mm256_add_ps(a, b);   // 向量加法
__m256 c = _mm256_mul_ps(a, b);   // 向量乘法

// FMA (Fused Multiply-Add): c = a*b + c
c = _mm256_fmadd_ps(a, b, c);     // 1条指令完成乘加

// 归约（求和）
float sum = _mm256_reduce_add_ps(a);
```

### 向量点积优化

**标量版本：**
```cpp
float dot_product_scalar(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];  // 2个操作：乘法、加法
    }
    return sum;
}
```

**AVX2优化版本：**
```cpp
float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();

    // 主循环：每次处理8个元素
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    // 水平求和（归约）
    float sum = _mm256_reduce_add_ps(sum_vec);

    // 处理剩余元素
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}
```

**性能对比：**
```
n=1024 的向量点积:

标量: 2.5 µs
AVX2: 0.8 µs
加速比: 3.1x (理论8x，实际受限于内存带宽)
```

### 矩阵乘法优化

**矩阵乘法：C = A × B**

标量实现：
```cpp
// C(m×n) = A(m×k) × B(k×n)
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        float sum = 0;
        for (int p = 0; p < k; ++p) {
            sum += A[i][p] * B[p][j];
        }
        C[i][j] = sum;
    }
}
```

SIMD优化策略：
```cpp
// 处理C的每一行
for (int i = 0; i < m; ++i) {
    // 每次处理8列
    for (int j = 0; j < n; j += 8) {
        __m256 sum = _mm256_setzero_ps();

        // 点积
        for (int p = 0; p < k; ++p) {
            __m256 a = _mm256_set1_ps(A[i][p]);  // 广播
            __m256 b = _mm256_loadu_ps(&B[p][j]);
            sum = _mm256_fmadd_ps(a, b, sum);
        }

        _mm256_storeu_ps(&C[i][j], sum);
    }
}
```

### 内存对齐

**为什么需要对齐？**

未对齐访问：
```
地址:  [0x1001] [0x1002] ... [0x1008]
需要: 2次内存访问
```

对齐访问：
```
地址:  [0x1000] [0x1004] ... [0x101C]
需要: 1次内存访问，快2倍！
```

**实现：**
```cpp
// 分配32字节对齐的内存
float* aligned_alloc(size_t n) {
    return (float*)aligned_alloc(32, n * sizeof(float));
}

// 对齐加载（更快）
__m256 a = _mm256_load_ps(ptr);   // 要求32字节对齐

// 非对齐加载（兼容性）
__m256 a = _mm256_loadu_ps(ptr);  // 任意地址，稍慢
```

### SIMD的局限性

1. **内存带宽瓶颈：** 理论加速8x，实际3-4x
2. **分支不友好：** 条件语句需要特殊处理
3. **编写难度：** 需要汇编级别的理解
4. **可移植性：** AVX2不是所有CPU都支持

---

## 其他优化技术

### Flash Attention

**问题：** 标准attention的显存占用

```
Attention矩阵: O(n²)
序列长度8K: 8K × 8K × 4 bytes = 256 MB (单层！)
```

**Flash Attention核心思想：**
- 分块计算，不存储完整注意力矩阵
- 利用GPU的共享内存（SRAM）
- 减少HBM（显存）访问

**效果：**
- 内存：O(n²) → O(n)
- 速度：2-4x更快
- 支持更长序列

### 投机解码（Speculative Decoding）

**思路：** 用小模型猜测，大模型验证

```
1. 小模型快速生成k个token
2. 大模型并行验证这k个token
3. 接受正确的，拒绝错误的
```

**加速比：** 2-3x（当小模型准确率高时）

### 批处理优化

**动态批处理：**
```
不同请求的序列长度不同：
Request 1: 10 tokens
Request 2: 50 tokens
Request 3: 100 tokens

传统方法：padding到100，浪费算力

优化：每个请求单独处理，动态组batch
```

### 量化+KV Cache组合

最佳实践：
```
模型权重: INT4量化 (8x压缩)
KV Cache: FP16 (保持精度)

原因:
- 权重量化对精度影响小
- KV Cache量化会累积误差
```

---

## 优化效果总结

### Llama-2-7B优化对比

| 优化方法 | TTFT | TPS | 内存 | 精度损失 |
|---------|------|-----|------|---------|
| 基线(FP16) | 500ms | 20 | 14GB | - |
| + KV Cache | 500ms | 100 | 15GB | 0% |
| + INT8 | 400ms | 150 | 8GB | <1% |
| + SIMD | 300ms | 200 | 8GB | <1% |
| + Flash Attn | 250ms | 250 | 8GB | <1% |
| **全部组合** | **250ms** | **250** | **8GB** | **<1%** |

**加速比：** 12.5x
**内存节省：** 43%

### 优化实施建议

**第一阶段（必须）：**
1. KV Cache - 最大性价比
2. INT8量化 - 内存受限时

**第二阶段（推荐）：**
3. SIMD优化 - CPU推理
4. Flash Attention - 长序列

**第三阶段（高级）：**
5. PagedAttention - 大批量
6. 投机解码 - 极致延迟

---

## 实践检查清单

**推理部署前：**
- [ ] 确定目标平台（GPU/CPU）
- [ ] 测量基线性能
- [ ] 确定关键指标（TTFT vs TPS）
- [ ] 评估内存限制

**优化实施：**
- [ ] 实现KV Cache
- [ ] 选择量化策略（INT8/INT4）
- [ ] 测试精度损失
- [ ] 性能对比测试

**生产环境：**
- [ ] 监控内存使用
- [ ] 记录延迟分布（P50/P95/P99）
- [ ] 设置超时和限流
- [ ] 准备降级方案

---

## 延伸阅读

**论文：**
1. FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
2. Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., 2023)
3. LLM.int8(): 8-bit Matrix Multiplication for Transformers (Dettmers et al., 2022)
4. GPTQ: Accurate Post-Training Quantization (Frantar et al., 2022)

**开源实现：**
- vLLM: PagedAttention实现
- llama.cpp: INT4/INT8量化
- TensorRT-LLM: NVIDIA推理优化
- Flash Attention 2: 官方实现

**下一步：**
- [RAG系统理论](03_RAG_System_Theory.md)
- [Prompt Engineering](04_Prompt_Engineering.md)
