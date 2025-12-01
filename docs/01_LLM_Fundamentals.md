# LLM基础理论

## 目录
1. [Transformer架构](#transformer架构)
2. [自注意力机制](#自注意力机制)
3. [位置编码](#位置编码)
4. [自回归生成](#自回归生成)
5. [模型参数规模](#模型参数规模)

---

## Transformer架构

### 核心思想

Transformer摒弃了RNN的序列依赖，完全基于注意力机制，实现并行化处理。

**架构组成：**

```
Input Tokens
    ↓
Embedding + Positional Encoding
    ↓
┌─────────────────────────────┐
│  Transformer Block (x N)    │
│  ├─ Multi-Head Attention    │
│  ├─ Add & Norm              │
│  ├─ Feed Forward Network    │
│  └─ Add & Norm              │
└─────────────────────────────┘
    ↓
Linear + Softmax
    ↓
Output Probabilities
```

### Decoder-Only架构（GPT系列）

现代LLM（如GPT、Llama）采用Decoder-Only架构：

- **单向注意力**：只能看到当前及之前的token
- **因果掩码**：防止信息泄露
- **自回归生成**：逐token生成

---

## 自注意力机制

### 数学定义

给定输入序列 X ∈ ℝ^(n×d)，自注意力计算：

```
Q = XW_Q  (Query)
K = XW_K  (Key)
V = XW_V  (Value)

Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

### 计算步骤

**1. 线性投影：**
```python
# d_model = 512, d_k = d_v = 64
Q = input @ W_Q  # (n, 512) @ (512, 64) = (n, 64)
K = input @ W_K  # (n, 512) @ (512, 64) = (n, 64)
V = input @ W_V  # (n, 512) @ (512, 64) = (n, 64)
```

**2. 计算注意力分数：**
```python
scores = Q @ K.T / sqrt(d_k)  # (n, n)
# 为什么除以√d_k？
# 防止点积过大导致softmax梯度消失
```

**3. 应用Softmax：**
```python
attention_weights = softmax(scores)  # (n, n)
# 每行和为1，表示当前token对所有token的注意力分布
```

**4. 加权求和：**
```python
output = attention_weights @ V  # (n, n) @ (n, 64) = (n, 64)
```

### 多头注意力（Multi-Head Attention）

**为什么需要多头？**
- 单个注意力头可能只关注某一方面特征
- 多头可以并行学习不同的注意力模式

**计算过程：**

```python
# h个头，每个头的维度为d_k
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W_O

where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

**Llama-2示例：**
- 模型维度：d_model = 4096
- 注意力头数：n_heads = 32
- 每个头的维度：d_k = 4096 / 32 = 128

### 因果掩码（Causal Mask）

对于自回归模型，必须防止"看到未来"：

```python
# 上三角掩码矩阵
mask = [[0, -∞, -∞, -∞],
        [0,  0, -∞, -∞],
        [0,  0,  0, -∞],
        [0,  0,  0,  0]]

scores = scores + mask  # 加上掩码后，未来位置的注意力为0
```

### 计算复杂度

**时间复杂度：** O(n²d)
- n: 序列长度
- d: 模型维度
- 瓶颈在于 QK^T 的矩阵乘法

**空间复杂度：** O(n²)
- 需要存储 (n×n) 的注意力矩阵
- 这是长序列的主要瓶颈！

---

## 位置编码

### 为什么需要位置编码？

自注意力是**置换不变**的，无法区分token顺序：
```
Attention([A, B, C]) = Attention([C, A, B])
```

必须显式注入位置信息！

### 绝对位置编码（Sinusoidal）

Transformer原论文使用的方法：

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**优点：**
- 确定性，不需要学习
- 可以外推到训练时未见过的长度

**示例：**
```python
import numpy as np

def sinusoidal_position_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe
```

### 可学习位置编码

GPT系列使用的方法：

```python
position_embeddings = nn.Embedding(max_seq_len, d_model)
pos_enc = position_embeddings(positions)
```

**缺点：** 无法外推到超过 max_seq_len 的序列

### 旋转位置编码（RoPE）

Llama、GPT-NeoX使用的方法，直接在Q、K上应用旋转变换：

```python
# 简化版本
def apply_rotary_emb(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
```

**优点：**
- 相对位置编码
- 外推性能好
- 不增加参数量

---

## 自回归生成

### 生成过程

LLM采用自回归方式逐token生成：

```
输入: "人工智能的未来"
↓
生成token 1: "是"
↓
输入: "人工智能的未来是"
↓
生成token 2: "充满"
↓
输入: "人工智能的未来是充满"
↓
生成token 3: "希望"
...
```

### 采样策略

**1. Greedy Decoding（贪婪解码）**
```python
next_token = argmax(logits)
```
- 总是选择概率最高的token
- 确定性，但可能重复、单调

**2. Temperature Sampling**
```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```
- temperature < 1: 更确定，更保守
- temperature > 1: 更随机，更有创意
- temperature = 1: 标准sampling

**3. Top-K Sampling**
```python
top_k_probs, top_k_indices = topk(probs, k)
next_token = sample(top_k_probs)
```
- 只从概率最高的K个token中采样
- 避免采样到低概率的奇怪token

**4. Top-P (Nucleus) Sampling**
```python
sorted_probs = sort(probs, descending=True)
cumsum = cumsum(sorted_probs)
cutoff = find_first(cumsum > p)
next_token = sample(sorted_probs[:cutoff])
```
- 动态调整候选集大小
- p=0.9: 从累积概率90%的最小集合中采样

### 推理的两个阶段

**Prefill（预填充）阶段：**
- 处理完整的输入prompt
- 计算所有token的K、V
- 并行计算，速度快
- 复杂度：O(n²d)

**Decode（解码）阶段：**
- 每次生成1个新token
- 只需计算新token的K、V
- 串行计算，速度慢
- 复杂度：O(nd) per token

**这就是为什么需要KV Cache！**

---

## 模型参数规模

### 参数组成

以Llama-2-7B为例：

```
总参数: 7B

分解：
├─ Token Embeddings:    32000 × 4096 = 131M
├─ 32个Transformer层:
│  ├─ Attention:
│  │  ├─ W_Q: 4096 × 4096 = 16.8M
│  │  ├─ W_K: 4096 × 4096 = 16.8M
│  │  ├─ W_V: 4096 × 4096 = 16.8M
│  │  └─ W_O: 4096 × 4096 = 16.8M
│  └─ FFN:
│     ├─ W1: 4096 × 11008 = 45M
│     ├─ W2: 11008 × 4096 = 45M
│     └─ W3: 4096 × 11008 = 45M (SwiGLU)
└─ Output Head: 4096 × 32000 = 131M
```

**每层参数：** ~186M
**32层总计：** ~6B
**加上Embeddings：** ~7B

### 内存占用

**FP32精度：**
```
7B params × 4 bytes = 28 GB
```

**FP16精度：**
```
7B params × 2 bytes = 14 GB
```

**INT8量化：**
```
7B params × 1 byte = 7 GB
```

**INT4量化：**
```
7B params × 0.5 byte = 3.5 GB
```

### 计算量（FLOPs）

生成1个token的计算量：

```
FLOPs ≈ 2 × n_params × seq_len
```

**示例：** Llama-2-7B，序列长度2048
```
FLOPs = 2 × 7B × 2048 ≈ 28 TFLOPs
```

在A100 GPU（312 TFLOPS）上：
```
理论时间 = 28 / 312 ≈ 90ms per token
```

实际会慢很多（内存带宽限制）！

---

## 关键洞察

### 1. 为什么自注意力有效？

- **全局感受野：** 每个token可以直接关注所有其他token
- **数据驱动：** 注意力模式由数据学习，不是人工设计
- **并行化：** 不同位置可以并行计算

### 2. 计算瓶颈在哪？

**Prefill阶段：**
- 瓶颈：QK^T矩阵乘法（O(n²d)）
- 优化方向：Flash Attention

**Decode阶段：**
- 瓶颈：从内存读取KV Cache
- 优化方向：KV Cache优化、量化

### 3. 为什么长序列困难？

- **注意力复杂度：** O(n²) - 序列翻倍，计算量4倍
- **KV Cache大小：** O(n) - 序列翻倍，内存2倍
- **位置编码：** 有些方法无法外推

### 4. Scaling Laws

模型性能（Loss）与规模的关系：

```
Loss ∝ N^(-α)

N: 模型参数量
α ≈ 0.076 (经验值)
```

**启示：**
- 10倍参数 → ~1.5倍性能提升
- 但计算成本增加10倍
- 存在最优规模权衡点

---

## 实践建议

### 1. 选择合适的模型规模

| 任务类型 | 推荐规模 | 示例模型 |
|---------|---------|---------|
| 简单分类/摘要 | 1B-3B | Phi-2, StableLM |
| 通用问答 | 7B-13B | Llama-2-7B, Mistral-7B |
| 复杂推理 | 30B-70B | Llama-2-70B, Mixtral-8x7B |
| 顶级性能 | 100B+ | GPT-4 |

### 2. 理解推理成本

**内存：** 主要用于存储模型权重和KV Cache
```
总内存 = 模型权重 + KV Cache + 激活值
       ≈ 模型大小 + batch_size × seq_len × hidden_dim × layers × 4
```

**计算：** 主要在矩阵乘法
```
每token延迟 ∝ 模型大小 / GPU算力
```

### 3. 优化优先级

1. **首先：** KV Cache（20x加速）
2. **其次：** 量化（4x内存节省）
3. **再次：** Flash Attention（2x加速）
4. **最后：** 批处理、投机解码等高级技术

---

## 延伸阅读

**核心论文：**
1. Attention Is All You Need (Vaswani et al., 2017) - Transformer原论文
2. Language Models are Few-Shot Learners (Brown et al., 2020) - GPT-3
3. LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)

**优化技术：**
1. FlashAttention (Dao et al., 2022)
2. PagedAttention (Kwon et al., 2023) - vLLM
3. Rotary Position Embedding (Su et al., 2021)

**下一步学习：**
- [推理优化理论](02_Inference_Optimization.md)
- [KV Cache详解](03_KV_Cache_Deep_Dive.md)
