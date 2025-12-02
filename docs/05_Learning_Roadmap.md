# AIGC完整学习路线图

## 目录
1. [学习路径总览](#学习路径总览)
2. [阶段一：基础理论](#阶段一基础理论)
3. [阶段二：核心技术](#阶段二核心技术)
4. [阶段三：项目实践](#阶段三项目实践)
5. [阶段四：深入优化](#阶段四深入优化)
6. [进阶方向](#进阶方向)

---

## 学习路径总览

### 完整时间线（3个月全职学习）

```
月份1：理论基础 + Python AI开发
  ├─ Week 1-2: LLM基础理论
  ├─ Week 3: Python AI生态系统
  └─ Week 4: 向量数据库和Embedding

月份2：核心技术实践
  ├─ Week 5-6: RAG系统开发
  ├─ Week 7: Prompt Engineering
  └─ Week 8: Mini项目快速原型

月份3：高性能推理
  ├─ Week 9-10: C++推理引擎
  ├─ Week 11: 量化和SIMD优化
  └─ Week 12: 整合和优化
```

### 学习目标检查清单

**理论知识：**
- [ ] 理解Transformer架构和自注意力机制
- [ ] 掌握向量嵌入和语义搜索原理
- [ ] 了解LLM推理过程和性能瓶颈
- [ ] 熟悉量化技术的数学基础

**技术能力：**
- [ ] 能够使用LangChain构建RAG应用
- [ ] 熟练编写和优化Prompt
- [ ] 掌握向量数据库的使用
- [ ] 能够实现KV Cache和INT8量化
- [ ] 会使用SIMD指令优化计算

**项目经验：**
- [ ] 完成一个RAG应用（代码问答系统）
- [ ] 实现一个高性能推理引擎
- [ ] 开发4个以上Mini原型项目
- [ ] 有完整的性能测试和优化经验

---

## 阶段一：基础理论

### Week 1-2: LLM基础理论

**学习内容：**

```
Day 1-2: Transformer架构
├─ 论文阅读："Attention Is All You Need"
├─ 视频：Stanford CS224N Lecture
├─ 实践：手写简化版Self-Attention
└─ 检查点：能解释QKV矩阵的作用

Day 3-4: 位置编码和多头注意力
├─ 实现：Sinusoidal Position Encoding
├─ 实现：Multi-Head Attention
├─ 对比：绝对 vs 相对位置编码
└─ 检查点：理解RoPE的优势

Day 5-6: 自回归生成
├─ 理论：Prefill vs Decode阶段
├─ 实践：实现贪婪解码和Top-K采样
├─ 分析：为什么Decode是瓶颈
└─ 检查点：计算生成100 token的FLOPs

Day 7: LLM架构演进
├─ GPT系列：GPT-1 → GPT-4的变化
├─ Llama系列：架构特点和优化
├─ 其他：Mixtral (MoE), Phi (小模型)
└─ 检查点：能说出5个主流LLM的特点

Day 8-10: 数学基础
├─ 线性代数：矩阵乘法、向量范数
├─ 概率论：Softmax、交叉熵
├─ 优化：梯度下降、Adam
└─ 检查点：推导Softmax的梯度

Day 11-14: 代码实践
├─ PyTorch基础
├─ 实现简化版Transformer
├─ 使用Hugging Face加载模型
└─ 检查点：运行GPT-2生成文本
```

**推荐资源：**

- **论文：**
  - Attention Is All You Need (必读)
  - Language Models are Few-Shot Learners (GPT-3)
  - LLaMA: Open and Efficient Foundation Language Models

- **课程：**
  - Stanford CS224N (NLP with Deep Learning)
  - Andrej Karpathy: "Let's build GPT"
  - Fast.ai: Practical Deep Learning

- **代码：**
  - minGPT (Andrej Karpathy)
  - nanoGPT (简化版GPT)
  - Annotated Transformer

### Week 3: Python AI生态

**Day 15-16: LangChain框架**
```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 学习核心概念
- Models (LLM wrappers)
- Prompts (模板和管理)
- Chains (组合多个步骤)
- Agents (自主决策)
```

**Day 17-18: 向量数据库**
```python
import chromadb

# 实践操作
- 创建collection
- 添加文档和embedding
- 相似度搜索
- 元数据过滤

# 对比不同向量数据库
- ChromaDB: 轻量级，Python友好
- FAISS: Meta开源，性能强
- Pinecone: 云服务，可扩展
- Weaviate: 功能丰富
```

**Day 19-21: OpenAI API深入**
```python
# Chat Completions API
- 角色设定（system, user, assistant）
- 流式输出
- 函数调用（Function Calling）
- Token计数和成本估算

# Embeddings API
- text-embedding-3-small vs large
- 批量处理
- 维度降低
```

### Week 4: 向量和检索

**Day 22-24: Embedding深入**
```
理论：
- Word2Vec, GloVe (传统方法)
- BERT, Sentence-BERT (现代方法)
- Contrastive Learning原理

实践：
- Fine-tune embedding模型
- 评估embedding质量
- 降维可视化（t-SNE, UMAP）
```

**Day 25-28: 检索算法**
```
实现：
- 暴力搜索
- LSH (Locality Sensitive Hashing)
- HNSW (Hierarchical NSW)
- IVF (Inverted File)

对比：
- 速度 vs 精度权衡
- 内存占用
- 构建索引时间
```

---

## 阶段二：核心技术

### Week 5-6: RAG系统开发

**学习路径：**

```
Day 29-30: 文档处理
├─ 加载：PDF, Markdown, 代码
├─ 分块策略：固定大小 vs 语义分块
├─ 元数据提取
└─ 项目：实现通用文档加载器

Day 31-32: 向量化和索引
├─ Batch embedding优化
├─ 增量索引
├─ 索引持久化
└─ 项目：索引10K+文档

Day 33-34: 检索优化
├─ Top-K调优
├─ 混合检索（关键词+语义）
├─ 重排序（MMR, Cross-Encoder）
└─ 项目：提升检索精度10%

Day 35-36: 生成优化
├─ Prompt工程
├─ 上下文压缩
├─ 引用来源
└─ 项目：实现可溯源的答案

Day 37-40: 代码库RAG特化
├─ AST-based分块（tree-sitter）
├─ 代码语义理解
├─ 多语言支持
└─ 项目：完成代码问答系统v1
```

**里程碑项目：代码问答系统**

```python
# 目标功能
1. 索引代码库（10+语言）
2. 语义代码搜索
3. 代码解释
4. Bug检测
5. 使用示例生成

# 性能目标
- 检索准确率：80%+
- 响应时间：<2秒
- 支持10K+行代码
```

### Week 7: Prompt Engineering

**Day 41-42: 基础技术**
```
实践：
- Zero-Shot vs Few-Shot
- Chain of Thought
- Self-Consistency

练习：
- 设计5个不同任务的prompt
- A/B测试不同prompt
- 测量质量提升
```

**Day 43-44: 高级策略**
```
学习：
- ReAct (Reasoning + Acting)
- Tree of Thought
- Meta-Prompting

项目：
- 实现Prompt优化器
- 自动生成Few-Shot示例
- Prompt版本管理
```

**Day 45-47: 项目应用**
```
将Prompt Engineering应用到：
- 代码审查（详细的检查清单）
- 代码生成（遵循最佳实践）
- Bug检测（系统化分析）

结果：
- 每个任务3+个优化的prompt模板
- 性能对比报告
```

### Week 8: Mini项目周

**快速原型实践：**

```
Day 48-49: Mini-RAG
├─ 目标：简化版RAG，500行
├─ 技术：FAISS + OpenAI
├─ 学习：RAG最小可行实现
└─ 成果：可运行的demo

Day 50-51: 量化工具
├─ 目标：INT8/INT4量化器
├─ 技术：NumPy数值计算
├─ 学习：量化数学和实现
└─ 成果：量化+精度评估

Day 52-53: Prompt优化器
├─ 目标：7种优化策略
├─ 技术：LLM API + 评估框架
├─ 学习：系统化Prompt工程
└─ 成果：A/B测试工具

Day 54-55: 基准测试工具
├─ 目标：性能测试框架
├─ 技术：异步调用 + 统计
├─ 学习：性能分析方法
└─ 成果：TTFT/TPS测量

Day 56: 整合和文档
├─ 代码重构
├─ 添加测试
├─ 编写README
└─ 发布到GitHub
```

---

## 阶段三：项目实践

### Week 9-10: C++推理引擎基础

**Day 57-59: C++17现代特性**
```cpp
学习：
- 智能指针（unique_ptr, shared_ptr）
- lambda表达式和函数对象
- STL容器和算法
- 移动语义

实践：
- 实现简单的矩阵类
- 使用智能指针管理内存
- STL算法练习
```

**Day 60-63: KV Cache实现**
```cpp
Day 60: 设计接口
class KVCache {
public:
    KVCache(size_t max_seq_len, size_t n_layers,
            size_t n_heads, size_t head_dim);
    void update_k(size_t layer, const float* k, size_t pos);
    const float* get_k(size_t layer) const;
    // ...
};

Day 61-62: 实现核心功能
- 预分配连续内存
- O(1)更新操作
- 内存对齐优化

Day 63: 测试和基准
- 单元测试
- 性能测试（Prefill vs Decode）
- 内存使用分析
```

**Day 64-68: 量化实现**
```cpp
Day 64-65: INT8对称量化
- 计算scale
- 量化/反量化
- 误差分析

Day 66-67: INT4分组量化
- 分组策略
- 打包/解包
- 精度vs压缩权衡

Day 68: INT8矩阵乘法
- 实现quantized matmul
- 性能对比
```

**Day 69-70: 整合测试**
```
- 组合优化测试
- 端到端基准测试
- 文档和README
```

### Week 11: SIMD优化

**Day 71-73: SIMD基础**
```cpp
Day 71: AVX2指令集
- 寄存器和指令
- 数据加载和存储
- 基本运算（+, -, *, FMA）

Day 72: 向量点积
float dot_avx2(const float* a, const float* b, int n);

Day 73: 矩阵乘法
void matmul_avx2(const float* A, const float* B,
                 float* C, int m, int k, int n);
```

**Day 74-77: 优化实战**
```
Day 74: 内存对齐
- 32字节对齐分配
- 对齐 vs 非对齐加载

Day 75-76: 量化+SIMD
- INT8 SIMD优化
- 混合精度计算

Day 77: 性能分析
- 使用perf工具
- 找出瓶颈
- 优化热点代码
```

### Week 12: 整合和优化

**Day 78-80: 完整推理流程**
```
实现：
- 完整的attention计算
- 多层Transformer
- 端到端推理

测试：
- 与PyTorch对比精度
- 性能基准测试
```

**Day 81-84: 高级优化**
```
Day 81: Flash Attention
- 理解算法
- 分块计算
- 内存优化

Day 82-83: PagedAttention（可选）
- 分页KV Cache
- 动态内存管理

Day 84: 批处理
- 批量推理
- 动态批处理
```

**Day 85-87: 文档和发布**
```
- 完整的README
- API文档
- 快速开始指南
- 性能报告
```

---

## 阶段四：深入优化

### 高级推理优化

**Flash Attention 2**
```
论文阅读：
- FlashAttention: Fast and Memory-Efficient Exact Attention
- FlashAttention-2: Faster Attention with Better Parallelism

实现：
- 分块计算策略
- GPU共享内存优化
- 性能提升：2-4x
```

**投机解码（Speculative Decoding）**
```
原理：
- 小模型快速生成K个token
- 大模型并行验证
- 接受正确的，拒绝错误的

实现：
- 双模型架构
- 并行验证逻辑
- 性能提升：2-3x
```

**批处理优化**
```
技术：
- Continuous Batching
- 动态调整批大小
- 请求调度策略

框架：
- vLLM实现分析
- TensorRT-LLM学习
```

### 高级RAG技术

**GraphRAG**
```
思想：
- 构建知识图谱
- 图结构检索
- 关系推理

实现：
- Neo4j + Vector Search
- 实体关系提取
- 图遍历算法
```

**Multi-Modal RAG**
```
扩展：
- 图片 + 文本检索
- 代码 + 文档联合
- 表格数据处理

模型：
- CLIP (图文)
- CodeBERT (代码)
```

**Agent-based RAG**
```
架构：
- ReAct框架
- 工具调用
- 多步推理

应用：
- 复杂问题分解
- 多数据源整合
```

---

## 进阶方向

### 方向1：模型训练和微调

**预训练（需要大量资源）**
```
理论：
- Tokenization (BPE, SentencePiece)
- 训练超参数调优
- 数据清洗和去重

实践：
- 训练小模型（1B参数）
- 使用Megatron-LM
- 分布式训练（DDP, FSDP)
```

**微调（Fine-tuning）**
```
技术：
- Full Fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)

框架：
- Hugging Face Transformers
- Axolotl
- LLaMA Factory

项目：
- 在特定领域微调模型
- 指令微调（Instruction Tuning）
- RLHF（强化学习）
```

### 方向2：生产部署

**模型服务**
```
框架：
- vLLM (高吞吐)
- TensorRT-LLM (低延迟)
- Text Generation Inference (Hugging Face)

优化：
- 模型并行
- Pipeline并行
- 混合精度推理
```

**监控和运维**
```
指标：
- Latency (P50/P95/P99)
- Throughput (QPS)
- GPU利用率
- 成本分析

工具：
- Prometheus + Grafana
- LangSmith (LLM专用)
- Weights & Biases
```

### 方向3：研究前沿

**跟踪最新进展**
```
资源：
- ArXiv: cs.CL, cs.LG
- Hugging Face Daily Papers
- Twitter/X: AI研究者
- Papers with Code

关键会议：
- NeurIPS
- ICML
- ACL, EMNLP
```

**研究方向：**
- 长上下文（100K+ tokens）
- 多模态（视觉+语言）
- 小模型高性能（Phi, Gemma）
- 高效推理（Flash Attention 3）
- Agent系统

---

## 学习建议

### 时间分配

**每天8小时学习：**
```
2小时：理论学习（论文、视频）
4小时：编码实践
1小时：阅读他人代码
1小时：写文档/博客总结
```

**每周计划：**
```
周一-周五：主线学习
周六：回顾总结，整理笔记
周日：自由探索，阅读论文
```

### 学习方法

**1. 主动学习**
```
✅ 动手实现，不只是看代码
✅ 提出问题，寻找答案
✅ 对比不同方法
✅ 做笔记，画图解释
```

**2. 项目驱动**
```
✅ 每个知识点都用项目巩固
✅ 从简单到复杂，逐步迭代
✅ 发布到GitHub，获得反馈
```

**3. 社区参与**
```
✅ 加入Discord/Slack社群
✅ 参与开源项目
✅ 写技术博客
✅ 回答他人问题（教是最好的学）
```

### 避免常见陷陷阱

**❌ 只看不练**
```
只看教程，不写代码 → 理解浅薄
解决：每学一个概念，立即实践
```

**❌ 追求完美**
```
一个项目永远不发布，不断重构 → 进度缓慢
解决：先完成MVP，再迭代优化
```

**❌ 盲目追新**
```
每天追最新的模型，不深入理解 → 浮于表面
解决：先掌握基础，再扩展广度
```

**❌ 孤立学习**
```
一个人闭门造车 → 缺少反馈和动力
解决：加入社群，分享进展
```

---

## 里程碑检查

### 第1个月结束

**能够：**
- [ ] 解释Transformer的工作原理
- [ ] 使用LangChain构建简单应用
- [ ] 理解向量检索的数学原理
- [ ] 独立实现一个Mini-RAG

**产出：**
- [ ] 2-3篇学习笔记/博客
- [ ] 1个RAG原型项目
- [ ] GitHub有commits记录

### 第2个月结束

**能够：**
- [ ] 设计和实现完整RAG系统
- [ ] 编写高质量的Prompt
- [ ] 优化检索和生成质量
- [ ] 评估RAG系统性能

**产出：**
- [ ] 1个生产级RAG项目（代码问答）
- [ ] 4个Mini项目
- [ ] Prompt优化案例集

### 第3个月结束

**能够：**
- [ ] 实现KV Cache和量化
- [ ] 使用SIMD优化计算
- [ ] 分析和优化推理性能
- [ ] 端到端部署LLM应用

**产出：**
- [ ] C++推理引擎项目
- [ ] 完整的性能测试报告
- [ ] 技术文档和使用指南
- [ ] 可展示的Portfolio

---

## 资源清单

### 必读论文（20篇）

**基础（5篇）：**
1. Attention Is All You Need
2. BERT: Pre-training of Deep Bidirectional Transformers
3. Language Models are Few-Shot Learners (GPT-3)
4. LLaMA: Open and Efficient Foundation Language Models
5. Training language models to follow instructions (InstructGPT)

**优化（8篇）：**
6. FlashAttention: Fast and Memory-Efficient Exact Attention
7. PagedAttention (vLLM)
8. LLM.int8(): 8-bit Matrix Multiplication
9. GPTQ: Accurate Post-Training Quantization
10. LoRA: Low-Rank Adaptation of Large Language Models
11. QLoRA: Efficient Finetuning of Quantized LLMs
12. Speculative Decoding
13. Flash Attention 2

**RAG（4篇）：**
14. Retrieval-Augmented Generation for Knowledge-Intensive Tasks
15. Dense Passage Retrieval for Open-Domain QA
16. HyDE: Precise Zero-Shot Dense Retrieval
17. Self-RAG: Learning to Retrieve, Generate, and Critique

**Prompt（3篇）：**
18. Chain-of-Thought Prompting
19. ReAct: Synergizing Reasoning and Acting
20. Tree of Thoughts

### 关键开源项目

**学习用：**
- nanoGPT (Andrej Karpathy)
- minGPT
- Annotated Transformer

**生产用：**
- vLLM
- TensorRT-LLM
- llama.cpp
- LangChain
- LlamaIndex

### 在线课程

1. **Stanford CS224N** - NLP with Deep Learning
2. **Fast.ai** - Practical Deep Learning
3. **DeepLearning.AI** - ChatGPT Prompt Engineering
4. **Hugging Face Course** - NLP Course

---

## 完整学习检查清单

> **使用方式：** 复制此检查清单到自己的学习笔记中，逐项打勾记录进度

### 📅 第1个月：基础理论

#### Week 1-2: LLM基础理论 ✅

**理论理解：**
- [ ] 能解释Transformer为什么比RNN更适合并行计算
- [ ] 能手写Self-Attention的公式并解释QKV的含义
- [ ] 理解Multi-Head Attention的作用（捕捉不同模式）
- [ ] 能说出至少3种位置编码方法及其优缺点
- [ ] 理解自回归生成的Prefill和Decode两个阶段
- [ ] 能计算生成N个token需要的计算量（FLOPs）

**代码实践：**
- [ ] 成功安装PyTorch和Transformers库
- [ ] 能用NumPy手写简化版Self-Attention（20行代码）
- [ ] 使用Hugging Face加载并运行GPT-2模型
- [ ] 实现贪婪解码、Top-K采样、Top-P采样
- [ ] 对比不同temperature参数对生成结果的影响

**论文阅读：**
- [ ] 阅读《Attention is All You Need》论文
- [ ] 理解GPT-3论文中的Scaling Law
- [ ] 浏览LLaMA/LLaMA2技术报告

---

#### Week 3: Python AI生态系统 ✅

**工具掌握：**
- [ ] 熟练使用Jupyter Notebook/Lab
- [ ] 掌握Pandas进行数据处理
- [ ] 了解LangChain的基本概念（Chains, Agents）
- [ ] 能使用OpenAI API完成基础调用

**环境搭建：**
- [ ] 配置好Python虚拟环境（venv/conda）
- [ ] 安装并测试CUDA（如有GPU）
- [ ] 熟悉pip/conda包管理

---

#### Week 4: 向量数据库和Embedding ✅

**理论理解：**
- [ ] 理解Embedding的本质（文本→高维向量）
- [ ] 能解释余弦相似度的计算方法
- [ ] 了解常见Embedding模型（text-embedding-3, BERT）
- [ ] 理解向量索引算法（HNSW, IVF）

**实践能力：**
- [ ] 使用OpenAI API生成Embedding
- [ ] 安装并使用ChromaDB存储向量
- [ ] 实现简单的语义搜索（100行代码）
- [ ] 对比不同Embedding模型的效果

---

### 📅 第2个月：核心技术实践

#### Week 5-6: RAG系统开发 ✅

**系统理解：**
- [ ] 能画出RAG系统的完整流程图
- [ ] 理解Chunking策略（固定大小 vs 语义分块）
- [ ] 掌握Top-K检索和MMR检索的区别
- [ ] 了解RAG评估指标（Relevance, Faithfulness）

**项目完成：**
- [ ] 完成代码问答RAG系统（参考本项目）
- [ ] 实现文档上传和索引功能
- [ ] 支持至少3种编程语言的代码检索
- [ ] 添加Gradio Web界面
- [ ] 编写完整的README和使用文档

**性能优化：**
- [ ] 测试并优化Chunk大小（500/1000/2000字符）
- [ ] 测试并优化Top-K值（3/5/10）
- [ ] 实现缓存机制减少API调用
- [ ] 检索准确率达到80%+

---

#### Week 7: Prompt Engineering ✅

**技术掌握：**
- [ ] 能熟练使用Few-Shot Prompting
- [ ] 理解Chain-of-Thought的原理和应用场景
- [ ] 掌握ReAct框架（Thought-Action-Observation）
- [ ] 了解Self-Consistency和ToT（Tree of Thoughts）

**实践经验：**
- [ ] 为至少5个不同任务编写优化的Prompt
- [ ] 进行A/B测试对比Prompt效果
- [ ] 实现简单的Prompt模板系统
- [ ] 测量并记录不同Prompt的成功率

---

#### Week 8: Mini项目快速原型 ✅

**项目完成：**
- [ ] 完成Mini-RAG项目（简化版RAG）
- [ ] 完成量化工具项目（INT8/INT4）
- [ ] 完成Prompt优化器项目
- [ ] 完成基准测试工具项目

**能力提升：**
- [ ] 能在1-2天内快速实现一个原型
- [ ] 掌握快速验证想法的方法
- [ ] 学会取舍（MVP vs 完美）

---

### 📅 第3个月：高性能推理

#### Week 9-10: C++推理引擎 ✅

**理论理解：**
- [ ] 理解KV Cache的原理和实现
- [ ] 掌握量化的数学基础（scale计算）
- [ ] 了解SIMD向量化原理
- [ ] 能计算内存带宽对性能的影响

**项目完成：**
- [ ] 实现完整的KV Cache系统
- [ ] 实现INT8对称量化
- [ ] 使用AVX2优化矩阵乘法
- [ ] 编写完整的单元测试

**性能达标：**
- [ ] KV Cache实现20倍加速
- [ ] INT8量化SQNR >40dB
- [ ] SIMD优化达到3倍加速
- [ ] 整体推理速度提升30倍

---

#### Week 11: 量化和SIMD优化 ✅

**深入理解：**
- [ ] 理解对称量化 vs 非对称量化
- [ ] 掌握Per-Channel和Per-Group量化
- [ ] 了解GPTQ、AWQ等高级量化方法
- [ ] 理解AVX2指令集的使用

**代码实践：**
- [ ] 实现多种量化方法并对比
- [ ] 优化关键热点代码（profiling）
- [ ] 使用内存对齐优化性能
- [ ] 编写性能基准测试

---

#### Week 12: 整合和优化 ✅

**项目整合：**
- [ ] 整合所有项目到一个代码库
- [ ] 编写统一的构建系统（CMake/Make）
- [ ] 完善所有项目的文档
- [ ] 准备项目演示材料

**性能优化：**
- [ ] 使用Profiler找出瓶颈
- [ ] 优化内存使用
- [ ] 减少不必要的内存拷贝
- [ ] 达到预期性能指标

---

### 📚 进阶学习检查清单

#### 模型训练与微调（可选）

- [ ] 理解LoRA的低秩分解原理
- [ ] 掌握QLoRA的4-bit量化技术
- [ ] 了解RLHF的三个训练阶段
- [ ] 理解DPO相比RLHF的优势
- [ ] 完成一次LoRA微调实验
- [ ] 评估微调模型的性能提升

#### Agent系统（可选）

- [ ] 理解Agent的四大组件（感知/规划/行动/记忆）
- [ ] 掌握ReAct框架的实现
- [ ] 了解Function Calling机制
- [ ] 实现一个简单的工具使用Agent
- [ ] 学习多Agent协作框架（AutoGen）

#### 生产部署（可选）

- [ ] 掌握vLLM的使用和配置
- [ ] 了解Continuous Batching原理
- [ ] 学会使用Docker部署模型
- [ ] 掌握Prometheus+Grafana监控
- [ ] 实现简单的负载均衡

#### 数据工程（可选）

- [ ] 掌握数据采集方法（爬虫/API）
- [ ] 学会使用MinHash去重
- [ ] 了解数据清洗最佳实践
- [ ] 掌握标注质量控制方法
- [ ] 学习Self-Instruct数据生成

#### MLOps（可选）

- [ ] 掌握W&B实验管理
- [ ] 学会使用DVC进行数据版本控制
- [ ] 了解CI/CD在ML中的应用
- [ ] 掌握模型监控和数据漂移检测
- [ ] 学习成本优化策略

---

### 🎯 阶段性自测

#### 第1个月末自测

**理论测试：**
- [ ] 在白板上画出Transformer架构并解释每个组件
- [ ] 计算一个简单Attention示例的输出
- [ ] 解释为什么Decode阶段是内存瓶颈

**实践测试：**
- [ ] 30分钟内从零搭建一个简单RAG系统
- [ ] 能快速诊断和修复常见错误
- [ ] 代码风格规范，有适当注释

---

#### 第2个月末自测

**项目评估：**
- [ ] RAG系统检索准确率 >80%
- [ ] 响应时间 <2秒（P95）
- [ ] 代码库有完整的README和测试

**能力评估：**
- [ ] 能独立设计一个RAG系统架构
- [ ] 能评估不同技术方案的优劣
- [ ] 能快速debug和解决问题

---

#### 第3个月末自测

**性能指标：**
- [ ] 推理引擎性能达标（30倍加速）
- [ ] 量化精度损失 <1%
- [ ] 所有测试用例通过

**综合能力：**
- [ ] 能从0到1开发一个完整AIGC项目
- [ ] 理解从理论到工程实现的完整链路
- [ ] 具备性能分析和优化能力
- [ ] 能向他人清晰讲解技术细节

---

### ✅ 最终检查清单

**知识体系：**
- [ ] 阅读完所有14个理论文档
- [ ] 理解所有核心概念和术语
- [ ] 能应用理论解决实际问题

**项目经验：**
- [ ] 完成3个以上完整项目
- [ ] 有至少10,000行代码积累
- [ ] 每个项目都有文档和测试

**技术能力：**
- [ ] Python AI开发（LangChain, Transformers）
- [ ] C++性能优化（SIMD, 量化）
- [ ] RAG系统设计与实现
- [ ] Prompt工程
- [ ] 性能分析和优化

**求职准备（如适用）：**
- [ ] 简历突出AIGC项目经验
- [ ] 准备项目演示材料
- [ ] 复习常见面试题
- [ ] 准备技术深度问题的回答

---

## 下一步行动

**立即开始：**

1. **今天：**
   - 阅读本文档的[LLM基础理论](01_LLM_Fundamentals.md)
   - 安装Python环境和必要库

2. **本周：**
   - 完成Week 1的学习计划
   - 实现简单的Self-Attention

3. **本月：**
   - 按照路线图完成第一阶段
   - 开始第一个RAG项目

**持续进步：**
- 每天记录学习日志
- 每周发布一篇总结
- 每月回顾并调整计划

---

**祝你在AIGC领域的学习之旅顺利！记住：最重要的是开始行动。🚀**
