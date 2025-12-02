# AIGC术语表与速查表

> **文档定位：** AIGC领域核心术语快速参考手册
> **适用对象：** 所有学习者（初学者查概念，进阶者查公式）
> **使用方式：** Ctrl+F 搜索术语，快速定位到定义

---

## 目录

1. [核心术语表（A-Z）](#核心术语表)
2. [重要公式速查](#重要公式速查)
3. [常用工具速查](#常用工具速查)
4. [文档导航索引](#文档导航索引)
5. [缩写对照表](#缩写对照表)

---

## 核心术语表

### A

**Attention（注意力机制）**
- **定义：** 神经网络中用于关注输入序列不同部分的机制
- **公式：** `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
- **应用：** Transformer的核心组件
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Autoregressive（自回归）**
- **定义：** 模型根据先前生成的token逐个生成下一个token
- **特点：** P(x₁...xₙ) = P(x₁)P(x₂|x₁)...P(xₙ|x₁...xₙ₋₁)
- **应用：** GPT、LLaMA等语言模型
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Agent（智能体）**
- **定义：** 能够感知环境、做出决策并执行行动的AI系统
- **组成：** 感知 + 规划 + 行动 + 记忆
- **框架：** ReAct, Plan-and-Execute
- **参考：** [08_Agent_Systems.md](./08_Agent_Systems.md)

**ALiBi（Attention with Linear Biases）**
- **定义：** 一种不使用位置编码的长上下文优化技术
- **优势：** 可外推到更长的序列
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

---

### B

**BLEU（Bilingual Evaluation Understudy）**
- **定义：** 机器翻译质量评估指标
- **范围：** 0-1（1表示完美匹配）
- **局限：** 只考虑n-gram重叠，不考虑语义
- **参考：** [07_LLM_Evaluation.md](./07_LLM_Evaluation.md)

**Beam Search（束搜索）**
- **定义：** 生成文本时保留top-k个候选序列的搜索算法
- **参数：** beam_size（通常3-5）
- **权衡：** 质量 vs 速度
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Batch Size（批大小）**
- **定义：** 每次训练迭代处理的样本数量
- **影响：** 显存占用、训练稳定性、收敛速度
- **典型值：** 8-64（训练），1-32（推理）
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

---

### C

**Chain-of-Thought（CoT, 思维链）**
- **定义：** 通过展示推理步骤来改进复杂推理任务的Prompt技术
- **示例：** "让我们一步步思考..."
- **效果：** 在数学、逻辑推理任务上显著提升
- **参考：** [04_Prompt_Engineering.md](./04_Prompt_Engineering.md)

**CLIP（Contrastive Language-Image Pre-training）**
- **定义：** OpenAI的视觉-语言对比学习模型
- **应用：** 图文匹配、零样本图像分类
- **架构：** 图像编码器 + 文本编码器
- **参考：** [09_Multimodal_AI.md](./09_Multimodal_AI.md)

**Constitutional AI**
- **定义：** 通过宪法原则（规则）引导模型行为的对齐技术
- **方法：** 让模型自我批评和修正输出
- **提出者：** Anthropic
- **参考：** [11_Safety_and_Alignment.md](./11_Safety_and_Alignment.md)

**Context Window（上下文窗口）**
- **定义：** 模型一次能处理的最大token数量
- **典型值：** GPT-3.5(4K), GPT-4(8K/32K), Claude(100K), GPT-4-Turbo(128K)
- **影响：** 决定能处理的文档长度
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Continuous Batching（连续批处理）**
- **定义：** vLLM等推理引擎的动态批处理技术
- **优势：** 吞吐量提升2-3倍
- **原理：** 不等待所有序列完成，持续添加新请求
- **参考：** [10_Production_Deployment.md](./10_Production_Deployment.md)

---

### D

**Decoder-Only（仅解码器）**
- **定义：** 只使用Transformer解码器的架构
- **代表模型：** GPT系列、LLaMA、Claude
- **特点：** 适合文本生成任务
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**DPO（Direct Preference Optimization）**
- **定义：** 直接从偏好数据优化模型的对齐技术
- **优势：** 比RLHF更简单，无需训练奖励模型
- **公式：** 最大化`log σ(β log π(y_w|x) / π_ref(y_w|x) - β log π(y_l|x) / π_ref(y_l|x))`
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**Data Drift（数据漂移）**
- **定义：** 生产环境数据分布与训练数据分布的偏移
- **检测方法：** KS检验、PSI（Population Stability Index）
- **影响：** 模型性能下降
- **参考：** [14_MLOps_Best_Practices.md](./14_MLOps_Best_Practices.md)

---

### E

**Embedding（嵌入）**
- **定义：** 将文本、图像等转换为固定长度的向量表示
- **维度：** 256-1536（常见）
- **用途：** 语义搜索、RAG、相似度计算
- **模型：** OpenAI text-embedding-3-small/large, Sentence-BERT
- **参考：** [03_RAG_System_Theory.md](./03_RAG_System_Theory.md)

**Encoder-Decoder（编码器-解码器）**
- **定义：** 同时使用编码器和解码器的Transformer架构
- **代表模型：** T5, BART
- **特点：** 适合翻译、摘要等seq2seq任务
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Evol-Instruct**
- **定义：** 通过增加深度和广度来演化指令的数据增强技术
- **方法：** 增加复杂度、改变场景、应用到新领域
- **应用：** 生成高质量指令数据
- **参考：** [13_Data_Engineering.md](./13_Data_Engineering.md)

---

### F

**Few-Shot Learning（少样本学习）**
- **定义：** 在Prompt中提供少量示例来引导模型
- **示例数：** 通常1-10个
- **优势：** 无需微调即可适应新任务
- **参考：** [04_Prompt_Engineering.md](./04_Prompt_Engineering.md)

**Fine-tuning（微调）**
- **定义：** 在预训练模型基础上，使用特定数据继续训练
- **类型：** Full Fine-tuning, LoRA, QLoRA
- **用途：** 适配特定领域或任务
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**Flash Attention**
- **定义：** 内存高效的注意力计算算法
- **优势：** 减少HBM访问，速度提升2-4倍
- **版本：** Flash Attention 2/3
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

**FSDP（Fully Sharded Data Parallel）**
- **定义：** PyTorch的分布式训练技术
- **优势：** 将模型、梯度、优化器状态分片到多GPU
- **应用：** 大模型训练
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

---

### G

**Gradient Accumulation（梯度累积）**
- **定义：** 累积多个小batch的梯度后再更新参数
- **用途：** 在有限显存下模拟大batch训练
- **公式：** `effective_batch_size = batch_size × accumulation_steps`
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**Greedy Decoding（贪婪解码）**
- **定义：** 每步选择概率最高的token
- **优势：** 速度快，确定性
- **劣势：** 可能陷入重复，质量不如采样
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**GPTQ（GPT Quantization）**
- **定义：** 一种后训练量化方法
- **特点：** 保持精度的同时压缩模型
- **应用：** 4-bit量化
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

---

### H

**Hallucination（幻觉）**
- **定义：** 模型生成的看似合理但实际错误的信息
- **类型：** 事实错误、逻辑错误、编造细节
- **检测：** 事实核查、引用验证
- **参考：** [07_LLM_Evaluation.md](./07_LLM_Evaluation.md)

**HumanEval**
- **定义：** OpenAI的代码生成能力评估基准
- **任务：** 164个Python编程问题
- **指标：** Pass@K（生成K个候选中至少1个通过）
- **参考：** [07_LLM_Evaluation.md](./07_LLM_Evaluation.md)

---

### I

**In-Context Learning（上下文学习）**
- **定义：** 通过Prompt中的示例学习，无需参数更新
- **机制：** Few-shot learning的理论基础
- **发现：** GPT-3展示的涌现能力
- **参考：** [04_Prompt_Engineering.md](./04_Prompt_Engineering.md)

**Instruction Tuning（指令微调）**
- **定义：** 使用指令-回复对微调模型以提升指令遵循能力
- **数据格式：** Alpaca, ShareGPT
- **代表模型：** InstructGPT, Alpaca
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**INT8/INT4 Quantization（整数量化）**
- **定义：** 将FP16/FP32权重转换为8-bit/4-bit整数
- **压缩比：** INT8(4x), INT4(8x)
- **精度损失：** INT8(<1%), INT4(1-3%)
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

---

### K

**KV Cache（键值缓存）**
- **定义：** 缓存注意力机制中的Key和Value矩阵
- **优势：** 避免重复计算，20倍加速
- **代价：** 占用额外显存（每token约1MB）
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

---

### L

**LangChain**
- **定义：** LLM应用开发框架
- **组件：** Chains, Agents, Memory, Tools
- **应用：** RAG、Agent、多步骤推理
- **参考：** [03_RAG_System_Theory.md](./03_RAG_System_Theory.md)

**LLaVA（Large Language and Vision Assistant）**
- **定义：** 开源视觉语言模型
- **架构：** CLIP视觉编码器 + LLM
- **能力：** 图像理解、VQA
- **参考：** [09_Multimodal_AI.md](./09_Multimodal_AI.md)

**LoRA（Low-Rank Adaptation）**
- **定义：** 通过低秩矩阵微调大模型的参数高效方法
- **公式：** `W' = W + BA`，其中B∈ℝᵈˣʳ, A∈ℝʳˣᵏ, r<<min(d,k)
- **优势：** 只需训练0.1-1%的参数
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**LLM（Large Language Model，大语言模型）**
- **定义：** 参数量在数十亿以上的语言模型
- **代表：** GPT-4, Claude, LLaMA, Gemini
- **能力：** 文本生成、理解、推理、翻译等
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

---

### M

**MMLU（Massive Multitask Language Understanding）**
- **定义：** 涵盖57个学科的综合评估基准
- **题目：** 选择题，涵盖STEM、人文、社科等
- **难度：** 从小学到专业水平
- **参考：** [07_LLM_Evaluation.md](./07_LLM_Evaluation.md)

**MoE（Mixture of Experts）**
- **定义：** 稀疏激活的专家混合架构
- **原理：** 每次只激活部分专家
- **优势：** 参数多但计算少
- **代表：** Mixtral, GPT-4(推测)
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

**Multi-Head Attention（多头注意力）**
- **定义：** 并行运行多个注意力头
- **公式：** `MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O`
- **头数：** 通常8-32
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

---

### P

**PEFT（Parameter-Efficient Fine-Tuning）**
- **定义：** 参数高效微调方法的统称
- **方法：** LoRA, Adapter, Prefix Tuning, P-Tuning
- **优势：** 节省显存和存储
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**Perplexity（困惑度）**
- **定义：** 语言模型的评估指标，越低越好
- **公式：** `PPL = exp(-1/N Σ log P(xᵢ|x₁...xᵢ₋₁))`
- **含义：** 模型对下一个token的平均不确定性
- **参考：** [07_LLM_Evaluation.md](./07_LLM_Evaluation.md)

**Positional Encoding（位置编码）**
- **定义：** 为序列中的token添加位置信息
- **类型：** 绝对位置（Sinusoidal, Learned）、相对位置（RoPE, ALiBi）
- **必要性：** Attention本身无法区分token顺序
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Prompt Engineering（提示工程）**
- **定义：** 设计有效提示词以引导模型输出的技术
- **技术：** Few-shot, CoT, ReAct, Self-Consistency
- **重要性：** 可使性能提升10-50%
- **参考：** [04_Prompt_Engineering.md](./04_Prompt_Engineering.md)

**PSI（Population Stability Index）**
- **定义：** 数据分布稳定性指标
- **范围：** <0.1(稳定), 0.1-0.2(轻微变化), >0.2(显著变化)
- **应用：** 数据漂移检测
- **参考：** [14_MLOps_Best_Practices.md](./14_MLOps_Best_Practices.md)

---

### Q

**QLoRA（Quantized LoRA）**
- **定义：** 结合量化和LoRA的高效微调方法
- **技术：** 4-bit NormalFloat量化 + LoRA
- **优势：** 在单卡上微调65B模型
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**Quantization（量化）**
- **定义：** 降低模型权重精度以压缩模型
- **类型：** PTQ（后训练量化）, QAT（量化感知训练）
- **精度：** FP32 → FP16 → INT8 → INT4
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

---

### R

**RAG（Retrieval-Augmented Generation）**
- **定义：** 检索增强生成，结合外部知识库的生成方法
- **流程：** 查询 → 检索相关文档 → 拼接上下文 → 生成回答
- **优势：** 减少幻觉、知识可更新
- **参考：** [03_RAG_System_Theory.md](./03_RAG_System_Theory.md)

**ReAct（Reasoning + Acting）**
- **定义：** 交替进行推理和行动的Agent框架
- **格式：** Thought → Action → Observation → ...
- **应用：** 工具使用、复杂任务分解
- **参考：** [08_Agent_Systems.md](./08_Agent_Systems.md)

**RLHF（Reinforcement Learning from Human Feedback）**
- **定义：** 从人类反馈中强化学习，用于模型对齐
- **步骤：** 1) SFT, 2) 训练奖励模型, 3) PPO优化
- **代表：** ChatGPT, Claude
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**RoPE（Rotary Position Embedding）**
- **定义：** 旋转位置编码，相对位置编码的一种
- **优势：** 可外推、保持相对位置信息
- **应用：** LLaMA, GPT-NeoX
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**ROUGE（Recall-Oriented Understudy for Gisting Evaluation）**
- **定义：** 文本摘要评估指标
- **类型：** ROUGE-N(n-gram), ROUGE-L(最长公共子序列)
- **范围：** 0-1
- **参考：** [07_LLM_Evaluation.md](./07_LLM_Evaluation.md)

---

### S

**Self-Attention（自注意力）**
- **定义：** 序列内部元素之间的注意力机制
- **公式：** `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- **作用：** 捕捉长距离依赖
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Self-Instruct**
- **定义：** 使用LLM生成新指令数据的方法
- **流程：** 种子指令 → LLM生成新指令 → LLM生成回复
- **应用：** 扩充训练数据
- **参考：** [13_Data_Engineering.md](./13_Data_Engineering.md)

**SFT（Supervised Fine-Tuning）**
- **定义：** 监督微调，RLHF的第一阶段
- **数据：** 高质量的指令-回复对
- **目标：** 提升基础指令遵循能力
- **参考：** [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md)

**SIMD（Single Instruction Multiple Data）**
- **定义：** 单指令多数据，向量化计算技术
- **指令集：** AVX2, AVX-512
- **加速比：** 3-8倍
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

**Speculative Decoding（投机解码）**
- **定义：** 使用小模型预测多个token，大模型验证
- **加速比：** 2-3倍
- **原理：** 并行验证多个候选
- **参考：** [02_Inference_Optimization.md](./02_Inference_Optimization.md)

**Stable Diffusion**
- **定义：** 开源文生图扩散模型
- **架构：** VAE + UNet + CLIP文本编码器
- **应用：** 图像生成、编辑、超分
- **参考：** [09_Multimodal_AI.md](./09_Multimodal_AI.md)

---

### T

**Temperature（温度）**
- **定义：** 控制采样随机性的参数
- **范围：** 0-2（通常）
- **效果：** 低温(0.1-0.5)=确定性，高温(0.8-1.5)=创造性
- **参考：** [04_Prompt_Engineering.md](./04_Prompt_Engineering.md)

**Token（词元）**
- **定义：** 文本的基本单位（可能是词、子词、字符）
- **分词器：** BPE, WordPiece, SentencePiece
- **比例：** 英文约0.75词/token，中文约1.5字/token
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Tokenization（分词）**
- **定义：** 将文本分解为token序列
- **方法：** BPE, WordPiece, Unigram
- **重要性：** 影响词汇表大小和模型性能
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Top-K Sampling**
- **定义：** 从概率最高的K个token中采样
- **典型值：** K=40-50
- **优势：** 避免低概率的荒谬输出
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Top-P Sampling（Nucleus Sampling）**
- **定义：** 从累积概率达到P的最小token集合中采样
- **典型值：** P=0.9-0.95
- **优势：** 动态调整候选数量
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**Transformer**
- **定义：** 基于自注意力的神经网络架构
- **提出：** 2017年《Attention is All You Need》
- **组成：** Encoder + Decoder（或仅其一）
- **参考：** [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md)

**TTFT（Time To First Token）**
- **定义：** 首token延迟，用户发送请求到收到第一个token的时间
- **影响：** 用户体验的关键指标
- **优化：** KV Cache, 批处理
- **参考：** [10_Production_Deployment.md](./10_Production_Deployment.md)

---

### V

**Vector Database（向量数据库）**
- **定义：** 专门存储和检索向量的数据库
- **代表：** ChromaDB, Pinecone, Milvus, Weaviate
- **用途：** RAG、语义搜索
- **参考：** [03_RAG_System_Theory.md](./03_RAG_System_Theory.md)

**vLLM**
- **定义：** 高吞吐量LLM推理引擎
- **核心技术：** PagedAttention, Continuous Batching
- **加速比：** 2-3倍吞吐量提升
- **参考：** [10_Production_Deployment.md](./10_Production_Deployment.md)

---

### Z

**Zero-Shot Learning（零样本学习）**
- **定义：** 无需示例，仅通过指令完成任务
- **优势：** 简单、通用
- **局限：** 性能通常不如Few-shot
- **参考：** [04_Prompt_Engineering.md](./04_Prompt_Engineering.md)

---

## 重要公式速查

### Attention机制

```
Self-Attention:
Q = XW_Q,  K = XW_K,  V = XW_V
Attention(Q,K,V) = softmax(QK^T / √d_k)V

Multi-Head Attention:
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
```

### 量化

```
INT8对称量化:
scale = max(|W|) / 127
W_int8 = round(W / scale)
W_dequant = W_int8 × scale

SQNR (Signal-to-Quantization-Noise Ratio):
SQNR = 10 × log₁₀(Var(W) / Var(W - W_dequant))
```

### 评估指标

```
Perplexity:
PPL = exp(-1/N Σᵢ log P(xᵢ|x₁...xᵢ₋₁))

BLEU:
BLEU = BP × exp(Σᵢ wᵢ log pᵢ)
其中BP为brevity penalty, pᵢ为n-gram precision

F1 Score:
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### LoRA

```
LoRA权重更新:
W' = W₀ + ΔW = W₀ + BA
其中 B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)

可训练参数比例:
ratio = 2×r×(d+k) / (d×k) ≈ 0.1% - 1%
```

### 向量相似度

```
余弦相似度:
cos_sim(A,B) = (A·B) / (||A|| × ||B||)

欧氏距离:
L2_dist(A,B) = ||A - B|| = √(Σᵢ(Aᵢ - Bᵢ)²)
```

---

## 常用工具速查

### 模型训练/微调

| 工具 | 用途 | 命令/链接 |
|------|------|----------|
| **Transformers** | 模型训练框架 | `pip install transformers` |
| **PEFT** | 参数高效微调 | `pip install peft` |
| **DeepSpeed** | 分布式训练 | `pip install deepspeed` |
| **Accelerate** | 分布式训练简化 | `pip install accelerate` |
| **W&B** | 实验管理 | `pip install wandb` |
| **MLflow** | 模型管理 | `pip install mlflow` |

### 推理优化

| 工具 | 用途 | 命令/链接 |
|------|------|----------|
| **vLLM** | 高吞吐推理 | `pip install vllm` |
| **TensorRT-LLM** | NVIDIA加速 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **llama.cpp** | CPU推理 | [GitHub](https://github.com/ggerganov/llama.cpp) |
| **GPTQ** | 量化 | `pip install auto-gptq` |
| **BitsAndBytes** | 量化 | `pip install bitsandbytes` |

### RAG/向量数据库

| 工具 | 用途 | 命令/链接 |
|------|------|----------|
| **LangChain** | RAG框架 | `pip install langchain` |
| **LlamaIndex** | 数据框架 | `pip install llama-index` |
| **ChromaDB** | 向量数据库 | `pip install chromadb` |
| **Pinecone** | 向量数据库 | `pip install pinecone-client` |
| **FAISS** | 向量搜索 | `pip install faiss-cpu` |

### 数据处理

| 工具 | 用途 | 命令/链接 |
|------|------|----------|
| **Datasets** | 数据集加载 | `pip install datasets` |
| **DVC** | 数据版本控制 | `pip install dvc` |
| **Scrapy** | 网页爬取 | `pip install scrapy` |
| **Pandas** | 数据分析 | `pip install pandas` |

### 监控/部署

| 工具 | 用途 | 命令/链接 |
|------|------|----------|
| **Prometheus** | Metrics收集 | [官网](https://prometheus.io/) |
| **Grafana** | 可视化 | [官网](https://grafana.com/) |
| **Docker** | 容器化 | [官网](https://www.docker.com/) |
| **Kubernetes** | 容器编排 | [官网](https://kubernetes.io/) |

---

## 文档导航索引

### 按学习阶段

**初学者路径：**
1. [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md) - 理解基础概念
2. [04_Prompt_Engineering.md](./04_Prompt_Engineering.md) - 学会使用LLM
3. [03_RAG_System_Theory.md](./03_RAG_System_Theory.md) - 构建第一个应用

**进阶开发者：**
1. [02_Inference_Optimization.md](./02_Inference_Optimization.md) - 性能优化
2. [08_Agent_Systems.md](./08_Agent_Systems.md) - 构建Agent
3. [10_Production_Deployment.md](./10_Production_Deployment.md) - 生产部署

**模型开发者：**
1. [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md) - 训练/微调
2. [07_LLM_Evaluation.md](./07_LLM_Evaluation.md) - 评估模型
3. [11_Safety_and_Alignment.md](./11_Safety_and_Alignment.md) - 对齐安全

**工程师：**
1. [13_Data_Engineering.md](./13_Data_Engineering.md) - 数据工程
2. [14_MLOps_Best_Practices.md](./14_MLOps_Best_Practices.md) - MLOps
3. [10_Production_Deployment.md](./10_Production_Deployment.md) - 生产部署

### 按主题

**基础理论：**
- [01_LLM_Fundamentals.md](./01_LLM_Fundamentals.md) - Transformer, Attention
- [05_Learning_Roadmap.md](./05_Learning_Roadmap.md) - 学习路线

**性能优化：**
- [02_Inference_Optimization.md](./02_Inference_Optimization.md) - KV Cache, 量化, SIMD

**应用开发：**
- [03_RAG_System_Theory.md](./03_RAG_System_Theory.md) - RAG系统
- [04_Prompt_Engineering.md](./04_Prompt_Engineering.md) - Prompt技术
- [08_Agent_Systems.md](./08_Agent_Systems.md) - Agent系统

**模型训练：**
- [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md) - 微调技术
- [13_Data_Engineering.md](./13_Data_Engineering.md) - 数据工程

**模型评估：**
- [07_LLM_Evaluation.md](./07_LLM_Evaluation.md) - 评估基准

**多模态：**
- [09_Multimodal_AI.md](./09_Multimodal_AI.md) - CLIP, Stable Diffusion

**生产部署：**
- [10_Production_Deployment.md](./10_Production_Deployment.md) - 部署方案
- [14_MLOps_Best_Practices.md](./14_MLOps_Best_Practices.md) - MLOps实践

**安全对齐：**
- [11_Safety_and_Alignment.md](./11_Safety_and_Alignment.md) - 安全技术

**求职准备：**
- [12_Job_Preparation_Guide.md](./12_Job_Preparation_Guide.md) - 简历面试

---

## 缩写对照表

| 缩写 | 全称 | 中文 |
|------|------|------|
| **AIGC** | AI-Generated Content | AI生成内容 |
| **API** | Application Programming Interface | 应用程序接口 |
| **AVX** | Advanced Vector Extensions | 高级向量扩展 |
| **BLEU** | Bilingual Evaluation Understudy | 双语评估替补 |
| **BPE** | Byte Pair Encoding | 字节对编码 |
| **CLIP** | Contrastive Language-Image Pre-training | 对比语言-图像预训练 |
| **CoT** | Chain-of-Thought | 思维链 |
| **DPO** | Direct Preference Optimization | 直接偏好优化 |
| **DVC** | Data Version Control | 数据版本控制 |
| **FAISS** | Facebook AI Similarity Search | Facebook AI相似度搜索 |
| **FSDP** | Fully Sharded Data Parallel | 全分片数据并行 |
| **GSM8K** | Grade School Math 8K | 小学数学8000题 |
| **HPA** | Horizontal Pod Autoscaler | 水平Pod自动扩缩容 |
| **INT8** | 8-bit Integer | 8位整数 |
| **KV Cache** | Key-Value Cache | 键值缓存 |
| **LLM** | Large Language Model | 大语言模型 |
| **LoRA** | Low-Rank Adaptation | 低秩适配 |
| **MMLU** | Massive Multitask Language Understanding | 大规模多任务语言理解 |
| **MoE** | Mixture of Experts | 专家混合 |
| **NER** | Named Entity Recognition | 命名实体识别 |
| **PEFT** | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| **PII** | Personally Identifiable Information | 个人身份信息 |
| **PPO** | Proximal Policy Optimization | 近端策略优化 |
| **PSI** | Population Stability Index | 群体稳定性指数 |
| **PTQ** | Post-Training Quantization | 后训练量化 |
| **QA** | Question Answering | 问答 |
| **QAT** | Quantization-Aware Training | 量化感知训练 |
| **QLoRA** | Quantized LoRA | 量化LoRA |
| **RAG** | Retrieval-Augmented Generation | 检索增强生成 |
| **ReAct** | Reasoning + Acting | 推理+行动 |
| **RLHF** | Reinforcement Learning from Human Feedback | 人类反馈强化学习 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码 |
| **ROUGE** | Recall-Oriented Understudy for Gisting Evaluation | 面向召回的摘要评估 |
| **SFT** | Supervised Fine-Tuning | 监督微调 |
| **SIMD** | Single Instruction Multiple Data | 单指令多数据 |
| **SQNR** | Signal-to-Quantization-Noise Ratio | 信号量化噪声比 |
| **TPS** | Tokens Per Second | 每秒token数 |
| **TTFT** | Time To First Token | 首token时间 |
| **VAE** | Variational Autoencoder | 变分自编码器 |
| **VQA** | Visual Question Answering | 视觉问答 |

---

## 使用建议

1. **快速查找：** 使用 Ctrl+F 搜索术语
2. **深入学习：** 点击"参考"链接查看详细文档
3. **对比学习：** 查看相关术语（如LoRA vs QLoRA）
4. **公式推导：** 在"重要公式速查"中查找数学细节
5. **工具选择：** 在"常用工具速查"中找到合适的库

---

## 持续更新

本文档会随着AIGC领域发展持续更新。如发现遗漏或错误，欢迎反馈！

**最后更新：** 2025-12-02
