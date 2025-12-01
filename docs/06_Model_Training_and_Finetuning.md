# 模型训练与微调

## 目录
1. [预训练基础](#预训练基础)
2. [微调技术](#微调技术)
3. [指令微调](#指令微调)
4. [对齐技术](#对齐技术)
5. [实践指南](#实践指南)

---

## 预训练基础

### 什么是预训练？

**定义：** 在大规模无标注文本上训练语言模型，学习通用的语言表示。

```
预训练目标：
给定前文 x₁, x₂, ..., xₜ₋₁，预测下一个词 xₜ

Loss = -∑ log P(xₜ | x₁, ..., xₜ₋₁)
```

### 训练数据

**数据规模：**
```
GPT-3:     300B tokens
LLaMA:     1.4T tokens
LLaMA-2:   2T tokens
Mistral:   未公开，估计1T+
```

**数据来源：**

| 类型 | 占比 | 来源示例 |
|-----|------|---------|
| 网页 | 60-70% | CommonCrawl, C4 |
| 书籍 | 10-15% | Books3, Gutenberg |
| 代码 | 5-10% | GitHub, StackOverflow |
| 论文 | 5-10% | ArXiv, PubMed |
| 对话 | 5% | Reddit, 论坛 |

**数据清洗流程：**

```python
# 1. 去重
from datasketch import MinHash, MinHashLSH

def deduplicate_documents(docs, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    unique_docs = []
    for i, doc in enumerate(docs):
        # 计算MinHash
        m = MinHash(num_perm=128)
        for word in doc.split():
            m.update(word.encode('utf8'))

        # 查找相似文档
        result = lsh.query(m)
        if not result:
            lsh.insert(f"doc_{i}", m)
            unique_docs.append(doc)

    return unique_docs

# 2. 质量过滤
def quality_filter(text):
    # 长度过滤
    if len(text) < 100 or len(text) > 100000:
        return False

    # 重复内容过滤
    lines = text.split('\n')
    unique_ratio = len(set(lines)) / len(lines)
    if unique_ratio < 0.3:  # 重复行太多
        return False

    # 语言过滤（保留英文/中文）
    import langdetect
    try:
        lang = langdetect.detect(text)
        if lang not in ['en', 'zh-cn']:
            return False
    except:
        return False

    # 有害内容过滤
    toxic_words = ['...']  # 敏感词列表
    if any(word in text.lower() for word in toxic_words):
        return False

    return True

# 3. PII移除
import re

def remove_pii(text):
    # 邮箱
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', text)
    # 电话
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    # IP地址
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', text)

    return text
```

### Tokenization

**BPE（Byte Pair Encoding）：**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 训练tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=32000,  # 词表大小
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
)

files = ["corpus.txt"]
tokenizer.train(files, trainer)

# 使用
encoded = tokenizer.encode("Hello, world!")
print(encoded.tokens)  # ['Hello', ',', 'world', '!']
print(encoded.ids)     # [1234, 45, 5678, 90]
```

**SentencePiece（LLaMA使用）：**

```python
import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='llama_tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe'
)

# 加载和使用
sp = spm.SentencePieceProcessor()
sp.load('llama_tokenizer.model')

text = "人工智能的未来"
tokens = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

print(tokens)  # ['▁人工', '智能', '的', '未来']
print(ids)     # [1234, 5678, 90, 1011]
```

### 训练超参数

**关键超参数：**

```python
# LLaMA-7B训练配置
config = {
    # 模型架构
    "n_layers": 32,
    "n_heads": 32,
    "d_model": 4096,
    "vocab_size": 32000,

    # 训练参数
    "batch_size": 4_000_000,  # tokens per batch
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,

    # 学习率调度
    "warmup_steps": 2000,
    "lr_scheduler": "cosine",
    "min_lr": 3e-5,

    # 优化器
    "optimizer": "AdamW",
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-8,

    # 正则化
    "dropout": 0.0,  # GPT通常不用dropout
    "attention_dropout": 0.0,
}
```

**学习率调度：**

```python
import math

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup阶段：线性增长
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = (current_step - num_warmup_steps) / \
                   (num_training_steps - num_warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 从1.0衰减到min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
```

### 分布式训练

**数据并行（DDP）：**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')

# 创建模型
model = GPT(config).cuda()
model = DDP(model, device_ids=[local_rank])

# 训练循环
for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

**完全分片数据并行（FSDP）：**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

# FSDP配置
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

model = FSDP(
    model,
    mixed_precision=mp_policy,
    sharding_strategy="FULL_SHARD",  # 分片所有参数
    cpu_offload=CPUOffload(offload_params=True),  # CPU offload
)
```

**DeepSpeed ZeRO：**

```python
import deepspeed

# DeepSpeed配置
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO-3: 分片参数+梯度+优化器状态
        "offload_optimizer": {
            "device": "cpu"  # Optimizer状态offload到CPU
        },
        "offload_param": {
            "device": "cpu"  # 参数offload到CPU
        }
    }
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

---

## 微调技术

### Full Fine-tuning

**标准微调流程：**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 准备数据
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048
    )

train_dataset = dataset.map(preprocess, batched=True)

# 训练
training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**问题：** 7B模型全量微调需要 ~28GB显存（FP32）或 ~14GB（FP16）

### LoRA（Low-Rank Adaptation）

**核心思想：** 不修改原始权重，只训练低秩分解矩阵

```
W = W₀ + ΔW
ΔW = BA

其中：
- W₀: 预训练权重（冻结）
- B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)
- r << min(d, k)，如 r=8, d=4096
```

**参数量对比：**
```
原始：d × k = 4096 × 4096 = 16M参数
LoRA：d × r + r × k = 4096×8 + 8×4096 = 65K参数
压缩比：16M / 65K ≈ 250倍
```

**实现：**

```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=8,                    # 秩
    lora_alpha=32,          # 缩放因子
    target_modules=[        # 应用LoRA的模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# 输出：trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# 训练（同Full Fine-tuning）
trainer = Trainer(...)
trainer.train()

# 保存（只保存LoRA权重）
model.save_pretrained("./llama-lora")  # 仅~3MB
```

**LoRA前向传播：**

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=32):
        super().__init__()
        # 预训练权重（冻结）
        self.weight = nn.Parameter(pretrained_weight, requires_grad=False)

        # LoRA权重
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = lora_alpha / r

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始输出 + LoRA增量
        result = F.linear(x, self.weight)
        result += (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result
```

### QLoRA（Quantized LoRA）

**创新点：** 4-bit量化基础模型 + LoRA微调

```
1. 基础模型：4-bit NormalFloat量化
2. LoRA权重：BF16
3. 梯度计算：BF16
4. 反向传播时：临时反量化到BF16
```

**NF4量化：**

```python
def quantize_nf4(tensor):
    """
    NormalFloat 4-bit量化
    假设权重服从正态分布，使用分位数量化
    """
    # 计算分位数（16个bin）
    quantiles = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495,
        0.0, 0.07958029955625534, 0.16093020141124725,
        0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0
    ])

    # 归一化到[-1, 1]
    absmax = tensor.abs().max()
    normalized = tensor / absmax

    # 量化到最近的分位数
    quantized = torch.zeros_like(tensor, dtype=torch.uint8)
    for i in range(len(tensor)):
        distances = (normalized[i] - quantiles).abs()
        quantized[i] = distances.argmin()

    return quantized, absmax

# QLoRA训练
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
```

**内存对比：**
```
Full Fine-tuning (FP16): 14GB
LoRA (FP16): 14GB + 0.1GB = 14.1GB
QLoRA (4-bit + LoRA): 3.5GB + 0.1GB = 3.6GB
```

7B模型可在单张RTX 3090（24GB）上微调！

### 其他PEFT方法

**Adapter Tuning：**

```python
# 在每个Transformer层插入Adapter
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Bottleneck结构
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接
```

**Prefix Tuning：**

```python
class PrefixTuning(nn.Module):
    def __init__(self, n_layers, n_heads, head_dim, prefix_len=10):
        super().__init__()
        # 可学习的prefix
        self.prefix_k = nn.Parameter(
            torch.randn(n_layers, prefix_len, n_heads, head_dim)
        )
        self.prefix_v = nn.Parameter(
            torch.randn(n_layers, prefix_len, n_heads, head_dim)
        )

    def forward(self, layer_idx, k, v):
        # 在K、V前添加prefix
        k = torch.cat([self.prefix_k[layer_idx], k], dim=1)
        v = torch.cat([self.prefix_v[layer_idx], v], dim=1)
        return k, v
```

---

## 指令微调

### 什么是指令微调？

**目标：** 让模型学会遵循指令（instruction following）

```
标准预训练：
输入: "The capital of France is"
输出: "Paris"

指令微调：
输入: "What is the capital of France?"
输出: "The capital of France is Paris."
```

### 数据格式

**Alpaca格式：**

```json
{
  "instruction": "将以下句子翻译成法语",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**ShareGPT格式（多轮对话）：**

```json
{
  "conversations": [
    {"from": "human", "value": "什么是机器学习？"},
    {"from": "gpt", "value": "机器学习是..."},
    {"from": "human", "value": "它有哪些应用？"},
    {"from": "gpt", "value": "主要应用包括..."}
  ]
}
```

### 构建指令数据集

**方法1：Self-Instruct（自动生成）**

```python
def self_instruct(seed_tasks, model, n_generate=1000):
    """
    使用少量种子任务，让LLM生成更多任务
    """
    generated_tasks = []

    for _ in range(n_generate):
        # 随机采样种子任务作为示例
        examples = random.sample(seed_tasks, k=3)

        prompt = f"""
        生成一个新的指令任务，类似于以下示例：

        示例1: {examples[0]}
        示例2: {examples[1]}
        示例3: {examples[2]}

        新任务：
        """

        # LLM生成新任务
        new_task = model.generate(prompt)

        # 质量过滤
        if is_valid_task(new_task):
            generated_tasks.append(new_task)

    return generated_tasks
```

**方法2：Evol-Instruct（进化指令）**

```python
def evol_instruct(instruction):
    """
    逐步增加指令复杂度
    """
    evolution_prompts = [
        # 深度进化
        f"为以下任务添加更多约束条件：\n{instruction}",

        # 广度进化
        f"创建一个类似但不同领域的任务：\n{instruction}",

        # 具体化
        f"使以下任务更具体和详细：\n{instruction}",

        # 增加推理
        f"为以下任务添加推理步骤要求：\n{instruction}",
    ]

    evolved = model.generate(random.choice(evolution_prompts))
    return evolved
```

### 训练流程

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# 加载指令数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 格式化为训练格式
def format_instruction(example):
    if example["input"]:
        prompt = f"""Below is an instruction that describes a task, paired with an input. Write a response.

### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        prompt = f"""Below is an instruction. Write a response.

### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""

    return {"text": prompt}

formatted_dataset = dataset.map(format_instruction)

# 训练
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="./llama-instruct",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
    )
)

trainer.train()
```

---

## 对齐技术

### RLHF（Reinforcement Learning from Human Feedback）

**三阶段流程：**

```
阶段1: 监督微调（SFT）
├─ 数据：高质量的指令-回答对
├─ 目标：学习基本的遵循指令能力
└─ 输出：SFT模型

阶段2: 训练奖励模型（RM）
├─ 数据：人类偏好对比数据
│  └─ 同一问题的多个回答，人类排序
├─ 目标：学习人类偏好
└─ 输出：奖励模型

阶段3: 强化学习优化（PPO）
├─ 使用RM引导SFT模型优化
├─ 平衡：奖励最大化 vs 与SFT模型不要偏离太远
└─ 输出：对齐后的模型
```

**奖励模型训练：**

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 使用最后一个token的hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward

# 偏好数据
class PreferenceDataset(Dataset):
    def __init__(self, data):
        # data: [(prompt, chosen, rejected), ...]
        self.data = data

    def __getitem__(self, idx):
        prompt, chosen, rejected = self.data[idx]
        return {
            "prompt": prompt,
            "chosen": prompt + chosen,
            "rejected": prompt + rejected,
        }

# 训练
def train_reward_model(model, dataset):
    for batch in dataloader:
        # 前向传播
        reward_chosen = model(batch["chosen"])
        reward_rejected = model(batch["rejected"])

        # 对比损失
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected))
        loss = loss.mean()

        # 反向传播
        loss.backward()
        optimizer.step()
```

**PPO训练：**

```python
def ppo_step(policy_model, reward_model, ref_model, prompt):
    """
    PPO算法单步
    """
    # 1. 生成回答
    response = policy_model.generate(prompt)

    # 2. 计算奖励
    reward = reward_model(prompt + response)

    # 3. 计算KL散度惩罚（防止偏离参考模型太远）
    logprobs_policy = policy_model.get_logprobs(prompt, response)
    logprobs_ref = ref_model.get_logprobs(prompt, response)
    kl_penalty = (logprobs_policy - logprobs_ref).sum()

    # 4. 总目标
    objective = reward - beta * kl_penalty

    # 5. PPO裁剪
    ratio = torch.exp(logprobs_policy - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    return loss
```

### DPO（Direct Preference Optimization）

**核心思想：** 直接优化偏好，无需训练奖励模型

```
DPO损失：
L = -E[log σ(β log π(y_w | x) / π_ref(y_w | x)
              - β log π(y_l | x) / π_ref(y_l | x))]

其中：
- y_w: 偏好的回答（winner）
- y_l: 不偏好的回答（loser）
- β: 温度参数
- π: 策略模型
- π_ref: 参考模型
```

**实现：**

```python
from trl import DPOTrainer

# DPO训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,  # KL惩罚系数
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./llama-dpo",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=5e-7,
    )
)

trainer.train()
```

**DPO vs RLHF对比：**

| 维度 | RLHF | DPO |
|-----|------|-----|
| 复杂度 | 高（3阶段） | 低（1阶段） |
| 稳定性 | 较差（RL不稳定） | 较好 |
| 效果 | 略好 | 接近RLHF |
| 训练速度 | 慢 | 快 |
| 内存 | 需要RM+Policy | 只需Policy |

---

## 实践指南

### 选择微调方法

**决策树：**

```
有足够GPU资源(80GB+) ?
├─ Yes → Full Fine-tuning
└─ No →
    数据量大(100K+) ?
    ├─ Yes → QLoRA
    └─ No → LoRA

需要多任务适配 ?
├─ Yes → Adapter（每个任务一个adapter）
└─ No → LoRA

需要极致压缩 ?
├─ Yes → QLoRA (4-bit)
└─ No → LoRA (16-bit)
```

### 数据准备

**数据量需求：**

```
任务类型           最少    推荐     最佳
简单分类/QA        1K      5K      20K
指令微调          10K     50K     100K+
领域适配          5K      20K     50K+
对话系统          10K     50K     200K+
```

**数据质量检查：**

```python
def check_data_quality(dataset):
    issues = []

    # 1. 长度检查
    lengths = [len(ex["output"]) for ex in dataset]
    if np.mean(lengths) < 10:
        issues.append("回答太短")
    if np.std(lengths) > 500:
        issues.append("长度差异过大")

    # 2. 多样性检查
    unique_ratio = len(set(dataset["instruction"])) / len(dataset)
    if unique_ratio < 0.8:
        issues.append("指令重复度高")

    # 3. 格式检查
    for ex in dataset:
        if not ex.get("instruction") or not ex.get("output"):
            issues.append(f"缺少必要字段: {ex}")
            break

    return issues
```

### 超参数调优

**LoRA超参数：**

```python
# 推荐配置
configs = {
    "小模型(1-3B)": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "learning_rate": 3e-4,
    },
    "中模型(7-13B)": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
    },
    "大模型(30B+)": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "learning_rate": 1e-4,
    }
}
```

**学习率：**

```
Full Fine-tuning:  1e-5 ~ 5e-5
LoRA:              1e-4 ~ 5e-4
QLoRA:             1e-4 ~ 3e-4

规律：参数越少，学习率可以越大
```

### 评估与验证

**训练中监控：**

```python
from transformers import TrainerCallback

class EvaluationCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # 生成测试
        model = kwargs["model"]
        test_prompts = [
            "解释什么是机器学习",
            "写一首关于春天的诗",
            "1+1等于多少？"
        ]

        for prompt in test_prompts:
            output = model.generate(prompt)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {output}")
```

**最终评估：**

```python
# 1. ROUGE分数（摘要任务）
from rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(predictions, references, avg=True)

# 2. BLEU分数（翻译任务）
from sacrebleu import corpus_bleu

bleu = corpus_bleu(predictions, [references])

# 3. 人工评估
def human_eval_template():
    return """
    评分标准（1-5分）：
    1. 准确性：回答是否正确？
    2. 完整性：是否全面？
    3. 流畅性：语言是否自然？
    4. 安全性：是否有害内容？

    总分：
    """
```

### 常见问题

**问题1：过拟合**

症状：
```
训练loss持续下降
验证loss上升或停滞
生成内容重复训练数据
```

解决：
```python
# 增加dropout
lora_config = LoraConfig(lora_dropout=0.1)

# 减少训练轮数
training_args = TrainingArguments(num_train_epochs=1)

# 增加数据量
# 使用数据增强
```

**问题2：灾难性遗忘**

症状：
```
微调后，通用能力下降
只能回答特定领域问题
```

解决：
```python
# 混合通用数据
mixed_dataset = concatenate_datasets([
    domain_specific_data,  # 70%
    general_data,          # 30%
])

# 使用较小的学习率
learning_rate = 1e-5
```

**问题3：模型不遵循指令**

解决：
```python
# 改进prompt格式
prompt_template = """
### Instruction:
{instruction}

### Response:
{response}

### End
"""

# 增加系统提示词
system_prompt = "你是一个有用的AI助手，总是遵循用户指令。"
```

---

## 延伸阅读

**核心论文：**
1. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
2. QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
3. Self-Instruct: Aligning LMs with Self-Generated Instructions (Wang et al., 2022)
4. Training language models to follow instructions with human feedback (Ouyang et al., 2022)
5. Direct Preference Optimization (Rafailov et al., 2023)

**开源工具：**
- Hugging Face PEFT
- Axolotl (微调框架)
- LLaMA Factory
- DeepSpeed
- TRL (Transformer Reinforcement Learning)

**下一步：**
- [LLM评估](07_LLM_Evaluation.md)
- [Agent系统](08_Agent_Systems.md)
