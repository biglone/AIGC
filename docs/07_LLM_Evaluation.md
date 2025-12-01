# LLM评估方法论

## 目录
1. [评估基准](#评估基准)
2. [自动评估方法](#自动评估方法)
3. [人工评估](#人工评估)
4. [问题诊断](#问题诊断)
5. [评估最佳实践](#评估最佳实践)

---

## 评估基准

### 通用能力评估

**MMLU (Massive Multitask Language Understanding)**

```
覆盖领域：57个任务
├─ STEM（数学、物理、化学、生物）
├─ 人文（历史、哲学、法律）
├─ 社会科学（经济、心理、政治）
└─ 其他（医学、商业等）

难度：高中到专家水平
格式：多选题（4选1）
评分：准确率（0-100%）
```

**示例题目：**
```
Question: In a supersonic fluid flowing past a body, weak disturbances are
A. Transmitted ahead of the body
B. Transmitted only along Mach lines
C. Transmitted

 only perpendicular to Mach lines
D. Not transmitted at all

Answer: B
```

**运行评估：**
```python
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf-causal",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["mmlu"],
    num_fewshot=5,
    batch_size=8
)

print(f"MMLU Score: {results['results']['mmlu']['acc']:.2%}")
# 输出: MMLU Score: 45.30%
```

**主流模型得分：**
| 模型 | MMLU | 参数 |
|-----|------|------|
| GPT-4 | 86.4% | 未知 |
| Claude-3-Opus | 86.8% | 未知 |
| Gemini Ultra | 90.0% | 未知 |
| Llama-2-70B | 68.9% | 70B |
| Llama-2-7B | 45.3% | 7B |
| Mistral-7B | 60.1% | 7B |

### 代码能力评估

**HumanEval**

```
任务：根据文档字符串生成Python函数
数量：164个编程问题
评分：Pass@K（K次尝试中至少1次通过）
```

**示例：**
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """
    Check if in given list of numbers, are any two numbers closer to each other
    than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    # 模型需要生成的代码
```

**评估实现：**
```python
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

# 1. 生成代码
problems = read_problems()
samples = []

for task_id, problem in problems.items():
    prompt = problem["prompt"]

    # 让模型生成代码（K次）
    for _ in range(K):
        completion = model.generate(prompt)
        samples.append({
            "task_id": task_id,
            "completion": completion
        })

write_jsonl("samples.jsonl", samples)

# 2. 执行测试
results = evaluate_functional_correctness("samples.jsonl")
print(f"Pass@{K}: {results['pass@k']:.2%}")
```

**主流模型得分：**
| 模型 | Pass@1 | Pass@10 |
|-----|--------|---------|
| GPT-4 | 67.0% | - |
| Claude-3-Opus | 84.9% | - |
| GPT-3.5-Turbo | 48.1% | - |
| Code Llama 70B | 53.7% | - |
| Llama-2-70B | 29.9% | - |

**MBPP (Mostly Basic Python Problems)**

```
任务：更简单的Python编程问题
数量：974个问题
格式：给定描述+测试用例，生成代码
难度：入门级
```

### 数学推理评估

**GSM8K (Grade School Math 8K)**

```
任务：小学数学应用题
数量：8,500个问题
格式：自然语言问题 → 数值答案
评分：精确匹配
```

**示例：**
```
Question: Natalia sold clips to 48 of her friends in April, and then she sold
half as many clips in May. How many clips did Natalia sell altogether in April
and May?

Answer: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48 + 24 = 72 clips altogether in April and May.
#### 72
```

**评估代码：**
```python
import re

def extract_answer(text):
    # 提取 #### 后的答案
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None

def evaluate_gsm8k(model, dataset):
    correct = 0
    for example in dataset:
        response = model.generate(example["question"])
        pred_answer = extract_answer(response)
        true_answer = extract_answer(example["answer"])

        if pred_answer == true_answer:
            correct += 1

    return correct / len(dataset)
```

**MATH**

```
难度：更高（竞赛数学）
覆盖：代数、几何、概率、数论等
评分：LaTeX格式答案匹配
```

### 中文评估

**C-Eval**

```
覆盖：52个学科
语言：中文
难度：高中到专业水平
格式：多选题
```

**CMMLU (Chinese Massive Multitask Language Understanding)**

```
覆盖：67个任务
专注：中文知识和文化
格式：多选题
```

**评估示例：**
```python
from lm_eval import evaluator

# C-Eval评估
results = evaluator.simple_evaluate(
    model="hf-causal",
    model_args="pretrained=THUDM/chatglm3-6b",
    tasks=["ceval"],
    num_fewshot=5
)

print(f"C-Eval Score: {results['results']['ceval']['acc']:.2%}")
```

### 其他重要基准

**HellaSwag（常识推理）：**
```
任务：句子补全
测试：常识推理能力
难度：人类95%, GPT-3.5 85%
```

**TruthfulQA（真实性）：**
```
任务：回答易产生虚假信息的问题
测试：模型是否会编造事实
评分：真实性百分比
```

**WinoGrande（代词消歧）：**
```
任务：选择代词正确指代
测试：语言理解深度
```

---

## 自动评估方法

### LLM-as-Judge

**核心思想：** 使用强大的LLM（如GPT-4）评估其他LLM的输出

**评估模板：**
```python
JUDGE_PROMPT = """
你是一个公正的评审员。请评估以下AI助手的回答。

问题：
{question}

AI回答：
{answer}

评分标准（1-10分）：
1. 准确性（3分）：信息是否正确？
2. 完整性（3分）：是否全面回答问题？
3. 有用性（2分）：对用户是否有帮助？
4. 清晰度（2分）：表达是否清晰？

请按照以下格式输出：
准确性：X/3
完整性：X/3
有用性：X/2
清晰度：X/2
总分：X/10
理由：[简要说明]
"""

def llm_judge(question, answer):
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    judgment = gpt4(prompt)
    score = parse_score(judgment)
    return score
```

**成对比较（Pairwise Comparison）：**

```python
PAIRWISE_PROMPT = """
问题：{question}

回答A：
{answer_a}

回答B：
{answer_b}

哪个回答更好？考虑准确性、完整性和有用性。

输出格式：
胜者：[A/B/平局]
理由：[说明]
"""

def pairwise_judge(question, answer_a, answer_b):
    prompt = PAIRWISE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )
    judgment = gpt4(prompt)

    if "胜者：A" in judgment:
        return "A"
    elif "胜者：B" in judgment:
        return "B"
    else:
        return "Tie"

# 计算Elo评分
def compute_elo(model_names, battles):
    """
    battles: [(model_a, model_b, winner), ...]
    winner: "A", "B", or "Tie"
    """
    from collections import defaultdict

    elo = {name: 1500 for name in model_names}
    K = 32  # Elo K-factor

    for model_a, model_b, winner in battles:
        # 计算期望胜率
        expected_a = 1 / (1 + 10**((elo[model_b] - elo[model_a])/400))
        expected_b = 1 - expected_a

        # 实际结果
        if winner == "A":
            score_a, score_b = 1, 0
        elif winner == "B":
            score_a, score_b = 0, 1
        else:  # Tie
            score_a, score_b = 0.5, 0.5

        # 更新Elo
        elo[model_a] += K * (score_a - expected_a)
        elo[model_b] += K * (score_b - expected_b)

    return elo
```

**MT-Bench（多轮对话评估）：**

```python
# MT-Bench: 80个多轮对话问题
mt_bench_example = {
    "question_1": "创作一个包含角色、设定和冲突的引人入胜的故事。",
    "question_2": "你能为上面的故事创作一首诗吗？",
    # 测试多轮对话和指令遵循能力
}

def evaluate_mt_bench(model):
    total_score = 0

    for conversation in mt_bench_conversations:
        # 第一轮
        response_1 = model.chat(conversation["question_1"])
        score_1 = gpt4_judge(conversation["question_1"], response_1)

        # 第二轮（有上下文）
        response_2 = model.chat(
            conversation["question_2"],
            history=[(conversation["question_1"], response_1)]
        )
        score_2 = gpt4_judge(conversation["question_2"], response_2)

        total_score += (score_1 + score_2) / 2

    return total_score / len(mt_bench_conversations)
```

### 参考答案对比

**ROUGE（摘要任务）：**

```python
from rouge import Rouge

def evaluate_rouge(predictions, references):
    """
    ROUGE-N: N-gram重叠
    ROUGE-L: 最长公共子序列
    """
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    return {
        "rouge-1": scores["rouge-1"]["f"],  # Unigram F1
        "rouge-2": scores["rouge-2"]["f"],  # Bigram F1
        "rouge-l": scores["rouge-l"]["f"],  # LCS F1
    }

# 示例
pred = "北京是中国的首都，人口众多。"
ref = "北京作为中国的首都，拥有庞大的人口。"

scores = evaluate_rouge([pred], [ref])
# {'rouge-1': 0.75, 'rouge-2': 0.44, 'rouge-l': 0.67}
```

**BLEU（翻译任务）：**

```python
from sacrebleu import corpus_bleu

def evaluate_bleu(predictions, references):
    """
    BLEU: 精确匹配的N-gram比例
    """
    # references: 可以有多个参考译文
    bleu = corpus_bleu(predictions, [references])
    return bleu.score  # 0-100

# 示例
pred = ["The cat sat on the mat."]
ref = ["The cat was sitting on the mat."]

score = evaluate_bleu(pred, ref)
# BLEU: 54.26
```

**BERTScore（语义相似度）：**

```python
from bert_score import score

def evaluate_bertscore(predictions, references):
    """
    基于BERT embeddings的语义相似度
    """
    P, R, F1 = score(
        predictions,
        references,
        lang="en",
        model_type="bert-base-uncased"
    )

    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# 优势：捕捉语义相似性，不仅仅是词汇重叠
pred = "The feline sat on the rug."
ref = "The cat sat on the mat."

scores = evaluate_bertscore([pred], [ref])
# F1: 0.91 (BLEU只有0.2，但语义相近)
```

---

## 人工评估

### 评分标准设计

**单点评分（Likert Scale）：**

```
评估维度：准确性
1分：完全错误，误导性信息
2分：大部分错误，有少量正确信息
3分：部分正确，但有明显错误
4分：基本正确，有小瑕疵
5分：完全正确，无错误

评估维度：有用性
1分：完全无用，不相关
2分：略有相关，但帮助很小
3分：有一定帮助
4分：很有帮助
5分：极其有用，完美解决问题
```

**成对偏好评估：**

```
问题：[展示问题]

回答A：[展示回答A]
回答B：[展示回答B]

问题：哪个回答更好？
选项：
○ A明显更好
○ A稍好
○ 差不多
○ B稍好
○ B明显更好

理由：[文本框]
```

### 众包评估流程

**1. 评估者筛选：**

```python
# 资格测试
qualification_test = [
    {
        "question": "评估这个回答",
        "answer_to_eval": "...",
        "ground_truth_score": 4,
        "tolerance": 1  # 允许±1分误差
    },
    # 10-20个测试样本
]

def qualify_annotator(responses):
    correct = 0
    for i, response in enumerate(responses):
        expected = qualification_test[i]["ground_truth_score"]
        tolerance = qualification_test[i]["tolerance"]

        if abs(response - expected) <= tolerance:
            correct += 1

    # 通过标准：80%以上
    return correct / len(responses) >= 0.8
```

**2. 评估任务设计：**

```
每个样本：3-5个评估者
批次大小：10-20个样本/批次
休息时间：每批次后休息
时间限制：每个样本2-5分钟
支付：$0.5-1.0/样本（取决于复杂度）
```

**3. 质量控制：**

```python
def compute_inter_annotator_agreement(annotations):
    """
    计算评估者一致性（Krippendorff's Alpha）
    """
    from simpledorff import calculate_krippendorffs_alpha

    # annotations: [[a1_s1, a2_s1, a3_s1],  # 样本1的3个评分
    #                [a1_s2, a2_s2, a3_s2],  # 样本2
    #                ...]

    alpha = calculate_krippendorffs_alpha(
        reliability_data=annotations
    )

    if alpha < 0.67:
        print("⚠️  一致性低，需要改进评估指南")
    elif alpha < 0.80:
        print("✓ 一致性可接受")
    else:
        print("✓✓ 一致性高")

    return alpha

# 识别不可靠的评估者
def identify_outliers(annotations):
    from scipy import stats

    # 计算每个评估者与其他人的平均偏差
    n_annotators = len(annotations[0])
    deviations = []

    for i in range(n_annotators):
        scores_i = [ann[i] for ann in annotations]
        others_avg = [
            np.mean([ann[j] for j in range(n_annotators) if j != i])
            for ann in annotations
        ]
        deviation = np.mean(np.abs(np.array(scores_i) - np.array(others_avg)))
        deviations.append(deviation)

    # Z-score检测异常
    z_scores = stats.zscore(deviations)
    outliers = [i for i, z in enumerate(z_scores) if abs(z) > 2]

    return outliers
```

**4. 聚合评分：**

```python
def aggregate_scores(annotations, method="median"):
    """
    聚合多个评估者的分数
    """
    if method == "mean":
        return np.mean(annotations, axis=1)
    elif method == "median":
        # 中位数更稳健，不受异常值影响
        return np.median(annotations, axis=1)
    elif method == "trimmed_mean":
        # 去掉最高和最低分后的平均
        return stats.trim_mean(annotations, 0.2, axis=1)
    elif method == "majority":
        # 众数（离散评分）
        from scipy.stats import mode
        return mode(annotations, axis=1)[0]
```

---

## 问题诊断

### 幻觉检测

**事实核查：**

```python
def detect_hallucination(claim, knowledge_base):
    """
    检测生成内容是否与知识库矛盾
    """
    # 1. 提取陈述
    claims = extract_claims(claim)

    hallucinations = []
    for c in claims:
        # 2. 检索相关事实
        facts = knowledge_base.retrieve(c)

        # 3. 蕴含检测
        entailment = nli_model.predict(c, facts)

        if entailment == "contradiction":
            hallucinations.append({
                "claim": c,
                "evidence": facts,
                "type": "factual_error"
            })
        elif entailment == "neutral" and len(facts) == 0:
            hallucinations.append({
                "claim": c,
                "type": "unsupported"
            })

    return hallucinations

# 示例
response = "埃菲尔铁塔建于1887年，高330米。"
issues = detect_hallucination(response, wiki_kb)
# [{"claim": "建于1887年", "type": "factual_error",
#   "truth": "建于1889年"}]
```

**自洽性检查：**

```python
def check_self_consistency(question, model, n=5):
    """
    生成多次，检查答案一致性
    """
    responses = []
    for _ in range(n):
        response = model.generate(question, temperature=0.7)
        answer = extract_answer(response)
        responses.append(answer)

    # 计算一致性
    from collections import Counter
    counts = Counter(responses)
    most_common_count = counts.most_common(1)[0][1]

    consistency = most_common_count / n

    if consistency < 0.6:
        return {
            "consistent": False,
            "confidence": "low",
            "answers": responses
        }
    else:
        return {
            "consistent": True,
            "confidence": "high",
            "majority_answer": counts.most_common(1)[0][0]
        }
```

**引用验证：**

```python
def verify_citations(response, sources):
    """
    验证回答中的引用是否准确
    """
    # 提取回答中的声称引用
    claims_with_citations = extract_citations(response)

    errors = []
    for claim, citation in claims_with_citations:
        # 在源文档中查找
        cited_text = sources.get(citation)

        if cited_text is None:
            errors.append({
                "claim": claim,
                "citation": citation,
                "error": "citation_not_found"
            })
        else:
            # 验证声称是否与引用内容一致
            entailment = nli_model.predict(claim, cited_text)
            if entailment == "contradiction":
                errors.append({
                    "claim": claim,
                    "citation": citation,
                    "cited_text": cited_text,
                    "error": "misattribution"
                })

    return errors
```

### 毒性检测

```python
from detoxify import Detoxify

def detect_toxicity(text):
    """
    检测有害内容
    """
    model = Detoxify('original')
    scores = model.predict(text)

    # 分类：毒性、严重毒性、淫秽、威胁、侮辱、身份攻击
    return {
        "toxic": scores['toxicity'] > 0.5,
        "scores": scores,
        "flags": [k for k, v in scores.items() if v > 0.5]
    }

# 示例
text = "You are stupid and worthless."
result = detect_toxicity(text)
# {'toxic': True, 'flags': ['toxicity', 'insult']}
```

### 偏见分析

```python
def analyze_bias(model, templates):
    """
    检测模型偏见（性别、种族等）
    """
    results = []

    # 性别偏见测试
    gendered_templates = [
        ("The {gender} worked as a {occupation}.",
         ["doctor", "nurse", "engineer", "teacher"]),
    ]

    for template, occupations in gendered_templates:
        for occupation in occupations:
            male_prob = model.get_probability(
                template.format(gender="man", occupation=occupation)
            )
            female_prob = model.get_probability(
                template.format(gender="woman", occupation=occupation)
            )

            bias_score = abs(male_prob - female_prob) / (male_prob + female_prob)

            results.append({
                "occupation": occupation,
                "male_prob": male_prob,
                "female_prob": female_prob,
                "bias_score": bias_score
            })

    return results

# 可视化偏见
import matplotlib.pyplot as plt

def plot_bias(results):
    occupations = [r["occupation"] for r in results]
    male_probs = [r["male_prob"] for r in results]
    female_probs = [r["female_prob"] for r in results]

    x = range(len(occupations))
    plt.bar([i - 0.2 for i in x], male_probs, width=0.4, label="Male")
    plt.bar([i + 0.2 for i in x], female_probs, width=0.4, label="Female")
    plt.xticks(x, occupations)
    plt.legend()
    plt.show()
```

---

## 评估最佳实践

### 构建评估流程

```
1. 定义目标
   ├─ 想要测试什么能力？
   ├─ 关心哪些指标？
   └─ 容忍什么样的错误？

2. 选择基准
   ├─ 通用能力 → MMLU, HellaSwag
   ├─ 代码能力 → HumanEval
   ├─ 数学推理 → GSM8K
   └─ 领域特定 → 自建数据集

3. 自动评估
   ├─ 运行标准基准测试
   ├─ LLM-as-Judge评估
   └─ 参考答案对比

4. 人工评估
   ├─ 设计评分标准
   ├─ 众包评估
   └─ 专家review

5. 问题诊断
   ├─ 幻觉检测
   ├─ 毒性检测
   └─ 偏见分析

6. 报告与改进
   ├─ 汇总指标
   ├─ 识别弱点
   └─ 迭代优化
```

### 评估数据集构建

```python
class EvaluationDataset:
    def __init__(self):
        self.examples = []

    def add_example(self, input_text, expected_output,
                   category=None, difficulty=None):
        self.examples.append({
            "id": len(self.examples),
            "input": input_text,
            "expected": expected_output,
            "category": category,
            "difficulty": difficulty,
            "metadata": {}
        })

    def balance_categories(self):
        """确保每个类别有足够样本"""
        from collections import Counter
        counts = Counter([ex["category"] for ex in self.examples])

        min_count = min(counts.values())
        balanced = []

        for category in counts:
            category_examples = [ex for ex in self.examples
                                if ex["category"] == category]
            balanced.extend(random.sample(category_examples, min_count))

        self.examples = balanced

    def split(self, test_size=0.2):
        """划分训练/测试集"""
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            self.examples,
            test_size=test_size,
            stratify=[ex["category"] for ex in self.examples]
        )

        return train, test
```

### 评估报告模板

```python
def generate_evaluation_report(model_name, results):
    report = f"""
# {model_name} 评估报告

## 总体表现

| 基准 | 得分 | 排名 |
|-----|------|------|
| MMLU | {results['mmlu']:.1%} | {get_rank('mmlu', results['mmlu'])} |
| HumanEval | {results['humaneval']:.1%} | {get_rank('humaneval', results['humaneval'])} |
| GSM8K | {results['gsm8k']:.1%} | {get_rank('gsm8k', results['gsm8k'])} |

## 详细分析

### 优势领域
{analyze_strengths(results)}

### 改进空间
{analyze_weaknesses(results)}

## 问题诊断

### 幻觉率
{results['hallucination_rate']:.1%}

### 毒性内容
{results['toxicity_rate']:.2%}

### 偏见分数
{results['bias_score']:.2f}

## 建议

{generate_recommendations(results)}
"""

    return report
```

---

## 延伸阅读

**论文：**
1. Holistic Evaluation of Language Models (HELM, Stanford)
2. Beyond the Imitation Game Benchmark (BIG-Bench, Google)
3. TruthfulQA: Measuring How Models Mimic Human Falsehoods
4. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**工具：**
- lm-evaluation-harness (EleutherAI)
- HELM (Stanford)
- OpenAI Evals
- AlpacaEval

**数据集：**
- MMLU, HellaSwag, TruthfulQA
- HumanEval, MBPP
- GSM8K, MATH
- C-Eval, CMMLU

**下一步：**
- [Agent系统](08_Agent_Systems.md)
- [多模态AI](09_Multimodal_AI.md)
