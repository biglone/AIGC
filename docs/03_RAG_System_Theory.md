# RAG系统理论

## 目录
1. [RAG基础概念](#rag基础概念)
2. [向量嵌入理论](#向量嵌入理论)
3. [检索算法](#检索算法)
4. [RAG评估指标](#rag评估指标)
5. [高级RAG技术](#高级rag技术)

---

## RAG基础概念

### 什么是RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

核心思想：
```
传统LLM: 问题 → LLM → 答案
         (依赖参数记忆，可能过时/幻觉)

RAG:     问题 → 检索相关文档 → 组合上下文 → LLM → 答案
         (基于事实，可验证，可更新)
```

### 为什么需要RAG？

**LLM的局限性：**

1. **知识截止日期**
   ```
   问: "2024年最新的Python版本是什么？"
   LLM: "我的训练数据截止到2023年，无法回答。"
   ```

2. **幻觉（Hallucination）**
   ```
   问: "公司内部的API文档在哪？"
   LLM: 编造一个不存在的链接
   ```

3. **领域知识不足**
   ```
   问: "我们代码库中Matrix类如何使用？"
   LLM: 给出通用答案，不是你项目特定的实现
   ```

**RAG的优势：**
- ✅ 始终使用最新数据
- ✅ 答案可溯源（提供引用）
- ✅ 无需重新训练模型
- ✅ 适合私有数据

### RAG架构

```
┌─────────────────────────────────────────────────────┐
│              1. 离线索引阶段                          │
├─────────────────────────────────────────────────────┤
│  原始文档 → 分块 → Embedding → 向量数据库             │
│  (一次性处理，可增量更新)                             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│              2. 在线查询阶段                          │
├─────────────────────────────────────────────────────┤
│  用户问题                                            │
│      ↓                                              │
│  问题Embedding                                       │
│      ↓                                              │
│  向量相似度搜索 (Top-K)                              │
│      ↓                                              │
│  检索到的文档片段                                     │
│      ↓                                              │
│  构建Prompt: [问题 + 检索文档]                       │
│      ↓                                              │
│  LLM生成答案                                         │
│      ↓                                              │
│  返回答案 + 来源引用                                  │
└─────────────────────────────────────────────────────┘
```

### 三个核心组件

**1. 文档处理器**
- 加载原始文档
- 分块（Chunking）
- 提取元数据

**2. 向量数据库**
- 存储文档嵌入
- 高效相似度搜索
- 元数据过滤

**3. 生成器**
- 接收问题+检索文档
- 生成自然语言答案
- 引用来源

---

## 向量嵌入理论

### 什么是Embedding？

将文本转换为高维向量，语义相似的文本在向量空间中距离近。

```python
"猫是一种动物" → [0.2, -0.5, 0.8, ..., 0.3]  (1536维)
"狗是一种宠物" → [0.3, -0.4, 0.7, ..., 0.2]  (相近)
"太阳系的行星" → [-0.8, 0.9, -0.1, ..., 0.5] (很远)
```

### Embedding模型

**常用模型：**

| 模型 | 维度 | 特点 | 适用场景 |
|-----|------|------|---------|
| text-embedding-3-small | 1536 | 快速、便宜 | 通用RAG |
| text-embedding-3-large | 3072 | 高精度 | 高要求场景 |
| sentence-transformers | 384-768 | 开源、可本地部署 | 离线应用 |
| BGE-large | 1024 | 中文优化 | 中文RAG |

**OpenAI Embedding API：**
```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="你的文本"
)

embedding = response.data[0].embedding  # 1536维向量
```

### Embedding的数学性质

**余弦相似度：**

```python
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)
```

**范围：** [-1, 1]
- 1: 完全相同
- 0: 正交（不相关）
- -1: 完全相反

**示例：**
```python
vec_cat = embed("猫")
vec_dog = embed("狗")
vec_car = embed("汽车")

similarity(vec_cat, vec_dog)  # 0.85 (高度相似)
similarity(vec_cat, vec_car)  # 0.12 (不相关)
```

### 语义搜索原理

**传统关键词搜索：**
```
文档: "Python是一种编程语言"
查询: "Python programming"
结果: 匹配（包含"Python"）

文档: "Python是一种编程语言"
查询: "编程语言"
结果: 不匹配（没有"编程语言"这个完整词）
```

**语义搜索：**
```python
query_vec = embed("什么是Python?")
doc_vec = embed("Python是一种编程语言")

similarity = cosine_similarity(query_vec, doc_vec)
# 0.78 (高相似度，虽然词汇不同)
```

**优势：**
- 理解同义词（"car" vs "automobile"）
- 跨语言（"猫" vs "cat"）
- 捕捉语义（"快乐" vs "高兴"）

---

## 检索算法

### 暴力搜索（Brute Force）

最简单的方法：计算查询与所有文档的相似度

```python
def brute_force_search(query_vec, doc_vecs, k=5):
    similarities = []
    for i, doc_vec in enumerate(doc_vecs):
        sim = cosine_similarity(query_vec, doc_vec)
        similarities.append((i, sim))

    # 排序并返回top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]
```

**复杂度：** O(N × D)
- N: 文档数量
- D: 向量维度

**问题：** 百万级文档时太慢！

### 近似最近邻搜索（ANN）

牺牲少许精度，大幅提升速度。

**HNSW（Hierarchical Navigable Small World）**

分层图结构：
```
Layer 2:  A -------- B
           |          |
Layer 1:  A --- C --- B --- D
           |    |     |     |
Layer 0:  A-C-E-F-B-D-G-H-I
```

搜索流程：
1. 从顶层开始
2. 找到该层最近邻
3. 下降到下一层
4. 重复直到底层

**复杂度：** O(log N)

**ChromaDB默认使用HNSW！**

### IVF（Inverted File Index）

**聚类思想：**

```
1. 将所有向量聚类成K个簇
2. 搜索时：
   - 找到最近的几个簇中心
   - 只在这些簇内搜索
```

**FAISS实现：**
```python
import faiss

# 训练索引（聚类）
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
index.train(doc_vectors)
index.add(doc_vectors)

# 搜索
index.nprobe = 10  # 搜索10个簇
D, I = index.search(query_vectors, k=5)
```

### 混合检索（Hybrid Search）

结合多种检索方法：

**方法1：稀疏 + 密集**
```python
# BM25（关键词）+ 向量搜索
keyword_results = bm25.search(query, k=20)
semantic_results = vector_db.search(query, k=20)

# 融合排序（RRF: Reciprocal Rank Fusion）
final_results = rerank(keyword_results, semantic_results)
```

**方法2：向量搜索 + 元数据过滤**
```python
# 先过滤
results = vector_db.search(
    query,
    filter={"language": "Python", "date": ">2023-01-01"},
    k=5
)
```

### 重排序（Re-ranking）

检索后的二次排序：

**MMR（Maximal Marginal Relevance）**

平衡相关性和多样性：

```python
def mmr(query_vec, doc_vecs, selected, lambda_param=0.5):
    scores = []
    for i, doc_vec in enumerate(doc_vecs):
        if i in selected:
            continue

        # 相关性
        relevance = cosine_similarity(query_vec, doc_vec)

        # 多样性（与已选文档的最大相似度）
        if selected:
            max_sim = max([
                cosine_similarity(doc_vec, doc_vecs[j])
                for j in selected
            ])
        else:
            max_sim = 0

        # MMR分数
        score = lambda_param * relevance - (1 - lambda_param) * max_sim
        scores.append((i, score))

    # 返回最高分
    return max(scores, key=lambda x: x[1])[0]
```

**lambda = 0.5：** 平衡
**lambda = 1.0：** 只考虑相关性
**lambda = 0.0：** 只考虑多样性

---

## RAG评估指标

### 检索质量评估

**1. Precision@K（精确率）**
```
Precision@K = (Top-K中相关文档数) / K
```

示例：
```
检索5个文档，其中3个相关
Precision@5 = 3 / 5 = 0.6
```

**2. Recall@K（召回率）**
```
Recall@K = (Top-K中相关文档数) / (所有相关文档数)
```

示例：
```
总共有10个相关文档，检索到3个
Recall@5 = 3 / 10 = 0.3
```

**3. MRR（Mean Reciprocal Rank）**
```
MRR = 1 / (第一个相关文档的排名)
```

示例：
```
第1个结果不相关
第2个结果不相关
第3个结果相关 ← 排名3
MRR = 1/3 = 0.33
```

**4. NDCG（Normalized Discounted Cumulative Gain）**

考虑排名位置的重要性：
```python
def dcg(relevances):
    return sum([
        (2**rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(relevances)
    ])

def ndcg(relevances, ideal_relevances):
    return dcg(relevances) / dcg(ideal_relevances)
```

### 生成质量评估

**1. 事实准确性（Faithfulness）**

答案是否基于检索文档？

```python
def check_faithfulness(answer, retrieved_docs):
    # 使用LLM判断
    prompt = f"""
    文档: {retrieved_docs}
    答案: {answer}

    答案中的所有陈述是否都能在文档中找到支持？
    回答: 是/否
    """
    return llm(prompt)
```

**2. 答案相关性（Relevance）**

答案是否回答了问题？

**3. 上下文相关性（Context Relevance）**

检索的文档是否与问题相关？

**4. 综合评估框架：RAGAS**

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ],
)

print(result)
```

---

## 高级RAG技术

### 文档分块策略

**固定大小分块：**
```python
def fixed_size_chunking(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
```

**问题：** 可能破坏语义完整性

**语义分块（代码库特化）：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 代码特化的分隔符
separators = [
    "\n\n",        # 段落
    "\nclass ",    # 类定义
    "\ndef ",      # 函数定义
    "\n",          # 行
    " ",           # 词
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=separators
)

chunks = splitter.split_text(code)
```

**AST-based分块（最佳）：**

使用tree-sitter解析代码AST，保持函数/类完整性：

```python
def ast_based_chunking(code, language="python"):
    tree = parser.parse(code, language)

    chunks = []
    for node in tree.root_node.children:
        if node.type in ["function_definition", "class_definition"]:
            chunk = {
                "content": node.text,
                "type": node.type,
                "start_line": node.start_point[0],
                "end_line": node.end_point[0],
            }
            chunks.append(chunk)

    return chunks
```

### 查询改写（Query Rewriting）

**问题：** 用户问题可能表达不清晰

**解决：** 用LLM改写查询

```python
def rewrite_query(original_query):
    prompt = f"""
    用户问题: {original_query}

    请将这个问题改写为3个更具体的检索查询，
    分别从不同角度表达相同的信息需求。

    返回JSON格式:
    ["查询1", "查询2", "查询3"]
    """

    rewritten = llm(prompt)
    return json.loads(rewritten)

# 使用
queries = rewrite_query("这个项目怎么用？")
# ["如何安装这个项目？", "项目的基本使用方法", "项目配置和初始化"]

# 多查询检索
all_results = []
for query in queries:
    results = vector_db.search(query, k=3)
    all_results.extend(results)

# 去重和重排序
final_results = deduplicate_and_rerank(all_results)
```

### HyDE（Hypothetical Document Embeddings）

**思路：** 让LLM生成假设性答案，用答案去检索

```python
def hyde_retrieval(question):
    # 1. 生成假设性文档
    hypothetical_doc = llm(f"""
        问题: {question}

        请写一段详细的答案（即使不确定）。
    """)

    # 2. 用假设文档的embedding去检索
    hypo_embedding = embed(hypothetical_doc)
    results = vector_db.search_by_vector(hypo_embedding, k=5)

    return results
```

**为什么有效？**
- 答案的embedding可能比问题更接近真实文档
- 特别适合技术文档检索

### 自查询（Self-Querying）

让LLM提取结构化查询参数：

```python
user_input = "找出2023年之后的Python相关文档"

# LLM提取查询意图
structured_query = llm(f"""
从以下输入提取查询意图:
{user_input}

返回JSON:
{{
    "query": "语义查询文本",
    "filter": {{"field": "value"}},
    "limit": 5
}}
""")

# 结构化检索
result = vector_db.search(
    query=structured_query["query"],
    filter=structured_query["filter"],
    k=structured_query["limit"]
)
```

### RAG-Fusion

融合多个检索结果：

```python
def rag_fusion(question, k=5):
    # 1. 生成多个查询变体
    queries = [
        question,
        rewrite(question, style="technical"),
        rewrite(question, style="concise"),
    ]

    # 2. 对每个查询检索
    all_results = {}
    for query in queries:
        results = vector_db.search(query, k=k*2)
        for rank, doc in enumerate(results):
            # Reciprocal Rank Fusion
            score = all_results.get(doc.id, 0)
            score += 1 / (rank + 60)  # k=60是常用值
            all_results[doc.id] = score

    # 3. 按融合分数排序
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results[:k]
```

### 父文档检索器

检索小块，返回大块：

```python
# 索引时：小块
small_chunks = split_text(document, chunk_size=200)
for chunk in small_chunks:
    vector_db.add(
        embedding=embed(chunk),
        metadata={"parent_id": document.id}
    )

# 检索时：
search_results = vector_db.search(query, k=5)
parent_ids = [r.metadata["parent_id"] for r in search_results]

# 返回完整文档
full_documents = [get_document(pid) for pid in parent_ids]
```

**优势：**
- 检索精确（小块语义聚焦）
- 上下文完整（大块信息丰富）

---

## 代码库RAG特化

### 代码语义理解

**问题：** 代码与自然语言不同

**解决：**

1. **代码注释提取**
   ```python
   def extract_docstrings(code):
       # 提取函数/类的docstring
       # 作为额外的检索文本
   ```

2. **函数签名索引**
   ```python
   def index_function_signature(function_node):
       return {
           "name": function_node.name,
           "params": function_node.parameters,
           "return_type": function_node.return_type,
       }
   ```

3. **导入关系**
   ```python
   # 记录依赖关系
   metadata = {
       "imports": ["numpy", "pandas"],
       "used_by": ["main.py", "utils.py"],
   }
   ```

### 多模态检索

代码 + 文档 + 注释联合检索：

```python
def multimodal_code_search(query):
    # 1. 代码检索
    code_results = code_index.search(query)

    # 2. 文档检索
    doc_results = doc_index.search(query)

    # 3. 融合
    combined = merge_and_rerank(code_results, doc_results)

    return combined
```

### 代码示例生成

结合检索和生成：

```python
def generate_code_example(query):
    # 1. 检索相似代码
    similar_code = vector_db.search(query, k=3)

    # 2. 生成示例
    prompt = f"""
    参考以下代码:
    {similar_code}

    问题: {query}

    请生成一个可运行的代码示例。
    """

    example = llm(prompt)
    return example
```

---

## 实践建议

### 选择Chunk大小

| 文档类型 | 推荐大小 | 重叠 |
|---------|---------|------|
| 长文章 | 1000-1500 | 200-300 |
| 技术文档 | 500-800 | 100-150 |
| 代码 | 800-1200 | 150-250 |
| FAQ | 200-400 | 50-100 |

**原则：**
- 足够大：包含完整语义
- 足够小：精确定位
- 有重叠：避免边界问题

### 优化检索质量

1. **使用元数据过滤**
   ```python
   results = db.search(
       query,
       filter={"date": ">2024-01-01", "type": "code"}
   )
   ```

2. **调整Top-K**
   ```
   K太小: 可能miss重要信息
   K太大: 噪声多，成本高

   推荐: 3-5 (通用), 5-10 (复杂问题)
   ```

3. **使用重排序**
   ```python
   # 粗检索：召回20个
   candidates = db.search(query, k=20)

   # 精排序：重排到5个
   final = rerank_model.rerank(query, candidates, top_k=5)
   ```

### 降低成本

**Embedding成本：**
```
text-embedding-3-small: $0.02 / 1M tokens
text-embedding-3-large: $0.13 / 1M tokens

建议: 大部分场景用small即可
```

**LLM成本：**
```
优化上下文长度 = 减少token消耗

bad:  10000 tokens上下文 × $0.001 = $0.01/次
good: 2000 tokens上下文 × $0.001 = $0.002/次
```

**缓存策略：**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_embedding(text):
    return openai.embeddings.create(input=text)
```

---

## 延伸阅读

**论文：**
1. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
2. Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)
3. Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE, Gao et al., 2022)

**工具和框架：**
- LangChain: RAG应用框架
- LlamaIndex: 数据连接框架
- ChromaDB: 向量数据库
- FAISS: Facebook向量检索库

**下一步：**
- [Prompt Engineering](04_Prompt_Engineering.md)
- [完整学习路线](05_Learning_Roadmap.md)
