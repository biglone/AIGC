#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mini-RAG 系统 - 简化版RAG实现
==========================

项目目标：
- 理解RAG（检索增强生成）核心原理
- 实现基础的向量检索和生成流程
- 适合快速学习和面试展示

技术要点：
1. 文档分块（Chunking）
2. 向量嵌入（Embeddings）
3. 相似度检索（Similarity Search）
4. 上下文增强生成（Context-Augmented Generation）

作者：面向C++工程师的AIGC学习
"""

import os
import json
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


# ============================================================
# 第一部分：文档处理
# ============================================================

@dataclass
class Document:
    """文档数据类"""
    content: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """
    文本分块器

    类比C++：类似于字符串分割器，但要保证语义完整性
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        """
        Args:
            chunk_size: 每个块的字符数
            overlap: 块之间的重叠字符数（避免切断完整语义）
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        将长文本切分成多个块

        策略：
        1. 优先按句子切分（。！？）
        2. 保持一定重叠避免信息丢失
        3. 每块大小大致相等
        """
        # 简单实现：按句子切分
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in '。！？\n':
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        # 合并句子成块
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


# ============================================================
# 第二部分：向量嵌入（简化版）
# ============================================================

class SimpleEmbedding:
    """
    简化的嵌入模型（用于演示）

    实际生产环境应该使用：
    - OpenAI Embeddings (text-embedding-ada-002)
    - Sentence Transformers
    - HuggingFace模型

    这里使用TF-IDF + 降维模拟向量化
    """

    def __init__(self, dim: int = 128):
        """
        Args:
            dim: 嵌入向量维度
        """
        self.dim = dim
        self.vocab = {}  # 词汇表
        self.idf = {}    # IDF值

    def fit(self, texts: List[str]):
        """训练嵌入模型（构建词汇表和IDF）"""
        # 构建词汇表
        doc_count = len(texts)
        word_doc_count = {}

        for text in texts:
            words = set(self._tokenize(text))
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1

        # 计算IDF
        for word, count in word_doc_count.items():
            self.idf[word] = np.log(doc_count / (count + 1))

        # 构建词汇表索引
        self.vocab = {word: idx for idx, word in enumerate(sorted(self.idf.keys()))}

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（实际应使用jieba等工具）"""
        # 这里简化为字符级别，实际应该用词级别
        import re
        # 提取中英文词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', text)
        return words

    def embed(self, text: str) -> np.ndarray:
        """
        将文本转换为向量

        简化版：TF-IDF向量 + 随机投影降维
        """
        words = self._tokenize(text)
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1

        # 创建TF-IDF向量
        vector = np.zeros(len(self.vocab))
        for word, count in word_count.items():
            if word in self.vocab:
                idx = self.vocab[word]
                tf = count / len(words) if words else 0
                idf = self.idf.get(word, 0)
                vector[idx] = tf * idf

        # 降维到指定维度（简单取前dim个维度 + 归一化）
        if len(vector) > self.dim:
            vector = vector[:self.dim]
        else:
            vector = np.pad(vector, (0, self.dim - len(vector)))

        # L2归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector


# ============================================================
# 第三部分：向量数据库（内存版）
# ============================================================

class VectorStore:
    """
    向量存储和检索

    类比C++：类似于std::vector<std::pair<Embedding, Data>>
    实际生产使用：Chroma, Pinecone, Milvus
    """

    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
        for doc in documents:
            # 计算嵌入向量
            embedding = self.embedding_model.embed(doc.content)
            self.documents.append(doc)
            self.embeddings.append(embedding)

    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        相似度检索

        Args:
            query: 查询文本
            k: 返回前k个最相似的文档

        Returns:
            [(Document, similarity_score), ...]
        """
        # 计算查询向量
        query_embedding = self.embedding_model.embed(query)

        # 计算余弦相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # 余弦相似度 = dot(A, B) / (||A|| * ||B||)
            # 因为向量已归一化，直接点积即可
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((self.documents[i], float(similarity)))

        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


# ============================================================
# 第四部分：RAG生成器
# ============================================================

class SimpleRAG:
    """
    简化RAG系统

    流程：
    1. 接收用户问题
    2. 检索相关文档（Retrieval）
    3. 构建Prompt（包含检索到的上下文）
    4. 调用LLM生成答案（Generation）
    """

    def __init__(self, vector_store: VectorStore, use_openai: bool = False):
        """
        Args:
            vector_store: 向量存储
            use_openai: 是否使用OpenAI API（False则返回模拟回答）
        """
        self.vector_store = vector_store
        self.use_openai = use_openai

        if use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("警告：未安装openai库，将使用模拟回答")
                self.use_openai = False

    def query(self, question: str, k: int = 3) -> Dict:
        """
        RAG查询

        Args:
            question: 用户问题
            k: 检索文档数量

        Returns:
            {
                'answer': str,
                'sources': List[Document],
                'similarities': List[float]
            }
        """
        # 第1步：检索相关文档
        results = self.vector_store.similarity_search(question, k=k)

        if not results:
            return {
                'answer': "抱歉，没有找到相关信息。",
                'sources': [],
                'similarities': []
            }

        # 第2步：构建上下文
        context = "\n\n".join([
            f"[文档{i+1}]\n{doc.content}"
            for i, (doc, _) in enumerate(results)
        ])

        # 第3步：构建Prompt
        prompt = self._build_prompt(question, context)

        # 第4步：生成答案
        if self.use_openai:
            answer = self._generate_with_openai(prompt)
        else:
            answer = self._generate_mock(question, results)

        return {
            'answer': answer,
            'sources': [doc for doc, _ in results],
            'similarities': [sim for _, sim in results]
        }

    def _build_prompt(self, question: str, context: str) -> str:
        """构建RAG Prompt"""
        return f"""请根据以下上下文回答问题。如果上下文中没有相关信息，请明确说明。

上下文：
{context}

问题：{question}

请提供详细、准确的回答："""

    def _generate_with_openai(self, prompt: str) -> str:
        """使用OpenAI生成答案"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，请根据提供的上下文准确回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成答案时出错：{str(e)}"

    def _generate_mock(self, question: str, results: List[Tuple[Document, float]]) -> str:
        """模拟生成答案（演示用）"""
        if not results:
            return "没有找到相关信息。"

        # 简单拼接最相关的文档内容
        top_doc, top_sim = results[0]

        answer = f"""根据检索到的信息（相似度：{top_sim:.3f}），以下是回答：

{top_doc.content[:200]}...

（注：这是模拟回答。实际应该使用OpenAI API生成更智能的答案。）
要使用真实LLM，请设置 OPENAI_API_KEY 环境变量并传入 use_openai=True。"""

        return answer


# ============================================================
# 第五部分：示例和测试
# ============================================================

def create_sample_knowledge_base() -> List[Document]:
    """创建示例知识库"""
    documents = [
        Document(
            content="KV Cache是Transformer推理优化的核心技术。它缓存了Attention计算中的Key和Value矩阵，避免重复计算。在自回归生成场景下，每生成一个新token，只需要计算新token的K/V，然后与历史K/V拼接即可。这样可以将时间复杂度从O(n²)降低到O(n)，推理速度提升20倍以上。",
            metadata={"source": "推理优化", "topic": "KV Cache"}
        ),
        Document(
            content="INT8量化是将FP32权重转换为INT8格式，可以节省75%的内存占用。量化公式为：q = round(x / scale)，其中scale = max(abs(x)) / 127。反量化时：x_dequant = q * scale。量化后精度损失通常小于1%，但推理速度可以提升2-3倍。主流框架如TensorRT、ONNXRuntime都支持INT8量化。",
            metadata={"source": "推理优化", "topic": "量化技术"}
        ),
        Document(
            content="Flash Attention是一种IO感知的Attention算法，由Stanford提出。传统Attention需要将Q、K、V矩阵完整加载到GPU HBM，中间结果占用大量显存。Flash Attention通过分块计算和重计算策略，将中间结果保存在SRAM中，大幅减少HBM读写。在长序列（如4K tokens）场景下，速度提升3-5倍，显存节省10-20倍。",
            metadata={"source": "推理优化", "topic": "Flash Attention"}
        ),
        Document(
            content="RAG（检索增强生成）是LLM应用的重要范式。它通过向量检索从外部知识库获取相关信息，然后将检索结果作为上下文提供给LLM生成答案。RAG的优势包括：1）可以使用最新数据无需重新训练；2）减少幻觉问题；3）答案可溯源。典型流程是：查询→向量检索→Prompt构建→LLM生成。",
            metadata={"source": "LLM应用", "topic": "RAG"}
        ),
        Document(
            content="Transformer架构由Attention机制组成。Self-Attention的核心公式是：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V。其中Q、K、V分别是Query、Key、Value矩阵，d_k是维度。Multi-Head Attention将Q、K、V投影到多个子空间并行计算，然后concat结果。这种机制让模型能够关注序列中不同位置的信息，是现代LLM的基础。",
            metadata={"source": "基础理论", "topic": "Transformer"}
        ),
        Document(
            content="SIMD（单指令多数据）是CPU向量化优化的核心技术。AVX2指令集可以一次处理8个float（256bit）。例如矩阵乘法中的内积计算，普通循环需要n次乘加，使用AVX2的_mm256_fmadd_ps指令可以并行8路计算，理论加速8倍。实际加速取决于内存带宽、缓存命中率等因素，通常能达到3-4倍。Intel MKL等库大量使用SIMD优化。",
            metadata={"source": "性能优化", "topic": "SIMD"}
        ),
        Document(
            content="Prompt Engineering是LLM应用的关键技能。常用技巧包括：1）Few-shot Learning：提供示例；2）Chain of Thought：引导逐步推理；3）角色设定：定义AI身份；4）格式约束：要求特定输出格式；5）上下文注入：提供背景信息。好的Prompt可以大幅提升LLM输出质量，某些场景下效果提升可达50%以上。",
            metadata={"source": "LLM应用", "topic": "Prompt Engineering"}
        ),
        Document(
            content="vLLM是高性能LLM推理框架，核心创新是PagedAttention。传统KV Cache使用连续内存，容易造成碎片和浪费。PagedAttention借鉴虚拟内存思想，将KV Cache分成固定大小的块（如512 tokens），按需分配。这样可以将GPU显存利用率从60%提升到90%以上，batch size增加2-3倍，吞吐量提升数倍。适合高并发推理服务。",
            metadata={"source": "推理框架", "topic": "vLLM"}
        ),
    ]
    return documents


def demo_mini_rag():
    """演示Mini-RAG系统"""
    print("=" * 60)
    print("Mini-RAG 系统演示")
    print("=" * 60)

    # 第1步：创建知识库
    print("\n[步骤1] 创建知识库...")
    documents = create_sample_knowledge_base()
    print(f"✓ 已加载 {len(documents)} 个文档")

    # 第2步：初始化嵌入模型
    print("\n[步骤2] 初始化嵌入模型...")
    embedding_model = SimpleEmbedding(dim=128)
    all_texts = [doc.content for doc in documents]
    embedding_model.fit(all_texts)
    print(f"✓ 词汇表大小：{len(embedding_model.vocab)}")

    # 第3步：构建向量库
    print("\n[步骤3] 构建向量库...")
    vector_store = VectorStore(embedding_model)
    vector_store.add_documents(documents)
    print(f"✓ 已索引 {len(vector_store.documents)} 个文档")

    # 第4步：创建RAG系统
    print("\n[步骤4] 创建RAG系统...")
    rag = SimpleRAG(vector_store, use_openai=False)
    print("✓ RAG系统已就绪")

    # 第5步：测试查询
    print("\n" + "=" * 60)
    print("开始测试查询")
    print("=" * 60)

    test_questions = [
        "什么是KV Cache？它如何优化推理性能？",
        "如何实现INT8量化？",
        "Flash Attention的原理是什么？",
        "RAG系统有什么优势？",
        "如何使用SIMD优化矩阵乘法？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"问题 {i}：{question}")
        print('─' * 60)

        result = rag.query(question, k=2)

        print(f"\n【检索结果】")
        for j, (source, sim) in enumerate(zip(result['sources'], result['similarities']), 1):
            print(f"\n文档{j} (相似度: {sim:.3f}):")
            print(f"  主题: {source.metadata.get('topic', 'N/A')}")
            print(f"  内容预览: {source.content[:80]}...")

        print(f"\n【生成答案】")
        print(result['answer'])

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

    # 性能统计
    print("\n【系统统计】")
    print(f"文档总数: {len(documents)}")
    print(f"向量维度: {embedding_model.dim}")
    print(f"平均文档长度: {np.mean([len(doc.content) for doc in documents]):.1f} 字符")
    print(f"检索方式: 余弦相似度")
    print(f"生成方式: {'OpenAI API' if rag.use_openai else '模拟生成'}")


def interactive_mode():
    """交互模式"""
    print("\n" + "=" * 60)
    print("进入交互模式（输入 'quit' 退出）")
    print("=" * 60)

    # 初始化系统
    documents = create_sample_knowledge_base()
    embedding_model = SimpleEmbedding(dim=128)
    embedding_model.fit([doc.content for doc in documents])
    vector_store = VectorStore(embedding_model)
    vector_store.add_documents(documents)
    rag = SimpleRAG(vector_store, use_openai=False)

    while True:
        question = input("\n请输入问题 > ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break

        if not question:
            continue

        result = rag.query(question, k=2)

        print(f"\n相似度: {result['similarities']}")
        print(f"\n回答:\n{result['answer']}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    import sys

    print("""
╔════════════════════════════════════════════════════════╗
║            Mini-RAG 系统 - 简化版实现                  ║
║                                                        ║
║  核心技术:                                             ║
║    • 文档分块 (Chunking)                              ║
║    • 向量嵌入 (Embeddings)                            ║
║    • 相似度检索 (Similarity Search)                   ║
║    • 上下文生成 (Context-Augmented Generation)        ║
║                                                        ║
║  适合场景:                                             ║
║    • RAG原理学习                                       ║
║    • 快速原型验证                                      ║
║    • 面试项目展示                                      ║
╚════════════════════════════════════════════════════════╝
    """)

    # 运行演示
    demo_mini_rag()

    # 可选：交互模式
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()

    print("\n提示：")
    print("1. 要使用真实LLM，设置 OPENAI_API_KEY 环境变量")
    print("2. 运行 python mini_rag.py --interactive 进入交互模式")
    print("3. 实际生产环境推荐使用 LangChain + Chroma + OpenAI Embeddings")
