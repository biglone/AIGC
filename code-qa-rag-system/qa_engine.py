"""
代码问答引擎

基于RAG的代码理解问答系统
"""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from code_indexer import CodeIndexer
from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SEARCH_TYPE,
    MMR_FETCH_K,
    MMR_LAMBDA,
    QA_PROMPT_TEMPLATE,
    CODE_REVIEW_PROMPT,
    CODE_EXPLAIN_PROMPT,
    logger
)


class CodeQAEngine:
    """
    代码问答引擎

    功能：
    1. 语义检索相关代码
    2. 生成准确的回答
    3. 代码审查
    4. 代码解释
    """

    def __init__(self):
        """初始化QA引擎"""
        logger.info("初始化代码问答引擎...")

        # 初始化LLM
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )

        # 初始化索引器（用于获取向量数据库）
        self.indexer = CodeIndexer()

        # 加载向量数据库
        try:
            self.vectorstore = self.indexer.get_vectorstore()
            logger.info("✅ 向量数据库加载成功")
        except Exception as e:
            logger.error(f"❌ 加载向量数据库失败: {e}")
            raise

        # 创建检索器
        search_kwargs = {"k": RETRIEVAL_TOP_K}
        if RETRIEVAL_SEARCH_TYPE == "mmr":
            search_kwargs.update({
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA
            })

        self.retriever = self.vectorstore.as_retriever(
            search_type=RETRIEVAL_SEARCH_TYPE,
            search_kwargs=search_kwargs
        )

        # 创建QA链
        self.qa_chain = self._create_qa_chain()

        logger.info("✅ 问答引擎初始化完成")

    def _create_qa_chain(self) -> RetrievalQA:
        """
        创建QA链
        """
        # 创建提示词模板
        prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # 创建RetrievalQA链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff"表示把所有文档塞进一个prompt
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt
            }
        )

        return qa_chain

    def query(self, question: str) -> Dict:
        """
        查询代码库

        Args:
            question: 用户问题

        Returns:
            包含答案和来源的字典：
            {
                'answer': str,
                'source_documents': List[Document],
                'question': str
            }
        """
        logger.info(f"处理问题: {question}")

        try:
            # 调用QA链
            result = self.qa_chain.invoke({"query": question})

            logger.info("✅ 问题处理完成")

            return {
                'answer': result['result'],
                'source_documents': result.get('source_documents', []),
                'question': question
            }

        except Exception as e:
            logger.error(f"❌ 查询失败: {e}")
            raise

    def review_code(self, code: str, filename: str = "unknown") -> str:
        """
        代码审查

        Args:
            code: 代码内容
            filename: 文件名

        Returns:
            审查结果
        """
        logger.info(f"审查代码: {filename}")

        prompt = CODE_REVIEW_PROMPT.format(code=code, filename=filename)

        try:
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"❌ 代码审查失败: {e}")
            raise

    def explain_code(self, code: str, filename: str = "unknown") -> str:
        """
        解释代码

        Args:
            code: 代码内容
            filename: 文件名

        Returns:
            代码解释
        """
        logger.info(f"解释代码: {filename}")

        prompt = CODE_EXPLAIN_PROMPT.format(code=code, filename=filename)

        try:
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            logger.error(f"❌ 代码解释失败: {e}")
            raise

    def search_similar_code(self, query: str, k: int = 5) -> List[Dict]:
        """
        搜索相似代码片段

        Args:
            query: 搜索查询
            k: 返回结果数量

        Returns:
            相似代码列表
        """
        logger.info(f"搜索相似代码: {query}")

        try:
            # 使用向量数据库搜索
            docs = self.vectorstore.similarity_search(query, k=k)

            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

            logger.info(f"✅ 找到 {len(results)} 个相似代码块")
            return results

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            raise

    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        搜索相似代码并返回相似度分数

        Args:
            query: 搜索查询
            k: 返回结果数量

        Returns:
            (Document, score) 元组列表
        """
        logger.info(f"搜索相似代码（带分数）: {query}")

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"✅ 找到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            raise

    def batch_query(self, questions: List[str]) -> List[Dict]:
        """
        批量查询

        Args:
            questions: 问题列表

        Returns:
            答案列表
        """
        logger.info(f"批量查询：{len(questions)} 个问题")

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"处理问题 {i}/{len(questions)}")
            try:
                result = self.query(question)
                results.append(result)
            except Exception as e:
                logger.error(f"问题 {i} 处理失败: {e}")
                results.append({
                    'answer': f"处理失败: {e}",
                    'source_documents': [],
                    'question': question
                })

        logger.info("✅ 批量查询完成")
        return results

    def get_retriever(self):
        """
        获取检索器（用于高级用法）

        Returns:
            检索器实例
        """
        return self.retriever

    def update_retrieval_params(self, top_k: Optional[int] = None,
                               search_type: Optional[str] = None):
        """
        更新检索参数

        Args:
            top_k: 检索文档数量
            search_type: 检索类型 ("similarity" 或 "mmr")
        """
        if top_k:
            search_kwargs = {"k": top_k}
        else:
            search_kwargs = {"k": RETRIEVAL_TOP_K}

        if search_type == "mmr":
            search_kwargs.update({
                "fetch_k": MMR_FETCH_K,
                "lambda_mult": MMR_LAMBDA
            })

        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type or RETRIEVAL_SEARCH_TYPE,
            search_kwargs=search_kwargs
        )

        # 重新创建QA链
        self.qa_chain = self._create_qa_chain()

        logger.info(f"✅ 检索参数已更新: top_k={top_k}, search_type={search_type}")


# ============================================================
# 测试代码
# ============================================================

def test_qa_engine():
    """
    测试问答引擎
    """
    print("=" * 70)
    print("测试代码问答引擎")
    print("=" * 70)

    try:
        # 初始化引擎
        print("\n[初始化] 加载问答引擎...")
        engine = CodeQAEngine()
        print("✅ 引擎加载成功")

    except Exception as e:
        print(f"❌ 引擎加载失败: {e}")
        print("请确保已经先运行 code_indexer.py 创建索引")
        return

    # 测试1：基础问答
    print("\n" + "=" * 70)
    print("[测试1] 基础问答")
    print("=" * 70)

    questions = [
        "这个项目有哪些主要的类和函数？",
        "如何加载代码文件？",
        "向量数据库使用的是什么？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        try:
            result = engine.query(question)

            print(f"\n回答:")
            print(result['answer'])

            if result.get('source_documents'):
                print(f"\n参考文件:")
                for j, doc in enumerate(result['source_documents'][:2], 1):
                    source = doc.metadata.get('source', 'unknown')
                    print(f"  [{j}] {source}")

        except Exception as e:
            print(f"❌ 查询失败: {e}")

        print("-" * 70)

    # 测试2：代码搜索
    print("\n" + "=" * 70)
    print("[测试2] 相似代码搜索")
    print("=" * 70)

    search_query = "文件加载和读取"
    print(f"\n搜索: {search_query}")

    try:
        results = engine.search_similar_code(search_query, k=3)
        print(f"✅ 找到 {len(results)} 个相似代码块\n")

        for i, result in enumerate(results, 1):
            print(f"[{i}] {result['metadata'].get('source', 'unknown')}")
            print(f"    ({result['metadata'].get('language', 'unknown')})")
            print(f"    {result['content'][:150]}...\n")

    except Exception as e:
        print(f"❌ 搜索失败: {e}")

    # 测试3：代码解释
    print("\n" + "=" * 70)
    print("[测试3] 代码解释")
    print("=" * 70)

    sample_code = """
def load_from_directory(directory: str) -> List[Dict]:
    documents = []
    for file_path in scan_files(directory):
        content = read_file(file_path)
        documents.append({'content': content})
    return documents
"""

    print(f"代码:\n{sample_code}")
    print("\n解释:")

    try:
        explanation = engine.explain_code(sample_code, "code_loader.py")
        print(explanation)

    except Exception as e:
        print(f"❌ 解释失败: {e}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    # 注意：运行此测试需要：
    # 1. 设置 OPENAI_API_KEY 环境变量
    # 2. 先运行 code_indexer.py 创建索引

    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请先设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY=sk-xxx")
    else:
        test_qa_engine()
