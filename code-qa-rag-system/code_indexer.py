"""
代码索引器

将代码文件转换为向量并存储到向量数据库
"""

from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from code_loader import CodeLoader
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    logger
)


class CodeIndexer:
    """
    代码索引器

    功能：
    1. 加载代码文件
    2. 智能切分代码块
    3. 向量化
    4. 存储到Chroma向量数据库
    """

    def __init__(self):
        """初始化"""
        self.code_loader = CodeLoader()

        # 初始化Embedding模型
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        # 初始化文本切分器（针对代码优化）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )

        logger.info("代码索引器初始化完成")

    def index_repository(self, repo_path: str) -> int:
        """
        索引整个代码库

        Args:
            repo_path: 代码库根目录

        Returns:
            索引的文件数量
        """
        logger.info(f"开始索引代码库: {repo_path}")

        # 1. 加载所有代码文件
        documents_data = self.code_loader.load_from_directory(repo_path)

        if not documents_data:
            logger.warning("未找到任何代码文件")
            return 0

        logger.info(f"已加载 {len(documents_data)} 个文件")

        # 2. 转换为LangChain Document格式
        documents = self._create_documents(documents_data)

        # 3. 切分代码块
        logger.info("切分代码块...")
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"切分完成，共 {len(splits)} 个代码块")

        # 4. 向量化并存储
        logger.info("开始向量化并存储...")
        self._create_vectorstore(splits)

        logger.info(f"✅ 索引完成！共处理 {len(documents_data)} 个文件")
        return len(documents_data)

    def index_files(self, file_paths: List[str]) -> int:
        """
        索引指定的文件列表

        Args:
            file_paths: 文件路径列表

        Returns:
            成功索引的文件数量
        """
        logger.info(f"开始索引 {len(file_paths)} 个文件")

        documents = []

        for file_path in file_paths:
            try:
                doc_data = self.code_loader.load_single_file(file_path)
                doc = Document(
                    page_content=doc_data['content'],
                    metadata=doc_data['metadata']
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"加载文件失败 {file_path}: {e}")
                continue

        if not documents:
            logger.warning("没有成功加载任何文件")
            return 0

        # 切分并存储
        splits = self.text_splitter.split_documents(documents)
        self._create_vectorstore(splits)

        logger.info(f"✅ 索引完成！共处理 {len(documents)} 个文件")
        return len(documents)

    def _create_documents(self, documents_data: List[dict]) -> List[Document]:
        """
        将加载的数据转换为LangChain Document格式
        """
        documents = []

        for doc_data in documents_data:
            # 添加文件头信息（帮助理解上下文）
            metadata = doc_data['metadata']
            header = f"""
File: {metadata['source']}
Language: {metadata['language']}
Lines: {metadata['lines']}

""".strip()

            # 组合成完整内容
            content = f"{header}\n\n{doc_data['content']}"

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def _create_vectorstore(self, splits: List[Document]):
        """
        创建向量数据库并存储

        Args:
            splits: 切分后的文档块
        """
        try:
            # 清空现有的向量数据库（如果存在）
            # 注意：这会删除所有旧数据！如果需要增量更新，需要修改这里

            # 创建新的向量数据库
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=str(VECTOR_DB_DIR),
                collection_name=COLLECTION_NAME
            )

            # Chroma v0.4+ 已自动持久化，不需要手动调用 persist()
            logger.info(f"向量数据库已保存到: {VECTOR_DB_DIR}")

        except Exception as e:
            logger.error(f"创建向量数据库失败: {e}")
            raise

    def update_index(self, repo_path: str, file_paths: Optional[List[str]] = None):
        """
        增量更新索引

        Args:
            repo_path: 代码库根目录
            file_paths: 需要更新的文件列表（None表示全量更新）
        """
        if file_paths is None:
            # 全量重建
            logger.info("执行全量索引重建")
            return self.index_repository(repo_path)
        else:
            # 增量更新
            logger.info(f"执行增量更新，共 {len(file_paths)} 个文件")
            return self.index_files(file_paths)

    def get_vectorstore(self) -> Chroma:
        """
        获取已有的向量数据库

        Returns:
            Chroma向量数据库实例
        """
        try:
            vectorstore = Chroma(
                persist_directory=str(VECTOR_DB_DIR),
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            return vectorstore
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            raise ValueError(
                "向量数据库不存在或损坏。请先运行 index_repository() 创建索引。"
            )

    def get_index_stats(self) -> dict:
        """
        获取索引统计信息

        Returns:
            统计信息字典
        """
        try:
            vectorstore = self.get_vectorstore()

            # 获取collection
            collection = vectorstore._collection

            stats = {
                'total_chunks': collection.count(),
                'collection_name': COLLECTION_NAME,
                'embedding_model': EMBEDDING_MODEL,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }

            return stats

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


# ============================================================
# 测试代码
# ============================================================

def test_indexer():
    """
    测试代码索引器
    """
    print("=" * 70)
    print("测试代码索引器")
    print("=" * 70)

    indexer = CodeIndexer()

    # 测试1：索引当前目录
    print("\n[测试1] 索引当前目录")
    try:
        num_files = indexer.index_repository(".")
        print(f"✅ 成功索引 {num_files} 个文件")
    except Exception as e:
        print(f"❌ 索引失败: {e}")
        return

    # 测试2：获取统计信息
    print("\n[测试2] 获取索引统计")
    try:
        stats = indexer.get_index_stats()
        print(f"✅ 统计信息:")
        print(f"  代码块数量: {stats.get('total_chunks', 0)}")
        print(f"  Collection: {stats.get('collection_name')}")
        print(f"  Embedding模型: {stats.get('embedding_model')}")
        print(f"  Chunk大小: {stats.get('chunk_size')}")
    except Exception as e:
        print(f"❌ 获取统计失败: {e}")

    # 测试3：测试检索
    print("\n[测试3] 测试检索功能")
    try:
        vectorstore = indexer.get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 搜索
        query = "如何加载代码文件？"
        docs = retriever.invoke(query)

        print(f"✅ 检索到 {len(docs)} 个相关代码块")
        if docs:
            print(f"\n最相关的代码块:")
            print(f"  文件: {docs[0].metadata.get('source', 'unknown')}")
            print(f"  内容预览: {docs[0].page_content[:200]}...")

    except Exception as e:
        print(f"❌ 检索测试失败: {e}")

    print()


if __name__ == "__main__":
    # 注意：运行此测试需要设置 OPENAI_API_KEY 环境变量
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请先设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY=sk-xxx")
    else:
        test_indexer()
