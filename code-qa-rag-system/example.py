"""
代码库问答系统 - 快速示例

展示如何使用代码问答系统的基本功能
"""

import os
from code_indexer import CodeIndexer
from qa_engine import CodeQAEngine


def example_basic_workflow():
    """
    示例1：基本工作流程
    """
    print("=" * 70)
    print("示例1：基本工作流程")
    print("=" * 70)

    # 步骤1：索引代码库
    print("\n[步骤1] 索引代码库")
    print("-" * 70)

    indexer = CodeIndexer()

    # 索引当前目录（你可以改成任何代码库路径）
    repo_path = "."
    print(f"正在索引: {repo_path}")

    num_files = indexer.index_repository(repo_path)
    print(f"✅ 成功索引 {num_files} 个文件")

    # 查看统计
    stats = indexer.get_index_stats()
    print(f"\n📊 索引统计:")
    print(f"  - 代码块数量: {stats.get('total_chunks', 0)}")
    print(f"  - Chunk大小: {stats.get('chunk_size')}")
    print(f"  - Embedding模型: {stats.get('embedding_model')}")

    # 步骤2：初始化问答引擎
    print("\n[步骤2] 初始化问答引擎")
    print("-" * 70)

    qa_engine = CodeQAEngine()
    print("✅ 问答引擎已就绪")

    # 步骤3：开始提问
    print("\n[步骤3] 开始提问")
    print("-" * 70)

    questions = [
        "这个项目的主要功能是什么？",
        "如何使用CodeLoader加载代码文件？",
        "向量数据库存储在哪里？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("-" * 50)

        result = qa_engine.query(question)

        print(f"回答:\n{result['answer']}")

        # 显示参考来源
        if result.get('source_documents'):
            print(f"\n📚 参考代码:")
            for j, doc in enumerate(result['source_documents'][:2], 1):
                source = doc.metadata.get('source', 'unknown')
                lines = doc.metadata.get('lines', 'N/A')
                print(f"  [{j}] {source} ({lines} 行)")

        print()


def example_code_search():
    """
    示例2：代码搜索
    """
    print("=" * 70)
    print("示例2：代码搜索")
    print("=" * 70)

    qa_engine = CodeQAEngine()

    # 搜索相似代码
    search_queries = [
        "文件加载",
        "向量化",
        "问答功能"
    ]

    for query in search_queries:
        print(f"\n🔍 搜索: {query}")
        print("-" * 70)

        results = qa_engine.search_similar_code(query, k=3)

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"\n[{i}] {metadata.get('source', 'unknown')}")
            print(f"    语言: {metadata.get('language', 'unknown')}")
            print(f"    行数: {metadata.get('lines', 'N/A')}")
            print(f"    内容预览:")
            print(f"    {result['content'][:200].strip()}...")


def example_code_analysis():
    """
    示例3：代码分析
    """
    print("=" * 70)
    print("示例3：代码分析")
    print("=" * 70)

    qa_engine = CodeQAEngine()

    # 示例代码
    sample_code = """
class CodeIndexer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def index_repository(self, repo_path: str) -> int:
        documents = self.code_loader.load_from_directory(repo_path)
        splits = self.text_splitter.split_documents(documents)
        self._create_vectorstore(splits)
        return len(documents)
"""

    # 解释代码
    print("\n[功能1] 代码解释")
    print("-" * 70)
    print(f"代码:\n{sample_code}")
    print("\n解释:")

    explanation = qa_engine.explain_code(sample_code, "code_indexer.py")
    print(explanation)

    # 审查代码
    print("\n" + "=" * 70)
    print("[功能2] 代码审查")
    print("-" * 70)

    buggy_code = """
def load_file(path):
    f = open(path, 'r')
    content = f.read()
    return content
"""

    print(f"代码:\n{buggy_code}")
    print("\n审查结果:")

    review = qa_engine.review_code(buggy_code, "example.py")
    print(review)


def example_interactive_qa():
    """
    示例4：交互式问答
    """
    print("=" * 70)
    print("示例4：交互式问答")
    print("=" * 70)
    print("\n输入问题，输入 'quit' 退出\n")

    qa_engine = CodeQAEngine()

    while True:
        try:
            question = input("\n❓ 你的问题: ").strip()

            if question.lower() in ['quit', 'exit', 'q', '退出']:
                print("再见！👋")
                break

            if not question:
                continue

            print("\n🤖 思考中...")

            result = qa_engine.query(question)

            print(f"\n💡 回答:\n{result['answer']}")

            # 显示来源
            if result.get('source_documents'):
                print(f"\n📚 参考:")
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    print(f"  [{i}] {doc.metadata.get('source', 'unknown')}")

        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    """
    主函数
    """
    # 检查API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误：未设置 OPENAI_API_KEY 环境变量")
        print("\n请先设置:")
        print("  export OPENAI_API_KEY=sk-xxx")
        print("\n或在代码中设置:")
        print("  os.environ['OPENAI_API_KEY'] = 'sk-xxx'")
        return

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                   代码库问答系统 - 示例程序                        ║
╚═══════════════════════════════════════════════════════════════════╝

请选择示例:

1. 基本工作流程（索引 → 问答）
2. 代码搜索（相似代码查找）
3. 代码分析（解释 + 审查）
4. 交互式问答（实时对话）

0. 退出
""")

    while True:
        try:
            choice = input("请选择 (0-4): ").strip()

            if choice == '0':
                print("再见！👋")
                break
            elif choice == '1':
                example_basic_workflow()
            elif choice == '2':
                example_code_search()
            elif choice == '3':
                example_code_analysis()
            elif choice == '4':
                example_interactive_qa()
            else:
                print("❌ 无效选择，请输入 0-4")

            input("\n按回车继续...")

        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
