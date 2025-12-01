"""
代码库问答系统 - Web界面

基于Gradio的交互式代码问答界面

运行：python app.py
访问：http://localhost:7860
"""

import gradio as gr
import os
from qa_engine import CodeQAEngine
from code_indexer import CodeIndexer


# 全局变量
qa_engine = None


def index_codebase(repo_path, progress=gr.Progress()):
    """
    索引代码库
    """
    try:
        progress(0, desc="开始索引...")

        if not os.path.exists(repo_path):
            return f"❌ 路径不存在: {repo_path}"

        # 创建索引器
        indexer = CodeIndexer()

        progress(0.3, desc="扫描代码文件...")

        # 索引代码库
        num_files = indexer.index_repository(repo_path)

        progress(0.9, desc="保存索引...")

        # 重新加载QA引擎
        global qa_engine
        qa_engine = CodeQAEngine()

        progress(1.0, desc="完成!")

        return f"""
✅ 索引完成！

📊 统计信息：
- 代码文件：{num_files} 个
- 索引路径：{repo_path}
- 向量数据库：已更新

现在可以开始提问了！
"""

    except Exception as e:
        return f"❌ 索引失败: {str(e)}"


def answer_question(question, chat_history):
    """
    回答问题
    """
    global qa_engine

    if qa_engine is None:
        return chat_history + [
            (question, "⚠️ 请先索引代码库！点击左侧'索引代码库'标签页。")
        ]

    try:
        # 调用QA引擎
        result = qa_engine.query(question)

        # 格式化回答
        answer = result['answer']

        # 添加来源文档
        if result.get('source_documents'):
            answer += "\n\n📚 **参考代码：**\n"
            for i, doc in enumerate(result['source_documents'][:2]):
                source = doc.metadata.get('source', '未知')
                answer += f"\n**[{i+1}] {source}**\n"
                answer += f"```\n{doc.page_content[:300]}...\n```\n"

        # 更新聊天历史
        chat_history.append((question, answer))
        return chat_history

    except Exception as e:
        chat_history.append((question, f"❌ 查询失败: {str(e)}"))
        return chat_history


def clear_chat():
    """
    清空聊天记录
    """
    return []


# 创建Gradio界面
with gr.Blocks(title="代码库问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 代码库问答系统

    基于RAG的智能代码理解助手 - 让AI帮你读懂代码库！
    """)

    with gr.Tabs():
        # Tab 1: 索引代码库
        with gr.Tab("📂 索引代码库"):
            gr.Markdown("""
            ### 第一步：索引你的代码库

            输入代码库路径，系统会自动：
            1. 扫描所有代码文件（.cpp, .h, .py等）
            2. 分析代码结构
            3. 向量化存储

            **示例路径**：
            - `/home/user/my_cpp_project`
            - `./examples/sample_cpp_project`
            - `C:\\Users\\user\\code\\project`
            """)

            repo_path_input = gr.Textbox(
                label="代码库路径",
                placeholder="/path/to/your/codebase",
                value="./examples/sample_cpp_project"
            )

            index_button = gr.Button("🚀 开始索引", variant="primary", size="lg")
            index_output = gr.Markdown(label="索引结果")

            index_button.click(
                fn=index_codebase,
                inputs=[repo_path_input],
                outputs=[index_output]
            )

            gr.Markdown("""
            ---
            **💡 提示**：
            - 首次索引可能需要几分钟
            - 支持增量更新
            - 建议代码有注释（提高准确率）
            """)

        # Tab 2: 代码问答
        with gr.Tab("💬 代码问答"):
            gr.Markdown("""
            ### 第二步：开始提问

            **示例问题**：
            - "这个项目的主要功能是什么？"
            - "Matrix类如何使用？"
            - "解释transpose()函数的实现"
            - "找出所有的bug"
            - "如何优化性能？"
            """)

            chatbot = gr.Chatbot(
                label="对话历史",
                height=400,
                avatar_images=(None, "🤖")
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="你的问题",
                    placeholder="问我关于代码的任何问题...",
                    scale=4
                )
                submit_button = gr.Button("发送", variant="primary", scale=1)

            with gr.Row():
                clear_button = gr.Button("清空对话")

            # 绑定事件
            submit_button.click(
                fn=answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "",  # 清空输入框
                outputs=[question_input]
            )

            question_input.submit(  # 支持回车发送
                fn=answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "",
                outputs=[question_input]
            )

            clear_button.click(
                fn=clear_chat,
                outputs=[chatbot]
            )

            # 示例问题
            gr.Examples(
                examples=[
                    ["这个项目有哪些主要的类和函数？"],
                    ["Matrix类的构造函数如何使用？"],
                    ["解释multiply()函数的实现逻辑"],
                    ["代码中有什么潜在的bug或性能问题？"],
                    ["如何添加一个新的矩阵运算功能？"]
                ],
                inputs=[question_input]
            )

        # Tab 3: 使用说明
        with gr.Tab("📖 使用说明"):
            gr.Markdown("""
            ## 🎯 快速上手

            ### 1. 索引代码库
            - 在"索引代码库"标签页输入路径
            - 点击"开始索引"
            - 等待完成（首次需要1-5分钟）

            ### 2. 开始提问
            - 切换到"代码问答"标签页
            - 输入问题，点击发送
            - AI会基于代码库回答

            ## 💡 提问技巧

            ### 代码理解
            ```
            ✅ 好的问题：
            - "解释Matrix::multiply()函数的实现"
            - "这个项目使用了哪些设计模式？"

            ❌ 不好的问题：
            - "这是什么？"（太模糊）
            - "帮我写代码"（不是问答系统的功能）
            ```

            ### Bug检测
            ```
            ✅ "检查代码中的内存泄漏问题"
            ✅ "找出所有未处理的异常"
            ✅ "分析潜在的性能瓶颈"
            ```

            ### 用法查询
            ```
            ✅ "如何创建一个3x3的矩阵？"
            ✅ "矩阵乘法的用法示例"
            ✅ "transpose()函数的参数是什么？"
            ```

            ## ⚙️ 高级功能

            ### 自定义Embedding模型
            编辑`config.py`：
            ```python
            EMBEDDING_MODEL = "text-embedding-3-large"  # 更高精度
            ```

            ### 调整检索数量
            ```python
            RETRIEVAL_TOP_K = 5  # 检索更多文档
            ```

            ### 支持新语言
            在`code_loader.py`中添加文件扩展名

            ## 📊 性能优化

            - **小项目（<100文件）**：实时响应
            - **中项目（100-1000文件）**：1-2秒
            - **大项目（>1000文件）**：建议分批索引

            ## 🐛 常见问题

            **Q: 索引很慢怎么办？**
            A: 可以先索引核心目录，或使用本地Embedding模型

            **Q: 回答不准确？**
            A:
            1. 检查代码是否有注释
            2. 增加检索文档数（RETRIEVAL_TOP_K）
            3. 使用更好的LLM模型（gpt-4o）

            **Q: 如何更新索引？**
            A: 重新索引即可，系统会自动覆盖

            ## 🔗 资源链接

            - [项目GitHub](https://github.com/your/repo)
            - [LangChain文档](https://python.langchain.com)
            - [OpenAI API](https://platform.openai.com)

            ---

            **由 LangChain + OpenAI 驱动 | MIT License**
            """)

    # 页脚
    gr.Markdown("""
    ---
    <div style='text-align: center; color: gray;'>
    ⚡ Powered by LangChain + OpenAI |
    📧 Questions? <a href='mailto:your@email.com'>Contact Us</a>
    </div>
    """)


# 启动函数
def main():
    """
    启动Gradio应用
    """
    print("=" * 70)
    print("代码库问答系统启动中...")
    print("=" * 70)

    # 检查API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  警告：未检测到OPENAI_API_KEY环境变量")
        print("请设置：export OPENAI_API_KEY=sk-xxx\n")

    # 尝试加载已有的向量数据库
    try:
        global qa_engine
        qa_engine = CodeQAEngine()
        print("✅ 已加载现有的向量数据库")
    except:
        print("ℹ️  未找到向量数据库，请先索引代码库")

    print("\n🚀 启动Web界面...")
    print("📍 访问地址：http://localhost:7860")
    print("\n" + "=" * 70)

    # 启动Gradio
    demo.queue().launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 设为True可以生成公网链接
        show_error=True
    )


if __name__ == "__main__":
    main()
