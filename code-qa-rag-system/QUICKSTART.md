# 🚀 快速开始指南

10分钟上手代码库问答系统！

## 第一步：环境准备

### 1. 安装依赖

```bash
cd project_code_qa
pip install -r requirements.txt
```

### 2. 设置API Key

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-your-api-key-here"

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Windows (CMD)
set OPENAI_API_KEY=sk-your-api-key-here
```

**如何获取API Key？**
- 访问 [OpenAI Platform](https://platform.openai.com/api-keys)
- 登录后点击 "Create new secret key"
- 复制API Key（只显示一次，请保存好）

### 3. 验证安装

```bash
python -c "import langchain; import openai; print('✅ 安装成功')"
```

---

## 第二步：索引你的代码库

有两种方式：

### 方式1：命令行索引（推荐新手）

```bash
python code_indexer.py
```

这会索引当前目录的所有代码文件。

### 方式2：Python脚本索引

```python
from code_indexer import CodeIndexer

indexer = CodeIndexer()

# 索引指定目录
indexer.index_repository("/path/to/your/codebase")

# 查看统计
stats = indexer.get_index_stats()
print(f"索引了 {stats['total_chunks']} 个代码块")
```

**索引需要多久？**
- 小项目（<100文件）：1-2分钟
- 中项目（100-1000文件）：5-10分钟
- 大项目（>1000文件）：15-30分钟

---

## 第三步：开始提问

### 方式1：Web界面（最简单）

```bash
python app.py
```

然后访问：http://localhost:7860

**Web界面功能：**
- ✅ 图形化索引界面
- ✅ 实时对话
- ✅ 历史记录
- ✅ 示例问题

### 方式2：交互式命令行

```bash
python example.py
```

选择 `4. 交互式问答`，然后输入问题。

### 方式3：Python脚本

```python
from qa_engine import CodeQAEngine

# 初始化引擎
engine = CodeQAEngine()

# 提问
result = engine.query("这个项目的主要功能是什么？")
print(result['answer'])
```

---

## 常用功能示例

### 1. 代码理解

```python
engine = CodeQAEngine()

# 问技术问题
result = engine.query("Matrix类如何实现矩阵乘法？")
print(result['answer'])

# 查看引用的代码
for doc in result['source_documents']:
    print(f"来源: {doc.metadata['source']}")
    print(f"内容: {doc.page_content[:200]}...")
```

### 2. 代码搜索

```python
# 搜索相似代码
results = engine.search_similar_code("错误处理", k=5)

for r in results:
    print(f"文件: {r['metadata']['source']}")
    print(f"代码: {r['content'][:150]}...")
```

### 3. 代码审查

```python
code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result
"""

review = engine.review_code(code, "example.py")
print(review)
# 输出：代码可优化为列表推导式，性能更好...
```

### 4. 代码解释

```python
explanation = engine.explain_code(code, "example.py")
print(explanation)
# 输出：这个函数将输入列表的每个元素乘以2...
```

---

## 🎯 提问技巧

### ✅ 好的问题

```
具体明确：
- "Matrix::multiply()函数如何实现？"
- "代码中如何处理文件读取错误？"
- "Vector类支持哪些操作？"

代码审查：
- "检查代码中的内存泄漏"
- "找出性能瓶颈"
- "有哪些潜在的bug？"

用法查询：
- "如何创建一个3x3的矩阵？"
- "transpose()函数的参数是什么？"
```

### ❌ 不好的问题

```
太模糊：
- "这是什么？"
- "代码怎么样？"

超出范围：
- "帮我写一个新功能"（不是问答系统的职责）
- "Python和Java哪个好？"（与代码库无关）
```

---

## ⚙️ 配置调优

编辑 `config.py` 进行自定义：

### 提高回答质量

```python
# 使用更强大的模型
LLM_MODEL = "gpt-4o"  # 默认是 gpt-4o-mini

# 检索更多上下文
RETRIEVAL_TOP_K = 5  # 默认是 3
```

### 提高准确率

```python
# 使用更好的Embedding模型
EMBEDDING_MODEL = "text-embedding-3-large"  # 默认是 small

# 减小Chunk大小（保持代码完整性）
CHUNK_SIZE = 800  # 默认是 1000
```

### 支持新语言

```python
SUPPORTED_EXTENSIONS = {
    '.cpp', '.h', '.py', '.java',
    '.rb',  # 添加 Ruby
    '.scala'  # 添加 Scala
}
```

---

## 🐛 常见问题

### Q1: 提示 "未找到向量数据库"

**原因**：还没有索引代码库

**解决**：
```bash
python code_indexer.py
```

### Q2: 索引很慢

**原因**：Embedding API调用较慢

**解决**：
- 使用更小的模型：`EMBEDDING_MODEL = "text-embedding-3-small"`
- 只索引核心目录
- 考虑使用本地Embedding模型（Ollama）

### Q3: 回答不准确

**可能原因和解决方案**：

1. **检索文档太少**
   ```python
   RETRIEVAL_TOP_K = 5  # 增加到5
   ```

2. **代码没有注释**
   - 建议给关键代码添加注释
   - AI依赖注释理解代码意图

3. **Chunk切分不当**
   ```python
   CHUNK_SIZE = 1500  # 增大chunk保持完整性
   ```

4. **模型能力不足**
   ```python
   LLM_MODEL = "gpt-4o"  # 使用更强的模型
   ```

### Q4: API调用失败

**检查**：
```bash
echo $OPENAI_API_KEY  # 确认已设置

# 测试API连接
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Q5: 内存不足

**解决**：
```python
# 减小批处理大小
CHUNK_SIZE = 500
RETRIEVAL_TOP_K = 2
```

---

## 📊 性能基准

测试环境：M1 Mac, 8GB RAM

| 项目规模 | 文件数 | 索引时间 | 查询延迟 |
|---------|--------|---------|---------|
| 小项目   | <50    | 1-2分钟  | 2-3秒   |
| 中项目   | 100-500 | 5-10分钟 | 3-5秒   |
| 大项目   | >1000  | 20-30分钟| 4-6秒   |

---

## 🔧 进阶用法

### 1. 自定义提示词

编辑 `config.py` 中的 `QA_PROMPT_TEMPLATE`：

```python
QA_PROMPT_TEMPLATE = """你是一个C++专家。基于以下代码回答问题。

代码：{context}
问题：{question}

要求：
1. 关注内存管理
2. 指出性能问题
3. 给出优化建议

回答："""
```

### 2. 批量问答

```python
questions = [
    "主要的类有哪些？",
    "如何编译这个项目？",
    "有哪些依赖库？"
]

results = engine.batch_query(questions)

for r in results:
    print(f"Q: {r['question']}")
    print(f"A: {r['answer']}\n")
```

### 3. 使用MMR检索（提高多样性）

```python
# 在 config.py 中
RETRIEVAL_SEARCH_TYPE = "mmr"  # 默认是 "similarity"
```

---

## 📚 完整示例项目

查看 `examples/sample_cpp_project/` 中的示例C++项目：

```bash
# 索引示例项目
python -c "
from code_indexer import CodeIndexer
indexer = CodeIndexer()
indexer.index_repository('./examples/sample_cpp_project')
"

# 提问
python -c "
from qa_engine import CodeQAEngine
engine = CodeQAEngine()
result = engine.query('Matrix类如何使用？')
print(result['answer'])
"
```

---

## 🎓 学习路径

1. **第1天**：完成环境准备和索引
2. **第2天**：尝试各种问题类型
3. **第3天**：调整配置，优化效果
4. **第4天**：集成到工作流程

---

## 💡 最佳实践

### 1. 索引前准备

- ✅ 添加README文件（AI会读取）
- ✅ 给关键函数加注释
- ✅ 使用清晰的文件结构

### 2. 提问技巧

- ✅ 包含具体的类名/函数名
- ✅ 一次问一个明确的问题
- ✅ 利用上下文连续提问

### 3. 优化策略

- ✅ 定期重建索引（代码更新后）
- ✅ 根据项目特点调整Chunk大小
- ✅ 使用合适的Embedding模型

---

## 🚀 下一步

- 阅读 [README.md](README.md) 了解架构
- 查看 [example.py](example.py) 学习更多用法
- 启动 [app.py](app.py) 使用Web界面
- 自定义 [config.py](config.py) 适配你的项目

---

## 🤝 获取帮助

- 📖 查看 [README.md](README.md)
- 💻 查看代码注释
- 🐛 [提交Issue](https://github.com/your/repo/issues)

---

**祝你使用愉快！🎉**
