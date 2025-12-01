"""
配置文件
"""

import os
from pathlib import Path

# ============================================================
# 基础配置
# ============================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# ============================================================
# OpenAI配置
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM模型
LLM_MODEL = "gpt-4o-mini"  # 便宜快速
LLM_TEMPERATURE = 0  # 确定性输出

# Embedding模型
EMBEDDING_MODEL = "text-embedding-3-small"  # 性价比高
# EMBEDDING_MODEL = "text-embedding-3-large"  # 高精度（贵）

# ============================================================
# 代码加载配置
# ============================================================

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = {
    # C/C++
    '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
    # Python
    '.py',
    # Java
    '.java',
    # JavaScript/TypeScript
    '.js', '.jsx', '.ts', '.tsx',
    # Go
    '.go',
    # Rust
    '.rs',
    # 其他
    '.swift', '.kt', '.rb', '.php'
}

# 忽略的目录
IGNORE_DIRS = {
    '.git', '.svn', '.hg',
    'node_modules', 'venv', 'env', '.venv',
    '__pycache__', '.pytest_cache',
    'build', 'dist', 'target',
    '.idea', '.vscode'
}

# 忽略的文件模式
IGNORE_PATTERNS = {
    '*.pyc', '*.pyo', '*.pyd',
    '*.so', '*.dylib', '*.dll',
    '*.o', '*.obj', '*.exe',
    '.DS_Store', 'Thumbs.db'
}

# ============================================================
# 文档切分配置
# ============================================================

# Chunk大小（字符数）
CHUNK_SIZE = 1000  # 代码建议稍大，保持函数完整性
CHUNK_OVERLAP = 200  # 重叠部分

# 分隔符（优先级从高到低）
SEPARATORS = [
    "\n\n",  # 段落
    "\nclass ",  # 类定义
    "\ndef ",    # 函数定义
    "\n",        # 行
    " ",         # 空格
    ""
]

# ============================================================
# 向量数据库配置
# ============================================================

# Collection名称
COLLECTION_NAME = "code_knowledge_base"

# 检索配置
RETRIEVAL_TOP_K = 3  # 检索文档数量
RETRIEVAL_SEARCH_TYPE = "similarity"  # similarity 或 mmr

# MMR参数（如果使用MMR）
MMR_FETCH_K = 20  # 候选文档数
MMR_LAMBDA = 0.5  # 多样性 vs 相似度平衡

# ============================================================
# QA提示词模板
# ============================================================

QA_PROMPT_TEMPLATE = """你是一个代码专家助手。基于以下代码片段回答用户的问题。

代码上下文：
{context}

问题：{question}

回答要求：
1. 基于提供的代码片段回答
2. 给出准确的技术解释
3. 如果代码中有问题，指出并给出建议
4. 引用具体的代码行或文件名
5. 如果代码片段不足以回答问题，明确说明需要更多上下文

回答："""

# ============================================================
# 代码分析提示词
# ============================================================

CODE_REVIEW_PROMPT = """你是一个资深代码审查专家。请审查以下代码：

代码：
{code}

文件：{filename}

请从以下角度分析：
1. **正确性**：代码逻辑是否正确？
2. **性能**：是否有性能问题？时间/空间复杂度如何？
3. **安全性**：是否有安全隐患（内存泄漏、缓冲区溢出等）？
4. **可读性**：代码是否清晰易懂？
5. **最佳实践**：是否遵循该语言的最佳实践？

请给出具体的改进建议。
"""

CODE_EXPLAIN_PROMPT = """你是一个代码教学专家。请解释以下代码：

代码：
{code}

文件：{filename}

请：
1. 用通俗易懂的语言解释代码功能
2. 说明关键算法和数据结构
3. 指出重要的设计模式（如果有）
4. 解释复杂的代码段
5. 给出使用示例

解释："""

# ============================================================
# 日志配置
# ============================================================

import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 配置日志
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)
