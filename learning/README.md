# 🎓 LLM零基础实践教程

> **适合人群：** 完全零基础，想通过实践理解LLM工作原理

## 📖 教程概览

这是一套**从实践中学习**的教程，通过运行代码、观察现象、理解原理的方式学习LLM。

**学习路径：**
```
01_hello_llm.py          → 体验LLM基础（30分钟）
02_understand_kv_cache.py → 理解优化原理（30分钟）
03_hands_on_rag.py       → 实战RAG系统（60分钟）[待创建]
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/Biglone/workspace/AIGC/learning

# 安装Python包
pip install openai tiktoken numpy
```

### 2. 设置API Key（仅第1个教程需要）

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**如何获取API key？**
1. 访问 https://platform.openai.com/
2. 注册/登录账号
3. 进入 API Keys 页面
4. 创建新的API key

**没有API key？**
- 可以直接跳到第2个教程（不需要API key）
- 或使用本地模型（见高级部分）

### 3. 运行第一个教程

```bash
python 01_hello_llm.py
```

**您将看到：**
- ✅ LLM如何逐词生成文本
- ✅ Temperature参数的作用
- ✅ 什么是注意力机制
- ✅ 什么是Token

### 4. 运行第二个教程

```bash
python 02_understand_kv_cache.py
```

**您将理解：**
- ✅ 为什么需要KV Cache
- ✅ 如何加速20倍
- ✅ 内存和速度的权衡

---

## 📚 教程详情

### 01_hello_llm.py - LLM基础体验

**学习目标：**
- 理解什么是"自回归生成"
- 理解"注意力机制"的作用
- 理解Token的概念

**运行时间：** ~5分钟

**代码亮点：**
```python
# 流式输出，看到逐词生成
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    stream=True  # 关键：流式输出
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

**关键概念：**
- **自回归生成** = 一次生成一个词
- **注意力** = 回头看之前的词
- **Token** = 模型的基本单位（可能是词、字符、子词）

---

### 02_understand_kv_cache.py - 理解优化原理

**学习目标：**
- 理解为什么自回归生成很慢
- 理解KV Cache优化原理
- 体验20倍加速效果

**运行时间：** ~10分钟

**代码亮点：**
```python
# 没有cache：每次都重新计算所有词
for i in range(1, seq_len + 1):
    Q = input[:i] @ W_Q  # 重复计算前面的词
    K = input[:i] @ W_K
    # ...

# 有cache：只计算新词
K_cache = []  # 保存之前的计算结果
for i in range(1, seq_len + 1):
    K_new = input[i] @ W_K  # 只计算新词
    K_cache.append(K_new)   # 保存
    # ...
```

**关键洞察：**
```
没有cache: 1² + 2² + 3² + ... + n² = O(n³)
有cache:   1 + 2 + 3 + ... + n = O(n²)
加速比:    20x+
```

---

## 🎯 学习建议

### 对于完全零基础的您

**第一天（1小时）：**
1. 运行 `01_hello_llm.py`（30分钟）
2. 运行 `02_understand_kv_cache.py`（30分钟）
3. **不要急于理解所有细节！**

**第二天（1小时）：**
1. 重新运行两个教程
2. 尝试修改参数，观察变化：
   - 修改 `max_tokens`（生成长度）
   - 修改 `temperature`（随机性）
   - 修改 `seq_len`（序列长度）

**第三天（2小时）：**
1. 阅读 `docs/01_LLM_Fundamentals.md`
2. 现在您应该能理解30-40%的内容
3. 遇到不懂的概念，回头看代码

**一周后：**
- 重新阅读 `docs/01_LLM_Fundamentals.md`
- 您会发现能理解60-70%
- 这就是"实践 → 理论 → 实践"的螺旋上升

---

## 💡 常见问题

### Q1: 没有OpenAI API key怎么办？

**方案1：** 跳过第1个教程，直接学第2个（不需要API）

**方案2：** 使用本地模型
```bash
# 使用Ollama运行本地模型
ollama pull llama2
# 修改代码，使用本地API
```

### Q2: 运行报错怎么办？

**检查清单：**
- [ ] Python版本 >= 3.8
- [ ] 已安装依赖：`pip install openai tiktoken numpy`
- [ ] API key正确（第1个教程）
- [ ] 网络连接正常

**常见错误：**
```bash
# 错误1：找不到模块
pip install openai tiktoken numpy

# 错误2：API key错误
export OPENAI_API_KEY="sk-..."  # 确保以sk-开头

# 错误3：网络问题
# 如在国内，可能需要代理
export https_proxy=http://127.0.0.1:7890
```

### Q3: 代码看不懂怎么办？

**不要慌！** 这很正常。

**学习策略：**
1. **先运行**，看现象
2. **再观察**，看输出
3. **后理解**，看注释
4. **最后改**，尝试修改参数

**记住：** 工程师的学习方式是"先会用，再理解原理"

### Q4: 数学公式还是看不懂？

**完全可以！**

大部分工程师都是这样：
1. 先知道"这个技术能做什么"
2. 再知道"怎么用这个技术"
3. 最后才理解"为什么这样设计"

**数学是用来"深入理解"的，不是"入门必需"的**

---

## 🔗 下一步

完成这两个教程后，您可以：

**路径1：深入理论**
- 重新阅读 `docs/01_LLM_Fundamentals.md`
- 现在您应该能看懂更多内容了

**路径2：继续实践**
- 查看 `mini_projects/` 的项目
- 运行 `code-qa-rag-system`

**路径3：工程实现**
- 查看 `llm-inference-engine` 的C++实现
- 理解真实的KV Cache代码

**推荐：** 路径2（继续实践）→ 路径1（深入理论）→ 路径3（工程实现）

---

## 📝 反馈

如果您在学习过程中：
- 遇到不理解的地方
- 发现代码bug
- 有更好的教学建议

欢迎：
1. 记录在学习笔记中
2. 与我讨论
3. 改进这套教程

---

## 🎓 学习目标检查清单

完成这两个教程后，您应该能：

- [ ] 用自己的话解释"什么是LLM"
- [ ] 用自己的话解释"什么是注意力机制"
- [ ] 理解为什么LLM生成文本需要时间
- [ ] 理解为什么KV Cache能加速
- [ ] 知道Temperature参数的作用
- [ ] 知道Token的概念

**如果还不能，没关系！** 重新运行代码，或者问我具体问题。

---

**记住：学习LLM不是一天的事，慢慢来，每天进步一点点！** 🚀
