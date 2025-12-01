# CLAUDE_CN.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供中文指导。

## 仓库概览

这是一个 AIGC（AI生成内容）作品集仓库，包含三个主要项目，展示了 LLM 优化和 RAG 应用。代码总量：约 10,000 行 C++17 和 Python 代码。

**项目列表：**
1. **llm-inference-engine** - 高性能 C++17 LLM 推理引擎，实现了 KV Cache、INT8 量化和 SIMD 优化
2. **code-qa-rag-system** - 基于 RAG 的代码问答系统，使用 LangChain、ChromaDB 和 OpenAI
3. **mini_projects** - 四个快速原型项目（Mini-RAG、量化工具、Prompt 优化器、基准测试工具）

## 构建与开发命令

### C++ 推理引擎 (llm-inference-engine)

**构建：**
```bash
cd llm-inference-engine
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**运行测试：**
```bash
# 从 build 目录运行
./test_kv_cache          # 测试 KV Cache 实现
./test_quantization      # 测试 INT8/INT4 量化
./benchmark              # 性能基准测试
```

**构建选项：**
- Release 构建（优化）：`cmake -DCMAKE_BUILD_TYPE=Release ..`
- 激进优化：`cmake -DCMAKE_CXX_FLAGS="-O3 -march=native" ..`
- AVX2 支持会自动检测并启用（如果可用）

**关键文件：**
- `cpp/include/kv_cache.h` - KV Cache 接口
- `cpp/include/quantization.h` - 量化接口
- `cpp/src/kv_cache.cpp` - KV Cache 实现
- `cpp/src/quantization.cpp` - 带 SIMD 优化的量化实现
- `CMakeLists.txt` - 构建配置

### RAG 问答系统 (code-qa-rag-system)

**环境设置：**
```bash
cd code-qa-rag-system
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

**索引代码库：**
```bash
python code_indexer.py  # 索引当前目录
```

**运行系统：**
```bash
python app.py           # 启动 Gradio Web 界面（端口 7860）
python example.py       # 交互式 CLI 示例
```

**关键文件：**
- `code_indexer.py` - 使用 tree-sitter 的代码加载和索引
- `qa_engine.py` - RAG 查询引擎
- `code_loader.py` - 多语言代码文件加载器
- `app.py` - Gradio Web 界面
- `config.py` - 所有配置（模型、分块大小、提示词）

**依赖项：**
- langchain, langchain-openai, openai
- chromadb（向量数据库）
- gradio（Web UI）
- tree-sitter, pygments（代码解析）

### Mini 项目 (mini_projects)

每个 mini 项目都是独立的：
```bash
cd mini_projects/mini_rag && python mini_rag.py
cd mini_projects/quantization_tool && python quantizer.py
cd mini_projects/prompt_optimizer && python prompt_optimizer.py
cd mini_projects/benchmark_tool && python benchmark.py
```

## 理论与学习资源

本仓库在 `docs/` 目录中包含全面的理论文档，以支持对 AIGC 技术的深入理解。

### 理论文档

**核心概念：**
1. **[LLM 基础](docs/01_LLM_Fundamentals.md)** - Transformer 架构、自注意力机制、自回归生成、模型扩展
2. **[推理优化](docs/02_Inference_Optimization.md)** - KV Cache 理论、量化数学、SIMD 向量化、性能分析
3. **[RAG 系统理论](docs/03_RAG_System_Theory.md)** - 向量嵌入、检索算法、评估指标、高级 RAG 技术
4. **[Prompt 工程](docs/04_Prompt_Engineering.md)** - Zero-shot/Few-shot、思维链、ReAct、优化策略
5. **[学习路线图](docs/05_Learning_Roadmap.md)** - 从基础到高级的完整 3 个月学习路径

**高级主题：**
6. **[模型训练与微调](docs/06_Model_Training_and_Finetuning.md)** - LoRA、QLoRA、RLHF、DPO、指令微调、分布式训练策略
7. **[LLM 评估](docs/07_LLM_Evaluation.md)** - MMLU、HumanEval、GSM8K 基准、LLM-as-Judge、幻觉检测、人工评估
8. **[Agent 系统](docs/08_Agent_Systems.md)** - ReAct 框架、函数调用、工具使用、规划策略、记忆管理、多 Agent 协作
9. **[多模态 AI](docs/09_Multimodal_AI.md)** - CLIP、LLaVA、GPT-4V、Stable Diffusion、ControlNet、Whisper、视觉-语言模型
10. **[生产部署](docs/10_Production_Deployment.md)** - vLLM、TensorRT-LLM、连续批处理、监控、成本优化、高可用性
11. **[安全与对齐](docs/11_Safety_and_Alignment.md)** - Prompt 注入防护、毒性检测、Constitutional AI、隐私保护、红队测试

**知识覆盖分析：**
- **[知识缺口分析](docs/00_Knowledge_Gap_Analysis.md)** - AIGC 知识体系完整性分析（93% 完成度）、学习路径和下一步建议

### 何时查阅理论文档

**实现之前：**
- 阅读 [LLM 基础](docs/01_LLM_Fundamentals.md) 了解为什么 KV Cache 能提供 20x 加速
- 查看 [推理优化](docs/02_Inference_Optimization.md) 了解量化权衡（INT8 vs INT4）
- 检查 [RAG 系统理论](docs/03_RAG_System_Theory.md) 了解分块策略和检索方法
- 参考 [Agent 系统](docs/08_Agent_Systems.md) 在构建带工具使用的自主 Agent 之前

**优化时：**
- [推理优化](docs/02_Inference_Optimization.md) 解释 SIMD 优化技术和内存带宽瓶颈
- [RAG 系统理论](docs/03_RAG_System_Theory.md) 涵盖混合搜索和重排策略
- [Prompt 工程](docs/04_Prompt_Engineering.md) 提供 A/B 测试框架和评估指标
- [生产部署](docs/10_Production_Deployment.md) 涵盖 vLLM 设置、监控和成本优化

**训练/微调时：**
- [模型训练与微调](docs/06_Model_Training_and_Finetuning.md) 解释 LoRA、QLoRA、RLHF 和 DPO，包含完整代码示例
- [LLM 评估](docs/07_LLM_Evaluation.md) 提供基准测试方法和幻觉检测技术
- [安全与对齐](docs/11_Safety_and_Alignment.md) 涵盖对齐技术和安全最佳实践

**构建多模态应用时：**
- [多模态 AI](docs/09_Multimodal_AI.md) 涵盖 CLIP 图文匹配、Stable Diffusion 生成、Whisper 语音识别

**学习时：**
- 从 [学习路线图](docs/05_Learning_Roadmap.md) 开始，获得结构化进阶路径
- 查阅 [知识缺口分析](docs/00_Knowledge_Gap_Analysis.md) 了解覆盖情况并选择学习路径
- 每个理论文档都包含数学推导、代码示例和实践指导
- 文档间交叉引用，便于深入研究特定主题

### 关键理论洞察

**LLM 推理：**
- 注意力复杂度：O(n²) 与序列长度相关 - 这就是为什么长上下文很昂贵
- Prefill vs Decode：Prefill 是计算密集型（快），Decode 是内存密集型（慢）
- KV Cache 用 1GB 内存换取 20x 加速，通过避免重复计算实现

**量化：**
- INT8：`scale = max(|W|) / 127`，提供 4x 压缩，精度损失 <1%
- 分组量化：每组独立 scale，将 SQNR 从 35dB 提升到 42dB
- 内存带宽：INT8 读取速度比 FP16 快 2x，直接提升解码速度

**RAG 检索：**
- 嵌入相似度：1536 维空间中的余弦相似度捕获语义含义
- 分块大小权衡：较大分块（1000+ 字符）保留上下文，较小分块（500 字符）提高精度
- Top-K 选择：大多数任务的最佳值是 3-5 个文档（更多会增加噪声，更少会遗漏信息）

**SIMD 优化：**
- AVX2：256 位寄存器 = 8x float32 并行（理论 8x，实际 3-4x，受限于内存带宽）
- 内存对齐：32 字节对齐加载比未对齐快 2x
- FMA 指令：融合乘加减少延迟并提高精度

## 架构与设计模式

### LLM 推理引擎架构

**核心优化策略：** 三层优化方法
1. **KV Cache** - 缓存 key/value 矩阵，避免自回归解码时的 O(n²) 重复计算
2. **INT8 量化** - 对称的按通道量化，可配置分组大小（32/64/128）
3. **SIMD 向量化** - AVX2 指令实现 8 路并行浮点运算

**内存布局：** 连续数组以实现缓存友好的访问模式。KV Cache 使用预分配数组，O(1) 更新。

**关键设计选择：**
- 静态库 + 共享库构建，提供灵活性
- 通过 pybind11 提供可选的 Python 绑定
- 仅头文件的工具函数，便于内联关键路径
- 独立的测试可执行文件，链接静态库

### RAG 系统架构

**流程管道：**
```
用户查询 → 查询处理 → 向量检索（ChromaDB）→ 上下文组装 → LLM 生成 → 响应
```

**代码分块策略：**
- 使用 RecursiveCharacterTextSplitter，具有代码感知的分隔符
- 优先级：段落分隔 → 类定义 → 函数定义 → 行
- 分块大小：1000 字符，重叠 200 字符以保留上下文
- 元数据跟踪：文件路径、语言、行号

**检索方法：**
- 嵌入：OpenAI text-embedding-3-small（1536 维）
- 搜索：余弦相似度（默认）或 MMR（多样性）
- Top-K：默认 3 个文档，可在 config.py 中配置

**LLM 配置：**
- 默认模型：gpt-4o-mini（成本高效）
- Temperature：0（确定性输出）
- 自定义提示词在 config.py 中，用于问答、代码审查、代码解释

### Mini 项目设计

**理念：** 快速原型（每个 1-2 天）验证概念，然后再进行完整实现。每个项目都是自包含的，依赖最少。

## 常见开发工作流

### 添加新的量化方法

1. 在 `llm-inference-engine/cpp/src/quantization.cpp` 中添加量化逻辑
2. 更新 `cpp/include/quantization.h` 中的接口
3. 在 `cpp/tests/test_quantization.cpp` 中添加测试用例
4. 重新构建并运行：`cd build && make && ./test_quantization`

### 修改 RAG 检索行为

1. 在 `code-qa-rag-system/config.py` 中编辑检索参数：
   - `RETRIEVAL_TOP_K` - 检索文档数量
   - `RETRIEVAL_SEARCH_TYPE` - "similarity" 或 "mmr"
   - `CHUNK_SIZE` / `CHUNK_OVERLAP` - 分块参数
2. 修改提示词，编辑 config.py 中的 `QA_PROMPT_TEMPLATE`
3. 如果修改了分块参数，需重新索引：`python code_indexer.py`

### 运行性能基准测试

**C++ 引擎：**
```bash
cd llm-inference-engine/build
./benchmark  # 输出 KV Cache、量化、SIMD 的时间对比
```

**Python 基准测试：**
```bash
cd mini_projects/benchmark_tool
python benchmark.py --model gpt-3.5-turbo --metric all
```

## 重要技术细节

### C++ 推理引擎

- **需要 C++17** - 使用了结构化绑定、std::optional、if constexpr
- **AVX2 可选** - 如果不可用，代码会自动回退到标量实现
- **内存对齐** - SIMD 代码需要 32 字节对齐以获得最佳性能
- **无外部依赖** - 纯 C++17 标准库（除了可选的 pybind11）

### RAG 系统

- **需要 OpenAI API** - 设置 OPENAI_API_KEY 环境变量
- **向量数据库持久化** - ChromaDB 将数据存储在 `data/vector_db/` 目录
- **支持的语言** - Python、C/C++、Java、JavaScript/TypeScript、Go、Rust、Swift、Kotlin、Ruby、PHP
- **忽略的目录** - 自动跳过 .git、node_modules、venv、build 等（见 config.py）

### 性能特征

**C++ 引擎：**
- KV Cache：自回归解码加速 20x
- INT8 量化：内存减少 4x，推理加速 2-3x
- 组合优化：总体加速约 30x

**RAG 系统：**
- 检索准确率：85%+（Top-5）
- 响应时间：大多数查询 <1s（P95）
- 可扩展到 100k+ 行代码

## 配置文件

- `llm-inference-engine/CMakeLists.txt` - C++ 构建配置
- `code-qa-rag-system/config.py` - 所有 RAG 系统设置（模型、提示词、分块）
- `code-qa-rag-system/requirements.txt` - Python 依赖
- `.gitignore` - 忽略构建产物、Python 缓存、向量数据库、凭证

## 测试

**C++ 测试：**
- KV Cache 和量化的单元测试
- 基准测试套件，测量 TTFT、吞吐量、内存使用
- 所有测试在 `llm-inference-engine/cpp/tests/`

**Python 测试：**
- `code-qa-rag-system/example.py` 中的使用示例
- 通过 Web UI 进行交互式测试：`python app.py`

## 环境设置

**C++ 开发：**
- GCC 7+ 或 Clang 5+，支持 C++17
- CMake 3.15+
- 可选：支持 AVX2 的 CPU（用于 SIMD）

**Python 开发：**
- Python 3.8+
- 建议使用虚拟环境
- RAG 系统和 mini 项目需要 OpenAI API key

## 项目结构上下文

```
.
├── llm-inference-engine/      # C++17 推理引擎
│   ├── cpp/
│   │   ├── include/           # 头文件
│   │   ├── src/               # 实现
│   │   └── tests/             # 测试可执行文件
│   ├── CMakeLists.txt         # 构建配置
│   └── QUICKSTART.md          # 详细设置指南
│
├── code-qa-rag-system/        # Python RAG 系统
│   ├── code_indexer.py        # 索引管道
│   ├── qa_engine.py           # 查询引擎
│   ├── code_loader.py         # 文件加载
│   ├── app.py                 # Web 界面
│   ├── config.py              # 配置
│   ├── requirements.txt       # 依赖
│   └── examples/              # 示例项目
│
└── mini_projects/             # 原型项目
    ├── mini_rag/              # 简化 RAG（539 行）
    ├── quantization_tool/     # INT8/INT4 量化器（642 行）
    ├── prompt_optimizer/      # Prompt 工程（712 行）
    └── benchmark_tool/        # 性能测试（671 行）
```

## 特别说明

- **ChromaDB 持久化：** 向量数据库存储在本地。删除 `data/vector_db/` 可重置。
- **重新索引：** 修改分块参数或添加新代码文件后需要重新索引。
- **SIMD 调试：** 如果遇到 AVX2 问题，可用 `-DCMAKE_CXX_FLAGS=""` 禁用并重新构建。
- **API 成本：** RAG 系统会调用 OpenAI API。通过 OpenAI 控制台监控使用情况。
