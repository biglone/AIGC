# 🚀 AIGC项目集合

> 从C++工程师到AIGC专家的3个月转型之路：完整的学习历程和项目实战

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Projects](https://img.shields.io/badge/Projects-3-brightgreen.svg)]()
[![Code Lines](https://img.shields.io/badge/Code-8K+-blue.svg)]()

---

## 📖 项目概览

本仓库包含3个完整的AIGC项目，展示了从理论到实践的完整技术栈：

| 项目 | 技术栈 | 代码量 | 核心成果 |
|-----|--------|-------|---------|
| **[C++推理引擎](#1-llm推理引擎)** | C++17, AVX2, CMake | 3,622行 | 30倍加速，75%内存节省 |
| **[RAG问答系统](#2-rag问答系统)** | Python, LangChain, ChromaDB | 3,636行 | 85%+准确率，<1秒响应 |
| **[Mini项目集](#3-mini项目集)** | Python, OpenAI API | 2,972行 | 4个原型，1周完成 |
| **总计** | - | **10,230行** | **完整AIGC技术栈** |

---

## 🎯 核心亮点

### 性能优化
- ⚡ **30倍推理加速** - KV Cache + 量化 + SIMD优化
- 💾 **75%内存节省** - INT8量化技术
- 🚀 **亚秒级响应** - RAG系统<1秒返回结果

### 技术深度
- 🔬 **底层优化** - C++ SIMD向量化、内存管理
- 🧠 **AI应用** - RAG架构、Prompt工程
- 📊 **性能测试** - 完整的benchmark框架

### 工程质量
- ✅ **完整测试** - 单元测试 + 性能测试
- 📝 **详细文档** - 每个项目都有完整README
- 🎨 **可运行** - 所有代码都可以直接运行

---

## 📚 理论文档体系

本仓库包含完整的AIGC理论+实践文档体系（**15个文档 + 配置模板库，98%知识覆盖**），从入门到精通：

### 核心理论（5个）

| 文档 | 内容 | 适合人群 |
|-----|------|----------|
| [01_LLM基础](./docs/01_LLM_Fundamentals.md) | Transformer、Attention、自回归生成 | 初学者 |
| [02_推理优化](./docs/02_Inference_Optimization.md) | KV Cache、量化、SIMD | 性能优化者 |
| [03_RAG系统](./docs/03_RAG_System_Theory.md) | 向量检索、Embedding、混合搜索 | 应用开发者 |
| [04_Prompt工程](./docs/04_Prompt_Engineering.md) | Few-shot、CoT、ReAct | 所有人 |
| [05_学习路线](./docs/05_Learning_Roadmap.md) | 3个月系统学习计划 | 转型者 |

### 进阶主题（6个）

| 文档 | 内容 | 适合人群 |
|-----|------|----------|
| [06_模型训练](./docs/06_Model_Training_and_Finetuning.md) | LoRA、QLoRA、RLHF、DPO | 模型开发者 |
| [07_模型评估](./docs/07_LLM_Evaluation.md) | MMLU、HumanEval、幻觉检测 | 研究者 |
| [08_Agent系统](./docs/08_Agent_Systems.md) | ReAct、工具使用、多Agent | 高级开发者 |
| [09_多模态AI](./docs/09_Multimodal_AI.md) | CLIP、Stable Diffusion、Whisper | 多模态开发者 |
| [10_生产部署](./docs/10_Production_Deployment.md) | vLLM、监控、成本优化 | 工程师 |
| [11_安全对齐](./docs/11_Safety_and_Alignment.md) | Prompt防护、Constitutional AI | 安全工程师 |

### 工程实践（2个）🆕

| 文档 | 内容 | 适合人群 |
|-----|------|----------|
| [13_数据工程](./docs/13_Data_Engineering.md) | 采集、清洗、标注、合成数据 | 数据工程师 |
| [14_MLOps](./docs/14_MLOps_Best_Practices.md) | 实验管理、CI/CD、监控 | ML工程师 |

### 实用指南（4个）🆕

| 文档 | 内容 | 适合人群 |
|-----|------|----------|
| [00_术语速查](./docs/00_Glossary_and_Quick_Reference.md) | 150+术语、公式、工具索引 | 所有人 |
| [05_学习路线](./docs/05_Learning_Roadmap.md) | 3个月计划 + 完整检查清单 | 初学者 |
| [12_求职准备](./docs/12_Job_Preparation_Guide.md) | 简历、面试、项目展示 | 求职者 |
| [15_项目实战](./docs/15_End_to_End_Project_Tutorial.md) | 3个端到端案例（智能客服等） | 实战者 |

### 配置模板库 🆕

| 模板 | 内容 | 用途 |
|-----|------|------|
| [templates/](./templates/) | Docker、K8s、CI/CD等配置 | 快速搭建环境 |

**文档特色：**
- 📖 理论深度：数学推导+原理解释
- 💻 代码实战：200+完整可运行示例
- 🎯 实用导向：工程实践+生产应用
- 🔗 体系完整：从基础到高级，逐步递进
- 🎓 学习追踪：完整的学习检查清单
- 🚀 快速启动：配置模板库，节省80%配置时间

**快速开始：**
1. **查概念** → [术语速查表](./docs/00_Glossary_and_Quick_Reference.md)
2. **定计划** → [学习路线+检查清单](./docs/05_Learning_Roadmap.md)
3. **学理论**：
   - 初学者：[01_LLM基础](./docs/01_LLM_Fundamentals.md) → [04_Prompt工程](./docs/04_Prompt_Engineering.md) → [03_RAG系统](./docs/03_RAG_System_Theory.md)
   - 开发者：[02_推理优化](./docs/02_Inference_Optimization.md) → [08_Agent系统](./docs/08_Agent_Systems.md) → [10_生产部署](./docs/10_Production_Deployment.md)
   - 工程师：[13_数据工程](./docs/13_Data_Engineering.md) → [14_MLOps](./docs/14_MLOps_Best_Practices.md)
   - 研究者：[06_模型训练](./docs/06_Model_Training_and_Finetuning.md) → [07_模型评估](./docs/07_LLM_Evaluation.md)
4. **做项目** → [端到端实战](./docs/15_End_to_End_Project_Tutorial.md)
5. **用模板** → [templates/](./templates/) 快速搭建环境

---

## 📂 项目详情

### 1️⃣ LLM推理引擎

**[→ 查看详细文档](./llm-inference-engine/README.md)**

基于C++17的高性能LLM推理引擎，实现三大核心优化：

**核心技术**：
- **KV Cache** - 缓存K/V矩阵，20倍加速
- **INT8量化** - 75%内存节省，<1%精度损失
- **SIMD优化** - AVX2向量化，3-4倍加速

**性能数据**：
```
首token延迟: 1000ms → 30ms (30倍)
内存占用:   4GB → 1GB (减少75%)
量化精度:   SQNR > 40dB
```

**技术栈**: C++17, AVX2 SIMD, CMake, Google Test

---

### 2️⃣ RAG问答系统

**[→ 查看详细文档](./code-qa-rag-system/README.md)**

基于RAG架构的智能代码问答系统：

**核心功能**：
- **语义分块** - 保持函数/类完整性
- **向量检索** - ChromaDB + OpenAI Embeddings
- **多语言支持** - Python, C++, Java, Go等10+语言
- **Web界面** - Gradio实现，零配置使用

**性能数据**：
```
检索准确率: 85%+ (Top-5)
响应时间:   <1秒 (P95)
代码库规模: 10万+行
```

**技术栈**: Python, LangChain, ChromaDB, OpenAI API, Gradio

---

### 3️⃣ Mini项目集

**[→ 查看详细文档](./mini_projects/README.md)**

4个快速原型项目，1周完成：

| 项目 | 代码量 | 功能 |
|-----|-------|------|
| Mini-RAG | 539行 | 简化版RAG实现 |
| 量化工具 | 642行 | INT8/INT4量化 |
| Prompt优化器 | 712行 | 7种优化策略 |
| 基准测试 | 671行 | TTFT/TPS测量 |

**技术栈**: Python, OpenAI API, FAISS, NumPy

---

## 🚀 快速开始

### 环境要求

```bash
# Python环境
Python 3.8+
pip install -r requirements.txt

# C++环境
GCC 7+ 或 Clang 5+
CMake 3.10+
支持AVX2的CPU
```

### 运行项目

```bash
# C++推理引擎
cd llm-inference-engine
mkdir build && cd build
cmake .. && make
./benchmarks/benchmark

# RAG问答系统
cd code-qa-rag-system
export OPENAI_API_KEY="your-key"
python app.py

# Mini项目
cd mini_projects
cd mini_rag && python mini_rag.py
```

---

## 📊 技术栈总览

### 编程语言
- **C++17** - 高性能推理引擎
- **Python 3.8+** - AI应用开发

### AI/ML框架
- **LangChain** - RAG框架
- **ChromaDB** - 向量数据库
- **OpenAI API** - LLM服务

### 性能优化
- **AVX2 SIMD** - 向量化计算
- **INT8/INT4量化** - 模型压缩
- **KV Cache** - 推理加速

### 开发工具
- **CMake** - C++构建系统
- **Google Test** - C++单元测试
- **Gradio** - Web界面

---

## 📈 学习历程

```
第1-2个月: 理论学习
├── Python AI开发基础
├── 机器学习算法
├── 深度学习框架
└── Transformer架构

第3个月: 项目实战
├── Week 1-2: C++推理引擎
├── Week 3-4: RAG问答系统
└── Week 5: Mini项目快速原型

成果:
✅ 3个完整项目
✅ 10,000+行代码
✅ 完整的技术栈
```

---

## 🎓 核心能力展示

### 1. 底层性能优化
- ✅ C++ SIMD向量化编程
- ✅ 内存管理和缓存优化
- ✅ 量化算法实现

### 2. AI应用开发
- ✅ RAG系统架构设计
- ✅ Prompt工程
- ✅ 向量检索优化

### 3. 工程实践
- ✅ 完整的测试框架
- ✅ 性能基准测试
- ✅ 文档和示例

### 4. 快速学习
- ✅ 3个月掌握完整技术栈
- ✅ 从理论到实践的转化能力
- ✅ 1周完成4个原型项目

---

## 🤝 贡献

欢迎提出建议和改进！

---

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 👤 作者

**Biglone**

- 背景: 资深C++工程师 → AIGC专家
- 转型: 3个月系统化学习
- 专注: LLM推理优化、RAG应用

GitHub: [@biglone](https://github.com/biglone)

---

**⭐ 如果这个项目对你有帮助，请给个star！**

---

*最后更新: 2025年12月*
