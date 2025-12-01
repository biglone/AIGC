# 🚀 AIGC Mini项目合集

> 4个快速原型项目，1周完成，总计2,564行代码，展示快速学习和实现能力

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 📋 项目概览

| 项目 | 代码行数 | 完成时间 | 核心功能 |
|-----|---------|---------|---------|
| **Mini-RAG** | 539行 | 1-2天 | 简化版RAG实现 |
| **量化工具** | 642行 | 1-2天 | INT8/INT4量化 |
| **Prompt优化器** | 712行 | 1-2天 | 7种优化策略 + A/B测试 |
| **基准测试工具** | 671行 | 1-2天 | TTFT/TPS性能指标 |
| **总计** | **2,564行** | **1周** | **完整原型验证** |

---

## 🎯 项目目标

这些Mini项目的设计目的：

1. **快速验证想法**：1-2天快速实现概念验证
2. **掌握核心原理**：通过简化实现深入理解技术本质
3. **展示学习能力**：从概念到代码的快速转化能力
4. **工程实践**：包含测试、文档、可运行示例

---

## 📦 项目详情

### 1️⃣ Mini-RAG（539行）

**简介**：简化版RAG（检索增强生成）系统实现

**核心功能**：
- 文档加载和分块（支持txt、pdf、markdown）
- 向量嵌入和存储（OpenAI Embeddings + FAISS）
- 相似度检索（Top-K）
- LLM生成（OpenAI API）

**技术栈**：
```
OpenAI API, FAISS, LangChain基础组件
```

**使用示例**：
```bash
cd mini_rag
python mini_rag.py

# 索引文档
python mini_rag.py --index --docs ./documents

# 提问
python mini_rag.py --query "什么是RAG？"
```

**学习重点**：
- RAG基本流程
- 向量检索原理
- Prompt工程基础

---

### 2️⃣ 量化工具（642行）

**简介**：实现INT8和INT4量化，包含性能和精度评估

**核心功能**：
- INT8对称量化
- INT4分组量化（可配置组大小）
- 精度评估（MSE、MAE、SQNR）
- 性能测试（推理速度对比）
- 可视化（量化前后分布对比）

**技术栈**：
```
NumPy, Matplotlib（可视化）
```

**使用示例**：
```bash
cd quantization_tool
python quantizer.py --model path/to/model.bin --bits 8

# 生成对比报告
python quantizer.py --model model.bin --bits 8 --report
```

**输出指标**：
- 内存节省：75%（INT8），87.5%（INT4）
- 精度损失：<1%（SQNR > 40dB）
- 推理加速：1.5-2x

**学习重点**：
- 量化基本原理
- 精度与性能权衡
- 分组量化策略

---

### 3️⃣ Prompt优化器（712行）

**简介**：实现7种Prompt优化策略，带A/B测试框架

**核心功能**：

**7种优化策略**：
1. **Few-Shot Learning**：添加示例
2. **Chain-of-Thought**：引导逐步思考
3. **Role Prompting**：角色设定
4. **Format Control**：输出格式控制
5. **Context Enhancement**：上下文增强
6. **Constraint Addition**：添加约束
7. **Negative Prompting**：明确不想要的

**A/B测试**：
- 自动生成多个变体
- 并行测试
- 指标对比（准确率、响应时间、成本）

**技术栈**：
```
OpenAI API, 并发测试框架
```

**使用示例**：
```bash
cd prompt_optimizer
python prompt_optimizer.py --strategy few-shot --input "分类这个评论"

# A/B测试
python prompt_optimizer.py --ab-test --strategies few-shot,cot,role
```

**学习重点**：
- Prompt工程最佳实践
- A/B测试方法
- 性能优化策略

---

### 4️⃣ 基准测试工具（671行）

**简介**：LLM推理性能测试工具，测量关键指标

**核心功能**：

**性能指标**：
- **TTFT**（Time To First Token）：首token延迟
- **TPS**（Tokens Per Second）：生成吞吐量
- **E2E Latency**：端到端延迟
- **内存占用**：峰值内存使用
- **并发性能**：多用户场景

**测试场景**：
- 单次推理
- 批量处理
- 并发压测
- 长序列测试

**技术栈**：
```
OpenAI API, 多线程, 性能监控
```

**使用示例**：
```bash
cd benchmark_tool
python benchmark.py --model gpt-3.5-turbo --metric all

# 并发测试
python benchmark.py --concurrent 10 --requests 100

# 生成报告
python benchmark.py --report --output benchmark_report.html
```

**输出报告**：
```
性能测试报告
================
TTFT (P50): 245ms
TTFT (P95): 380ms
TPS: 45 tokens/s
并发QPS: 12 requests/s
内存占用: 1.2GB
```

**学习重点**：
- 性能测试方法
- 关键指标定义
- 并发压测技巧

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
cd mini_projects

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置API密钥
export OPENAI_API_KEY="your-api-key-here"
```

### 运行示例

```bash
# 运行Mini-RAG
cd mini_rag && python mini_rag.py

# 运行量化工具
cd quantization_tool && python quantizer.py --demo

# 运行Prompt优化器
cd prompt_optimizer && python prompt_optimizer.py --demo

# 运行基准测试
cd benchmark_tool && python benchmark.py --quick-test
```

---

## 📊 项目统计

### 代码质量

| 指标 | 数值 |
|-----|------|
| 总代码行数 | 2,564行 |
| 平均每项目 | 641行 |
| 测试覆盖率 | 70%+ |
| 文档完整度 | 100% |

### 技术覆盖

| 技术领域 | 涉及项目 |
|---------|---------|
| RAG系统 | Mini-RAG |
| 量化优化 | 量化工具 |
| Prompt工程 | Prompt优化器 |
| 性能测试 | 基准测试工具 |
| OpenAI API | 全部4个 |
| 向量检索 | Mini-RAG |
| 并发编程 | 基准测试工具 |

---

## 🎓 学习价值

### 这些项目教会了什么：

1. **快速原型能力**
   - 1-2天实现完整功能
   - 快速迭代和验证

2. **核心技术理解**
   - RAG工作原理
   - 量化技术细节
   - Prompt工程技巧
   - 性能测试方法

3. **工程实践**
   - 代码组织结构
   - 错误处理
   - 测试和文档
   - 命令行工具开发

4. **技术选型**
   - 不同场景选择合适工具
   - 权衡复杂度和功能

---

## 🔄 与大项目的关系

这些Mini项目是大项目的前期探索：

```
Mini-RAG (539行)
    ↓ 扩展和优化
Code QA System (2,000行)

量化工具 (642行)
    ↓ 集成到推理引擎
LLM Inference Engine (2,200行)

Prompt优化器 (712行)
    ↓ 应用于实际项目
增强RAG系统的Prompt质量

基准测试工具 (671行)
    ↓ 验证性能优化
评估推理引擎性能
```

---

## 📈 项目时间线

```
第1天：Mini-RAG
    • 理解RAG原理
    • 实现基础流程
    • 测试验证

第2天：量化工具
    • 学习量化理论
    • 实现INT8/INT4
    • 精度评估

第3-4天：Prompt优化器
    • 研究Prompt技巧
    • 实现7种策略
    • A/B测试框架

第5天：基准测试工具
    • 定义性能指标
    • 实现测试框架
    • 生成报告

第6-7天：完善和文档
    • 代码重构
    • 添加测试
    • 编写文档
```

---

## 🤝 贡献

欢迎贡献！可以：
- 添加新的Mini项目
- 改进现有实现
- 增强文档和示例

---

## 📝 许可证

本项目采用MIT许可证

---

## 💡 最佳实践总结

### 从这些项目中学到的：

1. **从小做起**：先做简化版本，理解核心原理
2. **快速迭代**：1-2天完成MVP，快速验证想法
3. **注重质量**：即使是小项目，也要有测试和文档
4. **可复用**：设计成模块化，方便集成到大项目

---

**⭐ 如果这些项目对你有帮助，请给个star！**

---

*最后更新：2024年12月*
