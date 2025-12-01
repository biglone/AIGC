#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt优化工具 - 自动测试和优化Prompt
===================================

项目目标：
- 系统化测试不同Prompt策略
- 自动评估Prompt效果
- 快速找到最优Prompt

核心功能：
1. 多种Prompt模板
2. A/B测试对比
3. 自动评估打分
4. 最佳实践推荐

作者：面向C++工程师的AIGC学习
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================
# 第一部分：Prompt模板库
# ============================================================

class PromptStrategy(Enum):
    """Prompt策略枚举"""
    BASIC = "basic"                    # 基础直接提问
    ROLE_PLAY = "role_play"           # 角色扮演
    FEW_SHOT = "few_shot"             # Few-shot示例
    CHAIN_OF_THOUGHT = "cot"          # 思维链
    STRUCTURED = "structured"          # 结构化输出
    CONTEXT_RICH = "context_rich"     # 上下文丰富
    CONSTRAINT = "constraint"          # 约束引导


@dataclass
class PromptTemplate:
    """Prompt模板"""
    name: str
    strategy: PromptStrategy
    template: str
    description: str
    examples: List[str] = None


class PromptTemplateLibrary:
    """
    Prompt模板库

    包含各种常用的Prompt策略和模板
    """

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[PromptStrategy, PromptTemplate]:
        """初始化模板库"""
        return {
            # 基础模板
            PromptStrategy.BASIC: PromptTemplate(
                name="基础直接提问",
                strategy=PromptStrategy.BASIC,
                template="{question}",
                description="最简单的提问方式，适合简单明确的问题"
            ),

            # 角色扮演
            PromptStrategy.ROLE_PLAY: PromptTemplate(
                name="角色扮演",
                strategy=PromptStrategy.ROLE_PLAY,
                template="""你是一位{role}。

{question}

请以{role}的专业视角回答。""",
                description="让AI扮演特定角色，提高专业性和针对性",
                examples=["资深Python工程师", "机器学习专家", "系统架构师"]
            ),

            # Few-shot学习
            PromptStrategy.FEW_SHOT: PromptTemplate(
                name="Few-shot示例",
                strategy=PromptStrategy.FEW_SHOT,
                template="""请参考以下示例回答问题：

{examples}

现在请回答：
{question}""",
                description="提供示例引导输出格式和风格"
            ),

            # 思维链
            PromptStrategy.CHAIN_OF_THOUGHT: PromptTemplate(
                name="思维链（CoT）",
                strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                template="""{question}

请一步一步思考：
1. 首先，分析问题的核心
2. 然后，列出解决思路
3. 接着，详细推导过程
4. 最后，得出结论

让我们开始：""",
                description="引导逐步推理，提高复杂问题的准确率"
            ),

            # 结构化输出
            PromptStrategy.STRUCTURED: PromptTemplate(
                name="结构化输出",
                strategy=PromptStrategy.STRUCTURED,
                template="""{question}

请按以下格式回答：
## 概述
[简要说明]

## 详细分析
[具体内容]

## 示例代码
```python
[代码示例]
```

## 注意事项
[重要提示]""",
                description="要求特定格式输出，便于解析和阅读"
            ),

            # 上下文丰富
            PromptStrategy.CONTEXT_RICH: PromptTemplate(
                name="上下文丰富",
                strategy=PromptStrategy.CONTEXT_RICH,
                template="""背景信息：
{context}

基于以上背景，请回答：
{question}

要求：
- 结合背景信息
- 给出具体建议
- 说明理由""",
                description="提供充分上下文，提高答案相关性"
            ),

            # 约束引导
            PromptStrategy.CONSTRAINT: PromptTemplate(
                name="约束引导",
                strategy=PromptStrategy.CONSTRAINT,
                template="""{question}

约束条件：
{constraints}

请在满足以上约束的前提下回答。""",
                description="添加约束条件，精确控制输出"
            ),
        }

    def get_template(self, strategy: PromptStrategy) -> PromptTemplate:
        """获取指定策略的模板"""
        return self.templates.get(strategy)

    def list_templates(self) -> List[PromptTemplate]:
        """列出所有模板"""
        return list(self.templates.values())


# ============================================================
# 第二部分：Prompt生成器
# ============================================================

class PromptBuilder:
    """
    Prompt构建器

    根据模板和参数生成最终的Prompt
    """

    def __init__(self, library: PromptTemplateLibrary):
        self.library = library

    def build(self, strategy: PromptStrategy, **kwargs) -> str:
        """
        构建Prompt

        Args:
            strategy: 策略类型
            **kwargs: 模板参数

        Returns:
            生成的Prompt字符串
        """
        template = self.library.get_template(strategy)
        if not template:
            raise ValueError(f"Unknown strategy: {strategy}")

        try:
            prompt = template.template.format(**kwargs)
            return prompt
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")


# ============================================================
# 第三部分：Prompt测试器
# ============================================================

class PromptTester:
    """
    Prompt测试器

    自动测试不同Prompt并评估效果
    """

    def __init__(self, use_openai: bool = False):
        """
        Args:
            use_openai: 是否使用OpenAI API（False则返回模拟结果）
        """
        self.use_openai = use_openai
        self.client = None

        if use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("警告：未安装openai库，将使用模拟测试")
                self.use_openai = False

    def test_prompt(self, prompt: str, model: str = "gpt-3.5-turbo") -> Dict:
        """
        测试单个Prompt

        Args:
            prompt: Prompt文本
            model: LLM模型

        Returns:
            {
                'response': str,
                'tokens': int,
                'time': float,
                'cost': float
            }
        """
        if self.use_openai:
            return self._test_with_openai(prompt, model)
        else:
            return self._test_mock(prompt)

    def _test_with_openai(self, prompt: str, model: str) -> Dict:
        """使用OpenAI API测试"""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            elapsed = time.time() - start_time

            # 估算成本（GPT-3.5-turbo: $0.0015/1K input, $0.002/1K output）
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000

            return {
                'response': response.choices[0].message.content,
                'tokens': response.usage.total_tokens,
                'time': elapsed,
                'cost': cost,
                'success': True
            }

        except Exception as e:
            return {
                'response': f"Error: {str(e)}",
                'tokens': 0,
                'time': 0,
                'cost': 0,
                'success': False
            }

    def _test_mock(self, prompt: str) -> Dict:
        """模拟测试"""
        # 模拟延迟
        time.sleep(0.1)

        # 根据prompt长度模拟token数
        tokens = len(prompt.split()) * 1.3

        return {
            'response': f"[模拟回答] 这是针对prompt的模拟响应。\n\nPrompt长度: {len(prompt)}字符\n\n要获得真实响应，请设置OPENAI_API_KEY并使用use_openai=True。",
            'tokens': int(tokens),
            'time': 0.1,
            'cost': tokens * 0.002 / 1000,
            'success': True
        }

    def ab_test(self, prompts: Dict[str, str], model: str = "gpt-3.5-turbo") -> Dict:
        """
        A/B测试多个Prompt

        Args:
            prompts: {name: prompt_text}
            model: LLM模型

        Returns:
            测试结果对比
        """
        results = {}

        print(f"开始A/B测试 ({len(prompts)}个Prompt变体)...")
        print("=" * 60)

        for name, prompt in prompts.items():
            print(f"\n测试: {name}")
            result = self.test_prompt(prompt, model)
            results[name] = result

            if result['success']:
                print(f"  ✓ Tokens: {result['tokens']}")
                print(f"  ✓ 耗时: {result['time']:.2f}s")
                print(f"  ✓ 成本: ${result['cost']:.6f}")
            else:
                print(f"  ✗ 失败: {result['response']}")

        return results


# ============================================================
# 第四部分：Prompt评估器
# ============================================================

class PromptEvaluator:
    """
    Prompt评估器

    根据多个维度评估Prompt质量
    """

    @staticmethod
    def evaluate(prompt: str, response: str, criteria: List[str] = None) -> Dict:
        """
        评估Prompt和响应质量

        评估维度：
        1. Prompt清晰度
        2. 响应完整度
        3. 响应相关性
        4. 结构化程度
        5. 实用性

        Args:
            prompt: Prompt文本
            response: LLM响应
            criteria: 自定义评估标准

        Returns:
            评分和分析
        """
        scores = {}

        # 1. Prompt清晰度（基于长度和结构）
        prompt_clarity = PromptEvaluator._evaluate_clarity(prompt)
        scores['prompt_clarity'] = prompt_clarity

        # 2. 响应完整度（基于长度）
        response_completeness = PromptEvaluator._evaluate_completeness(response)
        scores['response_completeness'] = response_completeness

        # 3. 响应结构化程度
        response_structure = PromptEvaluator._evaluate_structure(response)
        scores['response_structure'] = response_structure

        # 4. 包含代码（如果适用）
        has_code = '```' in response
        scores['has_code'] = 1.0 if has_code else 0.0

        # 5. 综合评分
        overall = sum(scores.values()) / len(scores)
        scores['overall'] = overall

        return scores

    @staticmethod
    def _evaluate_clarity(prompt: str) -> float:
        """评估prompt清晰度"""
        # 简单启发式：100-300字符最佳
        length = len(prompt)
        if 100 <= length <= 300:
            return 1.0
        elif length < 50:
            return 0.5  # 太简单
        elif length > 500:
            return 0.7  # 太复杂
        else:
            return 0.8

    @staticmethod
    def _evaluate_completeness(response: str) -> float:
        """评估响应完整度"""
        # 基于长度：200-1000字符较好
        length = len(response)
        if 200 <= length <= 1000:
            return 1.0
        elif length < 100:
            return 0.3
        elif length > 2000:
            return 0.8
        else:
            return 0.7

    @staticmethod
    def _evaluate_structure(response: str) -> float:
        """评估响应结构化程度"""
        score = 0.5  # 基础分

        # 检查markdown标记
        if '#' in response:
            score += 0.2
        if '```' in response:
            score += 0.2
        if any(marker in response for marker in ['1.', '2.', '-', '*']):
            score += 0.1

        return min(score, 1.0)


# ============================================================
# 第五部分：实战演示
# ============================================================

def demo_prompt_strategies():
    """演示不同Prompt策略"""
    print("=" * 60)
    print("Prompt策略演示")
    print("=" * 60)

    # 初始化
    library = PromptTemplateLibrary()
    builder = PromptBuilder(library)

    # 测试问题
    question = "如何优化Python代码的性能？"

    print(f"\n原始问题：{question}\n")

    # 测试各种策略
    strategies = [
        (PromptStrategy.BASIC, {}),
        (PromptStrategy.ROLE_PLAY, {"role": "资深Python性能优化专家"}),
        (PromptStrategy.CHAIN_OF_THOUGHT, {}),
        (PromptStrategy.STRUCTURED, {}),
    ]

    for i, (strategy, params) in enumerate(strategies, 1):
        template = library.get_template(strategy)
        params['question'] = question

        prompt = builder.build(strategy, **params)

        print(f"\n{'─' * 60}")
        print(f"策略 {i}: {template.name}")
        print(f"描述: {template.description}")
        print(f"{'─' * 60}")
        print(prompt)


def demo_ab_testing():
    """演示A/B测试"""
    print("\n" + "=" * 60)
    print("A/B测试演示")
    print("=" * 60)

    library = PromptTemplateLibrary()
    builder = PromptBuilder(library)
    tester = PromptTester(use_openai=False)

    question = "解释什么是KV Cache"

    # 准备多个变体
    prompts = {
        "基础版": builder.build(PromptStrategy.BASIC, question=question),
        "角色扮演版": builder.build(
            PromptStrategy.ROLE_PLAY,
            role="LLM推理优化专家",
            question=question
        ),
        "思维链版": builder.build(PromptStrategy.CHAIN_OF_THOUGHT, question=question),
    }

    # 运行A/B测试
    results = tester.ab_test(prompts)

    # 评估结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    for name, result in results.items():
        if result['success']:
            scores = PromptEvaluator.evaluate(
                prompts[name],
                result['response']
            )

            print(f"\n{name}:")
            print(f"  综合评分: {scores['overall']:.2f}")
            print(f"  Prompt清晰度: {scores['prompt_clarity']:.2f}")
            print(f"  响应完整度: {scores['response_completeness']:.2f}")
            print(f"  结构化程度: {scores['response_structure']:.2f}")
            print(f"  包含代码: {'是' if scores['has_code'] > 0 else '否'}")
            print(f"  成本: ${result['cost']:.6f}")


def demo_best_practices():
    """展示Prompt最佳实践"""
    print("\n" + "=" * 60)
    print("Prompt工程最佳实践")
    print("=" * 60)

    practices = [
        {
            "title": "1. 明确任务和角色",
            "bad": "写一个函数",
            "good": "作为Python专家，请编写一个高效的二分查找函数，要求：\n- 输入：有序列表和目标值\n- 输出：目标值的索引\n- 包含注释和类型提示"
        },
        {
            "title": "2. 提供示例（Few-shot）",
            "bad": "将文本分类",
            "good": """请将文本分类为正面/负面/中性。

示例：
输入："这个产品太棒了！"
输出：正面

输入："质量一般"
输出：中性

现在请分类："{text}" """
        },
        {
            "title": "3. 分步引导（CoT）",
            "bad": "计算结果",
            "good": """请计算以下问题，并展示推理过程：

问题：一个长方形长10cm，宽5cm，求面积。

请这样回答：
步骤1：识别已知信息
步骤2：确定公式
步骤3：代入计算
步骤4：验证结果"""
        },
        {
            "title": "4. 约束输出格式",
            "bad": "介绍Python",
            "good": """请用JSON格式介绍Python：
{
  "name": "语言名称",
  "type": "语言类型",
  "features": ["特性1", "特性2"],
  "use_cases": ["应用场景1"]
}"""
        },
        {
            "title": "5. 添加上下文",
            "bad": "推荐算法",
            "good": """背景：我是C++工程师，要转型做AIGC推理优化。
目标：学习相关算法和技术。
限制：学习时间3个月。

基于以上背景，请推荐学习路径和重点算法。"""
        },
    ]

    for practice in practices:
        print(f"\n{practice['title']}")
        print("─" * 60)
        print("❌ 不好的示例:")
        print(f"   {practice['bad']}")
        print("\n✅ 好的示例:")
        print(f"   {practice['good']}")
        print()


def interactive_optimizer():
    """交互式Prompt优化器"""
    print("\n" + "=" * 60)
    print("交互式Prompt优化器")
    print("=" * 60)
    print("输入 'quit' 退出\n")

    library = PromptTemplateLibrary()
    builder = PromptBuilder(library)

    # 显示可用策略
    print("可用策略:")
    for i, template in enumerate(library.list_templates(), 1):
        print(f"  {i}. {template.name} - {template.description}")

    while True:
        print("\n" + "─" * 60)
        question = input("请输入问题 > ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            break

        if not question:
            continue

        print("\n选择策略（输入数字1-7，或直接回车使用基础策略）:")
        choice = input("> ").strip()

        if not choice:
            strategy = PromptStrategy.BASIC
        else:
            try:
                idx = int(choice) - 1
                templates = library.list_templates()
                if 0 <= idx < len(templates):
                    strategy = templates[idx].strategy
                else:
                    print("无效选择，使用基础策略")
                    strategy = PromptStrategy.BASIC
            except ValueError:
                print("无效选择，使用基础策略")
                strategy = PromptStrategy.BASIC

        # 构建prompt
        params = {'question': question}

        if strategy == PromptStrategy.ROLE_PLAY:
            role = input("请输入角色（如：Python专家）> ").strip()
            params['role'] = role if role else "专家"

        prompt = builder.build(strategy, **params)

        print("\n生成的Prompt:")
        print("─" * 60)
        print(prompt)
        print("─" * 60)

        # 评估prompt
        scores = PromptEvaluator.evaluate(prompt, prompt)  # 简化评估
        print(f"\nPrompt评分: {scores['prompt_clarity']:.2f}/1.0")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════╗
║          Prompt优化工具 - 自动化测试与优化             ║
║                                                        ║
║  核心功能:                                             ║
║    • 7种Prompt策略模板                                 ║
║    • A/B测试对比                                       ║
║    • 自动评估打分                                      ║
║    • 最佳实践展示                                      ║
║                                                        ║
║  适合场景:                                             ║
║    • 学习Prompt工程                                    ║
║    • 优化LLM应用                                       ║
║    • 面试项目展示                                      ║
╚════════════════════════════════════════════════════════╝
    """)

    # 1. 演示不同策略
    demo_prompt_strategies()

    # 2. A/B测试
    demo_ab_testing()

    # 3. 最佳实践
    demo_best_practices()

    # 4. 交互模式（可选）
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_optimizer()

    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    print("1. 根据任务选择合适的策略")
    print("2. 使用A/B测试找到最优prompt")
    print("3. 迭代优化，记录最佳实践")
    print("4. 设置OPENAI_API_KEY进行真实测试")
    print("\n运行 python prompt_optimizer.py --interactive 进入交互模式")
