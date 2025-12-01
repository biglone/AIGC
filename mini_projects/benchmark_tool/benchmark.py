#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
文本生成性能基准测试工具
=======================

项目目标：
- 测量LLM推理性能指标
- 对比不同模型/配置
- 生成详细性能报告

核心指标：
1. Time to First Token (TTFT) - 首token延迟
2. Tokens Per Second (TPS) - 吞吐量
3. Total Latency - 总延迟
4. Memory Usage - 内存占用
5. Cost - 成本

作者：面向C++工程师的AIGC学习
"""

import time
import psutil
import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics
import matplotlib.pyplot as plt


# ============================================================
# 第一部分：性能指标数据类
# ============================================================

@dataclass
class BenchmarkMetrics:
    """单次推理的性能指标"""
    # 延迟指标
    ttft: float  # Time to First Token (秒)
    total_time: float  # 总时间 (秒)
    tps: float  # Tokens Per Second

    # Token统计
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # 资源使用
    memory_used_mb: float  # 内存占用 (MB)
    cpu_percent: float  # CPU使用率

    # 成本（如果适用）
    cost_usd: float = 0.0

    def __str__(self):
        return f"""
性能指标:
  TTFT (首token延迟):  {self.ttft*1000:.2f} ms
  总延迟:              {self.total_time:.3f} s
  TPS (吞吐量):        {self.tps:.2f} tokens/s
  Token统计:           {self.input_tokens} → {self.output_tokens} (总: {self.total_tokens})
  内存占用:            {self.memory_used_mb:.2f} MB
  CPU使用率:           {self.cpu_percent:.1f}%
  成本:                ${self.cost_usd:.6f}
"""


@dataclass
class BenchmarkReport:
    """基准测试报告（多次运行的统计）"""
    model_name: str
    num_runs: int

    # 延迟统计
    ttft_mean: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float

    tps_mean: float
    tps_min: float
    tps_max: float

    total_time_mean: float

    # Token统计
    avg_input_tokens: float
    avg_output_tokens: float

    # 资源统计
    avg_memory_mb: float
    avg_cpu_percent: float

    # 成本统计
    total_cost_usd: float

    def __str__(self):
        return f"""
基准测试报告
{'=' * 60}
模型: {self.model_name}
运行次数: {self.num_runs}

延迟指标:
  TTFT平均:    {self.ttft_mean*1000:.2f} ms
  TTFT P50:    {self.ttft_p50*1000:.2f} ms
  TTFT P95:    {self.ttft_p95*1000:.2f} ms
  TTFT P99:    {self.ttft_p99*1000:.2f} ms
  总延迟平均:  {self.total_time_mean:.3f} s

吞吐量:
  TPS平均:     {self.tps_mean:.2f} tokens/s
  TPS最小:     {self.tps_min:.2f} tokens/s
  TPS最大:     {self.tps_max:.2f} tokens/s

Token统计:
  平均输入:    {self.avg_input_tokens:.1f} tokens
  平均输出:    {self.avg_output_tokens:.1f} tokens

资源使用:
  平均内存:    {self.avg_memory_mb:.2f} MB
  平均CPU:     {self.avg_cpu_percent:.1f}%

成本:
  总成本:      ${self.total_cost_usd:.6f}
  平均单次:    ${self.total_cost_usd/self.num_runs:.6f}
{'=' * 60}
"""


# ============================================================
# 第二部分：基准测试引擎
# ============================================================

class BenchmarkEngine:
    """
    基准测试引擎

    类比C++：类似于性能测试框架，测量各种性能指标
    """

    def __init__(self, use_openai: bool = False, model: str = "gpt-3.5-turbo"):
        """
        Args:
            use_openai: 是否使用OpenAI API（False使用模拟）
            model: 模型名称
        """
        self.use_openai = use_openai
        self.model = model
        self.client = None

        if use_openai:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("警告：未安装openai库，将使用模拟测试")
                self.use_openai = False

        # 记录所有运行的指标
        self.metrics_history: List[BenchmarkMetrics] = []

    def run_single(self, prompt: str, max_tokens: int = 100) -> BenchmarkMetrics:
        """
        运行单次基准测试

        Args:
            prompt: 输入prompt
            max_tokens: 最大生成token数

        Returns:
            性能指标
        """
        if self.use_openai:
            return self._benchmark_openai(prompt, max_tokens)
        else:
            return self._benchmark_mock(prompt, max_tokens)

    def _benchmark_openai(self, prompt: str, max_tokens: int) -> BenchmarkMetrics:
        """使用OpenAI API进行基准测试"""
        # 记录初始内存和CPU
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 开始计时
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=True  # 使用流式输出以测量TTFT
            )

            # 接收第一个token
            ttft = None
            full_response = ""

            for i, chunk in enumerate(response):
                if i == 0:
                    ttft = time.time() - start_time

                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    full_response += delta.content

            # 总时间
            total_time = time.time() - start_time

            # 记录最终内存和CPU
            final_memory = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()

            # Token统计（近似）
            input_tokens = len(prompt.split()) * 1.3  # 粗略估算
            output_tokens = len(full_response.split()) * 1.3
            total_tokens = input_tokens + output_tokens

            # 计算TPS
            tps = output_tokens / total_time if total_time > 0 else 0

            # 计算成本（GPT-3.5-turbo）
            cost = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000

            metrics = BenchmarkMetrics(
                ttft=ttft if ttft else total_time,
                total_time=total_time,
                tps=tps,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(total_tokens),
                memory_used_mb=final_memory - initial_memory,
                cpu_percent=cpu_percent,
                cost_usd=cost
            )

        except Exception as e:
            print(f"错误: {e}")
            # 返回失败指标
            metrics = BenchmarkMetrics(
                ttft=0, total_time=0, tps=0,
                input_tokens=0, output_tokens=0, total_tokens=0,
                memory_used_mb=0, cpu_percent=0, cost_usd=0
            )

        self.metrics_history.append(metrics)
        return metrics

    def _benchmark_mock(self, prompt: str, max_tokens: int) -> BenchmarkMetrics:
        """模拟基准测试（用于演示）"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # 模拟TTFT（首token延迟）
        import random
        ttft = random.uniform(0.05, 0.15)  # 50-150ms
        time.sleep(ttft)

        # 模拟逐token生成
        input_tokens = int(len(prompt.split()) * 1.3)
        output_tokens = min(max_tokens, random.randint(50, 100))

        # 模拟生成延迟（每个token约20-50ms）
        token_latency = random.uniform(0.02, 0.05)
        generation_time = output_tokens * token_latency
        time.sleep(min(generation_time, 0.5))  # 最多等待0.5s

        total_time = ttft + generation_time
        tps = output_tokens / total_time if total_time > 0 else 0

        final_memory = process.memory_info().rss / 1024 / 1024
        cpu_percent = random.uniform(10, 50)

        # 估算成本
        cost = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000

        metrics = BenchmarkMetrics(
            ttft=ttft,
            total_time=total_time,
            tps=tps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            memory_used_mb=final_memory - initial_memory,
            cpu_percent=cpu_percent,
            cost_usd=cost
        )

        self.metrics_history.append(metrics)
        return metrics

    def run_benchmark(self, prompts: List[str], num_runs: int = 5,
                     max_tokens: int = 100) -> BenchmarkReport:
        """
        运行完整基准测试

        Args:
            prompts: 测试prompt列表
            num_runs: 每个prompt运行次数
            max_tokens: 最大生成token数

        Returns:
            基准测试报告
        """
        print(f"开始基准测试...")
        print(f"  模型: {self.model}")
        print(f"  Prompts: {len(prompts)}个")
        print(f"  每个运行: {num_runs}次")
        print(f"  总运行次数: {len(prompts) * num_runs}")
        print("=" * 60)

        all_metrics = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}/{len(prompts)}")
            print(f"  内容: {prompt[:50]}...")

            for run in range(num_runs):
                print(f"  运行 {run+1}/{num_runs}...", end=' ')
                metrics = self.run_single(prompt, max_tokens)
                all_metrics.append(metrics)
                print(f"✓ TTFT={metrics.ttft*1000:.0f}ms, TPS={metrics.tps:.1f}")

        # 生成报告
        report = self._generate_report(all_metrics)

        return report

    def _generate_report(self, metrics_list: List[BenchmarkMetrics]) -> BenchmarkReport:
        """生成统计报告"""
        ttfts = [m.ttft for m in metrics_list]
        tpss = [m.tps for m in metrics_list]
        total_times = [m.total_time for m in metrics_list]

        # 计算百分位数
        ttfts_sorted = sorted(ttfts)
        n = len(ttfts_sorted)

        def percentile(data, p):
            idx = int(n * p / 100)
            return data[min(idx, n-1)]

        report = BenchmarkReport(
            model_name=self.model,
            num_runs=len(metrics_list),
            ttft_mean=statistics.mean(ttfts),
            ttft_p50=percentile(ttfts_sorted, 50),
            ttft_p95=percentile(ttfts_sorted, 95),
            ttft_p99=percentile(ttfts_sorted, 99),
            tps_mean=statistics.mean(tpss),
            tps_min=min(tpss),
            tps_max=max(tpss),
            total_time_mean=statistics.mean(total_times),
            avg_input_tokens=statistics.mean([m.input_tokens for m in metrics_list]),
            avg_output_tokens=statistics.mean([m.output_tokens for m in metrics_list]),
            avg_memory_mb=statistics.mean([m.memory_used_mb for m in metrics_list]),
            avg_cpu_percent=statistics.mean([m.cpu_percent for m in metrics_list]),
            total_cost_usd=sum([m.cost_usd for m in metrics_list])
        )

        return report

    def visualize_results(self, save_path: Optional[str] = None):
        """可视化基准测试结果"""
        if not self.metrics_history:
            print("没有可视化的数据")
            return

        metrics = self.metrics_history

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. TTFT分布
        ax1 = axes[0, 0]
        ttfts = [m.ttft * 1000 for m in metrics]  # 转换为ms
        ax1.hist(ttfts, bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('TTFT (ms)')
        ax1.set_ylabel('频数')
        ax1.set_title(f'首Token延迟分布\n平均: {statistics.mean(ttfts):.1f}ms')
        ax1.grid(True, alpha=0.3)

        # 2. TPS分布
        ax2 = axes[0, 1]
        tpss = [m.tps for m in metrics]
        ax2.hist(tpss, bins=20, edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('TPS (tokens/s)')
        ax2.set_ylabel('频数')
        ax2.set_title(f'吞吐量分布\n平均: {statistics.mean(tpss):.1f} tokens/s')
        ax2.grid(True, alpha=0.3)

        # 3. 总延迟分布
        ax3 = axes[0, 2]
        total_times = [m.total_time for m in metrics]
        ax3.hist(total_times, bins=20, edgecolor='black', alpha=0.7, color='orange')
        ax3.set_xlabel('总延迟 (s)')
        ax3.set_ylabel('频数')
        ax3.set_title(f'总延迟分布\n平均: {statistics.mean(total_times):.3f}s')
        ax3.grid(True, alpha=0.3)

        # 4. TTFT时间序列
        ax4 = axes[1, 0]
        ax4.plot(range(len(ttfts)), ttfts, marker='o', markersize=3)
        ax4.set_xlabel('运行次数')
        ax4.set_ylabel('TTFT (ms)')
        ax4.set_title('TTFT时间序列')
        ax4.grid(True, alpha=0.3)

        # 5. TPS时间序列
        ax5 = axes[1, 1]
        ax5.plot(range(len(tpss)), tpss, marker='o', markersize=3, color='green')
        ax5.set_xlabel('运行次数')
        ax5.set_ylabel('TPS (tokens/s)')
        ax5.set_title('TPS时间序列')
        ax5.grid(True, alpha=0.3)

        # 6. 成本累积
        ax6 = axes[1, 2]
        costs = [m.cost_usd for m in metrics]
        cumulative_cost = [sum(costs[:i+1]) for i in range(len(costs))]
        ax6.plot(range(len(cumulative_cost)), cumulative_cost, marker='o', markersize=3, color='red')
        ax6.set_xlabel('运行次数')
        ax6.set_ylabel('累积成本 (USD)')
        ax6.set_title(f'成本累积\n总计: ${sum(costs):.6f}')
        ax6.grid(True, alpha=0.3)

        plt.suptitle(f'基准测试结果可视化 - {self.model}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

    def export_results(self, filepath: str):
        """导出结果到JSON"""
        data = {
            'model': self.model,
            'metrics': [asdict(m) for m in self.metrics_history]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"结果已导出到: {filepath}")


# ============================================================
# 第三部分：对比测试
# ============================================================

class ComparativeBenchmark:
    """对比基准测试"""

    @staticmethod
    def compare_models(models: List[str], prompts: List[str],
                      num_runs: int = 3) -> Dict[str, BenchmarkReport]:
        """
        对比多个模型

        Args:
            models: 模型名称列表
            prompts: 测试prompts
            num_runs: 运行次数

        Returns:
            每个模型的报告
        """
        results = {}

        for model in models:
            print(f"\n{'=' * 60}")
            print(f"测试模型: {model}")
            print('=' * 60)

            engine = BenchmarkEngine(use_openai=False, model=model)
            report = engine.run_benchmark(prompts, num_runs)
            results[model] = report

        return results

    @staticmethod
    def print_comparison(reports: Dict[str, BenchmarkReport]):
        """打印对比表格"""
        print("\n" + "=" * 80)
        print("模型对比")
        print("=" * 80)

        print(f"{'模型':<20} {'TTFT(ms)':<12} {'TPS':<12} {'总延迟(s)':<12} {'成本($)':<12}")
        print("-" * 80)

        for model, report in reports.items():
            print(f"{model:<20} "
                  f"{report.ttft_mean*1000:<12.2f} "
                  f"{report.tps_mean:<12.2f} "
                  f"{report.total_time_mean:<12.3f} "
                  f"{report.total_cost_usd:<12.6f}")

        print("=" * 80)


# ============================================================
# 第四部分：实战演示
# ============================================================

def demo_single_benchmark():
    """单次基准测试演示"""
    print("=" * 60)
    print("演示1：单次基准测试")
    print("=" * 60)

    engine = BenchmarkEngine(use_openai=False, model="gpt-3.5-turbo-mock")

    prompt = "请解释什么是Transformer架构，并说明其核心组件。"

    print(f"\nPrompt: {prompt}")
    print("\n开始测试...")

    metrics = engine.run_single(prompt, max_tokens=100)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(metrics)


def demo_full_benchmark():
    """完整基准测试演示"""
    print("\n" + "=" * 60)
    print("演示2：完整基准测试")
    print("=" * 60)

    engine = BenchmarkEngine(use_openai=False, model="gpt-3.5-turbo-mock")

    # 准备测试prompts
    prompts = [
        "什么是KV Cache？",
        "解释INT8量化原理",
        "如何优化推理性能？",
    ]

    # 运行基准测试
    report = engine.run_benchmark(prompts, num_runs=3, max_tokens=100)

    print("\n" + report.__str__())

    # 可视化
    print("\n生成可视化图表...")
    engine.visualize_results()


def demo_comparative_benchmark():
    """对比基准测试演示"""
    print("\n" + "=" * 60)
    print("演示3：模型对比测试")
    print("=" * 60)

    models = [
        "gpt-3.5-turbo-mock",
        "gpt-4-mock",
        "claude-3-mock"
    ]

    prompts = [
        "解释Transformer",
        "什么是RAG？",
    ]

    results = ComparativeBenchmark.compare_models(models, prompts, num_runs=2)

    ComparativeBenchmark.print_comparison(results)


def demo_custom_benchmark():
    """自定义基准测试"""
    print("\n" + "=" * 60)
    print("演示4：自定义基准测试场景")
    print("=" * 60)

    # 场景：测试不同长度的输入
    print("\n场景：测试不同prompt长度的影响")

    engine = BenchmarkEngine(use_openai=False, model="test-model")

    test_cases = [
        ("短prompt", "解释KV Cache", 50),
        ("中prompt", "详细解释什么是KV Cache，它如何优化推理性能，包括实现原理和性能提升数据。", 100),
        ("长prompt", "作为LLM推理优化专家，请详细解释KV Cache技术。包括：1) 基本原理 2) 实现细节 3) 性能提升数据 4) 实际应用案例 5) 与其他优化技术的对比。请提供代码示例。" * 2, 150),
    ]

    results = []
    for name, prompt, max_tok in test_cases:
        print(f"\n测试: {name}")
        print(f"  Prompt长度: {len(prompt)} 字符")

        metrics = engine.run_single(prompt, max_tokens=max_tok)
        results.append((name, metrics))

        print(f"  TTFT: {metrics.ttft*1000:.2f} ms")
        print(f"  TPS: {metrics.tps:.2f} tokens/s")

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'场景':<15} {'TTFT(ms)':<12} {'TPS':<15} {'总延迟(s)':<12}")
    print("-" * 60)
    for name, m in results:
        print(f"{name:<15} {m.ttft*1000:<12.2f} {m.tps:<15.2f} {m.total_time:<12.3f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════╗
║       文本生成性能基准测试工具                          ║
║                                                        ║
║  核心指标:                                             ║
║    • TTFT (首Token延迟)                                ║
║    • TPS (吞吐量)                                      ║
║    • Total Latency (总延迟)                            ║
║    • Memory & CPU (资源占用)                           ║
║    • Cost (成本)                                       ║
║                                                        ║
║  功能:                                                 ║
║    • 单次基准测试                                      ║
║    • 完整基准测试（多次运行统计）                      ║
║    • 模型对比测试                                      ║
║    • 可视化报告                                        ║
║                                                        ║
║  适合场景:                                             ║
║    • 性能测试                                          ║
║    • 模型选型                                          ║
║    • 优化效果验证                                      ║
╚════════════════════════════════════════════════════════╝
    """)

    # 1. 单次测试
    demo_single_benchmark()

    # 2. 完整测试
    demo_full_benchmark()

    # 3. 对比测试
    demo_comparative_benchmark()

    # 4. 自定义测试
    demo_custom_benchmark()

    print("\n" + "=" * 60)
    print("所有演示完成")
    print("=" * 60)

    print("\n使用建议:")
    print("1. 使用OPENAI_API_KEY进行真实测试")
    print("2. 关注TTFT（用户体验）和TPS（成本效率）")
    print("3. 多次运行取平均值，关注P95/P99延迟")
    print("4. 对比不同配置找到最优方案")
    print("5. 结合成本分析做决策")

    print("\n关键性能目标（参考）:")
    print("  TTFT < 200ms   - 优秀（用户无感知）")
    print("  TTFT < 500ms   - 良好")
    print("  TTFT > 1000ms  - 需要优化")
    print("  TPS > 50       - 良好吞吐量")
    print("  TPS > 100      - 优秀吞吐量")
