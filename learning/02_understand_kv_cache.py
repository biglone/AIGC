"""
第二个实践：理解为什么需要KV Cache优化

这个脚本模拟：
1. 没有KV Cache时的重复计算
2. 有KV Cache时的优化效果
3. 为什么能加速20倍

不需要API key，纯本地演示
"""

import time
import numpy as np


def simulate_attention_without_cache(seq_len):
    """模拟没有KV Cache的注意力计算"""
    print(f"\n{'='*60}")
    print(f"模拟：生成 {seq_len} 个词（没有KV Cache）")
    print(f"{'='*60}")

    total_ops = 0
    d = 64  # 简化的维度

    print("\n每一步的计算量：")
    for i in range(1, seq_len + 1):
        # 每次都要重新计算所有之前的词
        ops = i * i * d  # O(n^2 * d)
        total_ops += ops
        if i <= 5 or i % 10 == 0 or i == seq_len:
            print(f"  生成第 {i:2d} 个词：需要计算 {i} x {i} = {i*i:4d} 次注意力 "
                  f"→ {ops:8d} 次运算")

    print(f"\n总计算量：{total_ops:,} 次运算")
    return total_ops


def simulate_attention_with_cache(seq_len):
    """模拟有KV Cache的注意力计算"""
    print(f"\n{'='*60}")
    print(f"模拟：生成 {seq_len} 个词（有KV Cache）")
    print(f"{'='*60}")

    total_ops = 0
    d = 64

    print("\n每一步的计算量：")
    for i in range(1, seq_len + 1):
        # 只需要计算新词与之前所有词的注意力
        ops = i * d  # O(n * d)
        total_ops += ops
        if i <= 5 or i % 10 == 0 or i == seq_len:
            print(f"  生成第 {i:2d} 个词：只需计算 {i} x 1 = {i:4d} 次注意力 "
                  f"→ {ops:8d} 次运算")

    print(f"\n总计算量：{total_ops:,} 次运算")
    return total_ops


def real_benchmark():
    """实际的性能对比（模拟）"""
    print(f"\n{'='*60}")
    print("实际性能测试：模拟矩阵运算")
    print(f"{'='*60}")

    seq_len = 100
    d_model = 512

    # 模拟没有cache的情况
    print("\n测试1：没有KV Cache")
    start = time.time()
    total_time_no_cache = 0
    for i in range(1, seq_len + 1):
        # 每次都重新计算
        Q = np.random.randn(i, d_model)
        K = np.random.randn(i, d_model)
        V = np.random.randn(i, d_model)

        step_start = time.time()
        scores = Q @ K.T  # (i, i)
        attention = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
        output = attention @ V
        step_time = time.time() - step_start
        total_time_no_cache += step_time

    time_no_cache = time.time() - start

    # 模拟有cache的情况
    print("\n测试2：有KV Cache")
    start = time.time()
    total_time_with_cache = 0

    # 预先分配cache
    K_cache = np.zeros((seq_len, d_model))
    V_cache = np.zeros((seq_len, d_model))

    for i in range(1, seq_len + 1):
        # 只计算新词
        Q_new = np.random.randn(1, d_model)
        K_new = np.random.randn(1, d_model)
        V_new = np.random.randn(1, d_model)

        step_start = time.time()
        # 更新cache
        K_cache[i-1] = K_new
        V_cache[i-1] = V_new

        # 只需要新Q与所有K的注意力
        scores = Q_new @ K_cache[:i].T  # (1, i)
        attention = np.exp(scores) / np.exp(scores).sum()
        output = attention @ V_cache[:i]
        step_time = time.time() - step_start
        total_time_with_cache += step_time

    time_with_cache = time.time() - start

    print(f"\n结果对比：")
    print(f"  没有KV Cache: {time_no_cache:.3f} 秒")
    print(f"  有KV Cache:   {time_with_cache:.3f} 秒")
    print(f"  加速比:       {time_no_cache/time_with_cache:.1f}x")

    return time_no_cache, time_with_cache


def visualize_memory_usage():
    """可视化内存使用"""
    print(f"\n{'='*60}")
    print("KV Cache的内存使用分析")
    print(f"{'='*60}")

    # Llama-2-7B的参数
    n_layers = 32
    n_heads = 32
    d_head = 128
    batch_size = 1
    seq_len = 2048

    # 每个token的KV cache大小
    kv_size_per_token = n_layers * 2 * n_heads * d_head * 2  # 2 for K and V, 2 bytes for FP16

    print(f"\n模型配置：Llama-2-7B")
    print(f"  - 层数: {n_layers}")
    print(f"  - 注意力头数: {n_heads}")
    print(f"  - 每个头的维度: {d_head}")
    print(f"  - 序列长度: {seq_len}")

    print(f"\nKV Cache内存占用：")
    print(f"  - 每个token: {kv_size_per_token / 1024:.2f} KB")
    print(f"  - {seq_len}个token: {kv_size_per_token * seq_len / 1024 / 1024:.2f} MB")

    print(f"\n💡 观察：")
    print(f"   - KV Cache用内存换时间")
    print(f"   - 序列越长，内存占用越大")
    print(f"   - 这就是为什么长文本对话很贵！")


def interactive_demo():
    """交互式演示"""
    print(f"\n{'='*60}")
    print("交互式演示：体验计算量差异")
    print(f"{'='*60}")

    while True:
        try:
            seq_len = input("\n请输入要生成的词数（建议10-100，输入0退出）: ")
            seq_len = int(seq_len)

            if seq_len == 0:
                break

            if seq_len < 1 or seq_len > 1000:
                print("请输入1-1000之间的数字")
                continue

            # 计算两种方式的计算量
            ops_no_cache = simulate_attention_without_cache(seq_len)
            ops_with_cache = simulate_attention_with_cache(seq_len)

            # 对比
            print(f"\n{'='*60}")
            print("📊 对比结果")
            print(f"{'='*60}")
            print(f"没有KV Cache: {ops_no_cache:,} 次运算")
            print(f"有KV Cache:   {ops_with_cache:,} 次运算")
            print(f"加速比:       {ops_no_cache/ops_with_cache:.1f}x")
            print(f"节省计算:     {(1 - ops_with_cache/ops_no_cache)*100:.1f}%")

        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            break


def main():
    """主函数"""
    print("\n" + "🎓 理解KV Cache优化 🎓".center(60))

    print("\n这个教程将帮助您理解：")
    print("1. 为什么自回归生成很慢")
    print("2. KV Cache如何优化")
    print("3. 为什么能加速20倍")
    print("4. 内存换时间的权衡\n")

    try:
        # 演示1：理论分析
        print("\n" + "第1部分：理论分析".center(60, "-"))
        ops_no_cache = simulate_attention_without_cache(20)
        ops_with_cache = simulate_attention_with_cache(20)

        print(f"\n💡 核心洞察：")
        print(f"   没有cache: 计算量 = 1+4+9+16+...+n² = O(n³)")
        print(f"   有cache:   计算量 = 1+2+3+4+...+n  = O(n²)")
        print(f"   加速比: {ops_no_cache/ops_with_cache:.1f}x")

        input("\n按回车继续...")

        # 演示2：实际性能
        print("\n" + "第2部分：实际性能测试".center(60, "-"))
        real_benchmark()

        input("\n按回车继续...")

        # 演示3：内存分析
        print("\n" + "第3部分：内存使用分析".center(60, "-"))
        visualize_memory_usage()

        input("\n按回车继续...")

        # 演示4：交互式
        print("\n" + "第4部分：交互式体验".center(60, "-"))
        interactive_demo()

        print("\n" + "=" * 60)
        print("🎉 恭喜！您已经理解了KV Cache优化")
        print("=" * 60)
        print("\n您现在知道了：")
        print("✅ 为什么自回归生成需要优化")
        print("✅ KV Cache如何避免重复计算")
        print("✅ 为什么能加速20倍")
        print("✅ 内存和速度的权衡")
        print("\n下一步：查看 llm-inference-engine 项目的实际C++实现")

    except KeyboardInterrupt:
        print("\n\n程序已退出")


if __name__ == "__main__":
    main()
