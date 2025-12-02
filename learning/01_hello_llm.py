"""
第一个LLM实践：体验自回归生成

这个脚本展示：
1. LLM如何一个词一个词地生成文本（自回归）
2. 为什么生成长文本需要时间
3. Temperature参数如何影响输出

使用前请设置：
export OPENAI_API_KEY="your-key"
"""

from openai import OpenAI
import sys

def demo1_basic_generation():
    """演示1：基础文本生成"""
    print("=" * 60)
    print("演示1：基础文本生成")
    print("=" * 60)

    client = OpenAI()

    prompt = "人工智能的未来"
    print(f"\n输入: {prompt}")
    print(f"输出: ", end="", flush=True)

    # stream=True 让我们看到逐个词的生成过程
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # 关键参数：流式输出
        max_tokens=50
    )

    # 逐个打印生成的词
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n\n💡 观察：您看到了吗？文字是一个一个蹦出来的！")
    print("   这就是'自回归生成' - LLM每次只生成一个词")


def demo2_temperature_effect():
    """演示2：Temperature参数的影响"""
    print("\n" + "=" * 60)
    print("演示2：Temperature参数如何影响输出")
    print("=" * 60)

    client = OpenAI()
    prompt = "今天天气真"

    print(f"\n输入: {prompt}")

    # 测试不同的temperature
    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        print("输出: ", end="")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=20
        )

        print(response.choices[0].message.content)

    print("\n💡 观察：")
    print("   - Temperature = 0.0: 每次输出都一样（确定性）")
    print("   - Temperature = 0.5: 比较保守，合理")
    print("   - Temperature = 1.0: 标准，有一定创造性")
    print("   - Temperature = 1.5: 很随机，可能不太合理")


def demo3_attention_visualization():
    """演示3：理解'注意力' - 通过任务展示"""
    print("\n" + "=" * 60)
    print("演示3：理解'注意力'机制")
    print("=" * 60)

    client = OpenAI()

    # 测试用例：需要"回头看"才能正确回答
    test_cases = [
        {
            "prompt": "小明今天去超市买了苹果。他很喜欢吃水果。问：谁喜欢吃水果？",
            "explanation": "模型需要注意到'他'指的是'小明'"
        },
        {
            "prompt": "猫坐在垫子上。它是黑色的。问：什么是黑色的？",
            "explanation": "需要判断'它'指的是'猫'还是'垫子'"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"输入: {case['prompt']}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": case['prompt']}],
            temperature=0,
            max_tokens=20
        )

        print(f"输出: {response.choices[0].message.content}")
        print(f"💡 这里用到了'注意力': {case['explanation']}")


def demo4_count_tokens():
    """演示4：理解Token（词）的概念"""
    print("\n" + "=" * 60)
    print("演示4：什么是Token（词）？")
    print("=" * 60)

    import tiktoken

    # GPT使用的分词器
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    test_texts = [
        "Hello World",
        "你好世界",
        "人工智能",
        "Artificial Intelligence"
    ]

    for text in test_texts:
        tokens = encoding.encode(text)
        print(f"\n文本: {text}")
        print(f"Token数量: {len(tokens)}")
        print(f"Token列表: {tokens}")
        print(f"解码回来: {[encoding.decode([t]) for t in tokens]}")

    print("\n💡 观察：")
    print("   - 英文单词通常是1个token")
    print("   - 中文字符通常是1-2个token")
    print("   - Token是模型的'基本单位'，每次生成1个token")


def main():
    """主函数"""
    print("\n" + "🎓 LLM零基础实践教程 🎓".center(60))
    print("\n这个教程包含4个演示，帮助您理解LLM的工作原理\n")

    # 检查API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: 请先设置 OPENAI_API_KEY 环境变量")
        print("   export OPENAI_API_KEY='your-key-here'")
        return

    try:
        # 演示1：基础生成
        demo1_basic_generation()
        input("\n按回车继续下一个演示...")

        # 演示2：Temperature
        demo2_temperature_effect()
        input("\n按回车继续下一个演示...")

        # 演示3：注意力
        demo3_attention_visualization()
        input("\n按回车继续下一个演示...")

        # 演示4：Token
        demo4_count_tokens()

        print("\n" + "=" * 60)
        print("🎉 恭喜！您已经完成了第一个LLM实践")
        print("=" * 60)
        print("\n您现在理解了：")
        print("✅ LLM如何逐词生成文本（自回归）")
        print("✅ Temperature参数的作用")
        print("✅ 什么是'注意力'机制")
        print("✅ 什么是Token（词）")
        print("\n下一步：运行 02_understand_kv_cache.py 理解为什么需要优化")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示：")
        print("1. 确保已安装: pip install openai tiktoken")
        print("2. 确保API key正确")
        print("3. 确保网络连接正常")


if __name__ == "__main__":
    main()
