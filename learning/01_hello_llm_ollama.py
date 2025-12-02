"""
第一个LLM实践：体验自回归生成（Ollama本地版本）

这个脚本展示：
1. LLM如何一个词一个词地生成文本（自回归）
2. 为什么生成长文本需要时间
3. Temperature参数如何影响输出

使用本地Ollama，不需要API key！

安装要求：
1. 安装Ollama: curl -fsSL https://ollama.com/install.sh | sh
2. 下载模型: ollama pull llama3.2:3b
"""

import requests
import json
import sys
import time


def check_ollama():
    """检查Ollama是否运行，并返回可用模型列表"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            if not models:
                print("❌ Ollama运行中，但没有安装模型")
                print("\n推荐安装以下模型之一：")
                print("  ollama pull qwen2.5:7b        # 最推荐：性能强，中文好")
                print("  ollama pull llama3.1:8b       # 通用模型")
                print("  ollama pull llama3.2:3b       # 轻量快速")
                return None

            print(f"✅ Ollama运行中，已安装 {len(models)} 个模型")
            return models
    except:
        print("❌ Ollama未运行")
        print("请先启动Ollama服务")
        print("安装: curl -fsSL https://ollama.com/install.sh | sh")
        return None


def select_model(models):
    """让用户选择要使用的模型"""
    print(f"\n{'='*60}")
    print("📋 可用模型列表")
    print(f"{'='*60}")

    for i, model in enumerate(models, 1):
        name = model['name']
        size_gb = model.get('size', 0) / (1024**3)
        modified = model.get('modified_at', '')[:10]
        print(f"  {i}. {name:35s} ({size_gb:5.1f} GB) - {modified}")

    print(f"\n推荐模型：")
    print(f"  - qwen2.5:7b/14b  : 中文能力强，性能好")
    print(f"  - llama3.1:8b     : 通用性好")
    print(f"  - deepseek-coder  : 编程专用")

    while True:
        try:
            choice = input(f"\n请选择模型编号 (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]['name']
                print(f"\n✅ 已选择: {selected}")
                return selected
            else:
                print(f"请输入 1-{len(models)} 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消")
            sys.exit(0)


def ollama_generate(prompt, model="llama3.2:3b", temperature=0.7, stream=True, max_tokens=50):
    """调用Ollama API生成文本"""
    url = "http://localhost:11434/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    response = requests.post(url, json=data, stream=stream)

    if stream:
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'response' in chunk:
                    yield chunk['response']
    else:
        result = response.json()
        return result.get('response', '')


def demo1_basic_generation(model):
    """演示1：基础文本生成"""
    print("=" * 60)
    print("演示1：基础文本生成")
    print("=" * 60)

    prompt = "人工智能的未来"
    print(f"\n输入: {prompt}")
    print(f"输出: ", end="", flush=True)

    # 流式输出，看到逐个词的生成
    for token in ollama_generate(prompt, model=model, stream=True, max_tokens=50):
        print(token, end="", flush=True)

    print("\n\n💡 观察：您看到了吗？文字是一个一个蹦出来的！")
    print("   这就是'自回归生成' - LLM每次只生成一个词")


def demo2_temperature_effect(model):
    """演示2：Temperature参数的影响"""
    print("\n" + "=" * 60)
    print("演示2：Temperature参数如何影响输出")
    print("=" * 60)

    prompt = "今天天气真"

    print(f"\n输入: {prompt}")

    # 测试不同的temperature
    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        print("输出: ", end="", flush=True)

        for token in ollama_generate(prompt, model=model, temperature=temp, stream=True, max_tokens=20):
            print(token, end="", flush=True)
        print()

    print("\n💡 观察：")
    print("   - Temperature = 0.0: 每次输出都一样（确定性）")
    print("   - Temperature = 0.5: 比较保守，合理")
    print("   - Temperature = 1.0: 标准，有一定创造性")
    print("   - Temperature = 1.5: 很随机，可能不太合理")


def demo3_attention_visualization(model):
    """演示3：理解'注意力' - 通过任务展示"""
    print("\n" + "=" * 60)
    print("演示3：理解'注意力'机制")
    print("=" * 60)

    # 测试用例：需要"回头看"才能正确回答
    test_cases = [
        {
            "prompt": "小明今天去超市买了苹果。他很喜欢吃水果。问：谁喜欢吃水果？请简短回答。",
            "explanation": "模型需要注意到'他'指的是'小明'"
        },
        {
            "prompt": "猫坐在垫子上。它是黑色的。问：什么是黑色的？请简短回答。",
            "explanation": "需要判断'它'指的是'猫'还是'垫子'"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"输入: {case['prompt']}")
        print(f"输出: ", end="", flush=True)

        for token in ollama_generate(case['prompt'], model=model, temperature=0, stream=True, max_tokens=30):
            print(token, end="", flush=True)

        print(f"\n💡 这里用到了'注意力': {case['explanation']}")


def demo4_generation_speed(model):
    """演示4：观察生成速度"""
    print("\n" + "=" * 60)
    print("演示4：观察生成速度（每个token的时间）")
    print("=" * 60)

    prompt = "请写一首关于人工智能的短诗"
    print(f"\n输入: {prompt}")
    print(f"\n开始生成...\n")

    token_times = []
    last_time = time.time()
    token_count = 0

    print("输出: ", end="", flush=True)
    for token in ollama_generate(prompt, model=model, stream=True, max_tokens=100):
        current_time = time.time()
        token_times.append(current_time - last_time)
        last_time = current_time
        token_count += 1

        print(token, end="", flush=True)

    avg_time = sum(token_times) / len(token_times) if token_times else 0
    tokens_per_sec = 1 / avg_time if avg_time > 0 else 0

    print(f"\n\n📊 性能统计：")
    print(f"   模型: {model}")
    print(f"   生成token数: {token_count}")
    print(f"   平均每token: {avg_time*1000:.1f} ms")
    print(f"   生成速度:   {tokens_per_sec:.1f} tokens/秒")
    print(f"\n💡 观察：这就是为什么长文本需要时间！")
    print(f"   生成1000个词大约需要 {1000/tokens_per_sec:.1f} 秒")


def demo5_model_comparison():
    """演示5：对比不同模型（如果有多个）"""
    print("\n" + "=" * 60)
    print("演示5：Jetson Thor性能展示")
    print("=" * 60)

    # 检查可用模型
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get('models', [])

        if len(models) > 1:
            print(f"\n您安装了 {len(models)} 个模型，对比一下：")
            prompt = "什么是人工智能？用一句话回答。"

            for model_info in models[:2]:  # 只测试前2个
                model_name = model_info['name']
                print(f"\n--- 模型: {model_name} ---")
                print("输出: ", end="", flush=True)

                start = time.time()
                for token in ollama_generate(prompt, model=model_name, stream=True, max_tokens=50):
                    print(token, end="", flush=True)
                elapsed = time.time() - start

                print(f"\n生成时间: {elapsed:.2f}秒")
        else:
            print(f"\n当前只有1个模型: {models[0]['name']}")
            print("您可以下载更多模型对比：")
            print("  ollama pull qwen2.5:7b")
            print("  ollama pull llama3.1:8b")

    except Exception as e:
        print(f"无法获取模型列表: {e}")


def main():
    """主函数"""
    print("\n" + "🎓 LLM零基础实践教程（Ollama本地版）🎓".center(60))
    print("\n这个教程包含5个演示，帮助您理解LLM的工作原理\n")
    print("✅ 完全本地运行，不需要API key")
    print("✅ 运行在您的Jetson Thor上")
    print()

    # 检查Ollama并获取模型列表
    models = check_ollama()
    if not models:
        return

    # 让用户选择模型
    selected_model = select_model(models)

    try:
        # 演示1：基础生成
        demo1_basic_generation(selected_model)
        input("\n按回车继续下一个演示...")

        # 演示2：Temperature
        demo2_temperature_effect(selected_model)
        input("\n按回车继续下一个演示...")

        # 演示3：注意力
        demo3_attention_visualization(selected_model)
        input("\n按回车继续下一个演示...")

        # 演示4：生成速度
        demo4_generation_speed(selected_model)
        input("\n按回车继续下一个演示...")

        # 演示5：模型对比
        demo5_model_comparison()

        print("\n" + "=" * 60)
        print("🎉 恭喜！您已经完成了第一个LLM实践（本地版）")
        print("=" * 60)
        print(f"\n使用模型: {selected_model}")
        print("\n您现在理解了：")
        print("✅ LLM如何逐词生成文本（自回归）")
        print("✅ Temperature参数的作用")
        print("✅ 什么是'注意力'机制")
        print("✅ 本地LLM的性能表现")
        print("\n下一步：运行 02_understand_kv_cache.py 理解为什么需要优化")

    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n提示：")
        print("1. 确保Ollama正在运行")
        print("2. 确保已下载模型: ollama pull llama3.2:3b")
        print("3. 检查网络和端口 11434 是否可用")


if __name__ == "__main__":
    main()
