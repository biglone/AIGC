"""
测试Ollama是否正常工作
"""
import requests
import json

print("测试1：检查Ollama服务")
print("="*60)
try:
    response = requests.get("http://localhost:11434")
    print(f"✅ Ollama服务正常: {response.text}")
except Exception as e:
    print(f"❌ 错误: {e}")
    exit(1)

print("\n测试2：列出模型")
print("="*60)
try:
    response = requests.get("http://localhost:11434/api/tags")
    models = response.json().get('models', [])
    print(f"找到 {len(models)} 个模型:")
    for m in models:
        print(f"  - {m['name']}")
except Exception as e:
    print(f"❌ 错误: {e}")
    exit(1)

print("\n测试3：简单生成（非流式）")
print("="*60)
try:
    data = {
        "model": "qwen2.5:7b",
        "prompt": "你好，请介绍一下你自己。",
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=data)
    result = response.json()
    print(f"模型: {result.get('model', 'unknown')}")
    print(f"输出: {result.get('response', '')}")
    print(f"完成: {result.get('done', False)}")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n测试4：流式生成")
print("="*60)
try:
    data = {
        "model": "qwen2.5:7b",
        "prompt": "人工智能是什么？用一句话回答。",
        "stream": True
    }
    print("输出: ", end="", flush=True)
    response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get('response', '')
            print(token, end="", flush=True)
    print()
    print("✅ 流式生成正常")
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n诊断完成！")
