#!/bin/bash

# LLM零基础实践教程 - 环境设置脚本

echo "======================================"
echo "  LLM零基础实践教程 - 环境设置"
echo "======================================"
echo ""

# 检查Python版本
echo "1. 检查Python版本..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ 错误: 未找到Python3"
    echo "请先安装Python 3.8+"
    exit 1
fi

echo "✅ Python已安装"
echo ""

# 创建虚拟环境
echo "2. 创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ 错误: 创建虚拟环境失败"
        echo "请先安装: sudo apt install python3-venv"
        exit 1
    fi
    echo "✅ 虚拟环境创建成功"
else
    echo "✅ 虚拟环境已存在"
fi
echo ""

# 激活虚拟环境
echo "3. 激活虚拟环境..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ 错误: 激活虚拟环境失败"
    exit 1
fi

echo "✅ 虚拟环境已激活"
echo ""

# 安装依赖
echo "4. 安装Python依赖..."
pip install --upgrade pip
pip install openai tiktoken numpy

if [ $? -ne 0 ]; then
    echo "❌ 错误: 依赖安装失败"
    exit 1
fi

echo "✅ 依赖安装成功"
echo ""

# 检查API key
echo "5. 检查OpenAI API Key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: 未设置 OPENAI_API_KEY 环境变量"
    echo ""
    echo "如果您有OpenAI API key，请设置："
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "如果没有，可以："
    echo "  - 跳过第1个教程（01_hello_llm.py）"
    echo "  - 直接运行第2个教程（02_understand_kv_cache.py）"
else
    echo "✅ OPENAI_API_KEY 已设置"
fi

echo ""
echo "======================================"
echo "  环境设置完成！"
echo "======================================"
echo ""
echo "📝 重要提示："
echo ""
echo "每次运行教程前，请先激活虚拟环境："
echo "  source venv/bin/activate"
echo ""
echo "然后运行："
echo "  python 01_hello_llm.py           # 需要API key"
echo "  python 02_understand_kv_cache.py # 不需要API key"
echo ""
echo "退出虚拟环境："
echo "  deactivate"
echo ""
echo "详细说明请查看 README.md"
echo ""
