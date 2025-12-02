# 🚀 在Jetson Thor上使用Ollama运行本地LLM

> 完全本地运行，不需要API key，充分利用Jetson Thor的强大性能！

## 快速开始（3步）

### 1️⃣ 安装Ollama

```bash
# 自动安装脚本（支持ARM架构）
curl -fsSL https://ollama.com/install.sh | sh
```

安装完成后，Ollama会自动启动服务。

**验证安装：**
```bash
# 检查版本
ollama --version

# 检查服务状态
curl http://localhost:11434
```

如果看到"Ollama is running"，说明安装成功！

---

### 2️⃣ 下载模型

根据您的需求选择模型大小：

```bash
# 推荐：快速入门（约2GB）
ollama pull llama3.2:3b

# 进阶：中文更好（约4.7GB）
ollama pull qwen2.5:7b

# 高级：性能更强（约4.7GB）
ollama pull llama3.1:8b
```

**Jetson Thor配置建议：**
- 16GB内存：可运行7B-8B模型
- 32GB内存：可运行13B模型
- 建议先下载3B模型测试

**查看已下载的模型：**
```bash
ollama list
```

---

### 3️⃣ 测试运行

```bash
# 交互式对话
ollama run llama3.2:3b

# 测试中文
ollama run llama3.2:3b "你好，介绍一下你自己"

# 退出：输入 /bye
```

---

## 🎓 运行教程

现在可以运行本地版教程了：

```bash
cd /home/Biglone/workspace/AIGC/learning

# 激活虚拟环境
source venv/bin/activate

# 安装依赖（只需要requests）
pip install requests

# 运行Ollama版教程
python 01_hello_llm_ollama.py
```

---

## 📊 性能对比

**Jetson Thor运行不同模型的性能参考：**

| 模型 | 参数量 | 内存占用 | 生成速度 | 适用场景 |
|-----|--------|----------|----------|----------|
| llama3.2:3b | 3B | ~2GB | ~30 tokens/s | 学习、快速测试 |
| qwen2.5:7b | 7B | ~4.7GB | ~15 tokens/s | 中文任务 |
| llama3.1:8b | 8B | ~4.7GB | ~12 tokens/s | 复杂推理 |
| qwen2.5:14b | 14B | ~8.5GB | ~8 tokens/s | 高性能需求 |

*实际速度取决于Jetson Thor的具体配置*

---

## 💡 常用命令

```bash
# 列出所有模型
ollama list

# 删除模型
ollama rm llama3.2:3b

# 查看模型信息
ollama show llama3.2:3b

# 停止Ollama服务
sudo systemctl stop ollama

# 启动Ollama服务
sudo systemctl start ollama

# 查看服务状态
sudo systemctl status ollama
```

---

## 🔧 高级配置

### 优化GPU使用

Jetson Thor有强大的GPU，Ollama会自动使用。查看GPU使用情况：

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用jetson工具
jtop
```

### 自定义模型参数

创建自定义配置文件：

```bash
# 创建Modelfile
cat > CustomModel << EOF
FROM llama3.2:3b
PARAMETER temperature 0.8
PARAMETER num_ctx 4096
EOF

# 创建自定义模型
ollama create my-custom-model -f CustomModel

# 使用自定义模型
ollama run my-custom-model
```

---

## 🐛 常见问题

### Q1: Ollama启动失败？

**检查：**
```bash
# 查看日志
sudo journalctl -u ollama -f

# 手动启动（调试）
ollama serve
```

### Q2: 模型下载很慢？

**解决：**
- 使用国内镜像（如果有）
- 或者手动下载模型文件后导入

### Q3: 内存不足？

**解决：**
```bash
# 使用更小的模型
ollama pull llama3.2:1b

# 或使用量化版本
ollama pull llama3.2:3b-q4_0  # 4bit量化
```

### Q4: 端口被占用？

**修改默认端口：**
```bash
# 设置环境变量
export OLLAMA_HOST=0.0.0.0:11435

# 重启服务
sudo systemctl restart ollama
```

---

## 🎯 学习路径

完成Ollama设置后：

1. **今天（30分钟）：**
   ```bash
   python 01_hello_llm_ollama.py
   ```
   理解自回归生成、注意力机制

2. **明天（30分钟）：**
   ```bash
   python 02_understand_kv_cache.py
   ```
   理解为什么需要优化

3. **后天（1小时）：**
   - 重读 `docs/01_LLM_Fundamentals.md`
   - 现在您能理解更多内容了

---

## 📚 推荐模型

### 中文任务
- **qwen2.5:7b** - 阿里Qwen系列，中文能力强
- **glm4:9b** - 智谱GLM4，中英双语

### 编程任务
- **codellama:7b** - Meta的代码专用模型
- **deepseek-coder:6.7b** - DeepSeek代码模型

### 通用任务
- **llama3.1:8b** - Meta最新，性能均衡
- **mistral:7b** - Mistral AI，速度快

### 下载命令
```bash
ollama pull qwen2.5:7b
ollama pull codellama:7b
ollama pull llama3.1:8b
```

---

## 🔗 资源链接

- **Ollama官网：** https://ollama.com/
- **模型库：** https://ollama.com/library
- **GitHub：** https://github.com/ollama/ollama
- **文档：** https://github.com/ollama/ollama/blob/main/docs/api.md

---

## ✨ 优势

**使用Ollama的好处：**
- ✅ 完全免费，无使用限制
- ✅ 数据隐私，不发送到云端
- ✅ 低延迟，本地运行
- ✅ 离线可用，不依赖网络
- ✅ 充分利用Jetson Thor的GPU性能

---

**准备好了吗？现在就开始：**

```bash
# 一键安装并下载模型
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# 运行教程
python 01_hello_llm_ollama.py
```

🚀 享受本地LLM的强大能力吧！
