# 配置模板库

本目录包含常用的配置文件模板，帮助快速启动AIGC项目。

## 📁 目录结构

```
templates/
├── docker/              # Docker相关配置
│   ├── Dockerfile.inference   # 推理服务Dockerfile
│   └── docker-compose.yml     # Docker Compose配置
├── kubernetes/          # Kubernetes配置
│   └── deployment.yaml        # K8s部署配置
├── cicd/               # CI/CD配置
│   └── github-actions.yml     # GitHub Actions工作流
└── training/           # 训练配置
    └── config.yaml            # 训练参数配置
```

## 🚀 使用方法

### 1. Docker部署

**单容器部署：**
```bash
# 复制Dockerfile
cp templates/docker/Dockerfile.inference ./Dockerfile

# 构建镜像
docker build -t my-llm-api:v1.0 .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  --gpus all \
  my-llm-api:v1.0
```

**Docker Compose部署：**
```bash
# 复制配置
cp templates/docker/docker-compose.yml ./

# 创建.env文件
cat > .env << EOF
OPENAI_API_KEY=your-key
POSTGRES_PASSWORD=your-password
GRAFANA_PASSWORD=admin
EOF

# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f llm-api
```

---

### 2. Kubernetes部署

```bash
# 复制配置
cp templates/kubernetes/deployment.yaml ./k8s/

# 创建secrets
kubectl create secret generic llm-secrets \
  --from-literal=openai-api-key=your-key

# 部署
kubectl apply -f k8s/deployment.yaml

# 查看状态
kubectl get pods -l app=llm-api
kubectl get svc llm-api-service

# 查看日志
kubectl logs -f deployment/llm-api
```

---

### 3. CI/CD配置

**GitHub Actions：**
```bash
# 复制到项目
mkdir -p .github/workflows
cp templates/cicd/github-actions.yml .github/workflows/ci.yml

# 配置Secrets（在GitHub仓库设置中）
# - DOCKER_USERNAME
# - DOCKER_PASSWORD
# - KUBE_CONFIG
# - SLACK_WEBHOOK
```

**触发流程：**
- 推送到main/develop分支 → 运行测试
- 创建Release → 构建镜像 + 部署

---

### 4. 训练配置

```bash
# 复制配置
cp templates/training/config.yaml ./configs/

# 修改配置
vim configs/config.yaml

# 使用配置训练
python train.py --config configs/config.yaml
```

---

## 📝 配置说明

### Dockerfile.inference

**关键配置：**
- 基础镜像：`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- Python版本：3.10
- 默认端口：8000
- 健康检查：每30秒检查/health端点

**自定义方法：**
1. 修改基础镜像版本
2. 调整worker数量
3. 添加环境变量
4. 修改启动命令

---

### docker-compose.yml

**包含服务：**
- llm-api：主API服务
- redis：缓存
- postgres：数据库
- prometheus：监控
- grafana：可视化

**端口映射：**
- 8000：API服务
- 6379：Redis
- 5432：PostgreSQL
- 9090：Prometheus
- 3000：Grafana

---

### deployment.yaml

**包含资源：**
- Deployment：应用部署
- Service：负载均衡
- HPA：自动扩缩容

**资源限制：**
- CPU：2-4核
- 内存：4-8GB
- GPU：1张

**扩缩容策略：**
- 最小副本：2
- 最大副本：10
- CPU阈值：70%
- 内存阈值：80%

---

### github-actions.yml

**工作流程：**
1. **测试（test）**
   - Lint检查
   - 格式检查
   - 单元测试
   - 覆盖率上传

2. **构建（build）**
   - Docker镜像构建
   - 推送到Docker Hub
   - 打标签

3. **部署（deploy）**
   - 部署到K8s
   - Slack通知

---

### config.yaml

**配置项：**
- **model**：模型选择和加载方式
- **lora**：LoRA参数配置
- **training**：训练超参数
- **data**：数据路径和预处理
- **wandb**：实验跟踪

**调参建议：**
- 小数据集：降低batch_size，增加epochs
- 大模型：启用gradient_checkpointing
- 快速验证：减少save_steps和eval_steps

---

## 🎯 最佳实践

### 1. 安全性

**敏感信息管理：**
```bash
# 使用环境变量
export OPENAI_API_KEY=your-key

# 使用.env文件（不要提交到Git）
echo ".env" >> .gitignore

# Kubernetes使用Secrets
kubectl create secret generic my-secret \
  --from-literal=api-key=your-key
```

### 2. 性能优化

**Docker：**
- 使用多阶段构建减小镜像体积
- 利用缓存加速构建
- 使用.dockerignore排除不需要的文件

**Kubernetes：**
- 合理设置资源requests和limits
- 使用HPA自动扩缩容
- 配置PDB（Pod Disruption Budget）保证可用性

### 3. 监控告警

**Prometheus指标：**
```python
# 在应用中暴露metrics
from prometheus_client import Counter, Histogram
from prometheus_client import make_asgi_app

# 添加到FastAPI
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**Grafana Dashboard：**
- 导入预置Dashboard
- 配置告警规则
- 设置通知渠道

---

## ❓ 常见问题

### Q1: GPU不可用？

**检查：**
```bash
# Docker
docker run --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Kubernetes
kubectl get nodes -o json | jq '.items[].status.allocatable'
```

### Q2: 端口冲突？

**解决：**
```bash
# 修改docker-compose.yml中的端口映射
ports:
  - "8001:8000"  # 使用8001而不是8000
```

### Q3: 构建慢？

**优化：**
- 使用国内镜像源
- 启用BuildKit
- 使用缓存层

---

## 📚 相关文档

- [Docker官方文档](https://docs.docker.com/)
- [Kubernetes官方文档](https://kubernetes.io/docs/)
- [GitHub Actions文档](https://docs.github.com/en/actions)
- [Prometheus文档](https://prometheus.io/docs/)

---

## 🔄 更新日志

- **2025-12-02**: 创建初始模板库
  - 添加Docker模板
  - 添加K8s模板
  - 添加CI/CD模板
  - 添加训练配置模板

---

**使用这些模板可以节省80%的配置时间，专注于业务逻辑开发！** 🚀
