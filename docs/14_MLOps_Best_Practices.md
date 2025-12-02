# MLOps最佳实践：LLM生产化完整指南

> **文档定位：** 从模型开发到生产部署的完整工程实践
> **适用对象：** ML工程师、DevOps工程师、技术负责人
> **前置知识：** LLM基础、Python编程、Docker/Kubernetes基础

---

## 目录

1. [模型开发流程](#1-模型开发流程)
2. [CI/CD Pipeline](#2-cicd-pipeline)
3. [模型监控](#3-模型监控)
4. [成本优化](#4-成本优化)
5. [实战案例](#5-实战案例)
6. [工具链推荐](#6-工具链推荐)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. 模型开发流程

### 1.1 实验管理

#### 1.1.1 Weights & Biases (W&B)

W&B 是最流行的实验跟踪工具，提供可视化、超参数追踪、模型版本管理等功能。

**基础使用**

```python
import wandb
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化W&B
wandb.init(
    project="llm-finetuning",
    name="gpt2-lora-run-1",
    config={
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 8,
        "model": "gpt2",
        "lora_r": 8,
        "lora_alpha": 16
    }
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=100,
    report_to="wandb",  # 自动记录到W&B
)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# 记录自定义指标
wandb.log({
    "custom_metric": 0.95,
    "inference_time_ms": 150
})

wandb.finish()
```

**进阶功能：超参数搜索**

```python
import wandb

# 定义搜索空间
sweep_config = {
    'method': 'bayes',  # 贝叶斯优化
    'metric': {
        'name': 'eval/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-4
        },
        'lora_r': {
            'values': [4, 8, 16, 32]
        },
        'lora_alpha': {
            'values': [8, 16, 32]
        },
        'batch_size': {
            'values': [4, 8, 16]
        }
    }
}

# 创建sweep
sweep_id = wandb.sweep(sweep_config, project="llm-finetuning")

# 定义训练函数
def train():
    # W&B会自动注入超参数
    config = wandb.config

    # 使用config中的超参数训练
    model = setup_model(
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha
    )

    trainer = setup_trainer(
        model=model,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size
    )

    trainer.train()

# 运行sweep（会自动运行多次实验）
wandb.agent(sweep_id, function=train, count=20)
```

#### 1.1.2 MLflow

MLflow 提供实验追踪、模型注册、模型部署等完整功能。

```python
import mlflow
import mlflow.pytorch

# 启动实验
mlflow.set_experiment("llm-finetuning")

with mlflow.start_run(run_name="gpt2-lora-experiment"):
    # 记录参数
    mlflow.log_params({
        "learning_rate": 2e-5,
        "epochs": 3,
        "model": "gpt2",
        "lora_r": 8
    })

    # 训练模型
    model = train_model()

    # 记录指标
    mlflow.log_metrics({
        "train_loss": 0.45,
        "eval_loss": 0.52,
        "perplexity": 12.3
    })

    # 保存模型
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="gpt2-lora-v1"
    )

    # 记录artifacts（配置文件、图表等）
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("training_curve.png")
```

**模型注册与版本管理**

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# 注册模型
model_uri = "runs:/<run_id>/model"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name="llm-chatbot"
)

# 模型版本管理
client.transition_model_version_stage(
    name="llm-chatbot",
    version=1,
    stage="Production"  # Staging, Production, Archived
)

# 加载特定版本
model = mlflow.pyfunc.load_model(
    model_uri="models:/llm-chatbot/Production"
)
```

---

### 1.2 版本控制

#### 1.2.1 代码版本控制（Git）

**Git工作流最佳实践**

```bash
# 功能分支工作流
git checkout -b feature/add-lora-support
# 开发...
git add .
git commit -m "feat: 添加LoRA微调支持"
git push origin feature/add-lora-support
# 创建Pull Request

# 主分支保护规则（GitHub/GitLab）
# - 要求代码审查
# - 要求CI通过
# - 禁止直接push到main
```

**Pre-commit Hooks**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88']

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']
```

安装与使用：

```bash
pip install pre-commit
pre-commit install

# 每次commit时自动运行
git commit -m "fix: 修复推理bug"
```

#### 1.2.2 数据版本控制（DVC）

DVC (Data Version Control) 用于管理大型数据集和模型文件。

```bash
# 安装DVC
pip install dvc dvc-s3

# 初始化
dvc init

# 配置远程存储（S3）
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc remote modify myremote access_key_id YOUR_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET

# 追踪数据文件
dvc add data/train.jsonl
dvc add models/checkpoint-1000.pt

# 提交到Git
git add data/train.jsonl.dvc models/checkpoint-1000.pt.dvc .gitignore
git commit -m "Add training data and model checkpoint"

# 推送数据到远程
dvc push

# 拉取数据（在另一台机器）
git pull
dvc pull
```

**数据Pipeline管理**

```yaml
# dvc.yaml
stages:
  prepare_data:
    cmd: python prepare_data.py
    deps:
      - prepare_data.py
      - data/raw/
    outs:
      - data/processed/train.jsonl
      - data/processed/eval.jsonl

  train:
    cmd: python train.py --config config.yaml
    deps:
      - train.py
      - data/processed/train.jsonl
      - config.yaml
    params:
      - config.yaml:
          - learning_rate
          - epochs
    outs:
      - models/checkpoint.pt
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python evaluate.py
    deps:
      - evaluate.py
      - models/checkpoint.pt
      - data/processed/eval.jsonl
    metrics:
      - eval_metrics.json:
          cache: false
```

运行Pipeline：

```bash
# 运行整个pipeline
dvc repro

# 查看metrics
dvc metrics show

# 比较不同实验
dvc metrics diff
```

---

### 1.3 实验复现

#### 1.3.1 随机种子管理

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """设置所有随机种子以确保复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保CUDA操作确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在训练开始时调用
set_seed(42)
```

#### 1.3.2 环境隔离（Docker）

**Dockerfile示例**

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements
COPY requirements.txt .

# 安装依赖（固定版本）
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache

# 启动命令
CMD ["python3", "train.py"]
```

**requirements.txt（固定版本）**

```txt
torch==2.1.0
transformers==4.35.0
datasets==2.14.0
peft==0.6.0
accelerate==0.24.0
wandb==0.16.0
```

**构建与运行**

```bash
# 构建镜像
docker build -t llm-training:v1.0 .

# 运行训练
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  llm-training:v1.0
```

#### 1.3.3 配置文件管理（Hydra）

```python
# config.yaml
model:
  name: gpt2
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.1

training:
  learning_rate: 2e-5
  batch_size: 8
  epochs: 3
  warmup_steps: 100
  gradient_accumulation_steps: 4

data:
  train_file: data/train.jsonl
  eval_file: data/eval.jsonl
  max_length: 512

wandb:
  project: llm-finetuning
  entity: my-team
```

**使用Hydra加载配置**

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # 打印配置
    print(OmegaConf.to_yaml(cfg))

    # 访问配置
    learning_rate = cfg.training.learning_rate
    model_name = cfg.model.name

    # 训练逻辑
    train(cfg)

if __name__ == "__main__":
    main()
```

**命令行覆盖配置**

```bash
# 覆盖学习率
python train.py training.learning_rate=1e-5

# 覆盖多个参数
python train.py training.batch_size=16 model.lora_r=16

# 使用不同配置文件
python train.py --config-name=config_prod
```

---

## 2. CI/CD Pipeline

### 2.1 持续集成

#### 2.1.1 GitHub Actions示例

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8

    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --max-line-length=88 --statistics

    - name: Format check with black
      run: black . --check

    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  model-test:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Run model tests
      run: |
        python -m pytest tests/model_tests/ -v

    - name: Check model accuracy regression
      run: |
        python scripts/check_accuracy.py --threshold 0.85
```

#### 2.1.2 模型测试

**单元测试示例**

```python
# tests/test_model.py
import pytest
import torch
from src.model import load_model, generate_text

@pytest.fixture
def model():
    """加载测试模型"""
    return load_model("gpt2", device="cpu")

def test_model_output_shape(model):
    """测试输出形状"""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    outputs = model(input_ids)

    assert outputs.logits.shape == (1, 5, model.config.vocab_size)

def test_generation(model):
    """测试文本生成"""
    prompt = "Hello, world"
    output = generate_text(model, prompt, max_length=20)

    assert isinstance(output, str)
    assert len(output) > len(prompt)
    assert output.startswith(prompt)

def test_model_determinism(model):
    """测试确定性"""
    torch.manual_seed(42)
    output1 = generate_text(model, "Test", max_length=10)

    torch.manual_seed(42)
    output2 = generate_text(model, "Test", max_length=10)

    assert output1 == output2
```

**准确率回归测试**

```python
# tests/test_accuracy.py
import json
import pytest
from src.evaluate import evaluate_model

def test_accuracy_no_regression():
    """确保准确率不下降"""
    # 加载基准结果
    with open('baselines/accuracy.json', 'r') as f:
        baseline = json.load(f)

    # 评估当前模型
    current_accuracy = evaluate_model("models/checkpoint-latest")

    # 允许1%的波动
    assert current_accuracy >= baseline['accuracy'] - 0.01, \
        f"准确率下降: {baseline['accuracy']:.4f} -> {current_accuracy:.4f}"

def test_inference_latency():
    """测试推理延迟"""
    import time

    model = load_model("models/checkpoint-latest")
    prompt = "Test prompt"

    start = time.time()
    for _ in range(10):
        generate_text(model, prompt)
    elapsed = (time.time() - start) / 10

    # 要求平均延迟 < 100ms
    assert elapsed < 0.1, f"推理太慢: {elapsed*1000:.2f}ms"
```

---

### 2.2 持续部署

#### 2.2.1 自动化部署流程

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          myorg/llm-api:${{ github.ref_name }}
          myorg/llm-api:latest

    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v4
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml
        images: |
          myorg/llm-api:${{ github.ref_name }}
        kubeconfig: ${{ secrets.KUBE_CONFIG }}

    - name: Run smoke tests
      run: |
        python scripts/smoke_test.py --url https://api.example.com

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: '部署完成: ${{ github.ref_name }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

#### 2.2.2 蓝绿部署

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api-blue
  labels:
    app: llm-api
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-api
      version: blue
  template:
    metadata:
      labels:
        app: llm-api
        version: blue
    spec:
      containers:
      - name: llm-api
        image: myorg/llm-api:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api-green
  labels:
    app: llm-api
    version: green
spec:
  replicas: 0  # 初始为0，部署时切换
  # ... 同上
```

**切换脚本**

```bash
#!/bin/bash
# deploy.sh

# 部署新版本到green
kubectl set image deployment/llm-api-green llm-api=myorg/llm-api:v2.0
kubectl scale deployment/llm-api-green --replicas=3

# 等待green就绪
kubectl rollout status deployment/llm-api-green

# 运行smoke tests
python smoke_test.py --target green

if [ $? -eq 0 ]; then
    echo "Smoke tests passed, switching traffic to green"

    # 切换流量（更新Service selector）
    kubectl patch service llm-api -p '{"spec":{"selector":{"version":"green"}}}'

    # 缩容blue
    kubectl scale deployment/llm-api-blue --replicas=0

    echo "Deployment complete!"
else
    echo "Smoke tests failed, rolling back"
    kubectl scale deployment/llm-api-green --replicas=0
    exit 1
fi
```

#### 2.2.3 金丝雀发布

```yaml
# k8s/canary.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: llm-api
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  service:
    port: 8000
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
    webhooks:
    - name: load-test
      url: http://loadtester.test/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://llm-api-canary:8000/v1/chat"
```

---

## 3. 模型监控

### 3.1 性能监控

#### 3.1.1 Prometheus + Grafana

**导出Metrics（Python）**

```python
# src/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义metrics
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Number of active requests'
)

TOKEN_COUNT = Counter(
    'llm_tokens_total',
    'Total number of tokens generated',
    ['model']
)

GPU_MEMORY = Gauge(
    'llm_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

# Decorator for tracking
def track_request(endpoint):
    def decorator(func):
        def wrapper(*args, **kwargs):
            ACTIVE_REQUESTS.inc()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                status = 'success'
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time

                REQUEST_COUNT.labels(
                    method='POST',
                    endpoint=endpoint,
                    status=status
                ).inc()

                REQUEST_LATENCY.labels(
                    method='POST',
                    endpoint=endpoint
                ).observe(duration)

                ACTIVE_REQUESTS.dec()

        return wrapper
    return decorator

# FastAPI集成
from fastapi import FastAPI
from prometheus_client import make_asgi_app

app = FastAPI()

# 添加metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/v1/chat/completions")
@track_request("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    # 生成响应
    response = generate_response(request)

    # 记录token数
    TOKEN_COUNT.labels(model=request.model).inc(response.usage.total_tokens)

    # 记录GPU内存
    import torch
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.memory_allocated(i)
        GPU_MEMORY.labels(gpu_id=str(i)).set(memory)

    return response
```

**Prometheus配置**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm-api'
    static_configs:
      - targets: ['llm-api:8000']
    metrics_path: '/metrics'
```

**Grafana Dashboard示例（JSON）**

```json
{
  "dashboard": {
    "title": "LLM API Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(llm_requests_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(llm_requests_total{status=\"error\"}[5m]) / rate(llm_requests_total[5m])"
          }
        ],
        "type": "graph"
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "llm_gpu_memory_bytes"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

### 3.2 模型监控

#### 3.2.1 数据漂移检测

```python
import numpy as np
from scipy import stats
from typing import List, Dict

class DataDriftDetector:
    """数据漂移检测"""

    def __init__(self, reference_data: np.ndarray):
        """
        Args:
            reference_data: 参考数据（训练集或初始生产数据）
        """
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)

    def detect_drift_ks(self, current_data: np.ndarray, threshold: float = 0.05) -> Dict:
        """
        使用Kolmogorov-Smirnov检验检测漂移

        Args:
            current_data: 当前数据
            threshold: p-value阈值

        Returns:
            漂移报告
        """
        n_features = self.reference_data.shape[1]
        drifted_features = []

        for i in range(n_features):
            # KS检验
            statistic, pvalue = stats.ks_2samp(
                self.reference_data[:, i],
                current_data[:, i]
            )

            if pvalue < threshold:
                drifted_features.append({
                    'feature_idx': i,
                    'pvalue': pvalue,
                    'statistic': statistic
                })

        return {
            'has_drift': len(drifted_features) > 0,
            'n_drifted_features': len(drifted_features),
            'drifted_features': drifted_features
        }

    def detect_drift_psi(self, current_data: np.ndarray, threshold: float = 0.1) -> Dict:
        """
        使用Population Stability Index (PSI)检测漂移

        PSI < 0.1: 无显著变化
        0.1 <= PSI < 0.2: 轻微变化
        PSI >= 0.2: 显著变化
        """
        def calculate_psi(expected, actual, bins=10):
            # 分箱
            breakpoints = np.linspace(
                min(expected.min(), actual.min()),
                max(expected.max(), actual.max()),
                bins + 1
            )

            expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
            actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

            # 避免除零
            expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
            actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

            # 计算PSI
            psi = np.sum((actual_percents - expected_percents) *
                        np.log(actual_percents / expected_percents))

            return psi

        n_features = self.reference_data.shape[1]
        psi_values = []

        for i in range(n_features):
            psi = calculate_psi(
                self.reference_data[:, i],
                current_data[:, i]
            )
            psi_values.append(psi)

        max_psi = max(psi_values)

        return {
            'has_drift': max_psi >= threshold,
            'max_psi': max_psi,
            'severity': 'high' if max_psi >= 0.2 else 'medium' if max_psi >= 0.1 else 'low',
            'psi_values': psi_values
        }

# 使用示例
"""
# 参考数据（训练集embeddings）
reference_embeddings = get_train_embeddings()
detector = DataDriftDetector(reference_embeddings)

# 定期检测（每天）
current_embeddings = get_production_embeddings(last_24h=True)
drift_report = detector.detect_drift_psi(current_embeddings)

if drift_report['has_drift']:
    alert("数据漂移检测到！", severity=drift_report['severity'])
"""
```

#### 3.2.2 输出质量监控

```python
from collections import defaultdict
import re

class OutputQualityMonitor:
    """输出质量监控"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def check_hallucination_indicators(self, response: str) -> Dict:
        """检测幻觉指标"""
        indicators = {
            'has_uncertainty': False,
            'has_made_up_facts': False,
            'confidence_score': 0.0
        }

        # 不确定性短语
        uncertainty_phrases = [
            "I'm not sure",
            "I don't know",
            "might be",
            "could be",
            "possibly"
        ]

        for phrase in uncertainty_phrases:
            if phrase.lower() in response.lower():
                indicators['has_uncertainty'] = True
                break

        # 可疑的具体数字（过于精确的统计）
        specific_numbers = re.findall(r'\d+\.?\d*%', response)
        if len(specific_numbers) > 3:
            indicators['has_made_up_facts'] = True

        return indicators

    def check_toxicity(self, response: str) -> float:
        """检测有害内容（需要调用Perspective API或本地模型）"""
        # 简化示例
        toxic_words = ['hate', 'stupid', 'kill']
        toxicity_score = sum(1 for word in toxic_words if word in response.lower())
        return min(toxicity_score / len(toxic_words), 1.0)

    def check_response_quality(self, response: str) -> Dict:
        """综合质量检查"""
        quality = {
            'length': len(response.split()),
            'has_code': '```' in response,
            'has_links': 'http' in response,
            'hallucination_risk': self.check_hallucination_indicators(response),
            'toxicity': self.check_toxicity(response)
        }

        # 质量评分
        score = 1.0
        if quality['hallucination_risk']['has_uncertainty']:
            score -= 0.2
        if quality['toxicity'] > 0.5:
            score -= 0.5

        quality['quality_score'] = max(score, 0.0)

        return quality

    def log_quality(self, request_id: str, response: str):
        """记录质量指标"""
        quality = self.check_response_quality(response)

        self.metrics['quality_scores'].append(quality['quality_score'])
        self.metrics['toxicity_scores'].append(quality['toxicity'])

        # 触发告警
        if quality['quality_score'] < 0.5:
            self.alert(f"Low quality response: {request_id}")

        if quality['toxicity'] > 0.7:
            self.alert(f"High toxicity detected: {request_id}")

    def get_summary(self) -> Dict:
        """获取质量摘要"""
        return {
            'avg_quality': np.mean(self.metrics['quality_scores']),
            'avg_toxicity': np.mean(self.metrics['toxicity_scores']),
            'low_quality_rate': sum(1 for s in self.metrics['quality_scores'] if s < 0.5) /
                               len(self.metrics['quality_scores'])
        }

    def alert(self, message: str):
        """发送告警"""
        print(f"ALERT: {message}")
        # 实际应用中发送到Slack/PagerDuty等
```

---

### 3.3 告警系统

#### 3.3.1 Prometheus Alertmanager配置

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK'

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'slack-notifications'

receivers:
  - name: 'slack-notifications'
    slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

**告警规则**

```yaml
# alerts.yml
groups:
  - name: llm_api_alerts
    interval: 30s
    rules:
      # 高错误率
      - alert: HighErrorRate
        expr: rate(llm_requests_total{status="error"}[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          description: "错误率超过5%: {{ $value | humanizePercentage }}"

      # 高延迟
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(llm_request_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          description: "P95延迟超过5秒: {{ $value }}s"

      # GPU内存不足
      - alert: GPUMemoryHigh
        expr: llm_gpu_memory_bytes / (16 * 1024^3) > 0.9
        for: 2m
        labels:
          severity: warning
        annotations:
          description: "GPU内存使用超过90%"

      # 数据漂移
      - alert: DataDrift
        expr: llm_data_drift_psi > 0.2
        for: 1h
        labels:
          severity: warning
        annotations:
          description: "检测到数据漂移: PSI={{ $value }}"
```

---

## 4. 成本优化

### 4.1 推理优化

#### 4.1.1 批处理（Batching）

```python
import asyncio
from collections import deque
from typing import List
import time

class DynamicBatcher:
    """动态批处理"""

    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.running = False

    async def add_request(self, request):
        """添加请求到队列"""
        future = asyncio.Future()
        self.queue.append((request, future))

        # 启动批处理循环
        if not self.running:
            asyncio.create_task(self.process_batches())

        return await future

    async def process_batches(self):
        """处理批次"""
        self.running = True

        while self.queue:
            batch = []
            futures = []

            # 收集批次
            start_time = time.time()
            while len(batch) < self.max_batch_size and self.queue:
                # 检查等待时间
                if batch and (time.time() - start_time) * 1000 > self.max_wait_ms:
                    break

                request, future = self.queue.popleft()
                batch.append(request)
                futures.append(future)

                # 短暂等待更多请求
                if len(batch) < self.max_batch_size:
                    await asyncio.sleep(0.001)

            # 批量处理
            if batch:
                try:
                    results = await self.batch_inference(batch)

                    # 返回结果
                    for future, result in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)

        self.running = False

    async def batch_inference(self, requests: List):
        """批量推理"""
        # 合并输入
        inputs = [req['input'] for req in requests]

        # 批量处理
        outputs = model.generate(inputs)

        return outputs

# FastAPI集成
batcher = DynamicBatcher(max_batch_size=32, max_wait_ms=100)

@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    result = await batcher.add_request(request.dict())
    return result
```

#### 4.1.2 请求合并（Request Coalescing）

```python
import hashlib
from typing import Dict, Optional
import asyncio

class RequestCache:
    """请求缓存（合并相同请求）"""

    def __init__(self, ttl_seconds: int = 60):
        self.cache: Dict[str, asyncio.Future] = {}
        self.ttl = ttl_seconds

    def _hash_request(self, request: Dict) -> str:
        """计算请求哈希"""
        # 只考虑影响结果的字段
        key = f"{request['model']}:{request['prompt']}:{request.get('temperature', 1.0)}"
        return hashlib.md5(key.encode()).hexdigest()

    async def get_or_compute(self, request: Dict, compute_func):
        """获取缓存或计算"""
        req_hash = self._hash_request(request)

        # 检查是否有进行中的相同请求
        if req_hash in self.cache:
            # 等待已有请求完成
            return await self.cache[req_hash]

        # 创建新future
        future = asyncio.Future()
        self.cache[req_hash] = future

        try:
            # 计算结果
            result = await compute_func(request)
            future.set_result(result)

            # 设置TTL后清除
            asyncio.create_task(self._cleanup_after_ttl(req_hash))

            return result
        except Exception as e:
            future.set_exception(e)
            del self.cache[req_hash]
            raise

    async def _cleanup_after_ttl(self, req_hash: str):
        """TTL后清除缓存"""
        await asyncio.sleep(self.ttl)
        self.cache.pop(req_hash, None)

# 使用示例
cache = RequestCache(ttl_seconds=60)

@app.post("/v1/completions")
async def completion(request: CompletionRequest):
    result = await cache.get_or_compute(
        request.dict(),
        compute_func=lambda r: model.generate(r['prompt'])
    )
    return result
```

---

### 4.2 资源管理

#### 4.2.1 自动扩缩容（Kubernetes HPA）

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  # 基于CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # 基于GPU
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "70"
  # 基于请求数
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

---

### 4.3 缓存策略

#### 4.3.1 多层缓存

```python
import redis
from functools import lru_cache
import pickle

class MultiLevelCache:
    """多层缓存：内存 → Redis → 计算"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.memory_cache_size = 1000

    @lru_cache(maxsize=1000)
    def _memory_cache(self, key: str):
        """L1: 内存缓存（最快）"""
        pass  # LRU cache装饰器已处理

    def _redis_cache_get(self, key: str) -> Optional[str]:
        """L2: Redis缓存"""
        value = self.redis_client.get(key)
        return pickle.loads(value) if value else None

    def _redis_cache_set(self, key: str, value: str, ttl: int = 3600):
        """设置Redis缓存"""
        self.redis_client.setex(key, ttl, pickle.dumps(value))

    async def get_or_generate(self, prompt: str, generate_func):
        """三层缓存策略"""
        key = f"llm:completion:{hash(prompt)}"

        # L1: 检查内存
        try:
            return self._memory_cache(key)
        except:
            pass

        # L2: 检查Redis
        cached = self._redis_cache_get(key)
        if cached:
            # 回填到内存缓存
            self._memory_cache.__setitem__(key, cached)
            return cached

        # L3: 生成新结果
        result = await generate_func(prompt)

        # 写入缓存
        self._redis_cache_set(key, result, ttl=3600)
        self._memory_cache.__setitem__(key, result)

        return result
```

---

## 5. 实战案例

### 5.1 完整的MLOps Pipeline示例

**场景：**聊天机器人从开发到生产的完整流程

**1. 开发阶段**

```bash
# 创建功能分支
git checkout -b feature/improve-response-quality

# 实验（自动记录到W&B）
python train.py --config configs/experiment_1.yaml

# 提交代码
git add .
git commit -m "feat: 改进回复质量"
git push origin feature/improve-response-quality
```

**2. CI测试（自动触发）**

```yaml
# GitHub Actions自动运行
# ✓ 代码格式检查
# ✓ 单元测试
# ✓ 模型准确率测试
# ✓ 性能回归测试
```

**3. 代码审查 & 合并**

```bash
# Pull Request审查通过后
git checkout main
git merge feature/improve-response-quality
```

**4. 模型注册**

```python
# 自动运行：注册新模型版本
import mlflow

mlflow.register_model(
    model_uri="runs:/xxx/model",
    name="chatbot-model"
)

# 设置为Staging
client.transition_model_version_stage(
    name="chatbot-model",
    version=2,
    stage="Staging"
)
```

**5. Staging环境测试**

```bash
# 部署到Staging
kubectl apply -f k8s/staging/

# 运行集成测试
python tests/integration_test.py --env staging

# 负载测试
k6 run load_test.js
```

**6. 金丝雀发布到生产**

```bash
# 标记release
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0

# 自动触发部署
# → 构建Docker镜像
# → 部署到10%流量
# → 监控metrics
# → 逐步增加到100%
```

**7. 监控 & 回滚（如needed）**

```bash
# 监控Grafana dashboard
# - 错误率：0.1% ✓
# - P95延迟：800ms ✓
# - 用户满意度：95% ✓

# 如有问题，一键回滚
kubectl rollback deployment/llm-api
```

---

## 6. 工具链推荐

### 实验管理
- **Weights & Biases**: ⭐⭐⭐⭐⭐ 实验跟踪
- **MLflow**: ⭐⭐⭐⭐⭐ 模型管理
- **TensorBoard**: ⭐⭐⭐ 可视化

### 版本控制
- **Git**: ⭐⭐⭐⭐⭐ 代码版本
- **DVC**: ⭐⭐⭐⭐ 数据/模型版本
- **LakeFS**: ⭐⭐⭐ 数据湖版本控制

### CI/CD
- **GitHub Actions**: ⭐⭐⭐⭐⭐ CI/CD
- **GitLab CI**: ⭐⭐⭐⭐ CI/CD
- **Jenkins**: ⭐⭐⭐ 传统CI

### 监控
- **Prometheus**: ⭐⭐⭐⭐⭐ Metrics收集
- **Grafana**: ⭐⭐⭐⭐⭐ 可视化
- **LangSmith**: ⭐⭐⭐⭐ LLM专用监控

### 部署
- **Kubernetes**: ⭐⭐⭐⭐⭐ 容器编排
- **Docker**: ⭐⭐⭐⭐⭐ 容器化
- **vLLM**: ⭐⭐⭐⭐⭐ LLM推理服务

---

## 7. 常见问题与解决方案

### Q1: 如何处理模型更新导致的不兼容问题？

**解决方案：版本化API**

```python
# 支持多个API版本
@app.post("/v1/completions")
async def completions_v1(request: CompletionRequestV1):
    return model_v1.generate(request)

@app.post("/v2/completions")
async def completions_v2(request: CompletionRequestV2):
    return model_v2.generate(request)

# 逐步迁移用户到v2
# 保留v1至少6个月
```

### Q2: 如何快速回滚有问题的部署？

**策略：**
1. **蓝绿部署**: 一键切换流量
2. **Feature Flags**: 动态关闭新功能
3. **自动回滚**: 监控触发自动回滚

### Q3: 如何平衡成本和性能？

**建议：**
1. **模型分级**: 简单请求用小模型，复杂请求用大模型
2. **Spot实例**: 非关键服务使用Spot节省70%
3. **缓存**: 相同请求缓存可节省80%成本

---

## 总结

本文档涵盖了LLM MLOps的完整实践：

1. **模型开发**: W&B实验管理、DVC版本控制、Docker环境隔离
2. **CI/CD**: 自动化测试、蓝绿部署、金丝雀发布
3. **监控**: Prometheus metrics、数据漂移检测、质量监控
4. **成本优化**: 批处理、缓存、自动扩缩容

**关键要点：**
- 自动化一切可自动化的流程
- 监控先行，提前发现问题
- 快速迭代，快速回滚
- 成本意识，持续优化

**下一步学习：**
- [10_Production_Deployment.md](./10_Production_Deployment.md) - 生产部署
- [13_Data_Engineering.md](./13_Data_Engineering.md) - 数据工程
- [07_LLM_Evaluation.md](./07_LLM_Evaluation.md) - 模型评估
