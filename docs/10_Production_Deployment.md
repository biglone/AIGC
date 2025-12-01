# LLM生产部署

## 目录
1. [模型服务化](#模型服务化)
2. [性能优化](#性能优化)
3. [监控与日志](#监控与日志)
4. [成本优化](#成本优化)
5. [高可用架构](#高可用架构)

---

## 模型服务化

### vLLM部署

**安装和启动：**
```bash
# 安装
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --tensor-parallel-size 1
```

**Python调用：**
```python
from openai import OpenAI

# vLLM提供OpenAI兼容接口
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)
```

**性能配置：**
```python
from vllm import LLM, SamplingParams

# 创建引擎
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # 2卡并行
    gpu_memory_utilization=0.9,  # GPU利用率
    max_num_batched_tokens=8192,  # 最大批处理
)

# 批量推理
prompts = ["问题1", "问题2", "问题3"]
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

outputs = llm.generate(prompts, sampling_params)
```

### TensorRT-LLM

**优化和部署：**
```bash
# 转换模型
python build.py \
    --model_dir ./llama-2-7b \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_output_len 512 \
    --output_dir ./trt_engines

# 运行推理
python run.py \
    --engine_dir ./trt_engines \
    --max_output_len 100 \
    --input_text "你好"
```

### FastAPI服务

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 运行：uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 性能优化

### 批处理策略

**动态批处理：**
```python
import asyncio
from collections import deque

class DynamicBatcher:
    def __init__(self, model, max_batch_size=8, max_wait_time=0.1):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.results = {}

    async def add_request(self, request_id, prompt):
        """添加请求到批处理队列"""
        future = asyncio.Future()
        self.queue.append((request_id, prompt, future))

        # 触发批处理
        if len(self.queue) >= self.max_batch_size:
            asyncio.create_task(self.process_batch())

        return await future

    async def process_batch(self):
        """处理一个批次"""
        if not self.queue:
            return

        # 等待accumulate更多请求
        await asyncio.sleep(self.max_wait_time)

        # 获取批次
        batch_size = min(len(self.queue), self.max_batch_size)
        batch = [self.queue.popleft() for _ in range(batch_size)]

        request_ids, prompts, futures = zip(*batch)

        # 批量推理
        outputs = self.model.generate(prompts)

        # 返回结果
        for future, output in zip(futures, outputs):
            future.set_result(output)

# 使用
batcher = DynamicBatcher(model)
result = await batcher.add_request("req_1", "你好")
```

### Continuous Batching

**vLLM的核心优势：**
```
传统批处理：
Batch 1: [req1(100 tokens), req2(50 tokens), req3(80 tokens)]
等待: req1完成，其他请求闲置

Continuous Batching:
Step 1: [req1, req2, req3]
Step 50: [req1, req3]  # req2完成，加入req4
Step 51: [req1, req3, req4]
```

### 模型并行

**张量并行（Tensor Parallelism）：**
```python
# 将一个大矩阵分片到多个GPU
# Linear层: Y = XW

# GPU 0: Y1 = X @ W1  (W的前半部分)
# GPU 1: Y2 = X @ W2  (W的后半部分)
# 合并: Y = concat(Y1, Y2)

# vLLM自动处理
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4  # 使用4张GPU
)
```

**Pipeline并行：**
```
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
```

---

## 监控与日志

### Prometheus + Grafana

**暴露指标：**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义指标
REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of requests',
    ['model', 'status']
)

REQUEST_LATENCY = Histogram(
    'llm_request_duration_seconds',
    'Request latency',
    ['model']
)

ACTIVE_REQUESTS = Gauge(
    'llm_active_requests',
    'Number of active requests',
    ['model']
)

# 在API中使用
@app.post("/generate")
async def generate(request: GenerateRequest):
    ACTIVE_REQUESTS.labels(model="llama-2-7b").inc()

    start_time = time.time()

    try:
        response = model.generate(request.prompt)

        REQUEST_COUNT.labels(model="llama-2-7b", status="success").inc()
        return response

    except Exception as e:
        REQUEST_COUNT.labels(model="llama-2-7b", status="error").inc()
        raise

    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(model="llama-2-7b").observe(duration)
        ACTIVE_REQUESTS.labels(model="llama-2-7b").dec()

# 启动metrics服务器
start_http_server(9090)
```

**Grafana仪表盘：**
```
面板1: QPS (Queries Per Second)
  rate(llm_requests_total[1m])

面板2: P95延迟
  histogram_quantile(0.95, llm_request_duration_seconds)

面板3: 错误率
  rate(llm_requests_total{status="error"}[1m]) / rate(llm_requests_total[1m])

面板4: 活跃请求数
  llm_active_requests
```

### 日志记录

```python
import logging
import json
from datetime import datetime

# 结构化日志
class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_request(self, request_id, prompt, response, latency):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "latency": latency,
            "model": "llama-2-7b"
        }
        self.logger.info(json.dumps(log_entry))

logger = StructuredLogger("llm_service")

# 使用
logger.log_request(
    request_id="req_123",
    prompt="你好",
    response="你好！有什么我可以帮助你的吗？",
    latency=0.523
)
```

---

## 成本优化

### 智能路由

```python
class ModelRouter:
    """根据任务复杂度路由到不同模型"""

    def __init__(self):
        self.models = {
            "simple": "gpt-3.5-turbo",      # $0.001/1K tokens
            "medium": "gpt-4o-mini",         # $0.0001/1K tokens
            "complex": "gpt-4",              # $0.03/1K tokens
        }

    def classify_task(self, prompt):
        """分类任务复杂度"""
        # 简单规则
        if len(prompt) < 50:
            return "simple"
        elif any(kw in prompt for kw in ["分析", "总结", "推理"]):
            return "complex"
        else:
            return "medium"

    def route(self, prompt):
        complexity = self.classify_task(prompt)
        model = self.models[complexity]
        return model, complexity

# 使用
router = ModelRouter()
model, _ = router.route("1+1等于多少？")  # gpt-3.5-turbo
model, _ = router.route("分析这个算法的时间复杂度...")  # gpt-4
```

### 缓存策略

```python
import hashlib
from functools import lru_cache
import redis

class ResponseCache:
    def __init__(self, redis_client):
        self.redis = redis_client

    def get_cache_key(self, prompt, model, temperature):
        """生成缓存key"""
        content = f"{prompt}|{model}|{temperature}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, prompt, model, temperature):
        """获取缓存"""
        key = self.get_cache_key(prompt, model, temperature)
        cached = self.redis.get(key)

        if cached:
            return json.loads(cached)
        return None

    def set(self, prompt, model, temperature, response, ttl=3600):
        """设置缓存"""
        key = self.get_cache_key(prompt, model, temperature)
        self.redis.setex(
            key,
            ttl,
            json.dumps(response)
        )

# 使用
cache = ResponseCache(redis.Redis())

@app.post("/generate")
async def generate(request):
    # 检查缓存
    cached = cache.get(
        request.prompt,
        request.model,
        request.temperature
    )

    if cached:
        return cached

    # 生成
    response = model.generate(request.prompt)

    # 缓存
    if request.temperature == 0:  # 只缓存确定性响应
        cache.set(request.prompt, request.model, 0, response)

    return response
```

### Spot实例

```python
# AWS Spot实例配置
{
    "SpotOptions": {
        "MaxPrice": "1.0",  # 最高价格
        "SpotInstanceType": "one-time",
        "InstanceInterruptionBehavior": "terminate"
    },
    "InstanceType": "p3.8xlarge",  # V100 x4
    "ImageId": "ami-xxxxx",
    "KeyName": "mykey"
}

# 容错处理
class SpotInstanceHandler:
    def on_interruption_warning(self):
        """收到中断警告（提前2分钟）"""
        # 1. 停止接收新请求
        self.accepting_requests = False

        # 2. 等待当前请求完成
        self.wait_for_completion()

        # 3. 保存检查点
        self.save_checkpoint()

    def wait_for_completion(self, timeout=90):
        """等待请求完成"""
        start = time.time()
        while self.active_requests > 0:
            if time.time() - start > timeout:
                break
            time.sleep(1)
```

---

## 高可用架构

### 负载均衡

```
           ┌─────────────┐
           │ Load Balancer│
           └──────┬───────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
    ┌───▼──┐  ┌───▼──┐  ┌───▼──┐
    │Model │  │Model │  │Model │
    │ Pod 1│  │ Pod 2│  │ Pod 3│
    └──────┘  └──────┘  └──────┘
```

**Kubernetes配置：**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm
  template:
    metadata:
      labels:
        app: llm
    spec:
      containers:
      - name: llm
        image: llm-service:v1
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 自动扩缩容

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: llm_active_requests
      target:
        type: AverageValue
        averageValue: "10"
```

---

## 延伸阅读

**框架：**
- vLLM
- TensorRT-LLM
- Text Generation Inference (TGI)
- Ray Serve

**监控：**
- Prometheus
- Grafana
- LangSmith
- Weights & Biases

**云服务：**
- AWS SageMaker
- Google Vertex AI
- Azure ML

**下一步：**
- [安全与对齐](11_Safety_and_Alignment.md)
