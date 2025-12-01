# 多模态AI系统

## 目录
1. [视觉-语言模型](#视觉-语言模型)
2. [图像生成](#图像生成)
3. [视频理解](#视频理解)
4. [音频处理](#音频处理)
5. [多模态应用](#多模态应用)

---

## 视觉-语言模型

### CLIP（对比学习）

**核心思想：** 图像和文本在同一嵌入空间中对齐

```python
import torch
from transformers import CLIPProcessor, CLIPModel

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 图文匹配
from PIL import Image
image = Image.open("cat.jpg")
texts = ["a cat", "a dog", "a car"]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print(probs)  # [0.92, 0.05, 0.03] - "a cat"概率最高
```

**Zero-Shot分类：**
```python
def zero_shot_classification(image, candidates):
    # 模板化候选标签
    texts = [f"a photo of a {label}" for label in candidates]

    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]

    results = [
        {"label": label, "score": score.item()}
        for label, score in zip(candidates, probs)
    ]

    return sorted(results, key=lambda x: x["score"], reverse=True)

# 使用
results = zero_shot_classification(
    image,
    ["cat", "dog", "bird", "car", "tree"]
)
```

### LLaVA（视觉指令微调）

**架构：** Vision Encoder + Projection + LLM

```
图像 → CLIP Vision Encoder → Projection Layer → LLaMA
                                                    ↓
                                            文本输入合并
                                                    ↓
                                            LLaMA生成回答
```

**使用示例：**
```python
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates

# 加载模型
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-7b",
    model_base=None,
    model_name="llava-v1.5-7b"
)

# 准备图像
from PIL import Image
image = Image.open("image.jpg")
image_tensor = image_processor.preprocess(image)

# 对话
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], "描述这张图片")
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

# 生成
input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    input_ids,
    images=image_tensor.unsqueeze(0),
    max_new_tokens=512
)

response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### GPT-4V 和 Gemini Vision

**GPT-4V使用：**
```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)
```

---

## 图像生成

### Stable Diffusion原理

**扩散过程：**
```
前向扩散：x₀ → x₁ → ... → xₜ (加噪)
反向去噪：xₜ → ... → x₁ → x₀ (生成图像)
```

**基本使用：**
```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# 生成图像
prompt = "a beautiful sunset over mountains, oil painting style"
negative_prompt = "blurry, low quality, distorted"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

image.save("output.png")
```

**关键参数：**
- `num_inference_steps`: 去噪步数（越多越精细，50-100）
- `guidance_scale`: CFG强度（3-15，越高越符合prompt）
- `negative_prompt`: 不想要的元素

### ControlNet（可控生成）

**使用边缘图控制：**
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2

# 加载ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny"
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet
).to("cuda")

# 准备控制图（Canny边缘）
image = Image.open("input.jpg")
image_np = np.array(image)
edges = cv2.Canny(image_np, 100, 200)
edges = Image.fromarray(edges)

# 生成
output = pipe(
    prompt="a beautiful landscape",
    image=edges,
    num_inference_steps=20
).images[0]
```

**其他控制类型：**
- Canny边缘
- 深度图（Depth）
- 姿态（OpenPose）
- 语义分割（Seg）
- 法线贴图（Normal）

### LoRA for Diffusion

**训练自定义风格：**
```python
from diffusers import DiffusionPipeline

# 加载基础模型
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1"
)

# 加载LoRA权重
pipe.load_lora_weights("path/to/lora", weight_name="style.safetensors")

# 生成
image = pipe("a castle in anime style").images[0]
```

---

## 视频理解

### 视频分类和字幕

```python
from transformers import VivitImageProcessor, VivitForVideoClassification
import av

# 加载模型
processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

# 读取视频帧
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

container = av.open("video.mp4")
indices = list(range(0, 32))  # 采样32帧
video = read_video_pyav(container, indices)

# 预测
inputs = processor(list(video), return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()

print(model.config.id2label[predicted_class])
```

---

## 音频处理

### Whisper（语音识别）

```python
import whisper

# 加载模型
model = whisper.load_model("base")

# 转录
result = model.transcribe("audio.mp3", language="zh")

print(result["text"])
# 输出：转录的文本

# 带时间戳的转录
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
```

### TTS（文本转语音）

```python
from transformers import VitsModel, AutoTokenizer

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

text = "Hello, how are you today?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

# 保存音频
import scipy.io.wavfile as wavfile
wavfile.write("output.wav", rate=16000, data=output.squeeze().numpy())
```

---

## 多模态应用

### 图文检索

```python
class MultimodalRetrieval:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_index = None

    def index_images(self, image_paths):
        """索引图像库"""
        embeddings = []

        for path in image_paths:
            image = Image.open(path)
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
            embeddings.append(embedding)

        self.image_index = torch.cat(embeddings)
        self.image_paths = image_paths

    def search(self, text_query, k=5):
        """文本搜索图像"""
        inputs = self.processor(text=text_query, return_tensors="pt")
        with torch.no_grad():
            text_embedding = self.clip_model.get_text_features(**inputs)

        # 计算相似度
        similarities = (text_embedding @ self.image_index.T).squeeze()
        top_k = similarities.topk(k)

        results = [
            {"path": self.image_paths[idx], "score": score.item()}
            for score, idx in zip(top_k.values, top_k.indices)
        ]

        return results
```

### OCR + LLM

```python
import easyocr
from openai import OpenAI

class DocumentUnderstanding:
    def __init__(self):
        self.reader = easyocr.Reader(['ch_sim', 'en'])
        self.client = OpenAI()

    def analyze_document(self, image_path, question):
        """分析文档并回答问题"""

        # 1. OCR提取文本
        result = self.reader.readtext(image_path)
        text = "\n".join([item[1] for item in result])

        # 2. LLM理解
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是文档分析专家"},
                {"role": "user", "content": f"文档内容：\n{text}\n\n问题：{question}"}
            ]
        )

        return response.choices[0].message.content
```

---

## 延伸阅读

**论文：**
1. CLIP: Learning Transferable Visual Models From Natural Language (Radford et al., 2021)
2. LLaVA: Visual Instruction Tuning (Liu et al., 2023)
3. High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)
4. Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet, Zhang et al., 2023)

**模型：**
- CLIP, BLIP-2, LLaVA, GPT-4V
- Stable Diffusion, DALL-E 3, Midjourney
- Whisper, TTS模型

**下一步：**
- [生产部署](10_Production_Deployment.md)
- [安全与对齐](11_Safety_and_Alignment.md)
