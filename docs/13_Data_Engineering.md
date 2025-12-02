# 数据工程实战：LLM数据准备完整指南

> **文档定位：** 从数据采集到数据集构建的完整工程实践
> **适用对象：** 希望训练/微调LLM的开发者、数据工程师
> **前置知识：** Python编程、机器学习基础、LLM基础原理

---

## 目录

1. [LLM数据准备全流程](#1-llm数据准备全流程)
2. [数据标注最佳实践](#2-数据标注最佳实践)
3. [合成数据生成](#3-合成数据生成)
4. [实战代码示例](#4-实战代码示例)
5. [工具链推荐](#5-工具链推荐)
6. [常见问题与解决方案](#6-常见问题与解决方案)

---

## 1. LLM数据准备全流程

### 1.1 数据采集

#### 1.1.1 网络爬取

**Common Crawl**

Common Crawl 是最大的公开网页爬取数据集，每月爬取数十亿网页。

```python
import requests
from bs4 import BeautifulSoup
import warc
import gzip

def download_common_crawl_sample():
    """下载 Common Crawl WARC 文件示例"""
    # Common Crawl WARC 文件URL（示例）
    warc_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/..."

    response = requests.get(warc_url, stream=True)

    with gzip.open('sample.warc.gz', 'wb') as f:
        f.write(response.content)

def parse_warc_file(warc_path):
    """解析 WARC 文件提取网页内容"""
    documents = []

    with gzip.open(warc_path, 'rb') as f:
        for record in warc.WARCFile(fileobj=f):
            if record['WARC-Type'] == 'response':
                # 提取HTML内容
                payload = record.payload.read()

                try:
                    soup = BeautifulSoup(payload, 'html.parser')

                    # 移除脚本和样式
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # 提取文本
                    text = soup.get_text()

                    # 清理空白
                    lines = (line.strip() for line in text.splitlines())
                    text = '\n'.join(line for line in lines if line)

                    documents.append({
                        'url': record['WARC-Target-URI'],
                        'date': record['WARC-Date'],
                        'text': text
                    })
                except Exception as e:
                    print(f"Error parsing record: {e}")
                    continue

    return documents

# 使用示例
docs = parse_warc_file('sample.warc.gz')
print(f"提取了 {len(docs)} 个文档")
```

**自定义网页爬虫**

```python
import scrapy
from scrapy.crawler import CrawlerProcess

class TextSpider(scrapy.Spider):
    name = 'text_spider'

    def __init__(self, start_urls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = start_urls
        self.documents = []

    def parse(self, response):
        # 提取文本内容
        text = ' '.join(response.css('p::text').getall())

        # 提取元数据
        title = response.css('title::text').get()

        self.documents.append({
            'url': response.url,
            'title': title,
            'text': text,
            'word_count': len(text.split())
        })

        # 跟随链接（限制深度）
        if response.meta.get('depth', 0) < 2:
            for href in response.css('a::attr(href)').getall():
                yield response.follow(href, self.parse)

# 使用示例
def crawl_websites(urls):
    """爬取指定网站列表"""
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_DELAY': 1,  # 礼貌爬取
    })

    spider = TextSpider(start_urls=urls)
    process.crawl(spider)
    process.start()

    return spider.documents
```

#### 1.1.2 API数据获取

**Reddit API**

```python
import praw
from datetime import datetime

def collect_reddit_data(subreddit_name, limit=1000):
    """从Reddit收集对话数据"""
    reddit = praw.Reddit(
        client_id='YOUR_CLIENT_ID',
        client_secret='YOUR_CLIENT_SECRET',
        user_agent='LLM Data Collector'
    )

    subreddit = reddit.subreddit(subreddit_name)
    conversations = []

    for submission in subreddit.top(limit=limit, time_filter='year'):
        # 获取帖子内容
        post = {
            'title': submission.title,
            'text': submission.selftext,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_utc': datetime.fromtimestamp(submission.created_utc),
            'comments': []
        }

        # 获取评论（展开所有评论）
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            post['comments'].append({
                'text': comment.body,
                'score': comment.score,
                'depth': comment.depth
            })

        conversations.append(post)

    return conversations

# 使用示例
data = collect_reddit_data('MachineLearning', limit=100)
print(f"收集了 {len(data)} 个对话")
```

**GitHub API**

```python
import requests
from typing import List, Dict

def collect_github_code(query: str, language: str = 'python', limit: int = 100):
    """从GitHub收集代码数据"""
    headers = {
        'Authorization': 'token YOUR_GITHUB_TOKEN',
        'Accept': 'application/vnd.github.v3+json'
    }

    code_samples = []
    page = 1

    while len(code_samples) < limit:
        url = f'https://api.github.com/search/code'
        params = {
            'q': f'{query} language:{language}',
            'per_page': 100,
            'page': page
        }

        response = requests.get(url, headers=headers, params=params)
        results = response.json()

        if 'items' not in results:
            break

        for item in results['items']:
            # 获取文件内容
            file_url = item['url']
            file_response = requests.get(file_url, headers=headers)
            file_data = file_response.json()

            # Base64解码
            import base64
            content = base64.b64decode(file_data['content']).decode('utf-8')

            code_samples.append({
                'repo': item['repository']['full_name'],
                'path': item['path'],
                'url': item['html_url'],
                'code': content,
                'stars': item['repository'].get('stargazers_count', 0)
            })

            if len(code_samples) >= limit:
                break

        page += 1

    return code_samples

# 使用示例
code_data = collect_github_code('machine learning', language='python', limit=50)
```

#### 1.1.3 开源数据集

**常用开源数据集列表**

| 数据集 | 规模 | 用途 | 下载方式 |
|--------|------|------|----------|
| **The Pile** | 825GB | 预训练 | `pip install datasets` |
| **RedPajama** | 1.2TB | 预训练 | Hugging Face |
| **C4** | 750GB | 预训练 | TensorFlow Datasets |
| **OpenWebText** | 38GB | GPT-2风格预训练 | Hugging Face |
| **Wikipedia** | 20GB | 知识密集型任务 | `pip install wikipedia` |
| **BookCorpus** | 5GB | 长文本理解 | Hugging Face |

**加载示例**

```python
from datasets import load_dataset

# 加载 The Pile 数据集
pile_dataset = load_dataset("EleutherAI/pile", split='train', streaming=True)

# 流式处理（避免内存溢出）
for i, example in enumerate(pile_dataset):
    text = example['text']
    meta = example['meta']

    # 处理文本
    print(f"Sample {i}: {text[:100]}...")

    if i >= 10:
        break

# 加载 Wikipedia
wiki_dataset = load_dataset("wikipedia", "20220301.en", split='train')
print(f"Wikipedia articles: {len(wiki_dataset)}")

# 加载 C4
c4_dataset = load_dataset("c4", "en", split='train', streaming=True)
```

---

### 1.2 数据清洗

#### 1.2.1 去重策略

**Exact Deduplication（精确去重）**

```python
from typing import List, Set
import hashlib

def exact_dedup(documents: List[str]) -> List[str]:
    """精确去重：基于文本哈希"""
    seen_hashes: Set[str] = set()
    unique_docs = []

    for doc in documents:
        # 计算MD5哈希
        doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()

        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs

# 使用示例
docs = ["Hello world", "Hello world", "Different text"]
unique = exact_dedup(docs)
print(f"原始: {len(docs)}, 去重后: {len(unique)}")
```

**MinHash（近似去重）**

MinHash 可以高效检测相似文档，适合大规模数据。

```python
from datasketch import MinHash, MinHashLSH
from typing import List, Tuple

def minhash_dedup(documents: List[str], threshold: float = 0.8) -> List[str]:
    """
    MinHash近似去重

    Args:
        documents: 文档列表
        threshold: Jaccard相似度阈值（0-1）
    """
    # 创建LSH索引
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    minhashes = {}
    unique_docs = []

    for i, doc in enumerate(documents):
        # 创建MinHash
        m = MinHash(num_perm=128)

        # 使用3-gram tokenization
        tokens = set()
        for j in range(len(doc) - 2):
            tokens.add(doc[j:j+3])

        for token in tokens:
            m.update(token.encode('utf-8'))

        # 查询相似文档
        result = lsh.query(m)

        if not result:
            # 没有相似文档，添加到索引
            lsh.insert(f"doc_{i}", m)
            minhashes[f"doc_{i}"] = m
            unique_docs.append(doc)

    return unique_docs

# 使用示例
docs = [
    "The quick brown fox jumps over the lazy dog",
    "The quick brown fox jumped over the lazy dog",  # 相似
    "Completely different sentence here"
]
unique = minhash_dedup(docs, threshold=0.8)
print(f"去重后: {len(unique)} 个文档")
```

**SimHash（快速近似去重）**

```python
import simhash

def simhash_dedup(documents: List[str], threshold: int = 3) -> List[str]:
    """
    SimHash去重

    Args:
        documents: 文档列表
        threshold: Hamming距离阈值（0-64）
    """
    seen_hashes = {}
    unique_docs = []

    for doc in documents:
        # 计算SimHash
        hash_value = simhash.Simhash(doc).value

        # 检查是否有相似文档
        is_duplicate = False
        for seen_hash in seen_hashes:
            hamming_dist = bin(hash_value ^ seen_hash).count('1')
            if hamming_dist <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_hashes[hash_value] = True
            unique_docs.append(doc)

    return unique_docs
```

#### 1.2.2 质量过滤

**困惑度过滤（Perplexity Filtering）**

使用小型语言模型计算困惑度，过滤低质量文本。

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PerplexityFilter:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

    def calculate_perplexity(self, text: str) -> float:
        """计算文本困惑度"""
        encodings = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss

        # 困惑度 = exp(loss)
        perplexity = torch.exp(loss).item()
        return perplexity

    def filter_by_perplexity(self, documents: List[str],
                            max_perplexity: float = 1000) -> List[str]:
        """过滤高困惑度（低质量）文档"""
        filtered = []

        for doc in documents:
            ppl = self.calculate_perplexity(doc)
            if ppl <= max_perplexity:
                filtered.append(doc)

        return filtered

# 使用示例
filter_model = PerplexityFilter()
docs = [
    "This is a well-written sentence.",
    "asdf qwer zxcv random chars",  # 高困惑度
]
filtered = filter_model.filter_by_perplexity(docs, max_perplexity=500)
```

**启发式规则过滤**

```python
import re
from typing import List

class HeuristicFilter:
    """基于规则的文本质量过滤"""

    def __init__(self):
        self.min_words = 10
        self.max_words = 10000
        self.min_avg_word_length = 3
        self.max_symbol_ratio = 0.3
        self.min_alpha_ratio = 0.6

    def filter_document(self, text: str) -> bool:
        """判断文档是否应该保留"""
        # 1. 长度检查
        words = text.split()
        if len(words) < self.min_words or len(words) > self.max_words:
            return False

        # 2. 平均单词长度
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < self.min_avg_word_length:
            return False

        # 3. 符号比例
        symbol_count = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        symbol_ratio = symbol_count / len(text)
        if symbol_ratio > self.max_symbol_ratio:
            return False

        # 4. 字母比例
        alpha_count = len(re.findall(r'[a-zA-Z]', text))
        alpha_ratio = alpha_count / len(text)
        if alpha_ratio < self.min_alpha_ratio:
            return False

        # 5. 重复行检查
        lines = text.split('\n')
        unique_lines = set(lines)
        if len(lines) > 10 and len(unique_lines) / len(lines) < 0.3:
            return False  # 太多重复行

        # 6. URL和代码检测
        url_count = len(re.findall(r'http[s]?://\S+', text))
        if url_count > 10:
            return False

        return True

    def filter_documents(self, documents: List[str]) -> List[str]:
        """批量过滤"""
        return [doc for doc in documents if self.filter_document(doc)]

# 使用示例
heuristic_filter = HeuristicFilter()
docs = [
    "This is a good quality document with meaningful content.",
    "a b c d e f",  # 太短
    "http://spam.com http://spam2.com " * 50,  # 太多URL
]
filtered = heuristic_filter.filter_documents(docs)
print(f"过滤后: {len(filtered)} 个文档")
```

#### 1.2.3 PII移除（个人身份信息）

**基于正则表达式**

```python
import re
from typing import Dict

class PIIRemover:
    """移除个人身份信息"""

    def __init__(self):
        # 定义PII模式
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }

    def remove_pii(self, text: str, replacement: str = '[REDACTED]') -> str:
        """移除所有PII"""
        cleaned = text

        for pii_type, pattern in self.patterns.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        return cleaned

    def detect_pii(self, text: str) -> Dict[str, int]:
        """检测PII出现次数"""
        counts = {}

        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            counts[pii_type] = len(matches)

        return counts

# 使用示例
pii_remover = PIIRemover()

text = """
Contact me at john.doe@example.com or call 555-123-4567.
My SSN is 123-45-6789.
"""

cleaned = pii_remover.remove_pii(text)
print("清洗后:", cleaned)

pii_stats = pii_remover.detect_pii(text)
print("检测到PII:", pii_stats)
```

**基于NER模型**

```python
from transformers import pipeline

class NERPIIRemover:
    """使用NER模型移除PII"""

    def __init__(self):
        # 加载NER模型
        self.ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )

    def remove_pii(self, text: str) -> str:
        """使用NER移除人名、组织名、地点"""
        entities = self.ner_pipeline(text)

        # 按位置倒序排序（从后往前替换，避免索引错乱）
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)

        cleaned = text
        for entity in entities:
            if entity['entity_group'] in ['PER', 'ORG', 'LOC']:
                # 替换实体
                start = entity['start']
                end = entity['end']
                cleaned = cleaned[:start] + '[REDACTED]' + cleaned[end:]

        return cleaned

# 使用示例
ner_remover = NERPIIRemover()
text = "John Smith works at Google in Mountain View."
cleaned = ner_remover.remove_pii(text)
print(cleaned)
```

#### 1.2.4 有毒内容过滤

**使用 Perspective API**

```python
import requests

class ToxicityFilter:
    """使用Perspective API过滤有毒内容"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    def get_toxicity_score(self, text: str) -> float:
        """获取毒性评分（0-1）"""
        data = {
            'comment': {'text': text},
            'requestedAttributes': {'TOXICITY': {}}
        }

        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            json=data
        )

        result = response.json()
        score = result['attributeScores']['TOXICITY']['summaryScore']['value']
        return score

    def filter_toxic_documents(self, documents: List[str],
                              threshold: float = 0.7) -> List[str]:
        """过滤高毒性文档"""
        filtered = []

        for doc in documents:
            try:
                score = self.get_toxicity_score(doc)
                if score < threshold:
                    filtered.append(doc)
            except Exception as e:
                print(f"Error processing document: {e}")
                continue

        return filtered

# 使用示例（需要API key）
# toxicity_filter = ToxicityFilter(api_key='YOUR_API_KEY')
# filtered = toxicity_filter.filter_toxic_documents(documents)
```

**使用本地分类器**

```python
from transformers import pipeline

class LocalToxicityFilter:
    """使用本地模型过滤有毒内容"""

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None
        )

    def get_toxicity_score(self, text: str) -> float:
        """获取毒性评分"""
        result = self.classifier(text)[0]

        # 找到toxic标签的分数
        for item in result:
            if item['label'] == 'toxic':
                return item['score']

        return 0.0

    def filter_documents(self, documents: List[str],
                        threshold: float = 0.5) -> List[str]:
        """批量过滤"""
        filtered = []

        for doc in documents:
            score = self.get_toxicity_score(doc)
            if score < threshold:
                filtered.append(doc)

        return filtered

# 使用示例
toxicity_filter = LocalToxicityFilter()
docs = [
    "This is a nice comment.",
    "I hate you so much!",  # 高毒性
]
filtered = toxicity_filter.filter_documents(docs, threshold=0.5)
```

---

### 1.3 数据增强

#### 1.3.1 回译（Back-translation）

```python
from transformers import MarianMTModel, MarianTokenizer

class BackTranslator:
    """回译数据增强"""

    def __init__(self, source_lang='en', pivot_lang='fr'):
        # 加载翻译模型
        self.src_to_pivot = self._load_model(source_lang, pivot_lang)
        self.pivot_to_src = self._load_model(pivot_lang, source_lang)

    def _load_model(self, src, tgt):
        model_name = f'Helsinki-NLP/opus-mt-{src}-{tgt}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return {'tokenizer': tokenizer, 'model': model}

    def translate(self, text: str, model_dict):
        """翻译文本"""
        tokenizer = model_dict['tokenizer']
        model = model_dict['model']

        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return result

    def back_translate(self, text: str) -> str:
        """执行回译"""
        # 英语 → 中间语言
        pivot_text = self.translate(text, self.src_to_pivot)

        # 中间语言 → 英语
        back_text = self.translate(pivot_text, self.pivot_to_src)

        return back_text

    def augment_dataset(self, texts: List[str]) -> List[str]:
        """数据增强：原始+回译"""
        augmented = []

        for text in texts:
            augmented.append(text)  # 原始
            augmented.append(self.back_translate(text))  # 回译

        return augmented

# 使用示例
bt = BackTranslator(source_lang='en', pivot_lang='fr')
original = "The weather is nice today."
augmented = bt.back_translate(original)
print(f"原始: {original}")
print(f"回译: {augmented}")
```

#### 1.3.2 改写（Paraphrasing）

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Paraphraser:
    """使用T5进行改写"""

    def __init__(self, model_name='ramsrigouthamg/t5_paraphraser'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def paraphrase(self, text: str, num_return_sequences: int = 3) -> List[str]:
        """生成多个改写版本"""
        input_text = f"paraphrase: {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=num_return_sequences,
            num_beams=num_return_sequences,
            temperature=1.5,
            do_sample=True
        )

        paraphrases = []
        for output in outputs:
            paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
            paraphrases.append(paraphrase)

        return paraphrases

# 使用示例
paraphraser = Paraphraser()
text = "Machine learning is a subset of artificial intelligence."
paraphrases = paraphraser.paraphrase(text, num_return_sequences=3)

print("原始:", text)
for i, para in enumerate(paraphrases, 1):
    print(f"改写{i}:", para)
```

#### 1.3.3 噪声注入

```python
import random
import re
from typing import List

class NoiseInjector:
    """文本噪声注入"""

    def __init__(self, noise_prob: float = 0.1):
        self.noise_prob = noise_prob

    def swap_chars(self, text: str) -> str:
        """随机交换相邻字符"""
        chars = list(text)

        for i in range(len(chars) - 1):
            if random.random() < self.noise_prob:
                chars[i], chars[i+1] = chars[i+1], chars[i]

        return ''.join(chars)

    def insert_chars(self, text: str) -> str:
        """随机插入字符"""
        chars = list(text)
        random_chars = 'abcdefghijklmnopqrstuvwxyz'

        i = 0
        while i < len(chars):
            if random.random() < self.noise_prob:
                chars.insert(i, random.choice(random_chars))
            i += 1

        return ''.join(chars)

    def delete_chars(self, text: str) -> str:
        """随机删除字符"""
        chars = list(text)
        filtered = [c for c in chars if random.random() > self.noise_prob]
        return ''.join(filtered)

    def substitute_chars(self, text: str) -> str:
        """随机替换字符"""
        chars = list(text)
        random_chars = 'abcdefghijklmnopqrstuvwxyz'

        for i in range(len(chars)):
            if chars[i].isalpha() and random.random() < self.noise_prob:
                chars[i] = random.choice(random_chars)

        return ''.join(chars)

    def add_typos(self, text: str) -> str:
        """模拟打字错误"""
        # 常见的打字错误映射
        typo_map = {
            'a': 's', 'b': 'v', 'c': 'x', 'd': 's', 'e': 'r',
            'f': 'd', 'g': 'f', 'h': 'g', 'i': 'u', 'j': 'h',
            'k': 'j', 'l': 'k', 'm': 'n', 'n': 'm', 'o': 'p',
            'p': 'o', 'q': 'w', 'r': 't', 's': 'a', 't': 'y',
            'u': 'i', 'v': 'b', 'w': 'q', 'x': 'z', 'y': 't',
            'z': 'x'
        }

        words = text.split()
        noisy_words = []

        for word in words:
            if random.random() < self.noise_prob:
                # 随机选择一个字符进行错误替换
                if len(word) > 1:
                    pos = random.randint(0, len(word) - 1)
                    char = word[pos].lower()

                    if char in typo_map:
                        chars = list(word)
                        chars[pos] = typo_map[char]
                        word = ''.join(chars)

            noisy_words.append(word)

        return ' '.join(noisy_words)

# 使用示例
noise_injector = NoiseInjector(noise_prob=0.05)

text = "This is a clean sentence."
print("原始:", text)
print("交换:", noise_injector.swap_chars(text))
print("插入:", noise_injector.insert_chars(text))
print("删除:", noise_injector.delete_chars(text))
print("替换:", noise_injector.substitute_chars(text))
print("打字错误:", noise_injector.add_typos(text))
```

---

### 1.4 数据集构建

#### 1.4.1 Instruction数据集

**Alpaca格式**

```python
import json
from typing import List, Dict

class AlpacaDatasetBuilder:
    """构建Alpaca格式的指令数据集"""

    def create_instruction_sample(self,
                                  instruction: str,
                                  input_text: str = "",
                                  output: str = "") -> Dict:
        """创建单个样本"""
        return {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }

    def build_qa_dataset(self, qa_pairs: List[tuple]) -> List[Dict]:
        """从Q&A对构建数据集"""
        dataset = []

        for question, answer in qa_pairs:
            sample = self.create_instruction_sample(
                instruction="回答下面的问题。",
                input_text=question,
                output=answer
            )
            dataset.append(sample)

        return dataset

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """保存为JSON格式"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

# 使用示例
builder = AlpacaDatasetBuilder()

qa_pairs = [
    ("什么是机器学习？", "机器学习是人工智能的一个分支..."),
    ("深度学习和机器学习的区别是什么？", "深度学习是机器学习的子集...")
]

dataset = builder.build_qa_dataset(qa_pairs)
builder.save_dataset(dataset, 'alpaca_dataset.json')
```

**ShareGPT格式**

```python
class ShareGPTDatasetBuilder:
    """构建ShareGPT格式的对话数据集"""

    def create_conversation(self, messages: List[Dict[str, str]]) -> Dict:
        """
        创建对话样本

        Args:
            messages: [{"role": "user|assistant", "content": "..."}]
        """
        return {
            "conversations": messages
        }

    def build_multiturn_dataset(self,
                               conversations: List[List[Dict]]) -> List[Dict]:
        """构建多轮对话数据集"""
        dataset = []

        for conv in conversations:
            sample = self.create_conversation(conv)
            dataset.append(sample)

        return dataset

# 使用示例
sharegpt_builder = ShareGPTDatasetBuilder()

conversations = [
    [
        {"role": "user", "content": "你好！"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
        {"role": "user", "content": "介绍一下机器学习"},
        {"role": "assistant", "content": "机器学习是..."}
    ]
]

dataset = sharegpt_builder.build_multiturn_dataset(conversations)
```

#### 1.4.2 偏好数据集（RLHF）

```python
class PreferenceDatasetBuilder:
    """构建偏好数据集（用于RLHF）"""

    def create_preference_sample(self,
                                prompt: str,
                                chosen: str,
                                rejected: str) -> Dict:
        """
        创建偏好样本

        Args:
            prompt: 输入提示
            chosen: 优选回复
            rejected: 劣选回复
        """
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    def build_from_rankings(self,
                           prompt: str,
                           responses: List[str],
                           rankings: List[int]) -> List[Dict]:
        """
        从排名构建配对数据

        Args:
            prompt: 提示
            responses: 回复列表
            rankings: 排名（1=最好）
        """
        pairs = []

        # 创建所有配对
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                if rankings[i] < rankings[j]:  # i 更好
                    pairs.append(self.create_preference_sample(
                        prompt, responses[i], responses[j]
                    ))
                else:  # j 更好
                    pairs.append(self.create_preference_sample(
                        prompt, responses[j], responses[i]
                    ))

        return pairs

# 使用示例
pref_builder = PreferenceDatasetBuilder()

prompt = "解释什么是神经网络"
responses = [
    "神经网络是一种机器学习模型，模仿人脑神经元的工作方式...",
    "就是一堆数学公式",
    "神经网络由多个层组成，包括输入层、隐藏层和输出层..."
]
rankings = [1, 3, 2]  # 1=最好，3=最差

pairs = pref_builder.build_from_rankings(prompt, responses, rankings)
print(f"生成了 {len(pairs)} 个偏好配对")
```

---

## 2. 数据标注最佳实践

### 2.1 标注平台选择

**常用标注平台对比**

| 平台 | 类型 | 优势 | 适用场景 |
|------|------|------|----------|
| **Label Studio** | 开源 | 免费、可定制 | 中小规模项目 |
| **Prodigy** | 商业 | 主动学习、效率高 | 需要快速迭代 |
| **Amazon SageMaker Ground Truth** | 云服务 | 集成AWS、自动化 | 大规模项目 |
| **Scale AI** | 众包 | 高质量、快速 | 需要专业标注 |
| **Labelbox** | 商业 | 协作、版本控制 | 团队协作 |

### 2.2 标注指南设计

**示例：文本分类标注指南**

```markdown
# 情感分类标注指南

## 标注目标
对用户评论进行情感分类：正面、负面、中性

## 标注规则

### 正面（Positive）
- 明确表达满意、喜欢、推荐
- 示例：
  - "这个产品太棒了！" → 正面
  - "强烈推荐给大家" → 正面

### 负面（Negative）
- 明确表达不满、失望、批评
- 示例：
  - "完全浪费钱" → 负面
  - "质量太差了" → 负面

### 中性（Neutral）
- 客观陈述事实，无明显情感倾向
- 疑问句通常为中性
- 示例：
  - "这个产品是蓝色的" → 中性
  - "什么时候发货？" → 中性

## 特殊情况

### 混合情感
- 如果同时包含正面和负面，选择主导情感
- 如果无法判断，标记为"需讨论"

### 讽刺
- "真是太'好'了（坏掉了三次）" → 负面

### 模糊情况
- 拿不准时，标记为"需讨论"，不要猜测

## 质量要求
- 每小时至少完成100条
- 准确率要求：>95%
- 不确定时及时沟通
```

### 2.3 质量控制

**多标注者一致性检查**

```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

class AnnotationQualityControl:
    """标注质量控制"""

    def calculate_agreement(self,
                          annotator1_labels: List[int],
                          annotator2_labels: List[int]) -> float:
        """计算Cohen's Kappa（标注者间一致性）"""
        kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
        return kappa

    def identify_disagreements(self,
                             annotations: Dict[str, List[int]],
                             threshold: float = 0.7) -> List[int]:
        """
        识别标注不一致的样本

        Args:
            annotations: {annotator_id: [labels]}
            threshold: 一致性阈值
        """
        disagreed_indices = []

        # 转换为numpy数组
        labels_matrix = np.array(list(annotations.values()))

        for i in range(labels_matrix.shape[1]):
            labels = labels_matrix[:, i]

            # 计算一致性（最常见标签的比例）
            unique, counts = np.unique(labels, return_counts=True)
            max_count = np.max(counts)
            agreement = max_count / len(labels)

            if agreement < threshold:
                disagreed_indices.append(i)

        return disagreed_indices

    def gold_standard_evaluation(self,
                                annotator_labels: List[int],
                                gold_labels: List[int]) -> Dict:
        """评估标注者准确率（与金标准对比）"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(gold_labels, annotator_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels, annotator_labels, average='weighted'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# 使用示例
qc = AnnotationQualityControl()

# 两个标注者的标签
annotator1 = [0, 1, 1, 0, 2, 1]
annotator2 = [0, 1, 0, 0, 2, 1]

kappa = qc.calculate_agreement(annotator1, annotator2)
print(f"Cohen's Kappa: {kappa:.2f}")
print(f"解释: {'几乎完美' if kappa > 0.8 else '需要改进'}")

# 识别不一致样本
annotations = {
    'annotator1': [0, 1, 1, 0, 2, 1],
    'annotator2': [0, 1, 0, 0, 2, 1],
    'annotator3': [0, 1, 1, 0, 1, 1]
}
disagreements = qc.identify_disagreements(annotations, threshold=0.7)
print(f"不一致样本索引: {disagreements}")
```

### 2.4 主动学习加速标注

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ActiveLearningAnnotator:
    """主动学习辅助标注"""

    def __init__(self, initial_labeled_data, initial_labels):
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(initial_labeled_data, initial_labels)

    def uncertainty_sampling(self, unlabeled_data: np.ndarray,
                           n_samples: int = 10) -> np.ndarray:
        """
        不确定性采样：选择模型最不确定的样本

        返回: 样本索引
        """
        # 获取预测概率
        probas = self.model.predict_proba(unlabeled_data)

        # 计算不确定性（熵）
        entropies = -np.sum(probas * np.log(probas + 1e-10), axis=1)

        # 选择熵最高的样本
        uncertain_indices = np.argsort(entropies)[-n_samples:]

        return uncertain_indices

    def diversity_sampling(self, unlabeled_data: np.ndarray,
                          n_samples: int = 10) -> np.ndarray:
        """
        多样性采样：选择与已有样本最不同的样本

        使用k-means聚类
        """
        from sklearn.cluster import KMeans

        # 聚类
        kmeans = KMeans(n_clusters=n_samples, random_state=42)
        kmeans.fit(unlabeled_data)

        # 选择每个簇的中心最近的样本
        selected_indices = []
        for i in range(n_samples):
            cluster_samples = np.where(kmeans.labels_ == i)[0]
            if len(cluster_samples) > 0:
                # 找到距离簇中心最近的样本
                distances = np.linalg.norm(
                    unlabeled_data[cluster_samples] - kmeans.cluster_centers_[i],
                    axis=1
                )
                closest = cluster_samples[np.argmin(distances)]
                selected_indices.append(closest)

        return np.array(selected_indices)

    def update_model(self, new_data, new_labels):
        """更新模型（增量学习）"""
        # 重新训练（实际应用中可使用增量学习算法）
        self.model.fit(new_data, new_labels)

# 使用示例
# 假设已有100个标注样本
initial_X = np.random.rand(100, 50)
initial_y = np.random.randint(0, 3, 100)

al = ActiveLearningAnnotator(initial_X, initial_y)

# 从1000个未标注样本中选择10个
unlabeled_X = np.random.rand(1000, 50)
uncertain_indices = al.uncertainty_sampling(unlabeled_X, n_samples=10)

print(f"建议标注样本: {uncertain_indices}")
```

---

## 3. 合成数据生成

### 3.1 Self-Instruct

Self-Instruct 使用LLM生成新的指令数据。

```python
from openai import OpenAI
import random

class SelfInstructor:
    """Self-Instruct数据生成"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.seed_instructions = []

    def load_seed_instructions(self, filepath: str):
        """加载种子指令"""
        import json
        with open(filepath, 'r') as f:
            self.seed_instructions = json.load(f)

    def generate_new_instructions(self, num_instructions: int = 10) -> List[str]:
        """生成新的指令"""
        # 随机选择种子指令
        seed_sample = random.sample(self.seed_instructions,
                                   min(5, len(self.seed_instructions)))

        prompt = f"""I gave you a few examples of instructions:

{chr(10).join(f'{i+1}. {inst}' for i, inst in enumerate(seed_sample))}

Please generate {num_instructions} new, diverse instructions that are different from the examples above.
Instructions should be clear and specific tasks.

New instructions:
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        # 解析生成的指令
        new_instructions = response.choices[0].message.content.strip().split('\n')
        new_instructions = [inst.strip('0123456789. ') for inst in new_instructions
                          if inst.strip()]

        return new_instructions

    def generate_response(self, instruction: str, input_text: str = "") -> str:
        """为指令生成回复"""
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def generate_dataset(self, num_samples: int = 100) -> List[Dict]:
        """生成完整数据集"""
        dataset = []

        # 生成指令
        instructions = []
        for _ in range(num_samples // 10):
            batch = self.generate_new_instructions(num_instructions=10)
            instructions.extend(batch)

        # 为每个指令生成回复
        for instruction in instructions[:num_samples]:
            response = self.generate_response(instruction)

            dataset.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })

        return dataset

# 使用示例（需要OpenAI API key）
"""
self_instructor = SelfInstructor(api_key='YOUR_API_KEY')

# 种子指令
seed = [
    "解释机器学习中的过拟合现象",
    "列举三种常见的深度学习优化器",
    "描述Transformer架构的核心组件"
]
self_instructor.seed_instructions = seed

# 生成数据集
dataset = self_instructor.generate_dataset(num_samples=20)
"""
```

### 3.2 Evol-Instruct

Evol-Instruct 通过增加深度和广度来演化指令。

```python
class EvolInstructor:
    """Evol-Instruct数据生成"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

        self.depth_prompts = [
            "Make the instruction more specific and detailed.",
            "Increase the complexity by adding constraints.",
            "Require multi-step reasoning in the response.",
        ]

        self.breadth_prompts = [
            "Rewrite the instruction with a different scenario.",
            "Change the topic while keeping the same format.",
            "Apply the instruction to a new domain.",
        ]

    def evolve_depth(self, instruction: str) -> str:
        """深度演化：增加复杂度"""
        evolution_prompt = random.choice(self.depth_prompts)

        prompt = f"""Original instruction: {instruction}

{evolution_prompt}

Evolved instruction:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def evolve_breadth(self, instruction: str) -> str:
        """广度演化：增加多样性"""
        evolution_prompt = random.choice(self.breadth_prompts)

        prompt = f"""Original instruction: {instruction}

{evolution_prompt}

Evolved instruction:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    def multi_round_evolution(self,
                             seed_instruction: str,
                             num_rounds: int = 3) -> List[str]:
        """多轮演化"""
        evolved = [seed_instruction]

        current = seed_instruction
        for i in range(num_rounds):
            if i % 2 == 0:
                current = self.evolve_depth(current)
            else:
                current = self.evolve_breadth(current)

            evolved.append(current)

        return evolved

# 使用示例
"""
evol = EvolInstructor(api_key='YOUR_API_KEY')

seed = "解释什么是神经网络"
evolved_instructions = evol.multi_round_evolution(seed, num_rounds=3)

for i, inst in enumerate(evolved_instructions):
    print(f"Round {i}: {inst}\n")
"""
```

### 3.3 RLAIF数据生成

使用AI反馈（而非人类反馈）生成偏好数据。

```python
class RLAIFDataGenerator:
    """RLAIF偏好数据生成"""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_responses(self, prompt: str, n: int = 4) -> List[str]:
        """为同一提示生成多个回复"""
        responses = []

        for _ in range(n):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9  # 高温度增加多样性
            )
            responses.append(response.choices[0].message.content)

        return responses

    def rank_responses(self, prompt: str, responses: List[str]) -> List[int]:
        """使用AI对回复进行排名"""
        ranking_prompt = f"""Given the following prompt and responses, rank them from best (1) to worst ({len(responses)}) based on:
- Accuracy
- Helpfulness
- Clarity
- Safety

Prompt: {prompt}

Responses:
{chr(10).join(f'{i+1}. {resp}' for i, resp in enumerate(responses))}

Provide only the ranking as a comma-separated list (e.g., "2,1,4,3"):
"""

        response = self.client.chat.completions.create(
            model="gpt-4",  # 使用更强的模型做评判
            messages=[{"role": "user", "content": ranking_prompt}],
            temperature=0
        )

        # 解析排名
        ranking_str = response.choices[0].message.content.strip()
        rankings = [int(r.strip()) for r in ranking_str.split(',')]

        return rankings

    def generate_preference_pairs(self, prompt: str) -> List[Dict]:
        """生成偏好配对"""
        # 生成多个回复
        responses = self.generate_responses(prompt, n=4)

        # AI排名
        rankings = self.rank_responses(prompt, responses)

        # 创建配对
        pairs = []
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                if rankings[i] < rankings[j]:  # i 更好
                    pairs.append({
                        "prompt": prompt,
                        "chosen": responses[i],
                        "rejected": responses[j]
                    })

        return pairs

# 使用示例
"""
rlaif = RLAIFDataGenerator(api_key='YOUR_API_KEY')

prompt = "解释量子计算的基本原理"
preference_pairs = rlaif.generate_preference_pairs(prompt)

print(f"生成了 {len(preference_pairs)} 个偏好配对")
"""
```

---

## 4. 实战代码示例

### 4.1 完整的数据处理Pipeline

```python
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

class LLMDataPipeline:
    """完整的LLM数据处理流水线"""

    def __init__(self, config: Dict):
        self.config = config

        # 初始化各个组件
        self.heuristic_filter = HeuristicFilter()
        self.pii_remover = PIIRemover()

    def process_raw_data(self, raw_documents: List[str]) -> List[str]:
        """处理原始数据"""
        print("开始数据处理...")

        # 1. 去重
        print("1. 去重...")
        documents = exact_dedup(raw_documents)
        print(f"   去重后: {len(documents)} 个文档")

        # 2. 质量过滤
        print("2. 质量过滤...")
        documents = self.heuristic_filter.filter_documents(documents)
        print(f"   过滤后: {len(documents)} 个文档")

        # 3. PII移除
        print("3. 移除PII...")
        documents = [self.pii_remover.remove_pii(doc) for doc in tqdm(documents)]

        # 4. 最终清洗
        print("4. 最终清洗...")
        documents = [self._final_clean(doc) for doc in documents]

        return documents

    def _final_clean(self, text: str) -> str:
        """最终清洗"""
        # 移除多余空白
        text = ' '.join(text.split())

        # 移除控制字符
        text = ''.join(char for char in text if char.isprintable() or char == '\n')

        return text

    def save_processed_data(self, documents: List[str], output_path: str):
        """保存处理后的数据"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in documents:
                # 每行一个文档（JSONL格式）
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + '\n')

        print(f"保存到: {output_path}")

    def run(self, input_path: str, output_path: str):
        """运行完整流水线"""
        # 加载原始数据
        print(f"加载数据: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_documents = [line.strip() for line in f if line.strip()]

        print(f"原始文档数: {len(raw_documents)}")

        # 处理数据
        processed = self.process_raw_data(raw_documents)

        # 保存结果
        self.save_processed_data(processed, output_path)

        print(f"处理完成！最终文档数: {len(processed)}")

# 使用示例
config = {
    "min_words": 10,
    "max_words": 10000,
    "remove_pii": True,
}

pipeline = LLMDataPipeline(config)
pipeline.run(
    input_path="raw_data.txt",
    output_path="processed/clean_data.jsonl"
)
```

### 4.2 数据质量评估脚本

```python
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class DataQualityAnalyzer:
    """数据质量分析"""

    def __init__(self, documents: List[str]):
        self.documents = documents

    def analyze_length_distribution(self):
        """分析长度分布"""
        lengths = [len(doc.split()) for doc in self.documents]

        stats = {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'p25': np.percentile(lengths, 25),
            'p75': np.percentile(lengths, 75),
            'p95': np.percentile(lengths, 95),
        }

        return stats, lengths

    def analyze_vocabulary(self):
        """分析词汇统计"""
        all_words = []
        for doc in self.documents:
            words = doc.lower().split()
            all_words.extend(words)

        word_counts = Counter(all_words)

        stats = {
            'total_words': len(all_words),
            'unique_words': len(word_counts),
            'vocabulary_richness': len(word_counts) / len(all_words),
            'top_10_words': word_counts.most_common(10)
        }

        return stats

    def detect_duplicates(self):
        """检测重复文档"""
        seen = {}
        duplicates = []

        for i, doc in enumerate(self.documents):
            doc_hash = hash(doc)
            if doc_hash in seen:
                duplicates.append({
                    'index': i,
                    'duplicate_of': seen[doc_hash]
                })
            else:
                seen[doc_hash] = i

        return duplicates

    def generate_report(self) -> str:
        """生成完整报告"""
        report = []
        report.append("=" * 50)
        report.append("数据质量分析报告")
        report.append("=" * 50)

        # 基本统计
        report.append(f"\n文档总数: {len(self.documents)}")

        # 长度分布
        length_stats, lengths = self.analyze_length_distribution()
        report.append("\n长度统计（单词数）:")
        report.append(f"  平均值: {length_stats['mean']:.1f}")
        report.append(f"  中位数: {length_stats['median']:.1f}")
        report.append(f"  标准差: {length_stats['std']:.1f}")
        report.append(f"  范围: [{length_stats['min']}, {length_stats['max']}]")
        report.append(f"  P95: {length_stats['p95']:.1f}")

        # 词汇统计
        vocab_stats = self.analyze_vocabulary()
        report.append("\n词汇统计:")
        report.append(f"  总词数: {vocab_stats['total_words']}")
        report.append(f"  唯一词数: {vocab_stats['unique_words']}")
        report.append(f"  词汇丰富度: {vocab_stats['vocabulary_richness']:.4f}")

        # 重复检测
        duplicates = self.detect_duplicates()
        report.append(f"\n重复文档数: {len(duplicates)}")

        return '\n'.join(report)

    def plot_length_distribution(self, save_path: str = None):
        """绘制长度分布图"""
        _, lengths = self.analyze_length_distribution()

        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, edgecolor='black')
        plt.xlabel('文档长度（单词数）')
        plt.ylabel('频数')
        plt.title('文档长度分布')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

# 使用示例
"""
# 加载数据
with open('processed/clean_data.jsonl', 'r') as f:
    documents = [json.loads(line)['text'] for line in f]

# 分析
analyzer = DataQualityAnalyzer(documents)
report = analyzer.generate_report()
print(report)

# 绘制分布图
analyzer.plot_length_distribution(save_path='length_dist.png')
"""
```

---

## 5. 工具链推荐

### 5.1 数据采集

| 工具 | 用途 | 推荐指数 |
|------|------|----------|
| **Scrapy** | 网页爬取框架 | ⭐⭐⭐⭐⭐ |
| **Beautiful Soup** | HTML解析 | ⭐⭐⭐⭐ |
| **Selenium** | 动态网页爬取 | ⭐⭐⭐ |
| **praw** | Reddit API | ⭐⭐⭐⭐ |
| **PyGithub** | GitHub API | ⭐⭐⭐⭐ |

### 5.2 数据处理

| 工具 | 用途 | 推荐指数 |
|------|------|----------|
| **Apache Spark** | 大规模数据处理 | ⭐⭐⭐⭐⭐ |
| **Dask** | 并行计算（Python） | ⭐⭐⭐⭐ |
| **Pandas** | 数据分析 | ⭐⭐⭐⭐⭐ |
| **datasketch** | MinHash去重 | ⭐⭐⭐⭐ |
| **simhash** | 快速去重 | ⭐⭐⭐ |

### 5.3 数据存储

| 工具 | 用途 | 推荐指数 |
|------|------|----------|
| **Parquet** | 列式存储格式 | ⭐⭐⭐⭐⭐ |
| **Arrow** | 内存格式 | ⭐⭐⭐⭐ |
| **JSONL** | 简单行式存储 | ⭐⭐⭐⭐ |
| **HDF5** | 大规模数据 | ⭐⭐⭐ |

### 5.4 数据标注

| 工具 | 用途 | 推荐指数 |
|------|------|----------|
| **Label Studio** | 开源标注平台 | ⭐⭐⭐⭐⭐ |
| **Prodigy** | 主动学习标注 | ⭐⭐⭐⭐ |
| **Doccano** | 文本标注 | ⭐⭐⭐ |

---

## 6. 常见问题与解决方案

### Q1: 如何处理超大规模数据集（TB级别）？

**解决方案：**

1. **流式处理**：使用 `datasets` 库的 streaming 模式
```python
from datasets import load_dataset

dataset = load_dataset("pile", split='train', streaming=True)

for example in dataset:
    process(example)
    # 处理一个扔一个，不占用大量内存
```

2. **分布式处理**：使用 Apache Spark
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataProcessing").getOrCreate()

df = spark.read.text("hdfs://data/*.txt")
processed = df.rdd.map(process_function)
processed.saveAsTextFile("hdfs://output/")
```

### Q2: 如何平衡数据质量和数据量？

**建议：**

1. **分层过滤**：
   - 第一轮：严格过滤（高质量）
   - 第二轮：宽松过滤（保量）
   - 混合使用，比例9:1

2. **质量评估**：
   - 抽样1000条人工评估
   - 计算质量分数
   - 调整过滤阈值

### Q3: 如何避免数据偏见？

**策略：**

1. **多样性采样**：确保覆盖不同来源、主题、风格
2. **平衡数据集**：检查性别、种族、地域分布
3. **偏见检测**：使用工具检测有害stereotypes

```python
def check_balance(documents, keywords_dict):
    """检查数据平衡性"""
    counts = {category: 0 for category in keywords_dict}

    for doc in documents:
        for category, keywords in keywords_dict.items():
            if any(kw in doc.lower() for kw in keywords):
                counts[category] += 1

    return counts

# 示例
keywords = {
    'science': ['physics', 'chemistry', 'biology'],
    'arts': ['painting', 'music', 'literature'],
    'sports': ['football', 'basketball', 'tennis']
}

balance = check_balance(documents, keywords)
print(balance)
```

---

## 总结

本文档涵盖了LLM数据工程的完整流程：

1. **数据采集**：网页爬取、API获取、开源数据集
2. **数据清洗**：去重、质量过滤、PII移除、有毒内容过滤
3. **数据增强**：回译、改写、噪声注入
4. **数据集构建**：Instruction、对话、偏好数据集
5. **数据标注**：平台选择、质量控制、主动学习
6. **合成数据**：Self-Instruct、Evol-Instruct、RLAIF

**关键要点：**

- 数据质量 > 数据量
- 多样性是关键
- 自动化 + 人工审核
- 持续迭代改进

**下一步学习：**

- [06_Model_Training_and_Finetuning.md](./06_Model_Training_and_Finetuning.md) - 使用数据训练模型
- [14_MLOps_Best_Practices.md](./14_MLOps_Best_Practices.md) - 数据版本管理
- [07_LLM_Evaluation.md](./07_LLM_Evaluation.md) - 评估数据质量
