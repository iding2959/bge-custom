# BGE-M3 OpenAI-Compatible API

BGE-M3 嵌入模型服务，提供兼容 OpenAI 的 API 端点，同时支持稠密向量和稀疏向量。

## 功能特性

- **Multi-Functionality**: 同时支持稠密向量、稀疏向量、ColBERT 多向量检索
- **Multi-Linguality**: 支持 100+ 语言
- **Multi-Granularity**: 支持从短句到最长 8192 tokens 的文档

## 快速开始

### 1. 启动服务

```bash
cd /home/iding/models/bge-custom
uv run python main.py
```

服务默认运行在 `http://localhost:8101`

### 2. API 端点概览

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 根端点 |
| `/health` | GET | 健康检查 |
| `/v1/embeddings` | POST | OpenAI 兼容（稠密 + 稀疏向量） |
| `/embeddings` | POST | 原生 BGE-M3 端点（稠密 + 稀疏向量） |
| `/score` | POST | 计算相似度分数 |
| `/models` | GET | 列出可用模型 |

---

## API 详细说明

### 健康检查

```bash
curl http://localhost:8101/health
```

响应：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

### OpenAI 兼容端点

#### 1. 稠密向量 (`/v1/embeddings`)

```python
import requests

response = requests.post("http://localhost:8101/v1/embeddings", json={
    "input": "Hello world",
    "model": "bge-m3"
})

print(response.json())
```

响应：
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "bge-m3",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

#### 稠密 + 稀疏向量 (`/v1/embeddings`)

```python
import requests

response = requests.post("http://localhost:8101/v1/embeddings", json={
    "input": "What is BGE M3?",
    "model": "bge-m3"
})

print(response.json())
```

响应：
```json
{
  "object": "list",
  "model": "bge-m3",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "dense": [0.1, 0.2, ...],
      "sparse": {
        "What": 0.08356,
        "is": 0.0814,
        "B": 0.1296,
        "GE": 0.252,
        "M": 0.1702,
        "3": 0.2695,
        "?": 0.04092
      }
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

---

### 原生 BGE-M3 端点

#### 1. 通用嵌入 (`/embeddings`)

```python
import requests

# 默认返回稠密 + 稀疏向量
response = requests.post("http://localhost:8101/embeddings", json={
    "input": "Hello world"
})

# 批量处理
response = requests.post("http://localhost:8101/embeddings", json={
    "input": ["文本1", "文本2", "文本3"]
})
```

响应：
```json
{
  "dense_vecs": [
    [0.1, 0.2, ...],
    [0.3, 0.4, ...]
  ],
  "sparse_vecs": [
    {"Hello": 0.5, "world": 0.3},
    {"文本": 0.8, "1": 0.2}
  ],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

**请求参数说明：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `input` | string or array | 必填 | 输入文本 |
| `input_type.dense` | boolean | true | 返回稠密向量 |
| `input_type.sparse` | boolean | true | 返回稀疏向量 |
| `input_type.colbert` | boolean | false | 返回 ColBERT 向量 |
| `batch_size` | integer | 12 | 批处理大小 |
| `max_length` | integer | 8192 | 最大序列长度 |

#### 2. 计算相似度分数 (`/score`)

```python
import requests

response = requests.post("http://localhost:8101/score", json={
    "sentences_1": ["What is BGE M3?"],
    "sentences_2": ["BGE M3 is an embedding model"],
    "weights": [0.4, 0.2, 0.4]
})

print(response.json())
```

响应：
```json
{
  "dense": [0.6259],
  "sparse": [0.1955],
  "colbert": [0.7796],
  "sparse+dense": [0.4825],
  "colbert+sparse+dense": [0.6013]
}
```

---

### 模型信息

```bash
curl http://localhost:8101/models/bge-m3
```

响应：
```json
{
  "id": "bge-m3",
  "object": "model",
  "embedding_dims": 1024,
  "max_length": 8192,
  "capabilities": {
    "dense": true,
    "sparse": true,
    "colbert": true
  }
}
```

---

## 使用示例

### Python 客户端

```python
import requests
from typing import List, Dict

class BGEClient:
    def __init__(self, base_url: str = "http://localhost:8101"):
        self.base_url = base_url

    def encode(self, texts: List[str], dense: bool = True, sparse: bool = True) -> Dict:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={
                "input": texts,
                "input_type": {
                    "dense": dense,
                    "sparse": sparse,
                    "colbert": False
                }
            }
        )
        return response.json()

# 使用示例
client = BGEClient()
result = client.encode(["Hello world", "BGE M3"])
print(f"稠密向量维度: {len(result['dense_vecs'][0])}")
print(f"稀疏向量: {result['sparse_vecs'][0]}")
```

---

## 配置

| 环境变量 | 默认值 | 描述 |
|----------|--------|------|
| `MODEL_PATH` | `/home/iding/models/bge-m3` | 模型路径 |
| `PORT` | `8101` | 服务端口 |

```bash
PORT=8080 uv run python main.py
```