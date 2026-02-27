# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本项目中工作时提供指导。

## 项目概述

BGE-M3 是一个基于 FastAPI 的嵌入模型服务，提供兼容 OpenAI 的 API 端点，用于 BAAI/bge-m3 嵌入模型。支持稠密向量、稀疏向量（词法权重）和 ColBERT 多向量表示。

## 常用命令

```bash
# 启动服务
uv run python main.py

# 或直接使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8101

# 自定义端口启动
PORT=8080 uv run python main.py

# 运行测试（需要安装 dev 依赖）
pytest
```

## 架构

应用为单文件 FastAPI 服务 (`main.py`)：

- **模型加载**：使用 `FlagEmbedding.BGEM3FlagModel`，通过 FastAPI lifespan 上下文管理器在启动时加载
- **设备选择**：自动检测 CUDA，可用时优先使用，否则回退到 CPU
- **默认模型路径**：`/home/iding/models/bge-m3`（可通过 `MODEL_PATH` 环境变量配置）

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 根端点，返回服务能力信息 |
| `/health` | GET | 健康检查（模型加载状态、设备信息） |
| `/v1/embeddings` | POST | OpenAI 兼容（返回 dense + sparse） |
| `/embeddings` | POST | BGE-M3 原生端点（支持 dense/sparse/colbert 配置） |
| `/score` | POST | 文本对相似度计算 |
| `/models` | GET | 列出可用模型 |
| `/models/bge-m3` | GET | 模型信息（1024 维，最大 8192 长度） |
| `/lexical/match` | POST | 稀疏向量间的词法匹配 |
| `/colbert/score` | POST | 多向量 ColBERT 评分 |

## 配置

环境变量：
- `MODEL_PATH` - BGE-M3 模型路径（默认：`/home/iding/models/bge-m3`）
- `PORT` - 服务端口（默认：`8101`）

## 关键实现细节

- 模型在启动时一次性加载并保存在内存中（全局 `model` 变量）
- 使用 `use_fp16=True` 加速 GPU 推理
- 启用全源 CORS
- Token 数量采用估算方式（约 4 字符 = 1 token）