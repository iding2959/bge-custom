"""Routes - embeddings endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any

from app.services import model_service

# ============== 默认配置常量 ==============
# 默认输出类型：dense + sparse，不含 colbert
DEFAULT_INPUT_TYPES = {"dense": True, "sparse": True, "colbert": False}


# ============== 请求模型 ==============

class EmbeddingRequest(BaseModel):
    """Embedding request (unified for both endpoints)."""
    input: Union[str, List[str]]
    model: str = "bge-m3"
    batch_size: int = 12
    max_length: int = 8192
    input_type: Optional[Dict[str, bool]] = None  # 仅原生端点使用


# ============== 响应模型 ==============

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    """Embedding data with dense and sparse vectors."""
    index: int
    dense: List[float]
    sparse: Dict[str, float]


# OpenAI 兼容响应格式
class OpenAIEmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    model: str
    data: List[EmbeddingData]
    usage: UsageInfo


# BGE-M3 原生响应格式
class BGEEmbeddingResponse(BaseModel):
    """BGE-M3 native embedding response."""
    dense_vecs: Optional[List[List[float]]] = None
    sparse_vecs: Optional[List[Dict[str, float]]] = None
    colbert_vecs: Optional[Any] = None
    usage: UsageInfo


def compute_token_count(texts: List[str], max_length: int = 8192) -> int:
    """Estimate token count (rough approximation)."""
    total_chars = sum(len(t) for t in texts)
    return total_chars // 4


router = APIRouter(prefix="", tags=["embeddings"])


async def _encode_texts(texts: List[str], batch_size: int, max_length: int, input_type: Optional[Dict[str, bool]] = None):
    """
    两个端点共用的核心编码逻辑。

    Returns:
        (dense_list, sparse_vecs, colbert_vecs, original_indices)
        - original_indices: 过滤空字符串后保留的输入在原列表中的索引，
          用于保证响应 index 字段与输入位置一一对应。
    """
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 过滤空字符串，防止 tokenizer.pad() 在旧版 transformers 上崩溃
    # 同时记录原始索引，确保响应 index 与输入位置对应
    indexed_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not indexed_texts:
        raise HTTPException(status_code=400, detail="Input must not be empty or contain only whitespace")
    original_indices = [i for i, _ in indexed_texts]
    filtered_texts = [t for _, t in indexed_texts]

    # 确定输出类型
    if input_type is None:
        input_types = DEFAULT_INPUT_TYPES
    else:
        input_types = input_type

    # 使用异步封装函数（包含信号量控制）
    output = await model_service.encode_async(
        filtered_texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=input_types.get("dense", True),
        return_sparse=input_types.get("sparse", True),
        return_colbert=input_types.get("colbert", False)
    )

    # 将 dense 向量转换为列表
    dense_vecs = output["dense_vecs"]
    dense_list = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in dense_vecs]

    # 获取稀疏向量
    sparse_vecs = output["lexical_weights"]

    # 将 colbert 向量转换为可序列化的列表
    colbert_vecs = output.get("colbert_vecs")
    if colbert_vecs is not None:
        colbert_vecs = [
            v.tolist() if hasattr(v, 'tolist') else v
            for v in colbert_vecs
        ]

    return dense_list, sparse_vecs, colbert_vecs, original_indices


@router.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def create_openai_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Returns both dense and sparse vectors.
    """
    # 将输入规范化为列表
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    batch_size = min(len(texts), request.batch_size)
    # OpenAI 兼容端点始终返回 dense + sparse，忽略 input_type
    dense_vecs, sparse_vecs, _, original_indices = await _encode_texts(
        texts, batch_size, request.max_length, None
    )

    # 构建 OpenAI 兼容响应，index 对应原始输入位置
    embeddings = []
    for pos, dense_vec in enumerate(dense_vecs):
        embeddings.append(
            EmbeddingData(
                dense=dense_vec,
                sparse=sparse_vecs[pos] if sparse_vecs is not None else {},
                index=original_indices[pos]
            )
        )

    prompt_tokens = compute_token_count(texts)

    return OpenAIEmbeddingResponse(
        data=embeddings,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
    )


@router.post("/embeddings", response_model=BGEEmbeddingResponse)
async def create_native_embeddings(request: EmbeddingRequest):
    """
    Native BGE-M3 embedding endpoint.
    Returns both dense and sparse vectors.
    """
    # 将输入规范化为列表
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    dense_vecs, sparse_vecs, colbert_vecs, _ = await _encode_texts(
        texts, request.batch_size, request.max_length, request.input_type
    )

    prompt_tokens = compute_token_count(texts, request.max_length)

    return BGEEmbeddingResponse(
        dense_vecs=dense_vecs,
        sparse_vecs=sparse_vecs,
        colbert_vecs=colbert_vecs,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
    )