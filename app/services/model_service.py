"""Model service - handles BGE-M3 model loading and encoding."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any

# 全局模型实例
model: Optional[BGEM3FlagModel] = None

# 用于异步推理的线程池
_executor: Optional[ThreadPoolExecutor] = None
_semaphore: Optional[asyncio.Semaphore] = None


def init_executor(max_workers: int = None):
    """初始化线程池和信号量，用于并发推理。"""
    global _executor, _semaphore
    max_workers = max_workers or int(os.getenv("MAX_CONCURRENT", "4"))
    _executor = ThreadPoolExecutor(max_workers=max_workers)
    _semaphore = asyncio.Semaphore(max_workers)
    print(f"线程池已初始化，工作线程数: {max_workers}")


def shutdown_executor():
    """关闭线程池。"""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)
        print("线程池已关闭")


def get_semaphore() -> Optional[asyncio.Semaphore]:
    """获取信号量，用于控制并发推理数量。"""
    return _semaphore


async def encode_async(
    texts: List[str],
    batch_size: int = 12,
    max_length: int = 8192,
    return_dense: bool = True,
    return_sparse: bool = True,
    return_colbert: bool = False
) -> Dict[str, Any]:
    """
    异步编码，使用信号量控制并发。

    Args:
        texts: 待编码文本列表
        batch_size: 批处理大小
        max_length: 最大序列长度
        return_dense: 返回稠密向量
        return_sparse: 返回稀疏向量
        return_colbert: 返回 ColBERT 向量

    Returns:
        包含 dense_vecs, lexical_weights, colbert_vecs 的字典
    """
    semaphore = get_semaphore()
    if semaphore is None:
        # 如果信号量未初始化，直接同步执行
        return encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert
        )

    async with semaphore:
        return await asyncio.to_thread(
            encode,
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert
        )


async def compute_score_async(
    sentences_1: List[str],
    sentences_2: List[str],
    weights: List[float] = None
) -> Dict[str, Any]:
    """异步计算相似度分数，使用信号量控制并发。"""
    semaphore = get_semaphore()
    if semaphore is None:
        return compute_score(sentences_1, sentences_2, weights)

    async with semaphore:
        return await asyncio.to_thread(
            compute_score,
            sentences_1,
            sentences_2,
            weights
        )


async def compute_lexical_match_async(sparse_1: Dict, sparse_2: Dict) -> float:
    """异步计算词法匹配分数，使用信号量控制并发。"""
    semaphore = get_semaphore()
    if semaphore is None:
        return compute_lexical_match(sparse_1, sparse_2)

    async with semaphore:
        return await asyncio.to_thread(
            compute_lexical_match,
            sparse_1,
            sparse_2
        )


async def colbert_score_async(colbert_1: Any, colbert_2: Any) -> float:
    """异步计算 ColBERT 分数，使用信号量控制并发。"""
    semaphore = get_semaphore()
    if semaphore is None:
        return colbert_score(colbert_1, colbert_2)

    async with semaphore:
        return await asyncio.to_thread(
            colbert_score,
            colbert_1,
            colbert_2
        )


def get_model() -> Optional[BGEM3FlagModel]:
    """Get the global model instance."""
    return model


def load_model(model_path: str) -> BGEM3FlagModel:
    """Load BGE-M3 model."""
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BGE-M3 model from {model_path}...")
    print(f"Using device: {device.upper()}" + (" (FP16 enabled)" if device == "cuda" else ""))
    model = BGEM3FlagModel(
        model_path,
        use_fp16=True,
        device=device
    )
    print("Model loaded successfully!")
    return model


def unload_model():
    """Unload model and free resources."""
    global model
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model unloaded.")


def encode(
    texts: List[str],
    batch_size: int = 12,
    max_length: int = 8192,
    return_dense: bool = True,
    return_sparse: bool = True,
    return_colbert: bool = False
) -> Dict[str, Any]:
    """
    Encode texts using BGE-M3 model.

    Args:
        texts: List of texts to encode
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        return_dense: Return dense vectors
        return_sparse: Return sparse vectors
        return_colbert: Return ColBERT vectors

    Returns:
        Dictionary with dense_vecs, lexical_weights, colbert_vecs
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert_vecs=return_colbert
    )
    return output


def compute_score(
    sentences_1: List[str],
    sentences_2: List[str],
    weights: List[float] = None
) -> Dict[str, Any]:
    """
    Compute similarity scores for text pairs.

    Args:
        sentences_1: First list of sentences
        sentences_2: Second list of sentences
        weights: [dense, sparse, colbert] weights

    Returns:
        Score results
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    if weights is None:
        weights = [0.4, 0.2, 0.4]

    sentence_pairs = [[i, j] for i in sentences_1 for j in sentences_2]

    scores = model.compute_score(
        sentence_pairs,
        max_passage_length=128,
        weights_for_different_modes=weights
    )
    return scores


def compute_lexical_match(sparse_1: Dict, sparse_2: Dict) -> float:
    """Compute lexical matching score between sparse vectors."""
    if model is None:
        raise RuntimeError("Model not loaded")
    return model.compute_lexical_matching_score(sparse_1, sparse_2)


def colbert_score(colbert_1: Any, colbert_2: Any) -> float:
    """Compute ColBERT score between multi-vector representations."""
    if model is None:
        raise RuntimeError("Model not loaded")
    return model.colbert_score(colbert_1, colbert_2)


def get_device() -> str:
    """Get current device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"