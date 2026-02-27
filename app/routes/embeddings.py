"""Routes - embeddings endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional, Dict

from app.services import model_service


# ============== Request/Response Models ==============

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: Union[str, List[str]]
    model: str = "bge-m3"
    encoding_format: str = "float"
    dimensions: Optional[int] = None
    timeout: Optional[float] = None
    batch_size: int = 12
    max_length: int = 8192


class DenseSparseEmbeddingObject(BaseModel):
    """Embedding object with dense and sparse vectors."""
    object: str = "embedding"
    index: int
    dense: List[float]
    sparse: Dict[str, float]


class DenseSparseEmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response with dense and sparse vectors."""
    object: str = "list"
    model: str
    data: List[DenseSparseEmbeddingObject]
    usage: UsageInfo


class BGEEmbeddingRequest(BaseModel):
    """BGE-M3 native embedding request."""
    input: Union[str, List[str]]
    input_type: Optional[Dict[str, bool]] = None
    batch_size: int = 12
    max_length: int = 8192


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


@router.post("/v1/embeddings", response_model=DenseSparseEmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Returns both dense and sparse vectors.
    """
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # Encode with both dense and sparse
    output = model_service.encode(
        texts,
        batch_size=len(texts) if len(texts) < request.batch_size else request.batch_size,
        max_length=request.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert=False
    )

    dense_vecs = output["dense_vecs"]
    sparse_vecs = output["lexical_weights"]

    # Build response
    embeddings = []
    for i, (dense_vec, sparse_vec) in enumerate(zip(dense_vecs, sparse_vecs)):
        dense_list = dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec
        embeddings.append(
            DenseSparseEmbeddingObject(
                dense=dense_list,
                sparse=sparse_vec,
                index=i
            )
        )

    prompt_tokens = compute_token_count(texts)

    return DenseSparseEmbeddingResponse(
        data=embeddings,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
    )


@router.post("/embeddings", response_model=BGEEmbeddingResponse)
async def create_native_embeddings(request: BGEEmbeddingRequest):
    """
    Native BGE-M3 embedding endpoint.
    Supports dense, sparse, and colbert vectors.
    """
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # Determine output types
    if request.input_type is None:
        input_types = {"dense": True, "sparse": True, "colbert": False}
    else:
        input_types = request.input_type

    # Encode
    output = model_service.encode(
        texts,
        batch_size=request.batch_size,
        max_length=request.max_length,
        return_dense=input_types.get("dense", True),
        return_sparse=input_types.get("sparse", True),
        return_colbert=input_types.get("colbert", False)
    )

    # Build response
    response = {}

    if input_types.get("dense", True):
        dense_vecs = output["dense_vecs"]
        response["dense_vecs"] = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in dense_vecs]

    if input_types.get("sparse", True):
        response["sparse_vecs"] = output["lexical_weights"]

    if input_types.get("colbert", False):
        response["colbert_vecs"] = output["colbert_vecs"]

    prompt_tokens = compute_token_count(texts, request.max_length)
    response["usage"] = {
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens
    }

    return response