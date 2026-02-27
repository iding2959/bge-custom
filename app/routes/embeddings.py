"""Routes - embeddings endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any

from app.services import model_service


# ============== Request Models ==============

class EmbeddingRequest(BaseModel):
    """Embedding request (unified for both endpoints)."""
    input: Union[str, List[str]]
    model: str = "bge-m3"
    batch_size: int = 12
    max_length: int = 8192
    input_type: Optional[Dict[str, bool]] = None  # Only used by native endpoint


# ============== Response Models ==============

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    """Embedding data with dense and sparse vectors."""
    index: int
    dense: List[float]
    sparse: Dict[str, float]


# OpenAI-compatible response format
class OpenAIEmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    model: str
    data: List[EmbeddingData]
    usage: UsageInfo


# Native BGE-M3 response format
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
    """Core encoding logic shared by both endpoints."""
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Determine output types
    if input_type is None:
        input_types = {"dense": True, "sparse": True, "colbert": False}
    else:
        input_types = input_type

    # Encode
    output = model_service.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=input_types.get("dense", True),
        return_sparse=input_types.get("sparse", True),
        return_colbert=input_types.get("colbert", False)
    )

    # Convert dense vectors to list
    dense_vecs = output["dense_vecs"]
    dense_list = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in dense_vecs]

    # Get sparse vectors
    sparse_vecs = output["lexical_weights"]

    return dense_list, sparse_vecs, output.get("colbert_vecs")


@router.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def create_openai_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Returns both dense and sparse vectors.
    """
    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    batch_size = min(len(texts), request.batch_size) if len(texts) < request.batch_size else request.batch_size
    dense_vecs, sparse_vecs, _ = await _encode_texts(texts, batch_size, request.max_length, request.input_type)

    # Build OpenAI-compatible response
    embeddings = []
    for i, (dense_vec, sparse_vec) in enumerate(zip(dense_vecs, sparse_vecs)):
        embeddings.append(
            EmbeddingData(
                dense=dense_vec,
                sparse=sparse_vec,
                index=i
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
    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    dense_vecs, sparse_vecs, colbert_vecs = await _encode_texts(texts, request.batch_size, request.max_length, request.input_type)

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