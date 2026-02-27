import torch
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import argparse
from contextlib import asynccontextmanager


# ============== CLI Arguments (for GPustack deployment) ==============
def parse_args():
    """Parse command line arguments for GPustack deployment."""
    parser = argparse.ArgumentParser(description="BGE-M3 API Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to BGE-M3 model (overrides MODEL_PATH env var)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Service port (overrides PORT env var)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Service host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--worker-ip",
        type=str,
        default=None,
        help="Worker IP address for binding"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bge-m3",
        help="Model name (default: bge-m3)"
    )
    return parser.parse_args()


args = parse_args()

# Model path configuration (CLI args > env var > default)
MODEL_PATH = args.model_path or os.getenv("MODEL_PATH", "/home/iding/models/bge-m3")
PORT = args.port or int(os.getenv("PORT", "8101"))
HOST = args.host if not args.worker_ip else args.worker_ip
MODEL_NAME = args.model_name

# Global model instance
model: BGEM3FlagModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    global model
    print(f"Loading BGE-M3 model from {MODEL_PATH}...")
    model = BGEM3FlagModel(
        MODEL_PATH,
        use_fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Model loaded successfully!")
    yield
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded.")


app = FastAPI(
    title="BGE-M3 OpenAI-Compatible API",
    description="BGE-M3 embedding model with OpenAI-compatible endpoints. Returns dense and sparse vectors.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Request/Response Models ==============

class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: Union[str, List[str]]
    model: str = "bge-m3"
    encoding_format: str = "float"  # "float" or "base64"
    dimensions: Optional[int] = None  # Not used for BGE-M3 (fixed 1024)
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


class SparseVector(BaseModel):
    """Sparse vector representation (token weights)."""
    index: int
    token: str
    weight: float


class BGEInputType(str):
    """Input type for BGE-M3."""
    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"


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
    colbert_vecs: Optional[Any] = None  # Complex nested structure
    usage: UsageInfo


# ============== Utility Functions ==============

def convert_sparse_to_list(sparse_weights: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Convert sparse weights to list format."""
    return sparse_weights


def compute_token_count(texts: List[str], max_length: int = 8192) -> int:
    """Estimate token count (rough approximation)."""
    # Rough estimate: ~4 characters per token
    total_chars = sum(len(t) for t in texts)
    return total_chars // 4


# ============== OpenAI-Compatible Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BGE-M3 OpenAI-Compatible API",
        "model": f"BAAI/{MODEL_NAME}",
        "endpoints": {
            "/v1/embeddings": "OpenAI-compatible embeddings endpoint (dense only)",
            "/embeddings": "Native BGE-M3 embeddings endpoint"
        },
        "capabilities": {
            "dense": True,
            "sparse": True,
            "colbert": True
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/v1/embeddings", response_model=DenseSparseEmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Returns both dense and sparse vectors.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # Encode with both dense and sparse
    output = model.encode(
        texts,
        batch_size=len(texts) if len(texts) < request.batch_size else request.batch_size,
        max_length=request.max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False
    )

    dense_vecs = output["dense_vecs"]
    sparse_vecs = output["lexical_weights"]

    # Build response
    embeddings = []
    for i, (dense_vec, sparse_vec) in enumerate(zip(dense_vecs, sparse_vecs)):
        # Convert dense to list if needed
        dense_list = dense_vec.tolist() if hasattr(dense_vec, 'tolist') else dense_vec

        embeddings.append(
            DenseSparseEmbeddingObject(
                dense=dense_list,
                sparse=sparse_vec,
                index=i
            )
        )

    # Compute usage (rough estimate)
    prompt_tokens = compute_token_count(texts)

    return DenseSparseEmbeddingResponse(
        data=embeddings,
        model=request.model,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens
        )
    )



# ============== Native BGE-M3 Endpoints ==============

@app.post("/embeddings", response_model=BGEEmbeddingResponse)
async def create_native_embeddings(request: BGEEmbeddingRequest):
    """
    Native BGE-M3 embedding endpoint.
    Supports dense, sparse, and colbert vectors.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    # Determine output types (use defaults if not provided)
    if request.input_type is None:
        input_types = {"dense": True, "sparse": True, "colbert": False}
    else:
        input_types = request.input_type

    # Encode
    output = model.encode(
        texts,
        batch_size=request.batch_size,
        max_length=request.max_length,
        return_dense=input_types.get("dense", True),
        return_sparse=input_types.get("sparse", True),
        return_colbert_vecs=input_types.get("colbert", False)
    )

    # Build response
    response = {}

    if input_types.get("dense", True):
        dense_vecs = output["dense_vecs"]
        # Convert numpy arrays to lists
        response["dense_vecs"] = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in dense_vecs]

    if input_types.get("sparse", True):
        response["sparse_vecs"] = output["lexical_weights"]

    if input_types.get("colbert", False):
        response["colbert_vecs"] = output["colbert_vecs"]

    # Compute usage
    prompt_tokens = compute_token_count(texts, request.max_length)
    response["usage"] = {
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens
    }

    return response


@app.post("/score")
async def compute_scores(request: Dict):
    """
    Compute similarity scores for text pairs.
    Supports dense, sparse, and colbert scoring.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sentences_1 = request.get("sentences_1", [])
    sentences_2 = request.get("sentences_2", [])
    weights = request.get("weights", [0.4, 0.2, 0.4])  # [dense, sparse, colbert]

    if not sentences_1 or not sentences_2:
        raise HTTPException(status_code=400, detail="sentences_1 and sentences_2 are required")

    sentence_pairs = [[i, j] for i in sentences_1 for j in sentences_2]

    scores = model.compute_score(
        sentence_pairs,
        max_passage_length=128,
        weights_for_different_modes=weights
    )

    return scores


@app.post("/lexical/match")
async def lexical_match(request: Dict):
    """
    Compute lexical matching score between sparse vectors.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sparse_1 = request.get("sparse_1", {})
    sparse_2 = request.get("sparse_2", {})

    if not sparse_1 or not sparse_2:
        raise HTTPException(status_code=400, detail="sparse_1 and sparse_2 are required")

    score = model.compute_lexical_matching_score(sparse_1, sparse_2)

    return {"score": score}


@app.post("/colbert/score")
async def colbert_score(request: Dict):
    """
    Compute ColBERT score between multi-vector representations.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    colbert_1 = request.get("colbert_1")
    colbert_2 = request.get("colbert_2")

    if colbert_1 is None or colbert_2 is None:
        raise HTTPException(status_code=400, detail="colbert_1 and colbert_2 are required")

    score = model.colbert_score(colbert_1, colbert_2)

    return {"score": score}


# ============== Model Info Endpoints ==============

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1704067200,  # Approximate
                "owned_by": "BAAI",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None,
                "embedding_dims": 1024,
                "capabilities": {
                    "dense": True,
                    "sparse": True,
                    "colbert": True
                }
            }
        ]
    }


@app.get("/models/{model_id}")
async def model_info(model_id: str):
    """Get BGE-M3 model information."""
    # Only respond to the actual model name
    if model_id != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": MODEL_NAME,
        "object": "model",
        "created": 1704067200,
        "owned_by": "BAAI",
        "embedding_dims": 1024,
        "max_length": 8192,
        "capabilities": {
            "dense": True,
            "sparse": True,
            "colbert": True
        },
        "description": "BGE-M3: Multi-Functionality, Multi-Linguality, and Multi-Granularity embedding model"
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)