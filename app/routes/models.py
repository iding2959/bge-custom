"""Routes - model info endpoints."""

from fastapi import APIRouter, HTTPException

from app.services import model_service


router = APIRouter(prefix="", tags=["models"])


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BGE-M3 OpenAI-Compatible API",
        "model": f"BAAI/{model_service.get_model().__class__.__name__}",
        "endpoints": {
            "/v1/embeddings": "OpenAI-compatible embeddings endpoint",
            "/embeddings": "Native BGE-M3 embeddings endpoint"
        },
        "capabilities": {
            "dense": True,
            "sparse": True,
            "colbert": True
        }
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_service.get_model() is not None,
        "device": model_service.get_device()
    }


@router.get("/models")
async def list_models(model_name: str = "bge-m3"):
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1704067200,
                "owned_by": "BAAI",
                "permission": [],
                "root": model_name,
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


@router.get("/models/{model_id}")
async def model_info(model_id: str, model_name: str = "bge-m3"):
    """Get model information."""
    if model_id != model_name:
        raise HTTPException(status_code=404, detail="Model not found")

    return {
        "id": model_name,
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