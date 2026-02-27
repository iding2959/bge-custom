"""Routes - score endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.services import model_service


router = APIRouter(prefix="", tags=["score"])


@router.post("/score")
async def compute_scores(request: Dict) -> Dict[str, Any]:
    """
    Compute similarity scores for text pairs.
    Supports dense, sparse, and colbert scoring.
    """
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sentences_1 = request.get("sentences_1", [])
    sentences_2 = request.get("sentences_2", [])
    weights = request.get("weights", [0.4, 0.2, 0.4])

    if not sentences_1 or not sentences_2:
        raise HTTPException(status_code=400, detail="sentences_1 and sentences_2 are required")

    scores = model_service.compute_score(sentences_1, sentences_2, weights)
    return scores


@router.post("/lexical/match")
async def lexical_match(request: Dict):
    """Compute lexical matching score between sparse vectors."""
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sparse_1 = request.get("sparse_1", {})
    sparse_2 = request.get("sparse_2", {})

    if not sparse_1 or not sparse_2:
        raise HTTPException(status_code=400, detail="sparse_1 and sparse_2 are required")

    score = model_service.compute_lexical_match(sparse_1, sparse_2)
    return {"score": score}


@router.post("/colbert/score")
async def colbert_score(request: Dict):
    """Compute ColBERT score between multi-vector representations."""
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    colbert_1 = request.get("colbert_1")
    colbert_2 = request.get("colbert_2")

    if colbert_1 is None or colbert_2 is None:
        raise HTTPException(status_code=400, detail="colbert_1 and colbert_2 are required")

    score = model_service.colbert_score(colbert_1, colbert_2)
    return {"score": score}