"""Routes - score endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.services import model_service


router = APIRouter(prefix="", tags=["score"])


@router.post("/score")
async def compute_scores(request: Dict) -> Dict[str, Any]:
    """
    计算文本对的相似度分数。
    支持 dense、sparse 和 colbert 评分方式。
    """
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sentences_1 = request.get("sentences_1", [])
    sentences_2 = request.get("sentences_2", [])
    weights = request.get("weights", [0.4, 0.2, 0.4])

    if not sentences_1 or not sentences_2:
        raise HTTPException(status_code=400, detail="sentences_1 and sentences_2 are required")

    scores = await model_service.compute_score_async(sentences_1, sentences_2, weights)
    return scores


@router.post("/lexical/match")
async def lexical_match(request: Dict):
    """计算稀疏向量之间的词法匹配分数。"""
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    sparse_1 = request.get("sparse_1", {})
    sparse_2 = request.get("sparse_2", {})

    if not sparse_1 or not sparse_2:
        raise HTTPException(status_code=400, detail="sparse_1 and sparse_2 are required")

    score = await model_service.compute_lexical_match_async(sparse_1, sparse_2)
    return {"score": score}


@router.post("/colbert/score")
async def colbert_score(request: Dict):
    """计算多向量 ColBERT 表示之间的分数。"""
    if model_service.get_model() is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    colbert_1 = request.get("colbert_1")
    colbert_2 = request.get("colbert_2")

    if colbert_1 is None or colbert_2 is None:
        raise HTTPException(status_code=400, detail="colbert_1 and colbert_2 are required")

    score = await model_service.colbert_score_async(colbert_1, colbert_2)
    return {"score": score}