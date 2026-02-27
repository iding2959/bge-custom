"""Model service - handles BGE-M3 model loading and encoding."""

import torch
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any, Optional

# Global model instance
model: Optional[BGEM3FlagModel] = None


def get_model() -> Optional[BGEM3FlagModel]:
    """Get the global model instance."""
    return model


def load_model(model_path: str) -> BGEM3FlagModel:
    """Load BGE-M3 model."""
    global model
    print(f"Loading BGE-M3 model from {model_path}...")
    model = BGEM3FlagModel(
        model_path,
        use_fp16=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
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