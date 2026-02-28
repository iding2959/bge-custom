"""BGE-M3 API Server - Entry Point."""

import os
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services import model_service
from app.routes import embeddings, score, models


# ============== CLI 参数（用于 GPustack 部署） ==============
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

# 配置（CLI 参数 > 环境变量 > 默认值）
MODEL_PATH = args.model_path or os.getenv("MODEL_PATH", "/home/iding/models/bge-m3")
PORT = args.port or int(os.getenv("PORT", "8101"))
HOST = args.host if not args.worker_ip else args.worker_ip
MODEL_NAME = args.model_name


# ============== FastAPI 应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, unload on shutdown."""
    model_service.load_model(MODEL_PATH)
    model_service.init_executor()
    yield
    model_service.shutdown_executor()
    model_service.unload_model()


app = FastAPI(
    title="BGE-M3 OpenAI-Compatible API",
    description="BGE-M3 embedding model with OpenAI-compatible endpoints.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(embeddings.router)
app.include_router(score.router)
app.include_router(models.router)


# ============== 入口 ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)