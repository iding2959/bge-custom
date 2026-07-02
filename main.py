"""BGE-M3 API Server - Entry Point."""

import os
os.environ["TQDM_DISABLE"] = "1"

# 在生产环境彻底禁用 tqdm 进度条输出
def _patch_tqdm():
    import tqdm
    _real_init = tqdm.tqdm.__init__
    def _noop_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _real_init(self, *args, **kwargs)
    tqdm.tqdm.__init__ = _noop_init

_patch_tqdm()

import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services import model_service
from app.routes import embeddings, score, models
from app.log import setup_logging


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
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent inference requests (overrides MAX_CONCURRENT env var, default: 4)"
    )
    return parser.parse_args()


args = parse_args()

# 配置（CLI 参数 > 环境变量 > 默认值）
MODEL_PATH = args.model_path or os.getenv("MODEL_PATH", "/mnt/models/bge-m3")
PORT = args.port or int(os.getenv("PORT", "8101"))
HOST = args.host if not args.worker_ip else args.worker_ip
MODEL_NAME = args.model_name
MAX_CONCURRENT = args.max_concurrent or int(os.getenv("MAX_CONCURRENT", "4"))

# 初始化日志系统
setup_logging()


# ============== FastAPI 应用 ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时卸载模型。"""
    model_service.load_model(MODEL_PATH)
    model_service.init_executor(max_workers=MAX_CONCURRENT)
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

# 请求日志中间件：记录所有非排除路径的请求/响应信息
from sololib.middleware import RequestLogMiddleware

app.add_middleware(
    RequestLogMiddleware,
    exclude_paths={
        "/api/fk/content/health",
        "/actuator/health",
        "/actuator/info",
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/favicon.ico",

    },
    exclude_path_prefixes={
        "/api/fk/content/admin",
        "/.well-known",
        "/dev-server",
    },
    log_request_body=True,
    log_response_body=True,
)

# 注册路由
app.include_router(embeddings.router)
app.include_router(score.router)
app.include_router(models.router)


# ============== 入口 ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)