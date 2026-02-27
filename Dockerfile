# 第一阶段：构建阶段
FROM docker.1ms.run/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# 1️⃣ 基础系统依赖（极少）
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*



# 2️⃣ 安装 uv（官方推荐方式）
RUN pip install --no-cache-dir uv

WORKDIR /app

# 3️⃣ 先拷贝依赖文件，最大化缓存命中
COPY pyproject.toml uv.lock ./

# 4️⃣ 安装依赖（不装项目本身）
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 5️⃣ 再拷贝代码
COPY . .

# 暴露 FastAPI 默认端口
EXPOSE 8000
ENV PATH="/app/.venv/bin:$PATH"

# 启动命令：支持 GPustack 部署参数
# 使用方式: python main.py --model-path {model_path} --port {port} --worker-ip {worker_ip} --model-name {model_name}
# 其他配置通过环境变量传入
CMD ["python", "main.py", "--host", "0.0.0.0"]
