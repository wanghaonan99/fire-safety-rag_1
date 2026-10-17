# ---------- fire-safety-rag 后端镜像 ----------
# 说明：默认 CPU 版 torch，避免拖入 2GB+ 的 CUDA 依赖
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

# 系统依赖：
#   libgl1 / libglib2.0-0 -> PaddleOCR(opencv) 运行需要
#   libpq5                -> psycopg2 运行需要
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# 数据目录（生产建议挂载 Render Disk 或改用对象存储）
RUN mkdir -p /app/data/uploads /app/data/indexes

EXPOSE 8000

# Render 会注入 $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
