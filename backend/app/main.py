"""FastAPI 入口：装配配置、CORS、路由、健康检查与生命周期钩子。

本地运行：
    uvicorn app.main:app --reload --port 8000
文档：
    http://localhost:8000/docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_admin, routes_eval, routes_ingest, routes_query
from app.config import settings
from app.models.response import HealthResponse
from app.utils.logging import get_logger, setup_logging

setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭钩子。预热动作都放在这里，且失败不阻断启动。"""
    logger.info("启动 %s（env=%s, debug=%s）", settings.app_name, settings.environment, settings.debug)

    # TODO(第二批): 初始化 LangFuse 回调处理器，注册到 LlamaIndex Settings.callback_manager
    if settings.langfuse_enabled:
        logger.info("LangFuse 已启用，host=%s", settings.langfuse_host)

    # 开发环境自动建表；生产环境请用 `alembic upgrade head`
    if not settings.is_prod:
        try:
            from app.db.session import init_db

            init_db()
        except Exception as exc:  # 数据库没配好也应能起来看 /docs
            logger.warning("自动建表失败（可稍后手动执行 Alembic）：%s", exc)

    yield

    logger.info("关闭 %s", settings.app_name)


app = FastAPI(
    title="工厂消防知识问答助手 API",
    description="基于 LlamaIndex + pgvector + DeepSeek 的消防法规/预案/培训资料 RAG 服务",
    version="0.2.0",
    lifespan=lifespan,
)

# ---------- CORS：允许 Vercel 前端跨域 ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求耗时日志 ----------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ---------- 全局异常兜底：永远返回结构化 JSON ----------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": f"内部错误: {exc}", "data": None},
    )


# ---------- 路由挂载 ----------
app.include_router(routes_ingest.router, prefix=settings.api_prefix)
app.include_router(routes_query.router, prefix=settings.api_prefix)
app.include_router(routes_eval.router, prefix=settings.api_prefix)
app.include_router(routes_admin.router, prefix=settings.api_prefix)


@app.get("/", tags=["meta"], summary="根路径")
def root() -> dict:
    return {"app": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="健康检查")
def health() -> HealthResponse:
    """Render / 前端用的健康检查；逐项探测依赖，单项失败不影响整体 200。"""
    from app.db.session import ping_db
    from app.utils.cache import cache_available, cache_backend

    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
        database=ping_db(),
        redis=cache_available(),
        cache_backend=cache_backend(),
        llm_configured=bool(settings.deepseek_api_key),
    )
