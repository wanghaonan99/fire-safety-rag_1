"""连通性体检：从当前网络实测各依赖服务是否可达（只读，不改任何东西）。"""

from __future__ import annotations

import socket
import ssl
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

DB = urlsplit(settings.database_url)


def tcp(host: str, port: int, timeout: float = 8.0) -> str:
    t = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return f"✓ 可达   TCP 建连 {time.perf_counter() - t:.2f}s"
    except Exception as exc:  # noqa: BLE001
        return f"✗ 不可达 {type(exc).__name__}: {str(exc)[:60]}"


def https(host: str, path: str = "/", timeout: float = 12.0) -> str:
    import httpx

    t = time.perf_counter()
    try:
        r = httpx.get(f"https://{host}{path}", timeout=timeout, follow_redirects=True)
        return f"✓ HTTP {r.status_code}   耗时 {time.perf_counter() - t:.2f}s"
    except Exception as exc:  # noqa: BLE001
        return f"✗ 失败 {type(exc).__name__}: {str(exc)[:60]}"


def tls(host: str, port: int = 443, timeout: float = 10.0) -> str:
    t = time.perf_counter()
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host):
                pass
        return f"✓ TLS 握手成功   {time.perf_counter() - t:.2f}s"
    except Exception as exc:  # noqa: BLE001
        return f"✗ 失败 {type(exc).__name__}: {str(exc)[:60]}"


def main() -> int:
    print("=" * 78)
    print("【数据库】Neon PostgreSQL（AWS ap-southeast-1 新加坡）")
    print("  主机:", DB.hostname)
    print("  TCP 5432 :", tcp(DB.hostname, 5432))

    print("=" * 78)
    print("【LLM】DeepSeek（国内服务）")
    print("  api.deepseek.com :", https("api.deepseek.com", "/v1/models"))

    print("=" * 78)
    print("【模型下载】HuggingFace")
    print("  huggingface.co   :", https("huggingface.co", "/BAAI/bge-m3/resolve/main/config.json"))
    print("  hf-mirror.com    :", https("hf-mirror.com", "/BAAI/bge-m3/resolve/main/config.json"))

    print("=" * 78)
    print("【OCR 模型源】百度 BOS（PaddleOCR 默认下载源，国内云）")
    print("  paddlex.bj.bcebos.com :", tcp("paddlex.bj.bcebos.com", 443))

    print("=" * 78)
    print("【包源】")
    print("  pypi.tuna.tsinghua.edu.cn :", tcp("pypi.tuna.tsinghua.edu.cn", 443))

    print("=" * 78)
    print("【部署目标（部署后浏览器才会访问）】")
    print("  vercel.com       :", https("vercel.com", "/"))
    print("  onrender.com     :", https("onrender.com", "/"))
    print("=" * 78)
    print("说明：本脚本从你当前网络直接发起请求，未使用任何代理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
