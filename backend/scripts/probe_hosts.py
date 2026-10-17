"""分层连通性测试：DNS → TCP 443 → TLS 握手 → HTTP（浏览器 UA）。

用来区分"网络层被阻断"和"应用层/Cloudflare 拦截脚本"：
  · TCP 连不上或 TLS 握手失败  → 网络层封锁（换域名也救不了，需要代理/换平台）
  · TLS 通但 HTTP 被 reset     → 应用层/Cloudflare（换个 UA 或真实浏览器可能就通）
  · HTTP 200                   → 之前那次 reset 可能只是拦脚本
"""

from __future__ import annotations

import socket
import ssl
import time

TARGETS = [
    ("onrender.com", "/", "Render 主站（与我们后端同后缀）"),
    ("dashboard.render.com", "/", "Render 控制台（同公司，不同子域）"),
    ("render.com", "/", "Render 官网主域（不带 on）"),
    ("fire-safety-rag-api.onrender.com", "/health", "假设的服务域名（未部署，仅看解析/握手）"),
    ("vercel.app", "/", "Vercel 应用域（前端会用的域名）"),
    ("your-app.vercel.app", "/", "假设的前端域名（未部署）"),
    ("api.deepseek.com", "/", "对照组：确定可用"),
]

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def tcp(host: str, port: int = 443, timeout: float = 8.0) -> str:
    t = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return f"OK   ({time.perf_counter() - t:.2f}s)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {str(exc)[:44]}"


def tls(host: str, timeout: float = 8.0) -> str:
    t = time.perf_counter()
    try:
        for family, stype, proto, _, sa in socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM):
            ctx = ssl.create_default_context()
            with socket.create_connection(sa, timeout=timeout) as raw:
                with ctx.wrap_socket(raw, server_hostname=host) as s:
                    ver = s.version()
            return f"OK   {ver} ({time.perf_counter() - t:.2f}s)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {str(exc)[:44]}"
    return "FAIL 无地址"


def http(host: str, path: str, timeout: float = 12.0) -> str:
    import httpx

    t = time.perf_counter()
    try:
        r = httpx.get(
            f"https://{host}{path}",
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": BROWSER_UA, "Accept": "text/html,application/json,*/*"},
        )
        return f"HTTP {r.status_code} ({time.perf_counter() - t:.2f}s)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {str(exc)[:44]}"


def main() -> int:
    print(f"{'主机':<38}{'TCP':<10}{'TLS':<24}{'HTTP(浏览器UA)':<34}说明")
    print("-" * 130)
    for host, path, note in TARGETS:
        dns = ""
        try:
            socket.getaddrinfo(host, 443, socket.AF_INET)
        except Exception:  # noqa: BLE001
            dns = "DNS 失败 → "
        print(f"{host:<38}{tcp(host):<10}{tls(host):<24}{dns + http(host, path):<34}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
