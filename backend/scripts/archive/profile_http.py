"""测稳态 HTTP 延迟：连续问 3 个**不同**问题，排除缓存影响。

用法：cd backend && python scripts/profile_http.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTIONS = [
    "消防安全重点单位多久组织演练一次？",
    "灭火和应急疏散预案应当包括哪些内容？",
    "演练结束后需要做哪些总结讲评工作？",
]


def main() -> int:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        latencies = []
        for i, q in enumerate(QUESTIONS, 1):
            t = time.perf_counter()
            r = client.post("/api/query", json={"question": q, "top_k": 5, "use_cache": False}, timeout=600)
            dt = time.perf_counter() - t
            if r.status_code != 200:
                print(f"  第 {i} 问失败 HTTP {r.status_code}: {r.text[:200]}")
                continue
            d = r.json()["data"]
            latencies.append(dt)
            print(f"  第 {i} 问: {dt:6.2f}s  出处 {len(d.get('sources', []))} 条  答案 {len(d.get('answer', ''))} 字")

        if len(latencies) > 1:
            steady = latencies[1:]
            print(f"\n首次 {latencies[0]:.1f}s（含模型加载）")
            print(f"稳态平均 {sum(steady) / len(steady):.1f}s（后续每一次提问）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
