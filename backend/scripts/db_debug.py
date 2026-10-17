"""数据库状态体检：向量表与业务表是否一致、HyPE 是否真的写进去了。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.config import apply_hf_cache_env  # noqa: E402
from app.db.session import engine  # noqa: E402


def main() -> int:
    print("HF 缓存目录:", apply_hf_cache_env() or "(未接管，沿用系统 HF_HOME)")
    with engine.connect() as c:
        total = c.execute(text("SELECT count(*) FROM data_fire_chunks")).scalar()
        print(f"\n向量表 data_fire_chunks 总行数: {total}")
        rows = c.execute(
            text(
                "SELECT metadata_->>'document_id' AS doc, count(*) AS n "
                "FROM data_fire_chunks GROUP BY 1 ORDER BY 2 DESC"
            )
        ).all()
        for doc, n in rows:
            print(f"  document_id={doc}  ->  {n} 行")

        print("\ndocuments 表:")
        for r in c.execute(
            text("SELECT id, title, status, chunk_count, is_scanned, created_at FROM documents ORDER BY created_at")
        ).all():
            print(f"  {r[0]}  {r[1][:24]:26} status={r[2]:8} chunks={r[3]:<4} scanned={r[4]} {r[5]}")

        print("\nchunks 表行数:", c.execute(text("SELECT count(*) FROM chunks")).scalar())
        hype = c.execute(
            text("SELECT count(*) FROM chunks WHERE jsonb_array_length(hypothetical_questions) > 0")
        ).scalar()
        print("其中带 HyPE 假想问题的:", hype)
        sample = c.execute(
            text(
                "SELECT content, hypothetical_questions FROM chunks "
                "WHERE jsonb_array_length(hypothetical_questions) > 0 LIMIT 1"
            )
        ).first()
        if sample:
            print("\n  样例切块(前 60 字):", (sample[0] or "")[:60].replace("\n", " "))
            print("  生成的假想问题:", sample[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
