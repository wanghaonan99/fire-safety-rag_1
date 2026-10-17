"""评估集上线前校验：保证每条 ground_truth_ids 都真实存在于库里。

为什么必须查：清理切块（如删目次碎片）或重建索引后，评估集里可能留下**悬空引用**——
那些题会被检索评估**永远判为未命中**，把 HitRate 压低，而你会误以为是检索能力不行。

用法：python scripts/validate_golden_set.py [评估集路径]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/retrieval_golden.jsonl")
    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]

    with engine.connect() as c:
        known = {r[0] for r in c.execute(text("SELECT id::text FROM chunks")).all()}
        titles = {r[0]: r[1] for r in c.execute(text("SELECT id::text, title FROM documents")).all()}
        doc_of = {
            r[0]: r[1]
            for r in c.execute(text("SELECT id::text, document_id::text FROM chunks")).all()
        }

    dangling: list[tuple[int, str, list[str]]] = []
    doc_dist: Counter[str] = Counter()
    for i, item in enumerate(items, 1):
        ids = [str(x) for x in item.get("ground_truth_ids") or []]
        missing = [x for x in ids if x not in known]
        if missing:
            dangling.append((i, item.get("question", "")[:56], missing))
        elif ids:
            doc_dist[titles.get(doc_of.get(ids[0], ""), "?")[:34]] += 1

    print(f"评估集 {src.name}：{len(items)} 条")
    print(f"库中切块 {len(known)} 个\n")

    print("按来源资料分布：")
    for d, n in doc_dist.most_common():
        print(f"  {n:>4} 条  {d}")

    print()
    if dangling:
        print(f"❌ 发现 {len(dangling)} 条悬空引用（这些题会永远判未命中）：")
        # 草稿/勾选清单里有原始编号 _no，能直接给出可执行的删除命令
        preselect = Path("data/eval/retrieval_golden_preselect.jsonl")
        no_of: dict[str, int] = {}
        if preselect.exists():
            for line in preselect.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    no_of[(row.get("question") or "").strip()] = row.get("_no")

        numbers: list[int] = []
        for no, q, missing in dangling:
            orig = no_of.get(q)
            if orig is not None:
                numbers.append(orig)
            tag = f"原始编号 #{orig}" if orig is not None else "未在勾选清单中找到"
            print(f"  [第{no}条 · {tag}] {q} → 缺失 {len(missing)} 个切块 id")

        if numbers:
            print("\n处理命令（直接复制执行）：")
            print(f"  python scripts/drop_golden_items.py {' '.join(map(str, numbers))}")
            print("  python scripts/build_golden_set.py")
        return 2

    print("✅ 全部 ground_truth_ids 均存在，可以跑检索基线")
    print(
        "\n  curl.exe -X POST http://127.0.0.1:8000/api/eval/retrieval "
        '-H "Content-Type: application/json" '
        f'-d \'{{"dataset_path":"./{src.as_posix()}","top_k":10,"run_name":"baseline-v1","use_rerank":true}}\''
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
