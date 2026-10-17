"""评估集草稿预筛：把"需要重点人工核对"的条目挑出来，并生成便于阅读的复核文件。

用法：
    cd backend
    python scripts/review_golden_set.py                    # 默认看草稿
    python scripts/review_golden_set.py data/eval/retrieval_golden.jsonl

它做的是**机械检查**（不能替代人工），覆盖五类常见问题：
  A 来源太短          —— 来源切块 <80 字**且没有完整句子**（标题行/碎片）；
                          注意"短但完整"的定义条款是**正常**的，不会误报
  B 标准答案与原文脱节 —— 模型自由发挥，答案不是原文内容
  C 数字对不上        —— 答案里的数值在原文中找不到（幻觉数值，消防场景最危险）
  D 语义重复          —— 与前面某题高度相似
  E 答案过短/过长     —— 标注质量差
  T 模板占位句        —— 导则类标准特有：原文只写"说明…/简述…"，答案必然靠模型发挥

输出：控制台摘要 + data/eval/<名字>_review.md（可读的复核清单）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402

MIN_CHUNK_CHARS = 80          # A：来源切块最少字数
MIN_ANSWER_OVERLAP = 0.40     # B：标准答案与原块的 2-gram 覆盖率下限
DUPLICATE_SIMILARITY = 0.75   # D：题目之间的 2-gram 相似度上限
MIN_ANSWER_CHARS = 10
MAX_ANSWER_CHARS = 400

# 导则类标准里有大量"模板占位句"（只说要写什么，不给具体内容），
# 从这种切块出的题**无法从原文回答**，必须人工剔除
TEMPLATE_MARKERS = ("说明", "简述", "明确", "描述", "包括", "规定")
TEMPLATE_MAX_CHARS = 150

NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(max(len(s) - 1, 0))}


def number_present(num: str, source_digits: str) -> bool:
    r"""判断数字是否"出现在原文里"，容忍日期/编号的不同写法。

    例：原文 `2020-04-01`，答案写 `2020年4月1日` → 数字 4、1 都应视为出现。
    直接拿 `\d+` 分词逐字比对会产生大量误报（这正是第一版检查器的问题）。
    """
    candidates = {num, num.lstrip("0") or "0"}
    if len(num) <= 2:
        candidates.add(num.zfill(2))
    return any(c in source_digits for c in candidates)


SECTION_NUM = re.compile(r"第\s*[\d一二三四五六七八九十]+\s*[章节条]|\b\d+(?:\.\d+)+\b")


def looks_like_template(text: str) -> bool:
    """识别"模板占位句"切块：短、只出现 说明/简述/明确 这类动词、且没有**数据性**数字。

    ⚠️ 必须先剥掉章节号（6.1 / 5.3.2 / 第6条）——它们是编号不是数据，
    否则"6.1 编制目的 简述预案编制的目的和作用。"会被误判成"含具体数值"。
    """
    t = SECTION_NUM.sub(" ", text).strip()
    if len(t) > TEMPLATE_MAX_CHARS:
        return False
    if NUM_PATTERN.search(t):
        return False
    return any(m in t for m in TEMPLATE_MARKERS)


def load_chunk_titles() -> dict[str, str]:
    """document_id -> 文档标题，用于按资料统计题目分布。"""
    sql = text("SELECT id, title FROM documents")
    with engine.connect() as conn:
        return {r[0]: r[1] for r in conn.execute(sql).all()}


def load_chunks() -> dict[str, dict]:
    """一次把全部切块读回来（避免 N+1 查询）。"""
    sql = text("SELECT id, document_id, chunk_index, content, heading_path FROM chunks")
    with engine.connect() as conn:
        rows = conn.execute(sql).all()
    return {
        r[0]: {
            "document_id": r[1],
            "chunk_index": r[2],
            "content": r[3] or "",
            "heading_path": r[4] or "",
        }
        for r in rows
    }


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/eval/retrieval_golden_draft.jsonl")
    if not src.exists():
        print(f"找不到评估集：{src}")
        return 1

    items = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = load_chunks()
    titles = load_chunk_titles()
    print(f"评估集：{src.name}   共 {len(items)} 条；库中切块 {len(chunks)} 个\n")

    seen_questions: list[tuple[int, set[str]]] = []
    report: list[dict] = []
    flag_count: dict[str, int] = {}

    for idx, item in enumerate(items, 1):
        q = (item.get("question") or "").strip()
        a = (item.get("ground_truth") or "").strip()
        ids = item.get("ground_truth_ids") or []
        qtype = (item.get("type") or "").strip()
        flags: list[str] = []

        # 取来源切块（可能多个 id，取并集文本）
        src_text = "\n".join(chunks.get(i, {}).get("content", "") for i in ids)
        heading = " / ".join(filter(None, (chunks.get(i, {}).get("heading_path", "") for i in ids)))
        missing_ids = [i for i in ids if i not in chunks]

        if missing_ids:
            flags.append("X-标注的切块不存在")
        if src_text and len(src_text) < MIN_CHUNK_CHARS:
            # ⚠️ 只按字数判断会大量误报：技术标准里"短但完整"的定义条款很常见，
            #    例如"消防产品是专门用于火灾预防、灭火救援和火灾防护、避难、逃生的产品。"（39 字），
            #    完全可回答。真正有问题的是"短且没有完整句子"的碎片（标题行、模板句）。
            has_sentence = "。" in src_text
            cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", src_text))
            if not has_sentence or cjk_chars < 20:
                flags.append(f"A-来源仅 {len(src_text)} 字且无完整句子")
        if src_text and looks_like_template(src_text):
            flags.append("T-模板占位句（原文只说要写什么，没给具体内容）")

        overlap = 0.0
        if src_text and a:
            bg_a = bigrams(a)
            overlap = len(bg_a & bigrams(src_text)) / max(len(bg_a), 1)
            if overlap < MIN_ANSWER_OVERLAP:
                flags.append(f"B-答案与原块重合仅 {overlap:.0%}")

        # 数字核对：答案里的数字必须能在原块找到（容忍日期写法差异）
        if src_text:
            source_digits = "".join(NUM_PATTERN.findall(src_text))
            bad_nums = sorted(
                n for n in set(NUM_PATTERN.findall(a)) if not number_present(n, source_digits)
            )
            if bad_nums:
                flags.append("C-答案含原文没有的数字 " + ",".join(bad_nums))

        q_bg = bigrams(q)
        for prev_idx, prev_bg in seen_questions:
            if q_bg and prev_bg:
                sim = len(q_bg & prev_bg) / max(min(len(q_bg), len(prev_bg)), 1)
                if sim > DUPLICATE_SIMILARITY:
                    flags.append(f"D-与第 {prev_idx} 题重复({sim:.0%})")
                    break
        seen_questions.append((idx, q_bg))

        if len(a) < MIN_ANSWER_CHARS:
            flags.append("E-答案过短")
        if len(a) > MAX_ANSWER_CHARS:
            flags.append("E-答案过长")

        for f in flags:
            key = f.split("-")[0]
            flag_count[key] = flag_count.get(key, 0) + 1

        report.append(
            {
                "idx": idx, "question": q, "answer": a, "heading": heading,
                "chunk_index": chunks.get(ids[0], {}).get("chunk_index") if ids else None,
                "ids": ids, "overlap": overlap, "flags": flags,
                "type": qtype,
                "doc": titles.get(chunks.get(ids[0], {}).get("document_id", ""), "?") if ids else "?",
                "chunk_preview": src_text[:300].replace("\n", " "),
            }
        )

    # ---------- 控制台摘要 ----------
    clean = [r for r in report if not r["flags"]]
    print("=" * 76)
    print(f"可直接采用（无任何机械问题）：{len(clean)} / {len(report)}")
    print(f"需要重点核对              ：{len(report) - len(clean)}")
    print()
    label = {
        "A": "来源切块太短", "B": "答案与原块脱节", "C": "数字对不上原文",
        "D": "题目重复", "E": "答案长度异常", "X": "标注切块不存在",
        "T": "模板占位句",
    }
    for k in sorted(flag_count):
        print(f"  {k} {label.get(k, k):14} {flag_count[k]} 条")

    # 按资料 / 题型统计分布（人工按策略配额挑选时的依据）
    print("\n按来源资料分布（对照策略配额：GB51283 12~15 · 消防法 8~10 · 导则 5~8）：")
    by_doc: dict[str, int] = {}
    for r in report:
        by_doc[r["doc"][:34]] = by_doc.get(r["doc"][:34], 0) + 1
    for d, n in sorted(by_doc.items(), key=lambda x: -x[1]):
        print(f"  {n:3} 条  {d}")

    print("\n按题型分布（对照策略：事实型 40% · 归纳型 30% · 易混淆型 20%）：")
    by_type: dict[str, int] = {}
    for r in report:
        by_type[r["type"] or "(未标注)"] = by_type.get(r["type"] or "(未标注)", 0) + 1
    for t, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {n:3} 条  {t}")

    # ---------- 写复核文件 ----------
    out = src.with_name(src.stem + "_review.md")
    lines = [
        f"# 评估集草稿复核清单（{src.name}）",
        "",
        f"- 总数 **{len(items)}** 条，机械检查无问题 **{len(clean)}** 条，需重点核对 **{len(report) - len(clean)}** 条",
        "- 标记含义：A 来源太短 · B 答案与原块脱节 · C 数字对不上原文 · D 与前面题目重复 · E 答案长度异常 · T 模板占位句",
        "- **机械检查不能替代阅读**：无标记也可能问得没意义，有标记也可能只是改写幅度大",
        "",
        "## 无标记（建议优先采用）",
        "",
    ]
    for r in clean:
        lines += [
            f"### {r['idx']}. {r['question']}",
            f"- 题型：{r['type'] or '(未标注)'} · 来源：《{r['doc']}》第 {r['chunk_index']} 块 · {r['heading'] or '(无标题)'} · 重合 {r['overlap']:.0%}",
            f"- 标准答案：{r['answer']}",
            f"- 原文：{r['chunk_preview'][:200]}",
            "",
        ]
    lines += ["", "## 需重点核对", ""]
    for r in [x for x in report if x["flags"]]:
        lines += [
            f"### {r['idx']}. {r['question']}",
            f"- ⚠️ {'；'.join(r['flags'])}",
            f"- 题型：{r['type'] or '(未标注)'} · 来源：《{r['doc']}》第 {r['chunk_index']} 块 · {r['heading'] or '(无标题)'} · 重合 {r['overlap']:.0%}",
            f"- 标准答案：{r['answer']}",
            f"- 原文：{r['chunk_preview'][:300]}",
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n复核清单已写入：{out}")

    # ---------- 勾选清单：把 64 条原样列出，加上 _flags 与 _keep，人工只需改 _keep ----------
    preselect = src.with_name("retrieval_golden_preselect.jsonl")
    with preselect.open("w", encoding="utf-8") as f:
        for r in report:
            row = {
                "_no": r["idx"],
                "question": r["question"],
                "ground_truth": r["answer"],
                "ground_truth_ids": r["ids"],
                "type": r["type"],
                "_keep": not r["flags"],          # 机械检查通过的先默认保留，人工再逐条确认
                "_flags": r["flags"],
                "_doc": r["doc"],
                "_heading": r["heading"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"勾选清单已写入：{preselect}（默认保留 {len(clean)} 条，人工改 _keep 即可）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
