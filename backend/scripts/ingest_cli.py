"""批量导入 CLI：把一个目录里的消防资料全部导进知识库。

用法（在 backend/ 目录，且已激活虚拟环境）：
    python -m scripts.ingest_cli --dir ./data/uploads --category law
    python -m scripts.ingest_cli --file ./data/uploads/建筑设计防火规范.pdf --category law
    python -m scripts.ingest_cli --dir ./资料 --category plan --no-ocr

设计为幂等：同一文件重复跑会被 file_hash 拦住（除非 --force）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 `python scripts/ingest_cli.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.indexer import index_file  # noqa: E402
from app.core.ingestion.loader import SUPPORTED_SUFFIXES  # noqa: E402
from app.db.session import init_db, session_scope  # noqa: E402
from app.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("ingest_cli")


def collect_files(target: Path, recursive: bool = True) -> list[Path]:
    if target.is_file():
        return [target]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in target.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description="fire-safety-rag 批量导入工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="要导入的目录")
    group.add_argument("--file", type=str, help="单个文件")
    parser.add_argument("--category", default="general", choices=["law", "plan", "training", "general"])
    parser.add_argument("--no-ocr", action="store_true", help="禁用扫描件 OCR 降级")
    parser.add_argument("--no-hype", action="store_true", help="禁用 HyPE 假想问题生成（省 LLM 调用）")
    parser.add_argument("--force", action="store_true", help="已索引的文件也重新导入")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="全量重建：清空向量表与文档/切块/会话表后再导入（换 embedding 模型时用）",
    )
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    init_db()   # 确保业务表存在

    if args.reset:
        from app.db.vector_store import drop_vector_table

        logger.warning("--reset：清空向量表与文档/切块数据，准备全量重建")
        drop_vector_table()
        with session_scope() as db:
            from sqlalchemy import text

            db.execute(text("TRUNCATE chunks, documents, chat_messages, chat_sessions RESTART IDENTITY CASCADE"))
        logger.info("已清空业务数据")

    target = Path(args.dir or args.file)
    files = collect_files(target, recursive=not args.no_recursive)
    if not files:
        logger.warning("没有找到可导入的文件（支持：%s）", sorted(SUPPORTED_SUFFIXES))
        return 1

    logger.info("待导入 %s 个文件，类别=%s", len(files), args.category)

    # HyPE 需要 LLM；不传的话 indexer 里会直接跳过假想问题生成（曾经踩过这个坑）
    from app.config import settings
    from app.dependencies import llm_dep

    llm = None
    if settings.enable_hype and not args.no_hype:
        llm = llm_dep()
        logger.info("HyPE 已启用：将为每个切块生成假想问题（额外消耗 LLM 调用）")
    else:
        logger.info("HyPE 未启用（--no-hype 或 ENABLE_HYPE=false）")

    ok, skipped, failed = 0, 0, 0
    for i, path in enumerate(files, 1):
        logger.info("[%s/%s] %s", i, len(files), path.name)
        try:
            with session_scope() as db:
                _, count, was_skipped = index_file(
                    db, path, doc_category=args.category,
                    use_ocr_fallback=not args.no_ocr, force=args.force, llm=llm,
                )
            if was_skipped:
                skipped += 1
            else:
                ok += 1
                logger.info("  -> %s 块", count)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("  -> 失败: %s", exc)

    logger.info("导入结束：成功 %s，跳过 %s，失败 %s", ok, skipped, failed)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
