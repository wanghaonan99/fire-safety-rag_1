"""MinerU 转换器：把复杂 PDF（扫描件 / 方正书版乱码 / 表格多的国标）转成 Markdown。

为什么单独做成"离线转换器"而不是塞进服务里：
1. MinerU 依赖重（base 包 = ONNX + llama.cpp，standard 档模型约 2GB、需 8GB 内存），
   塞进 FastAPI 进程会让 Render 免费套餐直接 OOM；
2. 它和 llama-index / paddleocr 的依赖容易打架，装到独立环境最省心；
3. 转换是一次性离线动作，产物 Markdown 可以进版本库、可复核、可手工修订，
   比"每次导入都跑模型"更可控。

用法（需要先有一个装了 MinerU 的独立环境，见 README「复杂 PDF 怎么办」）：
    # 只转换，产出 .md
    python scripts/mineru_convert.py "D:\\资料\\某国标.pdf" -o "D:\\资料\\某国标.md"

    # 转换后直接入库（走本项目的 loader -> chunker -> HyPE -> pgvector）
    python scripts/mineru_convert.py "D:\\资料\\某国标.pdf" --ingest --category law

MinerU 4.x 的 CLI：
    mineru-kit parse <pdf> -o <out.md> --tier basic|standard|advanced
    （basic：ONNX，模型约 0.8GB，最低 2GB 内存；standard：约 2GB，最低 8GB 内存）
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger("mineru_convert")

# MinerU 不同大版本的命令名：4.x 用 mineru-kit / mineru，1.x 是 magic-pdf
CANDIDATE_COMMANDS = ["mineru-kit", "mineru", "magic-pdf"]


def find_mineru() -> str | None:
    """在 PATH 里找 MinerU 可执行文件。"""
    for name in CANDIDATE_COMMANDS:
        path = shutil.which(name)
        if path:
            logger.info("使用 MinerU 命令：%s（%s）", name, path)
            return name
    return None


def build_command(cmd: str, pdf: Path, out: Path, tier: str, pages: str | None) -> list[str]:
    """按 MinerU 大版本拼命令。"""
    if cmd == "magic-pdf":          # 1.x
        return [cmd, "-p", str(pdf), "-o", str(out.parent), "-m", "auto"]
    if cmd == "mineru":             # 4.x 的 doclib 子命令（会建本地文档库）
        args = [cmd, "parse", str(pdf), "--output", str(out), "--tier", tier]
        if pages:
            args += ["--pages", pages]
        return args
    args = [cmd, "parse", str(pdf), "-o", str(out), "--tier", tier]   # 4.x 无状态转换
    if pages:
        args += ["--pages", pages]
    return args


def run_mineru(pdf: Path, out: Path, tier: str, pages: str | None) -> Path:
    """调用 MinerU 把 PDF 转成 Markdown，返回产出文件路径。"""
    cmd = find_mineru()
    if cmd is None:
        raise SystemExit(
            "未找到 MinerU 命令（mineru-kit / mineru / magic-pdf）。\n"
            "请先在独立环境安装（建议不要装进本项目的 .venv，避免依赖冲突）：\n"
            "    pip install uv\n"
            "    uv venv .mineru --python 3.12\n"
            '    uv pip install --python .mineru "mineru>=4.0,<5"\n'
            "然后激活该环境再运行本脚本，或把它的 Scripts 目录加入 PATH。"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(cmd, pdf, out, tier, pages)
    logger.info("执行：%s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        logger.error("MinerU 失败 stdout:\n%s", (result.stdout or "")[-2000:])
        logger.error("MinerU 失败 stderr:\n%s", (result.stderr or "")[-2000:])
        raise SystemExit(f"MinerU 退出码 {result.returncode}")

    # MinerU 不总是把结果精确放在 -o 指定路径：可能在同级目录生成 <stem>.md 或 <stem>/auto/<stem>.md
    candidates = [
        out,
        out.with_suffix(".md"),
        pdf.parent / f"{pdf.stem}.md",
        out.parent / pdf.stem / "auto" / f"{pdf.stem}.md",
        out.parent / pdf.stem / "vlm" / f"{pdf.stem}.md",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            logger.info("MinerU 产出：%s", c)
            return c
    # 兜底：在输出目录里找最新的 .md
    found = sorted(out.parent.glob("**/*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if found:
        logger.info("MinerU 产出（兜底查找）：%s", found[0])
        return found[0]
    raise SystemExit("MinerU 执行成功但没找到 Markdown 产物，请检查输出目录")


def markdown_as_html_document(md_path: Path):
    """把 Markdown 交给本项目已有的 loader（.md 走纯文本分支，标题 # 会被 chunker 识别）。"""
    from app.core.ingestion.loader import load_file

    return load_file(md_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="用 MinerU 把复杂 PDF 转成 Markdown 并可选入库")
    parser.add_argument("pdf", help="待转换的 PDF 路径")
    parser.add_argument("-o", "--output", help="输出 Markdown 路径（默认与 PDF 同目录同名）")
    parser.add_argument("--tier", default="basic", choices=["flash", "basic", "standard", "advanced"],
                        help="解析档位：basic≈0.8GB模型/2GB内存；standard≈2GB模型/8GB内存")
    parser.add_argument("--pages", help="页码范围，如 1-10 或 all（默认 MinerU 自己的默认范围）")
    parser.add_argument("--ingest", action="store_true", help="转换后直接导入知识库")
    parser.add_argument("--category", default="law", choices=["law", "plan", "training", "general"])
    parser.add_argument("--keep-md", action="store_true", help="入库后保留 Markdown（默认保留）")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF 不存在：{pdf}")

    out = Path(args.output) if args.output else pdf.with_suffix(".md")
    md_path = run_mineru(pdf, out, args.tier, args.pages)

    text = md_path.read_text(encoding="utf-8", errors="ignore")
    from app.core.ingestion.quality import quality_breakdown

    logger.info("Markdown 质量：%s", quality_breakdown(text))
    print(f"\n转换完成：{md_path}\n字符数：{len(text)}\n---- 前 400 字 ----\n{text[:400]}\n")

    if args.ingest:
        from app.core.ingestion.indexer import index_file
        from app.db.session import init_db, session_scope

        init_db()
        with session_scope() as db:
            doc, count, skipped = index_file(
                db, md_path, title=pdf.stem, doc_category=args.category, use_ocr_fallback=False
            )
            logger.info("入库完成：%s 块（跳过=%s）", count, skipped)
            print(f"已入库：{doc.title}，{count} 块，document_id={doc.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
