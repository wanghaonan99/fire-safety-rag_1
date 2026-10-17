"""Embedding 设备与吞吐自检：确认是否真的用上了显卡。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.ingestion.embedder import build_embed_model, resolve_device  # noqa: E402


def main() -> int:
    import torch

    print("torch 版本      :", torch.__version__)
    print("CUDA 编译支持   :", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU             :", torch.cuda.get_device_name(0))
        print("显存            : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("resolve_device()->", resolve_device())
    print()

    texts = [f"第 {i} 条 消防设施应当定期检查维护，确保完好有效，并做好检查记录。" for i in range(64)]
    model = build_embed_model()

    # 预热（首次会加载权重）
    model.get_text_embedding_batch(texts[:4])
    t0 = time.perf_counter()
    model.get_text_embedding_batch(texts)
    dt = time.perf_counter() - t0

    device = getattr(model, "device", None) or getattr(model, "_device", None) or resolve_device()
    print(f"\n64 条文本向量化：{dt:.2f}s（{64 / dt:.1f} 条/秒）  实际设备={device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
