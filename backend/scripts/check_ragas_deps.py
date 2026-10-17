"""检查 RAGAS 相关依赖是否可用。"""

from __future__ import annotations

import importlib
import sys

PACKAGES = ["ragas", "langchain_openai", "langchain_community", "datasets", "langchain_huggingface"]


def main() -> int:
    ok = 0
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [OK]   {name:24} {version}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name:24} {type(exc).__name__}: {str(exc)[:70]}")
    print(f"\n{ok}/{len(PACKAGES)} 可用")
    return 0 if ok == len(PACKAGES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
