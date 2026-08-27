#!/usr/bin/env python3
"""Fail closed unless an exported Hugging Face pack looks like finalized NVFP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def human(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024
    return str(n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    args = p.parse_args()
    root = args.path
    errors: list[str] = []

    qpath = root / "hf_quant_config.json"
    if not qpath.is_file():
        errors.append("missing hf_quant_config.json")
        qj = {}
    else:
        qj = json.loads(qpath.read_text())
    algo = (
        (qj.get("quantization") or {}).get("quant_algo")
        or (qj.get("quantization_config") or {}).get("quant_algo")
        or qj.get("quant_algo")
    )
    if str(algo).upper() != "NVFP4":
        errors.append(f"quant_algo is not NVFP4: {algo!r}")

    excludes: list = []
    if isinstance(qj.get("quantization"), dict):
        excludes = list(qj["quantization"].get("exclude_modules") or [])
    if not any("layers.45" in str(x) for x in excludes):
        errors.append("hf_quant_config.json exclude_modules missing layers.45 (BF16 MTP)")

    parts = sorted(root.glob("__shard_part_*.safetensors"))
    if parts:
        errors.append(f"temporary shard parts remain: {len(parts)}")
    shards = sorted(p for p in root.glob("*.safetensors") if not p.name.startswith("__"))
    if not shards:
        errors.append("no finalized safetensors files")

    index = root / "model.safetensors.index.json"
    if len(shards) > 1:
        if not index.is_file():
            errors.append("multiple shards but model.safetensors.index.json is missing")
        else:
            idx = json.loads(index.read_text())
            if not idx.get("weight_map"):
                errors.append("model index has empty/missing weight_map")

    total = sum(p.stat().st_size for p in shards)
    print(f"path: {root}")
    print(f"quant_algo: {algo}")
    print(f"final_shards: {len(shards)}")
    print(f"weights: {human(total)} ({total:,} bytes)")
    if errors:
        print("VALIDATION: FAIL")
        for e in errors:
            print(f"- {e}")
        return 1
    print("VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
