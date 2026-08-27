#!/usr/bin/env python3
"""Inspect /proc mappings and export destination without ptrace."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--process-match", default="quant_glm53_nvfp4.py")
    p.add_argument("--write", type=Path, default=None, help="Optional progress.json destination")
    return p.parse_args()


def pids(match: str) -> list[int]:
    out: list[int] = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        if match in cmd and "python" in cmd:
            out.append(int(p.name))
    return out


def mapped_layers(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{pid}/maps").read_text(errors="replace")
    except Exception:
        return []
    return sorted({int(m.group(1)) for m in re.finditer(r"layers\.(\d+)\.", text)})


def rss_gib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / (1024 * 1024), 2)
    except Exception:
        pass
    return 0.0


def main() -> None:
    args = parse_args()
    state = {}
    if args.state.exists():
        try:
            state = json.loads(args.state.read_text())
        except Exception:
            pass
    ps = pids(args.process_match)
    parts = sorted(args.dest.glob("__shard_part_*.safetensors")) if args.dest.exists() else []
    finals = sorted(
        p for p in args.dest.glob("*.safetensors")
        if not p.name.startswith("__")
    ) if args.dest.exists() else []
    bytes_total = sum(p.stat().st_size for p in args.dest.rglob("*") if p.is_file()) if args.dest.exists() else 0
    row: dict = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": state.get("phase"),
        "alive": bool(ps),
        "pids": ps,
        "dest_bytes": bytes_total,
        "temporary_parts": len(parts),
        "final_shards": len(finals),
        "has_quant_cfg": (args.dest / "hf_quant_config.json").is_file(),
    }
    if ps:
        layers = mapped_layers(ps[0])
        row["pid"] = ps[0]
        row["rss_gib"] = rss_gib(ps[0])
        row["mapped_layers"] = layers
        if parts or finals:
            row["export_stage"] = "writing_shards"
        elif layers:
            row["export_stage"] = f"resmooth_layer_{layers[-1]}/44"
        else:
            row["export_stage"] = "export_prep_or_load"
    else:
        row["export_stage"] = "process_not_found"
    text = json.dumps(row, indent=2) + "\n"
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
