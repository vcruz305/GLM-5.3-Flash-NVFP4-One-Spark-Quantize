#!/usr/bin/env python3
"""Quantize GLM-5.3-Flash BF16 to experts-only NVFP4 on one GB10.

This is a quantization/export workflow, not a one-Spark serving claim.
Reference method: layerwise ModelOpt PTQ + Accelerate disk offload +
streaming HF export + per-shard reverse tensor mappings.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Complete GLM-5.3-Flash-BF16 directory")
    p.add_argument("--out", type=Path, required=True, help="NVFP4 export directory")
    p.add_argument("--offload", type=Path, required=True, help="Accelerate offload scratch directory")
    p.add_argument("--state", type=Path, required=True, help="Small JSON status file")
    p.add_argument("--modelopt-state", type=Path, required=True, help="ModelOpt metadata receipt")
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--gpu-memory", default="40GiB")
    p.add_argument("--cpu-memory", default="24GiB")
    p.add_argument("--max-shard-size", default="4GB")
    p.add_argument("--calib-chunks", type=int, default=18)
    p.add_argument("--calib-seq", type=int, default=512)
    p.add_argument("--wipe-offload", action="store_true", help="Delete offload scratch before starting")
    return p.parse_args()


def set_state(path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d: dict = {}
    if path.exists() and path.stat().st_size:
        try:
            d = json.loads(path.read_text())
        except Exception:
            d = {}
    d.update(kw)
    d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(d, indent=2) + "\n")


def load_quant_cfg(recipe_path: Path):
    if not recipe_path.is_file():
        raise SystemExit(f"missing ModelOpt recipe: {recipe_path}")
    from modelopt.recipe import load_recipe

    recipe = load_recipe(str(recipe_path))
    cfg = recipe.quantize.model_dump()
    print("RECIPE", recipe_path, flush=True)
    return cfg


def build_calib(tokenizer, n_chunks: int, seq_len: int):
    texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Mixture-of-experts models route tokens to a subset of experts. ",
        "Write a Python function that binary-searches a sorted list. ",
        "Explain why NVFP4 uses E2M1 values and FP8 block scales. ",
        "Tool call: search_web(query='GLM-5.3-Flash ModelOpt NVFP4'). ",
        "Solve: if 3x + 7 = 22, what is x? Show the steps. ",
        'JSON: {"role":"assistant","content":"ok"} ',
        "长上下文与稀疏注意力用于降低推理成本。 ",
    ]
    blob = "".join(texts)
    blob *= max(100, n_chunks * 12)
    enc = tokenizer(blob, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    needed = n_chunks * seq_len
    if ids.numel() < needed:
        raise RuntimeError(f"calibration text produced {ids.numel()} tokens; need {needed}")
    chunks = [ids[i * seq_len : (i + 1) * seq_len].unsqueeze(0) for i in range(n_chunks)]
    print("CALIB_CHUNKS", len(chunks), "seq", seq_len, flush=True)

    def forward_loop(model):
        import torch

        print("CALIB_START", flush=True)
        with torch.no_grad():
            for i, cids in enumerate(chunks, start=1):
                try:
                    dev = next(model.parameters()).device
                except StopIteration:
                    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                try:
                    model(input_ids=cids.to(dev))
                except Exception:
                    model(input_ids=cids)
                print(f"CALIB_CHUNK {i}/{len(chunks)}", flush=True)
        print("CALIB_DONE", flush=True)

    return forward_loop


def patch_streaming_export() -> None:
    """Reference workaround for ModelOpt's offloaded GLM5Next split assertion.

    The export remains offloaded; reverse rules are applied per finalized shard.
    Re-test this against every new ModelOpt release.
    """
    import modelopt.torch.export.unified_export_hf_streaming as streaming

    def _allow_offloaded_reverse(_model) -> None:
        print("SKIP_SPLIT_ASSERT glm5_next; reverse per shard after export", flush=True)

    streaming._assert_no_split_rules = _allow_offloaded_reverse
    print("PATCHED _assert_no_split_rules", flush=True)

    try:
        import modelopt.torch.export.unified_export_hf as ue

        original = ue.requantize_resmooth_fused_llm_layers

        def logged(model):
            print("RESMOOTH_START", flush=True)
            t0 = time.time()
            out = original(model)
            print(f"RESMOOTH_DONE sec={time.time() - t0:.0f}", flush=True)
            return out

        ue.requantize_resmooth_fused_llm_layers = logged
        if hasattr(streaming, "requantize_resmooth_fused_llm_layers"):
            streaming.requantize_resmooth_fused_llm_layers = logged
    except Exception as exc:
        print("PATCH_RESMOOTH_LOG_FAIL", type(exc).__name__, exc, flush=True)


def ensure_mtp_bf16_excludes(data: dict) -> bool:
    """Layer 45 NextN is grafted BF16; vLLM treats missing excludes as packed NVFP4."""
    needed = [
        "model.language_model.layers.45*",
        "model.language_model.layers.45.mlp.experts*",
        "model.language_model.layers.45.mlp.gate",
        "model.language_model.layers.45.mlp.shared_experts*",
        "model.language_model.layers.45.self_attn*",
    ]
    changed = False
    for block_key in ("quantization", "quantization_config"):
        block = data.get(block_key)
        if not isinstance(block, dict):
            continue
        for list_key in ("exclude_modules", "ignore"):
            lst = block.get(list_key)
            if not isinstance(lst, list):
                continue
            for item in needed:
                if item in lst:
                    continue
                ins = None
                for i, existing in enumerate(lst):
                    if "layers.44" in str(existing):
                        ins = i + 1
                if ins is None:
                    vis = next((i for i, e in enumerate(lst) if e == "model.visual*"), None)
                    ins = vis if vis is not None else len(lst)
                lst.insert(ins, item)
                changed = True
    return changed


def reverse_exported_shards(model, export_dir: Path) -> None:
    from modelopt.torch.export.quant_aware_conversion import (
        _build_reverse_rules,
        apply_reverse_rules,
        build_reverse_name_mapper,
        revert_quant_config_names,
    )
    from safetensors.torch import load_file, save_file

    split_rules, rename_rules, _expert_fused = _build_reverse_rules(model)
    print("REVERSE_RULES", "splits", len(split_rules), "renames", len(rename_rules), flush=True)
    if not split_rules and not rename_rules:
        return

    shards = sorted(
        p for p in export_dir.glob("*.safetensors")
        if p.is_file() and not p.name.startswith("__") and not p.name.endswith(".rev")
    )
    if not shards:
        raise RuntimeError("no finalized safetensors shards found for reverse mapping")

    weight_map: dict[str, str] = {}
    for sf in shards:
        print("REVERSE_SHARD", sf.name, "bytes", sf.stat().st_size, flush=True)
        state = load_file(str(sf))
        try:
            reversed_state = apply_reverse_rules(state, split_rules, rename_rules)
        except Exception as exc:
            print("REVERSE_SHARD_WARN keep fused", sf.name, type(exc).__name__, exc, flush=True)
            reversed_state = state
        tmp = sf.with_name(sf.name + ".rev")
        save_file(reversed_state, str(tmp))
        os.replace(tmp, sf)
        for key in reversed_state:
            weight_map[key] = sf.name
        del state, reversed_state

    index_path = export_dir / "model.safetensors.index.json"
    if index_path.exists() and weight_map:
        data = json.loads(index_path.read_text())
        data["weight_map"] = weight_map
        index_path.write_text(json.dumps(data, indent=2) + "\n")

    try:
        mapper = build_reverse_name_mapper(model)
    except Exception as exc:
        print("NAME_MAPPER_WARN", type(exc).__name__, exc, flush=True)
        mapper = None

    for name in ("hf_quant_config.json", "config.json"):
        path = export_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        changed = False
        if mapper is not None:
            if isinstance(data.get("quantization"), dict):
                revert_quant_config_names(data["quantization"], mapper)
                changed = True
            if isinstance(data.get("quantization_config"), dict):
                revert_quant_config_names(data["quantization_config"], mapper)
                changed = True
        if ensure_mtp_bf16_excludes(data):
            changed = True
        if changed:
            path.write_text(json.dumps(data, indent=2) + "\n")


def save_modelopt_metadata(model, path: Path) -> None:
    import modelopt.torch.opt as mto
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mto.modelopt_state(model), path)
    print("MTO_STATE", path, "bytes", path.stat().st_size, flush=True)


def quant_algo(export_dir: Path) -> str | None:
    path = export_dir / "hf_quant_config.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    return (
        (data.get("quantization") or {}).get("quant_algo")
        or (data.get("quantization_config") or {}).get("quant_algo")
        or data.get("quant_algo")
    )


def main() -> int:
    args = parse_args()
    import torch
    import transformers
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    print("transformers", transformers.__version__, flush=True)
    print("modelopt", getattr(__import__("modelopt"), "__version__", "?"), flush=True)

    if not args.src.is_dir():
        raise SystemExit(f"source directory missing: {args.src}")
    if args.wipe_offload and args.offload.exists():
        shutil.rmtree(args.offload)
    args.offload.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(str(args.src), trust_remote_code=True)
    print("ARCH", cfg.architectures, "model_type", cfg.model_type, flush=True)
    if getattr(cfg, "quantization_config", None):
        raise SystemExit("source already has quantization_config; start from GLM-5.3-Flash-BF16")

    set_state(args.state, phase="load")
    print("LOAD", args.src, flush=True)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        str(args.src),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        offload_folder=str(args.offload),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print("LOADED", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(args.src), trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(str(args.src), trust_remote_code=True)
    except Exception as exc:
        processor = None
        print("NO_PROCESSOR", type(exc).__name__, exc, flush=True)

    quant_cfg = load_quant_cfg(args.recipe)
    set_state(args.state, phase="quantize")
    print("QUANTIZE experts-only NVFP4", flush=True)
    mtq.quantize(model, quant_cfg, build_calib(tokenizer, args.calib_chunks, args.calib_seq))
    save_modelopt_metadata(model, args.modelopt_state)

    set_state(args.state, phase="export")
    try:
        from modelopt.torch.export.unified_export_hf import has_accelerate_offload
        offloaded = has_accelerate_offload(model)
        print("OFFLOADED", offloaded, flush=True)
        if not offloaded:
            raise RuntimeError("model is not Accelerate-offloaded; refusing full-state export on GB10")
    except ImportError:
        print("OFFLOAD_CHECK unavailable in this ModelOpt version", flush=True)

    for p in list(args.out.glob("model-*.safetensors")) + list(args.out.glob("model.safetensors")):
        p.unlink()
    for name in ("hf_quant_config.json", "model.safetensors.index.json"):
        p = args.out / name
        if p.exists():
            p.unlink()

    patch_streaming_export()
    print("EXPORT", args.out, flush=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=str(args.out), max_shard_size=args.max_shard_size)
    print("EXPORT_RAW_DONE", flush=True)
    reverse_exported_shards(model, args.out)

    tokenizer.save_pretrained(str(args.out))
    if processor is not None:
        try:
            processor.save_pretrained(str(args.out))
        except Exception as exc:
            print("PROCESSOR_SAVE_WARN", type(exc).__name__, exc, flush=True)
    for name in ("chat_template.jinja", "LICENSE", "generation_config.json"):
        src = args.src / name
        if src.is_file():
            shutil.copy2(src, args.out / name)

    algo = quant_algo(args.out)
    print("QUANT_ALGO", algo, flush=True)
    if str(algo).upper() != "NVFP4":
        raise SystemExit(f"export is not validated NVFP4: quant_algo={algo!r}")
    shards = sorted(args.out.glob("*.safetensors"))
    print("NVFP4_EXPORT_OK", "shards", len(shards), flush=True)
    set_state(args.state, phase="export_done", quant_algo=str(algo), shards=len(shards))
    return 0


if __name__ == "__main__":
    args_for_error = None
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print("FATAL", type(exc).__name__, exc, flush=True)
        raise
