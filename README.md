# Quantizing GLM-5.3-Flash to NVFP4 on One DGX Spark

> [!IMPORTANT]
> **This repository is about quantizing and exporting GLM-5.3-Flash on one DGX Spark. It does _not_ claim that the resulting NVFP4 checkpoint serves entirely on one Spark.**
>
> Reference output: **177.15 GiB**. A DGX Spark exposes about **121 GiB usable UMA**, so the finished pack is larger than one Spark's memory.

A reproducible NVIDIA Model Optimizer workflow for converting `zai-org/GLM-5.3-Flash-BF16` into an **experts-only NVFP4** Hugging Face checkpoint on a single GB10 system by combining layerwise PTQ, Accelerate disk offload, streaming export, and shard-wise tensor-name reversal.

## Hardware acknowledgment

I do **not** currently own a DGX Spark. This reference run was accomplished using a **loaner DGX Spark provided by Weschera**. Their hardware access made it possible to develop, debug, and document this single-GB10 quantization/export workflow for the community.

## Reference result

| Item | Reference run |
|---|---:|
| Source | `zai-org/GLM-5.3-Flash-BF16` |
| Model | GLM-5.3-Flash, 320B total / 18B active |
| Hardware | 1× NVIDIA DGX Spark / GB10 — loaner provided by Weschera |
| Usable memory | ~121 GiB unified memory + 15 GiB swap |
| Source checkpoint | ~598.5 GiB BF16 |
| Quant policy | Routed MoE experts: NVFP4 W4A4; FP8 KV config |
| Output | **177.15 GiB** |
| Output shards | **48 safetensors** |
| Output bytes | **190,218,104,932** |
| Reference ModelOpt | `0.46.0rc2` |
| Reference Transformers | `5.16.1` |
| HF pack | `vcruz305/GLM-5.3-Flash-NVFP4` |

GLM-5.3-Flash is a 320B-parameter / 18B-active MoE model with native multimodality and hybrid sparse + linear attention. This repo starts from the **BF16** release, not the already-quantized FP8 checkpoint.

## What is novel here?

The NVFP4 policy itself is NVIDIA's **experts-only** ModelOpt recipe. This repo does **not** claim a new quantization algorithm.

The engineering contribution is the **single-GB10 execution path** for a source checkpoint far larger than Spark memory:

```text
~598.5 GiB BF16 checkpoint
        ↓
Accelerate device_map=auto + disk offload
        ↓
ModelOpt layerwise experts-only PTQ
        ↓
Routed experts → NVFP4 W4A4
Other non-targeted tensors remain at source precision
        ↓
ModelOpt state metadata only
        ↓
Offloaded streaming HF export
        ↓
4 GB temporary shards
        ↓
Reverse GLM5Next tensor mappings one shard at a time
        ↓
Validate hf_quant_config.json + finalized shards
        ↓
~177.15 GiB NVFP4 pack
```

The reference run used NVIDIA's `nvfp4_experts_only-kv_fp8_layerwise_offload` recipe, which sets layerwise calibration and `calib_mutates_weights=false` so calibrated weights can return to meta/offloaded state instead of accumulating in memory.

## Why the normal path fails on one Spark

GB10 unified memory is one physical pool. Treating `max_memory={0: ..., "cpu": ...}` as two independent piles of RAM can overcommit the machine.

Reference attempts:

| Load budget | Result |
|---|---|
| GPU `24GiB` + CPU `48GiB` | filled 15/15 GiB swap during load; aborted |
| GPU `6GiB` + CPU `8GiB` | Accelerate tried to offload the entire model to disk and refused |
| **GPU `40GiB` + CPU `24GiB`** | **load + layerwise calibration survived** |

The other major wall is export: GLM5Next has fused conversion rules (`gate_proj + up_proj`, stacked `down_proj`, fused conv1d projections). ModelOpt's streaming exporter can reject offloaded export when it sees tensor split rules and suggests exporting without offload — which cannot fit this checkpoint on 121 GiB UMA.

The reference workaround keeps the model offloaded, bypasses that conservative split assertion, streams 4 GB shards, and applies the reverse mappings **per shard** afterward. This is intentionally isolated in the script and should be re-evaluated against future ModelOpt releases.

## Disk requirement

Plan for **more than 1.2 TB free NVMe** during the job:

- ~598.5 GiB BF16 source
- ~550+ GiB Accelerate offload scratch at peak
- ~190–200 GiB packed output during export/finalization

Do not begin the quantization phase until the BF16 destination is complete.

## Installation

Use a GB10-compatible PyTorch/CUDA environment. NVIDIA's current ModelOpt package installs with:

```bash
pip install -U 'nvidia-modelopt[all]'
pip install -U 'transformers>=5.16.1' accelerate safetensors huggingface_hub
```

The **reference receipt** used ModelOpt `0.46.0rc2` and Transformers `5.16.1`. If reproducing the receipt exactly, pin to those versions rather than assuming a newer exporter behaves identically.

Verify GLM5Next directly:

```bash
python - <<'PY'
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
import transformers
print(transformers.__version__)
print(Glm5NextForConditionalGeneration)
PY
```

## Recipe

`recipes/nvfp4_experts_only-kv_fp8_layerwise_offload.yaml` is NVIDIA's Apache-2.0 recipe retained with its license header for reproducibility.

It targets routed MoE experts with NVFP4 W4A4 and configures FP8 KV, while enabling layerwise calibration with `calib_mutates_weights=false` for disk-offloaded PTQ.

## Run

Example directory layout:

```text
/mnt/models/GLM-5.3-Flash-BF16
/mnt/models/GLM-5.3-Flash-NVFP4
/mnt/work/glm53-flash/offload
```

Run:

```bash
python scripts/quant_glm53_nvfp4.py \
  --src /mnt/models/GLM-5.3-Flash-BF16 \
  --out /mnt/models/GLM-5.3-Flash-NVFP4 \
  --offload /mnt/work/glm53-flash/offload \
  --state /mnt/work/glm53-flash/state.json \
  --modelopt-state /mnt/work/glm53-flash/modelopt_state.pt \
  --recipe recipes/nvfp4_experts_only-kv_fp8_layerwise_offload.yaml \
  --gpu-memory 40GiB \
  --cpu-memory 24GiB \
  --max-shard-size 4GB \
  --calib-chunks 18 \
  --calib-seq 512
```

### Monitor the silent export phase

`quant.log` can appear frozen while `requantize_resmooth_fused_llm_layers` walks the model. On the reference machine, `/proc/<pid>/maps` advancing through decoder-layer tensor mappings was a better liveness signal.

```bash
python scripts/export_progress.py \
  --dest /mnt/models/GLM-5.3-Flash-NVFP4 \
  --state /mnt/work/glm53-flash/state.json \
  --process-match quant_glm53_nvfp4.py
```

Typical stages:

```text
resmooth_layer_38/44
writing_shards
```

An empty output directory during resmooth is **not** proof of a hang.

## Validate before uploading

```bash
python scripts/validate_pack.py /mnt/models/GLM-5.3-Flash-NVFP4
```

The validator requires:

- finalized `.safetensors` files
- no temporary `__shard_part_*` files
- `hf_quant_config.json`
- `quant_algo=NVFP4`
- a non-empty model index when sharded

Do not equate "calibration completed" with "checkpoint exported".

## Reference timing

The successful clean path on the reference Spark was approximately:

| Phase | Time |
|---|---:|
| Load | ~15–20 min |
| Layerwise calibration, 45/45 | **1h 24m** |
| Resmooth / export preparation | **~1h** |
| Streaming shard write | **~1–1.5h** |

The overall debugging campaign took roughly 12 hours because several failed paths repeated the expensive load/calibration stages. The point of this repository is to make the successful path reproducible instead of rediscovered.

## Known limitations

- **Not a one-Spark serving recipe.** The 177.15 GiB output exceeds ~121 GiB Spark UMA.
- The reference run is a **quantization/export receipt**, not a serving benchmark.
- Some `self_attn.conv1d.weight` tensors may remain fused if ModelOpt cannot reverse the 3D conversion rule; validate against the loader you plan to serve with.
- The streaming split-assert bypass is a version-sensitive workaround, not an upstream API guarantee.
- Do not assume the exact memory budget is optimal on every GB10 software stack.

## Reproducibility rules

1. Start from the BF16 model.
2. Use the experts-only offload recipe; do not silently substitute full-model or weight-only NVFP4.
3. Keep one quantization process alive at a time.
4. Do not materialize `model.state_dict()` to save the calibrated model; save ModelOpt metadata only.
5. Keep export offloaded.
6. Do not upload until `validate_pack.py` passes.
7. Report quantization/export and serving as separate claims.

## Credits

- GLM-5.3-Flash: Z.ai
- NVIDIA Model Optimizer / NVFP4 recipe: NVIDIA
- Loaner DGX Spark hardware access: Weschera
- Single-DGX-Spark offload/export workflow and reference reproduction: Victor Cruz (`vcruz305`)

## License

The repository code is Apache-2.0. The vendored NVIDIA recipe retains NVIDIA's original Apache-2.0 copyright and license header.
