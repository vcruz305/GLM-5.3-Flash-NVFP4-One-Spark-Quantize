# Reference reproduction notes

These values describe the successful reference run. They are receipts, not universal tuning constants.

## Hardware

- NVIDIA DGX Spark / GB10
- ~121 GiB unified memory available to the OS/workload
- 15 GiB swap
- NVMe scratch

## Software

- ModelOpt 0.46.0rc2
- Transformers 5.16.1 (`glm5_next`)
- GB10-compatible PyTorch + CUDA 13 environment

## ModelOpt policy

The quantization policy is NVIDIA's experts-only NVFP4 recipe:

- routed MoE experts: NVFP4 W4A4
- shared / non-targeted modules: source precision
- FP8 KV configuration in the exported quant metadata
- layerwise max calibration
- `calib_mutates_weights=false`

## Working load budget

```python
max_memory={0: "40GiB", "cpu": "24GiB"}
offload_state_dict=True
low_cpu_mem_usage=True
device_map="auto"
```

On GB10 UMA these caps are dispatch limits inside one unified pool, not separate physical memories.

## Calibration

Reference workload used 18 × 512-token calibration chunks. The purpose was a practical amax calibration pass, not a benchmark dataset.

Reference layerwise pass: 45/45 decoder layers, about 1h24m.

## Export

The export is the unusual part:

1. Confirm the model remains Accelerate-offloaded.
2. Bypass ModelOpt's conservative `_assert_no_split_rules` guard for the reference version.
3. `export_hf_checkpoint(..., max_shard_size="4GB")`.
4. During the silent resmooth phase, watch `/proc/<pid>/maps` for mapped decoder layers advancing.
5. After finalized shards exist, apply GLM5Next reverse rules one shard at a time.
6. Rebuild the index `weight_map` from the final shard contents.
7. Reverse quant-config tensor names when possible.
8. Require `hf_quant_config.json` and `quant_algo=NVFP4` before upload.

## Final receipt

- 48 safetensors shards
- 190,218,104,932 bytes
- 177.15 GiB
- `quant_algo=NVFP4`
- `kv_cache_quant_algo=FP8` in the reference pack

The output exceeds one Spark's UMA. This is why the project language must always say **quantized/exported on one Spark**, not **runs on one Spark**.
