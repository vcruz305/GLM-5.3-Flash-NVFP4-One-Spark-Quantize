# Troubleshooting

## Swap fills during `from_pretrained`

Do not treat GPU and CPU `max_memory` values as additive physical memory on GB10 UMA. The reference 24GiB GPU + 48GiB CPU dispatch overcommitted the machine and filled swap.

Use the reference 40GiB + 24GiB dispatch as a starting point, then re-measure your software stack.

## Accelerate says the whole model is being offloaded to disk

The dispatch caps are too small. A clean offload directory with 6GiB + 8GiB reproduced this failure.

## Calibration finished but output is empty

That can be normal. ModelOpt's resmooth/streaming-preparation phase may walk all 45 decoder layers before the first shard file appears.

Use `scripts/export_progress.py`; watch for mapped layers advancing.

## Export says reverse tensor-level split rules are unsupported with offload

That is the GLM5Next export wall this repo works around for the reference ModelOpt version. Exporting without offload is not viable for a ~598.5 GiB source on ~121 GiB UMA.

The script isolates the split assertion bypass and applies reverse rules per shard. Re-test against newer ModelOpt versions before assuming the workaround is still needed.

## `SIGUSR1` kills the job

Do not use it as a stack-dump trick on the live Python process. The script ignores `SIGUSR1` defensively on Linux.

## `quant.log` looks frozen

Expected during the silent resmooth phase. Observe `/proc/<pid>/maps`, process RSS/I/O, and destination growth instead.

## Temporary `__shard_part_*` files exist

The pack is not finalized. Do not upload it yet.

## `hf_quant_config.json` is absent

The pack is not validated. Do not upload it.

## Some 3D `self_attn.conv1d.weight` tensors remain fused

The reference run observed `QuantConversionUnsupportedError` for some 3D reverse rules. The script logs the condition and retains the fused tensor for that shard instead of destroying the export. Validate against your intended server/loader before calling the pack serve-ready.
