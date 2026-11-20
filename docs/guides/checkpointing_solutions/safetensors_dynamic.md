(safetensors-dynamic)=

# Dynamic SafeTensors Loading (`safetensors_dynamic`)

`source_checkpoint_layout="safetensors_dynamic"` loads Hugging Face (HF) `.safetensors` checkpoints directly into sharded MaxText device weights at startup—converting shapes, stacking layers/experts, and optionally dequantizing block-scaled weights in memory without an offline `to_maxtext` conversion or an intermediate Orbax copy on GCS.

## Overview & When to Use

| Capability                                     | Offline `to_maxtext` + Orbax                                         | `safetensors_dynamic` (`safetensors_dynamic_streaming=False`, default)                                        | `safetensors_dynamic` (`safetensors_dynamic_streaming=True`, experimental)                    |
| :--------------------------------------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------------------------------- |
| **Workflow**                                   | Run offline `to_maxtext` on CPU, write Orbax to GCS, then load Orbax | Read sharded `.safetensors` directly via Orbax `SafetensorsLayout`, then transform/stack on device            | Stream decoder layers in `~4 GiB/host` chunks via `SafetensorsReader`, transforming in flight |
| **Extra GCS storage**                          | 2× (HF SafeTensors + Orbax copy)                                     | 0× extra converted checkpoint storage (`hf://` caches raw HF files under `<base_output_directory>/hf_cache/`) | 0× extra converted checkpoint storage                                                         |
| **Peak HBM during load**                       | Target MaxText weights                                               | Whole HF checkpoint (sharded) + MaxText weights (`~2×` weights, less when quantized)                          | MaxText target weights + up to 2 read calls (`~8 GiB/host`)                                   |
| **Entry points**                               | Training, `from_pretrained`, standalone                              | Training (`setup_training_state`), `from_pretrained`, `standalone_checkpointer`, LoRA base weights            | Same (`load_state_if_possible` dispatches both modes)                                         |
| **Pathways (`enable_single_controller=True`)** | Supported                                                            | Not supported (rejected by config validation)                                                                 | Not supported                                                                                 |

Use `safetensors_dynamic` when loading Hugging Face checkpoints directly for cold training runs, post-training, or evaluation (`from_pretrained`), especially for large models where single-host CPU `to_maxtext` hits host RAM limits or storing duplicate TB-scale Orbax checkpoints is undesirable. For repeated restarts of smaller models where a pre-converted Orbax checkpoint already exists, standard `source_checkpoint_layout="orbax"` restore remains faster.

## Quick Start

### 1. Load from a GCS or Local SafeTensors Directory

Point `load_parameters_path` at a directory containing `.safetensors` files (for example, `gs://...` or a local filesystem path) and set `source_checkpoint_layout="safetensors_dynamic"`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_dynamic_run \
    model_name=llama3.1-70b \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="gs://my-bucket/hf_checkpoints/Llama-3.1-70B/" \
    base_output_directory="gs://my-bucket/output/"
```

### 2. Load Directly from Hugging Face Hub (`hf://` or Auto-Lookup)

When `load_parameters_path` is a Hugging Face repo ID (`hf://<org>/<repo>` or `<org>/<repo>`) or is left empty (resolved via `HF_IDS[model_name]` with exact key matching, preserving `-Instruct` suffixes), **Host 0** caches missing `.safetensors` files (and `config.json` when present) into `<base_output_directory>/hf_cache/<repo_id>/` using a 32-worker process pool (`32 MB` chunks), while all other hosts wait on `sync_global_devices("dynamic_hf_download_complete")`.

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_hf_run \
    model_name=llama3.1-8b-Instruct \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="hf://meta-llama/Llama-3.1-8B-Instruct" \
    base_output_directory="gs://my-bucket/output/" \
    hf_access_token="${HF_TOKEN}"
```

If `load_parameters_path` is omitted (`""`), `base_output_directory` must start with `gs://`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_auto_hf_run \
    model_name=qwen3-30b-a3b \
    source_checkpoint_layout="safetensors_dynamic" \
    base_output_directory="gs://my-bucket/output/"
```

### 3. Enable Experimental Layer-by-Layer Streaming

To minimize transient HBM usage on large checkpoints where holding the raw HF state and converted MaxText weights simultaneously would exceed device memory, enable `safetensors_dynamic_streaming=True`:

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
    run_name=my_streaming_run \
    model_name=qwen3.5-397b-a17b \
    source_checkpoint_layout="safetensors_dynamic" \
    safetensors_dynamic_streaming=True \
    load_parameters_path="gs://my-bucket/hf_checkpoints/Qwen3.5-397B-A17B/" \
    base_output_directory="gs://my-bucket/output/"
```

### 4. Inference & Logit Verification (`from_pretrained`)

`model_creation_utils.from_pretrained` supports `source_checkpoint_layout="safetensors_dynamic"` (via `_load_safetensors_dynamic`, which frees initial random device arrays with `_free_device_memory` before calling `checkpointing.load_state_if_possible`). Inference and verification scripts such as `forward_pass_logit_checker` can load `.safetensors` directly:

```bash
python3 -m tests.utils.forward_pass_logit_checker src/maxtext/configs/base.yml \
    model_name=gemma3-4b \
    source_checkpoint_layout="safetensors_dynamic" \
    load_parameters_path="gs://my-bucket/hf_checkpoints/gemma-3-4b-it/" \
    weight_dtype=bfloat16 \
    enable_checkpointing=false \
    --run_hf_model=True \
    --hf_model_path="google/gemma-3-4b-it" \
    --max_kl_div=0.015
```

## Configuration Reference

| Flag                                   | Default      | Behavior with `safetensors_dynamic`                                                                                                                                                                                                   |
| :------------------------------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `source_checkpoint_layout`             | `"orbax"`    | Set to `"safetensors_dynamic"` to bypass Orbax checkpoint restoration and convert HF `.safetensors` on load (`src/maxtext/configs/types.py`, `base.yml`).                                                                             |
| `safetensors_dynamic_streaming`        | `False`      | `False`: loads all HF tensors sharded via Orbax `SafetensorsLayout` then transforms in memory (`load_dynamic.py`). `True` (experimental): streams decoder layers in bounded read calls via `hf_streaming_load.py`.                    |
| `load_parameters_path`                 | `""`         | Accepts a `gs://` directory, local directory, `hf://<org>/<repo>`, or `<org>/<repo>`. If empty, looks up the exact `model_name` key in `maxtext.utils.globals.HF_IDS`.                                                                |
| `base_output_directory`                | `""`         | Must have a `gs://` prefix whenever reading from HF Hub (`hf://`, `<org>/<repo>`, or empty `load_parameters_path`), because Host 0 caches shards to `<base_output_directory>/hf_cache/`.                                              |
| `hf_access_token`                      | `None`       | Optional Hugging Face token passed to `huggingface_hub.HfFileSystem` for gated repositories.                                                                                                                                          |
| `weight_dtype`                         | `"bfloat16"` | Loaded weights are cast to each target leaf's dtype in `abstract_params` (exceptions like DeepSeek FP32 router biases keep their target collection dtype).                                                                            |
| `scan_layers`                          | `True`       | Works with both `scan_layers=True` (including nested `-local_layers` block scans) and `scan_layers=False`. Note that `verify_and_sync_scan_layers` skips Orbax metadata checks when `source_checkpoint_layout="safetensors_dynamic"`. |
| `enable_single_controller`             | `False`      | Must be `False`. MaxText config validation raises a `ValueError` if combined with `source_checkpoint_layout="safetensors_dynamic"`.                                                                                                   |
| `enable_single_replica_ckpt_restoring` | `False`      | Applies only to Orbax checkpoint restores; ignored by `safetensors_dynamic`.                                                                                                                                                          |

## Architecture

The implementation spans four modules under `src/maxtext/checkpoint_conversion/utils/` and integrates cleanly into `checkpointing.load_state_if_possible`:

- **`src/maxtext/common/checkpointing.py`**: `load_state_if_possible` extracts parameter-only leaves (`_abstract_params`) for both Linen and NNX models, invokes `load_dynamic.load_safetensors_dynamic_state(path, params, maxtext_config)`, verifies every expected parameter leaf is materialized (`_raise_on_weight_mismatch`), and updates NNX state via `nnx.replace_by_pure_dict` (or `nnx.update` in `from_pretrained`).
- **`src/maxtext/checkpoint_conversion/utils/load_dynamic.py`**: Orchestrates HF Hub → GCS caching (`build_gcs_cache_worker`, `_hub_files_to_cache`), config/mapping resolution (`get_hf_config_and_mappings`), legacy prefix normalization (`rename_legacy_hf_keys` for older multimodal layouts such as Gemma 3), and dispatches to either the Orbax loader or `hf_streaming_load`.
- **`src/maxtext/checkpoint_conversion/utils/tensor_handling.py`**: Shared transformation, stacking, and resharding helpers used by both `load_dynamic` and `hf_streaming_load` (`get_hf_loading_function`, `single_stack_axis`, `stacked_axes`, `_binary_chunked_stack`, `_apply_hook_fns_any_host`, `_reshard`, and variable collection handling for `MoEBiasVar` / `Tid2EidVar`).
- **`src/maxtext/checkpoint_conversion/utils/hf_streaming_load.py` & `safetensors_reader.py`**: Implements the experimental streaming backend (`safetensors_dynamic_streaming=True`).

### Default Non-Streaming Path (`safetensors_dynamic_streaming=False`)

1. **Sharded Safetensors Read (`load_sharded_hf_state`)**:
   - Opens Orbax `Context(checkpoint_layout=CheckpointLayout.SAFETENSORS)` with `SafetensorsOptions(max_over_read_ratio=1.0)` on multi-process runs so fine-grained MoE files (>1,024 tensors) do not fall back to full-span reads.
   - Disables `gcsfs` 5 MiB read-ahead inside `_no_gcs_read_ahead()` (`default_block_size = 0`) so small ranged reads over GCS do not download extra megabytes per tensor.
   - On multi-process runs, `_leading_axis_shardings` shards row-major tensors along axis `0` only on a 2D `("process", "local")` mesh so every host reads one contiguous byte range per tensor; single-process runs use Orbax `construct_maximal_shardings`.
2. **In-Memory Transform, Stacking & Reshard (`transform_hf_state_to_mt_state`)**:
   - Tracks `remaining_uses` per HF tensor key and pops each tensor from `hf_state` immediately after its final use (handling reused HF sources and composite `(weight, scale)` or split `(wi_0, wi_1)` keys).
   - Executes `HOOK_FNS[model_key]` via `_apply_hook_fns_any_host`, falling back to `1 GiB` sliced `multihost_utils.process_allgather` only if a NumPy-only hook hits a non-addressable cross-host shard.
   - Stacks layers/experts in device memory using `_binary_chunked_stack`, keeping tensors already split across all devices in read layout until the final stack is resharded (`_reshard` chunks transfers >`1 GiB` along an unsplit axis to avoid device-memory OOMs).

### Experimental Streaming Path (`safetensors_dynamic_streaming=True`)

1. **Header & Write Plan (`build_plan` / `pack_groups`)**:
   - `SafetensorsReader` parses `.safetensors` headers once on creation and performs parallel `32 MiB` ranged reads across `64` threads directly onto local TPU chips.
   - `load_hf_params_streaming` groups HF sources by decoder layer (`READ_BYTES_PER_HOST = 4 GiB` per host) and preallocates the destination MaxText arrays.
2. **Prefetched Layer-by-Layer Streaming**:
   - Reads the next layer group in a background thread (`prefetch=True`) while running per-shard device hooks and jitted `dynamic_update_slice` writes for the current group, blocking until writes complete (`jax.block_until_ready`) and dropping `hf_arrays` before advancing.

## On-the-Fly DeepSeek FP8 / MXFP4 Dequantization

For quantized Hugging Face DeepSeek checkpoints (DeepSeek-V3/V3.2 FP8 `_scale_inv` and DeepSeek-V4 hybrid FP8 + packed MXFP4 `.scale`), the non-streaming path dequantizes directly into BF16 during loading instead of requiring an offline `deepseek_dequantize.py` pass:

- **Automatic Detection (`load_dynamic.py`)**: `get_hf_config_and_mappings` reads `quantization_config` from the checkpoint's `config.json` when available, and on the non-streaming path also inspects `raw_hf_state.keys()` for `*_scale_inv` or `*.scale` tensors so quantized weights pair only with scale tensors actually present in the checkpoint (leaving BF16 checkpoints and offline `to_maxtext.py` unchanged).
- **Per-Shard `shard_map` Conversion (`param_mapping.py`)**:
  - `dequantize_block_scaled` handles both **FP8 (`float8_e4m3fn`)** with float or E8M0 block scales (`128×128`, `32×32`, or arbitrary `weight_block_size`, including non-divisible dimensions) and **packed MXFP4 (`int8`/`uint8`)** unpacked from two 4-bit E2M1 nibbles per byte with IEEE E8M0 `.scale` decoding (`jax.lax.bitcast_convert_type(..., jnp.float8_e8m0fnu).astype(jnp.float32)`).
  - Runs per shard via `jax.shard_map` to avoid XLA block-partitioning bugs at scale, keeping row-aligned scales (such as `1×32` MXFP4 `.scale`) co-sharded with zero collectives and replicating smaller/unaligned scales cleanly on the weight's mesh.
- **Note on `"F8_E8M0"` Headers**: Orbax `SafetensorsLayout` supports `I8`, `U8`, `F8_E4M3`, `F8_E5M2`, and float scale tensors (`_scale_inv` / `U8` `.scale`), but `"F8_E8M0"` in safetensors headers is not currently supported by `SafetensorsLayout` or `SafetensorsReader`.

## Troubleshooting

- **`ValueError: source_checkpoint_layout='safetensors_dynamic' is not supported on the Pathways backend`**: Pathways single-controller (`enable_single_controller=True`) is unsupported; run on McJAX or convert offline with `to_maxtext.py`.
- **`ValueError: base_output_directory with gs:// prefix is required for huggingface downloads`**: Set `base_output_directory="gs://<bucket>/..."` when passing an HF repo ID or omitting `load_parameters_path`, or point `load_parameters_path` to an already-downloaded `gs://` or local `.safetensors` directory.
- **Weight-mismatch errors after load**: Verify `model_name` (including `-Instruct` variants), `scan_layers`, and architectural overrides match the `.safetensors` checkpoint, as `_raise_on_weight_mismatch` fails if any expected MaxText parameter remains unmaterialized.
