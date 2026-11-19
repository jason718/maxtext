# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The NNX Unified TrainState."""

import fnmatch
import math
from typing import Any

from flax import nnx
from flax.nnx.training import optimizer as nnx_opt
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, reshard

from maxtext.utils import max_logging

_GIB_BYTES = 1 << 30
_RESIDUAL16_OPT_KEYS = ("inner", "master_residuals")


def _path_to_key_tuple(path: tuple[Any, ...]) -> tuple[Any, ...]:
  """Normalizes a JAX tree path or Flax flat-state path into raw key primitives."""
  out = []
  for entry in path:
    if hasattr(entry, "key"):
      out.append(entry.key)
    elif hasattr(entry, "idx"):
      out.append(entry.idx)
    elif hasattr(entry, "name"):
      out.append(entry.name)
    else:
      out.append(entry)
  return tuple(out)


def _flatten_leaves_by_key_tuple(tree: Any) -> dict[tuple[Any, ...], Any]:
  """Flattens a nested dict / nnx.State tree into `{key_tuple: leaf}`."""
  if not tree:
    return {}
  return {
      _path_to_key_tuple(p): leaf
      for p, leaf in jax.tree_util.tree_leaves_with_path(
          tree,
          is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct, nnx.Variable)),
      )
  }


def is_residual16_opt_state_dict(opt_state: Any) -> bool:
  """Returns True when `opt_state` has the Residual16Optimizer `{inner, master_residuals}` structure."""
  if not isinstance(opt_state, (dict, nnx.State)):
    return False
  return "inner" in opt_state and ("master_residuals" in opt_state or set(opt_state.keys()) == {"inner"})


def split_f32_to_bf16_residual16(x_f32: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Splits a float32 array into a rounded bfloat16 working weight and signed int16 residual.

  Every non-NaN IEEE-754 float32 bit pattern (including +/-0.0, subnormals, exact halfway ties
  rounded away from zero so -32768 fits in signed int16, +/-finfo.max, and +/-Inf) reconstructs
  bit-for-bit via `reconstruct_f32_from_bf16_residual16`. NaN bit patterns are canonicalized to
  quiet NaN (0x7FC00000) so adding 0x8000 never wraps payload bits such as 0xFFFFFFFF into +0.0.
  """
  x_f32 = jnp.asarray(x_f32, dtype=jnp.float32)
  bits = jax.lax.bitcast_convert_type(x_f32, jnp.uint32)
  bits = jnp.where(jnp.isnan(x_f32), jnp.uint32(0x7FC00000), bits)
  w_u16 = ((bits + jnp.uint32(0x8000)) >> jnp.uint32(16)).astype(jnp.uint16)
  w_bf16 = jax.lax.bitcast_convert_type(w_u16, jnp.bfloat16)
  res_i16 = jax.lax.bitcast_convert_type((bits & jnp.uint32(0xFFFF)).astype(jnp.uint16), jnp.int16)
  return w_bf16, res_i16


def reconstruct_f32_from_bf16_residual16(w_bf16: jax.Array, res_i16: jax.Array) -> jax.Array:
  """Reconstructs the exact float32 master weight from a `(bfloat16, int16)` residual16 pair."""
  w_bf16 = jnp.asarray(w_bf16, dtype=jnp.bfloat16)
  res_i16 = jnp.asarray(res_i16, dtype=jnp.int16)
  abstract_mesh = jax.sharding.get_abstract_mesh()
  if not abstract_mesh.empty and any(t == AxisType.Explicit for t in abstract_mesh.axis_types):
    r_spec = jax.typeof(res_i16).sharding.spec
    if jax.typeof(w_bf16).sharding.spec != r_spec:
      w_bf16 = reshard(w_bf16, r_spec)
  w_u32 = jax.lax.bitcast_convert_type(w_bf16, jnp.uint16).astype(jnp.uint32)
  r_u32 = jax.lax.bitcast_convert_type(res_i16.astype(jnp.int32), jnp.uint32)
  bits = (w_u32 << jnp.uint32(16)) + r_u32
  return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _reconstruct_f32_concrete_for_save(w_bf16: jax.Array, res_i16: jax.Array) -> jax.Array:
  """Reconstructs a concrete float32 array for checkpoint saving onto pinned_host when available."""
  if isinstance(w_bf16, jax.core.Tracer) or isinstance(res_i16, jax.core.Tracer):
    return reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
  w_sharding = getattr(w_bf16, "sharding", None)
  compute_sharding = w_sharding.with_memory_kind("device") if isinstance(w_sharding, NamedSharding) else None
  w_dev = (
      jax.device_put(w_bf16, compute_sharding)
      if compute_sharding is not None and getattr(w_sharding, "memory_kind", None) == "pinned_host"
      else w_bf16
  )
  r_dev = (
      jax.device_put(res_i16, compute_sharding)
      if compute_sharding is not None and getattr(res_i16, "sharding", None) != compute_sharding
      else res_i16
  )
  f32_arr = reconstruct_f32_from_bf16_residual16(w_dev, r_dev)
  if hasattr(w_sharding, "with_memory_kind"):
    try:
      host_sharding = w_sharding.with_memory_kind("pinned_host")
      f32_host = jax.device_put(f32_arr, host_sharding)
      f32_host.block_until_ready()
      del f32_arr
      return f32_host
    except (ValueError, RuntimeError):
      pass
  return f32_arr


def _split_f32_concrete_for_restore(
    raw_val: jax.Array,
    target_w_sharding: Any,
    target_res_sharding: Any,
) -> tuple[jax.Array, jax.Array]:
  """Splits a concrete restored weight into `(w_bf16, res_i16)` placed on target shardings."""
  raw_sharding = getattr(raw_val, "sharding", None)
  raw_dev = (
      jax.device_put(raw_val, raw_sharding.with_memory_kind("device"))
      if isinstance(raw_sharding, NamedSharding) and raw_sharding.memory_kind == "pinned_host"
      else raw_val
  )
  w_bf16, res_i16 = split_f32_to_bf16_residual16(raw_dev)
  if target_w_sharding is not None:
    w_bf16 = jax.device_put(w_bf16, target_w_sharding)
  if target_res_sharding is not None:
    res_i16 = jax.device_put(res_i16, target_res_sharding)
  return w_bf16, res_i16


def should_use_residual16_for_param(path: tuple[Any, ...], var: nnx.Variable, config: Any = None) -> bool:
  """Returns whether `var` at `path` should be stored as `bf16` working weight + `int16` residual."""
  val = var.get_value() if isinstance(var, nnx.Variable) else var
  dtype = getattr(val, "dtype", None)
  if dtype not in (jnp.float32, jnp.bfloat16):
    return False
  str_keys = tuple(str(k) for k in _path_to_key_tuple(path))
  # GatedDeltaNet / KDA log-decay and dt bias parameters (`A_log`, `dt_bias`) are computed in float32
  # during the forward recurrence (`g = -exp(A_log) * softplus(...)`) to prevent rounding amplification
  # across cumulative decay sums; preserve float32 working weights whenever initialized in float32.
  if dtype == jnp.float32 and any(k in ("A_log", "dt_bias") for k in str_keys):
    return False
  if config is not None:
    # Auxiliary-loss-free MoE router bias (`routed_bias`) is updated directly on `gate.bias.value`
    # outside Optax in `train.py`; keep it in float32 so small `routed_bias_update_rate` increments
    # accumulate without bfloat16 rounding loss.
    if getattr(config, "routed_bias", False) and "gate" in str_keys and str_keys and str_keys[-1] == "bias":
      return False
    if getattr(config, "float32_gate_logits", False) and any(k in ("gate", "shared_expert_gate") for k in str_keys):
      return False
    keep_fp32 = getattr(config, "keep_in_fp32_modules", None) or ()
    if isinstance(keep_fp32, str):
      keep_fp32 = (keep_fp32,)
    if keep_fp32:
      dotted_path = ".".join(str_keys)
      slash_path = "/".join(str_keys)
      module_dotted = ".".join(str_keys[:-1]) if len(str_keys) > 1 else dotted_path
      if any(
          p in str_keys
          or fnmatch.fnmatch(dotted_path, p)
          or fnmatch.fnmatch(slash_path, p)
          or fnmatch.fnmatch(module_dotted, p)
          for p in keep_fp32
      ):
        return False
    weight_dtype = getattr(config, "weight_dtype", jnp.float32)
    if weight_dtype in (jnp.bfloat16, "bfloat16"):
      # Under weight_dtype=bfloat16, leave intentionally float32 parameters in float32.
      return dtype == jnp.bfloat16
  return True


def _collect_mu_specs(opt_state_node: Any, out: dict[tuple[Any, ...], Any]) -> None:
  """Collects per-parameter PartitionSpecs from `mu` trees inside an Optax state."""
  if isinstance(opt_state_node, dict):
    if "mu" in opt_state_node and isinstance(opt_state_node["mu"], (dict, nnx.State)):
      for p, leaf in jax.tree_util.tree_leaves_with_path(opt_state_node["mu"]):
        if hasattr(leaf, "dtype"):
          out.setdefault(_path_to_key_tuple(p), jax.typeof(leaf).sharding.spec)
    for v in opt_state_node.values():
      _collect_mu_specs(v, out)
  elif isinstance(opt_state_node, (tuple, list)):
    for v in opt_state_node:
      _collect_mu_specs(v, out)
  elif hasattr(opt_state_node, "_fields"):
    for field_name in opt_state_node._fields:
      val = getattr(opt_state_node, field_name)
      if field_name == "mu" and isinstance(val, (dict, nnx.State)):
        for p, leaf in jax.tree_util.tree_leaves_with_path(val):
          if hasattr(leaf, "dtype"):
            out.setdefault(_path_to_key_tuple(p), jax.typeof(leaf).sharding.spec)
      else:
        _collect_mu_specs(val, out)


class Residual16Optimizer(nnx.Optimizer):
  """NNX Optimizer storing working weights in bfloat16 and low-16-bit residuals in int16.

  Model parameters stay in `bfloat16` (2 bytes/param) for forward/backward passes, while
  `self.opt_state["master_residuals"]` stores the signed `int16` low 16 bits (2 bytes/param).
  During `update`, exact `float32` master weights are reconstructed on the fly, passed to
  `self.tx.update` + `optax.apply_updates`, and split back into `(bfloat16, int16)` — saving
  2 bytes/param vs keeping both `float32` master weights (4 bytes) and `bfloat16` weights.
  """

  def __init__(
      self,
      model: nnx.Module,
      tx: Any,
      config: Any = None,
      *,
      wrt: nnx.filterlib.Filter = nnx.Param,
      graph: bool | None = None,
  ):
    # pylint: disable=super-init-not-called
    self.graph = graph
    self.step = nnx_opt.OptState(jnp.array(0, dtype=jnp.uint32))
    self.tx = tx
    self.wrt = wrt

    params_state = nnx.state(model, wrt, graph=graph)
    split_cache: dict[tuple[Any, ...], tuple[jax.Array, jax.Array, jax.Array]] = {}

    def _prepare_model_var(jax_path, var):
      key_tuple = _path_to_key_tuple(jax_path)
      if not should_use_residual16_for_param(key_tuple, var, config):
        return var
      val = var.get_value()
      if val.dtype == jnp.float32:
        w_bf16, res_i16 = split_f32_to_bf16_residual16(val)
        f32_val = val
      else:
        w_bf16 = jnp.asarray(val, dtype=jnp.bfloat16)
        res_i16 = jnp.zeros(val.shape, dtype=jnp.int16)
        f32_val = w_bf16.astype(jnp.float32)
      split_cache[key_tuple] = (w_bf16, res_i16, f32_val)
      return var.replace(value=w_bf16)

    updated_model_params = jax.tree_util.tree_map_with_path(
        _prepare_model_var,
        params_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    f32_params_for_tx = jax.tree_util.tree_map_with_path(
        lambda jax_path, var: (
            var.replace(value=split_cache[_path_to_key_tuple(jax_path)][2])
            if _path_to_key_tuple(jax_path) in split_cache
            else var
        ),
        params_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    eligible_state = params_state.filter(lambda path, var: _path_to_key_tuple(path) in split_cache)
    residual_vars = jax.tree_util.tree_map_with_path(
        lambda jax_path, var: var.replace(value=split_cache[_path_to_key_tuple(jax_path)][1]),
        eligible_state,
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

    nnx.update(model, updated_model_params)
    inner_opt_state = nnx_opt.to_opt_state(tx.init(f32_params_for_tx))
    master_residuals = nnx_opt.to_opt_state(residual_vars) if split_cache else {}
    self.opt_state = nnx.data(
        {
            "inner": inner_opt_state,
            "master_residuals": master_residuals,
        }
    )

  def update(self, model: nnx.Module, grads: Any, /, **kwargs) -> Any:
    """Reconstructs float32 master weights, runs `tx.update`, and splits back to `(bf16, int16)`."""
    param_arrays = nnx.as_pure(nnx.state(model, self.wrt, graph=self.graph))
    grad_arrays = nnx.as_pure(nnx.state(grads, self.wrt, graph=self.graph))
    opt_state_arrays = nnx.as_pure(self.opt_state)
    kwargs_arrays = nnx.as_pure(kwargs)

    inner_state = opt_state_arrays["inner"]
    residuals_pure = opt_state_arrays["master_residuals"]
    flat_residuals = _flatten_leaves_by_key_tuple(residuals_pure)

    abstract_mesh = jax.sharding.get_abstract_mesh()
    has_explicit_axes = not abstract_mesh.empty and any(t == AxisType.Explicit for t in abstract_mesh.axis_types)
    opt_specs: dict[tuple[Any, ...], Any] = {}
    if has_explicit_axes:
      for k, res_leaf in flat_residuals.items():
        opt_specs[k] = jax.typeof(res_leaf).sharding.spec
      _collect_mu_specs(inner_state, opt_specs)

    def _reconstruct_param(jax_path, w):
      key_tuple = _path_to_key_tuple(jax_path)
      if key_tuple in flat_residuals:
        return reconstruct_f32_from_bf16_residual16(w, flat_residuals[key_tuple])
      if has_explicit_axes and key_tuple in opt_specs and jax.typeof(w).sharding.spec != opt_specs[key_tuple]:
        w = reshard(w, opt_specs[key_tuple])
      return w

    f32_params = jax.tree_util.tree_map_with_path(_reconstruct_param, param_arrays)
    if has_explicit_axes and opt_specs:
      grad_arrays = jax.tree_util.tree_map_with_path(
          lambda p, g: (
              reshard(g, opt_specs[_path_to_key_tuple(p)])
              if _path_to_key_tuple(p) in opt_specs and jax.typeof(g).sharding.spec != opt_specs[_path_to_key_tuple(p)]
              else g
          ),
          grad_arrays,
      )

    updates, new_inner_state = self.tx.update(grad_arrays, inner_state, f32_params, **kwargs_arrays)
    new_f32_params = jax.tree.map(
        lambda p, u: None if p is None else jnp.asarray(p + u, dtype=p.dtype), f32_params, updates
    )

    updated_residuals_by_key: dict[tuple[Any, ...], jax.Array] = {}

    def _split_updated_param(jax_path, p_new):
      key_tuple = _path_to_key_tuple(jax_path)
      if key_tuple in flat_residuals:
        w_bf16, res_i16 = split_f32_to_bf16_residual16(p_new)
        updated_residuals_by_key[key_tuple] = res_i16
        return w_bf16
      return p_new

    new_working_params = jax.tree_util.tree_map_with_path(_split_updated_param, new_f32_params)
    new_residuals = (
        jax.tree_util.tree_map_with_path(
            lambda jax_path, _: updated_residuals_by_key[_path_to_key_tuple(jax_path)],
            residuals_pure,
        )
        if flat_residuals
        else residuals_pure
    )

    nnx.update(model, new_working_params)
    nnx.update(
        self.opt_state,
        nnx.state({"inner": new_inner_state, "master_residuals": new_residuals}, graph=self.graph),
    )
    self.step[...] += 1
    return updates

  def apply_master_param_transform(self, model: nnx.Module, transform_fn) -> None:
    """Applies `transform_fn(jax_path, param_f32)` on reconstructed master weights and re-splits."""
    param_arrays = nnx.as_pure(nnx.state(model, self.wrt, graph=self.graph))
    opt_state_arrays = nnx.as_pure(self.opt_state)
    residuals_pure = opt_state_arrays["master_residuals"]
    flat_residuals = _flatten_leaves_by_key_tuple(residuals_pure)

    f32_params = jax.tree_util.tree_map_with_path(
        lambda p, w: (
            reconstruct_f32_from_bf16_residual16(w, flat_residuals[_path_to_key_tuple(p)])
            if _path_to_key_tuple(p) in flat_residuals
            else w
        ),
        param_arrays,
    )
    transformed_f32 = jax.tree_util.tree_map_with_path(transform_fn, f32_params)

    updated_residuals_by_key: dict[tuple[Any, ...], jax.Array] = {}

    def _resplit(jax_path, p_new):
      key_tuple = _path_to_key_tuple(jax_path)
      if key_tuple in flat_residuals:
        w_bf16, res_i16 = split_f32_to_bf16_residual16(p_new)
        updated_residuals_by_key[key_tuple] = res_i16
        return w_bf16
      return p_new

    new_working_params = jax.tree_util.tree_map_with_path(_resplit, transformed_f32)
    new_residuals = (
        jax.tree_util.tree_map_with_path(
            lambda p, _: updated_residuals_by_key[_path_to_key_tuple(p)],
            residuals_pure,
        )
        if flat_residuals
        else residuals_pure
    )
    nnx.update(model, new_working_params)
    nnx.update(
        self.opt_state,
        nnx.state({"inner": opt_state_arrays["inner"], "master_residuals": new_residuals}, graph=self.graph),
    )


def create_train_state(
    model: nnx.Module,
    tx: Any,
    config: Any = None,
    *,
    wrt: nnx.filterlib.Filter = nnx.Param,
) -> "TrainStateNNX":
  """Creates a `TrainStateNNX` using `Residual16Optimizer` when `fp32_master_storage == 'residual16'`."""
  if getattr(config, "fp32_master_storage", "f32") == "residual16":
    optimizer: nnx.Optimizer = Residual16Optimizer(model, tx, config=config, wrt=wrt)
  else:
    optimizer = nnx.Optimizer(model, tx, wrt=wrt)
  return TrainStateNNX(model, optimizer)


def _per_device_leaf_bytes(aval: Any, sharding: Any) -> int:
  """Computes per-device byte footprint of an optimizer state leaf (0 for scalars)."""
  shape = getattr(aval, "shape", ())
  dtype = getattr(aval, "dtype", None)
  if dtype is None or len(shape) == 0:
    return 0
  shard_shape = sharding.shard_shape(shape) if isinstance(sharding, NamedSharding) else shape
  return int(math.prod(int(d) for d in shard_shape)) * int(jnp.dtype(dtype).itemsize)


def pick_device_resident_optimizer_leaves(
    opt_abstract_tree: Any,
    opt_sharding_tree: Any,
    device_resident_gib: float,
) -> tuple[set[tuple[Any, ...]], int, int]:
  """Selects optimizer leaves to keep on HBM (`device`) up to `device_resident_gib` GiB/device.

  Candidate non-scalar leaves are sorted deterministically by `(-per_device_bytes, canonical_path)`
  so every SPMD process/host makes identical device vs `pinned_host` placement decisions.

  Returns:
    `(resident_key_tuples, resident_bytes_per_device, offloaded_bytes_per_device)`.
  """
  budget_bytes = int(float(device_resident_gib) * _GIB_BYTES) if device_resident_gib > 0.0 else 0
  aval_leaves, _ = jax.tree_util.tree_flatten_with_path(
      opt_abstract_tree,
      is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
  )
  sharding_leaves, _ = jax.tree_util.tree_flatten_with_path(
      opt_sharding_tree,
      is_leaf=lambda x: isinstance(x, NamedSharding),
  )
  candidates: list[tuple[int, str, tuple[Any, ...]]] = []
  total_bytes = 0
  for (aval_path, aval), (_, sharding) in zip(aval_leaves, sharding_leaves):
    key_tuple = _path_to_key_tuple(aval_path)
    leaf_bytes = _per_device_leaf_bytes(aval, sharding)
    total_bytes += leaf_bytes
    if leaf_bytes > 0:
      canonical_path = "/".join(f"{type(k).__name__}:{k}" for k in key_tuple)
      candidates.append((leaf_bytes, canonical_path, key_tuple))

  candidates.sort(key=lambda item: (-item[0], item[1]))
  resident: set[tuple[Any, ...]] = set()
  resident_bytes = 0
  for leaf_bytes, _, key_tuple in candidates:
    if resident_bytes + leaf_bytes <= budget_bytes:
      resident.add(key_tuple)
      resident_bytes += leaf_bytes

  return resident, resident_bytes, total_bytes - resident_bytes


def partition_optimizer_memory_shardings(
    abstract_state: nnx.State,
    state_mesh_shardings: nnx.State,
    device_resident_gib: float,
) -> nnx.State:
  """Partitions optimizer `NamedSharding`s between HBM (`device`) and `pinned_host` under a GiB budget."""
  resident_keys, resident_bytes, offloaded_bytes = pick_device_resident_optimizer_leaves(
      abstract_state.optimizer,
      state_mesh_shardings.optimizer,
      device_resident_gib=device_resident_gib,
  )
  max_logging.log(
      f"optimizer_device_resident_gib={device_resident_gib:.3f}: keeping {len(resident_keys)} optimizer leaves "
      f"({resident_bytes / _GIB_BYTES:.3f} GiB/device) on HBM and offloading "
      f"{offloaded_bytes / _GIB_BYTES:.3f} GiB/device to pinned_host."
  )

  # Explicitly tag non-offloaded state leaves with memory_kind="device" so JAX's donate_argnums
  # buffer aliasing pairs matching memory spaces cleanly when optimizer leaves mix device & pinned_host.
  state_mesh_shardings = jax.tree_util.tree_map(
      lambda s: s.with_memory_kind("device") if isinstance(s, NamedSharding) and s.memory_kind is None else s,
      state_mesh_shardings,
      is_leaf=lambda x: isinstance(x, NamedSharding),
  )

  def _assign_opt_memory(jax_path, sharding: NamedSharding) -> NamedSharding:
    key_tuple = _path_to_key_tuple(jax_path)
    kind = "device" if key_tuple in resident_keys else "pinned_host"
    return sharding.with_memory_kind(kind)

  state_mesh_shardings.optimizer = jax.tree_util.tree_map_with_path(
      _assign_opt_memory,
      state_mesh_shardings.optimizer,
      is_leaf=lambda x: isinstance(x, NamedSharding),
  )
  return state_mesh_shardings


class TrainStateNNX(nnx.Module):
  """A unified container for NNX models and optimizers.

  This replaces Linen's TrainState for checkpointing.

  Linen TrainState pytree:
    {"params": {...}, "opt_state": {}...}
  TrainStateNNX state pytree:
    {"model": {...}, "optimizer": {"opt_state": {...}}}
  """

  def __init__(
      self,
      model: nnx.Module,
      optimizer: nnx.Optimizer | None,
  ):
    self.model = model
    self.optimizer = optimizer

  def apply_gradients(self, grads: Any, **kwargs):
    """Mimics the Linen apply_gradients function.

    Updates the optimizer state, applies updates to parameters, and increments
    the step counter. Only updates `self.model`. Extra kwargs (e.g. loss/grad_norm
    for the skip-step-on-spikes optimizer) are forwarded to the optax update.
    """
    if self.optimizer is None:
      raise RuntimeError(
          "Cannot call apply_gradients on a TrainStateNNX initialized without"
          " an optimizer. This usually happens when the state was created for"
          " inference only."
      )
    self.optimizer.update(self.model, grads, **kwargs)


# On-disk checkpoint format.
#
# A run saves in the same on-disk layout that Linen used, so old and new checkpoints are
# interchangeable. The NNX state pure dict differs from Linen's in three ways, all
# reshaped below at save time:
#   1. top-level keys: {model, optimizer:{step, opt_state}} -> {params:{params:...}, step, opt_state}
#   2. weights: model/... -> params/params/... (Linen `params` collection)
#   3. opt_state: int-keyed dict (empty entries skipped) -> list with None for EmptyState,
#      and mu/nu wrapped under the `params` collection
# Persistent NNX state Linen's params/opt_state/step layout can't hold -- rngs/dropout,
# batch stats, and any custom non-Param variable -- is split out by Variable type and stored
# under an `nnx_aux` subtree of `items` (see `to_checkpoint_dict`), so it survives a resume
# instead of resetting.

_NNX_RNG_STATE_KEYS = ("rngs", "dropout")


def _cast_step(step, dtype):
  """Casts the step's dtype, handling both concrete arrays and abstract ShapeDtypeStruct.

  NNX stores step as uint32 and Linen as int32; this also runs on the abstract
  (SDS) state when building a restore target, so it can't assume concrete
  values.
  """
  if isinstance(step, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(step.shape, dtype, sharding=getattr(step, "sharding", None))
  sharding = getattr(step, "sharding", None)
  if isinstance(sharding, NamedSharding) and sharding.memory_kind == "pinned_host":
    step_dev = jax.device_put(step, sharding.with_memory_kind("device"))
    casted = jnp.asarray(step_dev, dtype=dtype)
    return jax.device_put(casted, sharding) if dtype == jnp.uint32 else casted
  return jnp.asarray(step, dtype=dtype)


def _strip_rng_state(tree):
  """Removes the NNX-only 'rngs'/'dropout' subtrees that Linen doesn't carry.

  Subtrees that become empty after stripping are dropped too, so the result has
  no keys Linen wouldn't also have.
  """
  if not isinstance(tree, dict):
    return tree
  out = {}
  for k, v in tree.items():
    if k in _NNX_RNG_STATE_KEYS:
      continue
    stripped = _strip_rng_state(v)
    if isinstance(stripped, dict) and not stripped:
      continue
    out[k] = stripped
  return out


def _wrap_mu_nu_with_params(state):
  """Wraps mu/nu under an inner 'params' key (the Linen collection)."""
  if not isinstance(state, dict):
    return state
  return {k: {"params": v} if k in ("mu", "nu") and isinstance(v, dict) else v for k, v in state.items()}


def _as_chain_index(key):
  """Returns the int index for an int or digit-string key, else None."""
  if isinstance(key, int):
    return key
  if isinstance(key, str) and key.isdigit():
    return int(key)
  return None


def _opt_state_to_linen(opt_state):
  """Reshapes the NNX opt_state to Linen's on-disk layout.

  A chain (optax.adamw) is an int-keyed dict -> list-with-None. An un-chained
  optimizer (adam_pax) is flat (count/mu/nu) and stays a dict.
  """
  if not isinstance(opt_state, dict):
    return opt_state
  indices = [_as_chain_index(k) for k in opt_state.keys()]
  if not indices or any(i is None for i in indices):
    return _wrap_mu_nu_with_params(opt_state)
  chain = [None] * (max(indices) + 1)
  for key, idx in zip(opt_state.keys(), indices):
    chain[idx] = _wrap_mu_nu_with_params(opt_state[key])
  return chain


def to_linen_checkpoint_dict(nnx_pure_dict):
  """Reshapes a TrainStateNNX pure dict ({model, optimizer}) into the Linen on-disk layout."""
  if not isinstance(nnx_pure_dict, dict):
    return nnx_pure_dict
  result = {}
  optimizer = nnx_pure_dict.get("optimizer")
  raw_opt_state = optimizer.get("opt_state") if isinstance(optimizer, dict) else None
  is_res16 = is_residual16_opt_state_dict(raw_opt_state)
  flat_residuals = (
      _flatten_leaves_by_key_tuple(raw_opt_state.get("master_residuals", {}))
      if is_res16 and isinstance(raw_opt_state, dict)
      else {}
  )

  if "model" in nnx_pure_dict:
    model_params = _strip_rng_state(nnx_pure_dict["model"])
    if flat_residuals:

      def _reconstruct_or_promote(jax_path, leaf):
        key_tuple = _path_to_key_tuple(jax_path)
        if key_tuple not in flat_residuals:
          return leaf
        if isinstance(leaf, jax.ShapeDtypeStruct):
          return jax.ShapeDtypeStruct(leaf.shape, jnp.float32, sharding=getattr(leaf, "sharding", None))
        return _reconstruct_f32_concrete_for_save(leaf, flat_residuals[key_tuple])

      model_params = jax.tree_util.tree_map_with_path(
          _reconstruct_or_promote,
          model_params,
          is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
      )
    result["params"] = {"params": model_params}

  if isinstance(optimizer, dict):
    if "step" in optimizer:
      # NNX stores step as uint32; Linen uses int32.
      result["step"] = _cast_step(optimizer["step"], jnp.int32)
    if "opt_state" in optimizer:
      inner_opt = raw_opt_state["inner"] if is_res16 and isinstance(raw_opt_state, dict) else raw_opt_state
      result["opt_state"] = _opt_state_to_linen(inner_opt)
  return result


def reshape_restored_weights_for_residual16(weights: dict[str, Any], linen_state: nnx.State) -> dict[str, Any]:
  """Splits restored float32 checkpoint weights into `(bf16, int16)` when `linen_state` uses residual16."""
  if "optimizer" not in linen_state or "opt_state" not in linen_state["optimizer"]:
    return weights
  opt_state_target = linen_state["optimizer"]["opt_state"]
  if not is_residual16_opt_state_dict(opt_state_target):
    return weights

  abstract_model_pure = linen_state["model"].to_pure_dict() if "model" in linen_state else {}
  res_state = opt_state_target["master_residuals"] if "master_residuals" in opt_state_target else {}
  abstract_res_pure = res_state.to_pure_dict() if hasattr(res_state, "to_pure_dict") else res_state
  flat_abstract_model = _flatten_leaves_by_key_tuple(abstract_model_pure)
  flat_abstract_res = _flatten_leaves_by_key_tuple(abstract_res_pure)

  out = dict(weights)
  split_residuals_by_key: dict[tuple[Any, ...], Any] = {}
  if "model" in out and flat_abstract_res:

    def _split_model_leaf(jax_path, leaf):
      key_tuple = _path_to_key_tuple(jax_path)
      if key_tuple in flat_abstract_res and isinstance(leaf, jax.Array):
        target_w_sds = flat_abstract_model.get(key_tuple)
        target_res_sds = flat_abstract_res[key_tuple]
        w_bf16, res_i16 = _split_f32_concrete_for_restore(
            leaf,
            getattr(target_w_sds, "sharding", None),
            getattr(target_res_sds, "sharding", None),
        )
        split_residuals_by_key[key_tuple] = res_i16
        return w_bf16
      return leaf

    out["model"] = jax.tree_util.tree_map_with_path(
        _split_model_leaf,
        out["model"],
        is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
    )

  restored_residuals = (
      jax.tree_util.tree_map_with_path(
          lambda p, sds: split_residuals_by_key.get(_path_to_key_tuple(p), sds),
          abstract_res_pure,
          is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
      )
      if flat_abstract_res
      else {}
  )
  opt_dict = dict(out.get("optimizer", {}))
  new_opt_state: dict[str, Any] = {}
  if "opt_state" in opt_dict:
    new_opt_state["inner"] = opt_dict["opt_state"]
  if flat_abstract_res:
    new_opt_state["master_residuals"] = restored_residuals
  if new_opt_state:
    opt_dict["opt_state"] = new_opt_state
  if opt_dict:
    out["optimizer"] = opt_dict
  return out


def promote_abstract_params_for_residual16(model_params_state: nnx.State, abstract_state: Any) -> nnx.State:
  """Promotes residual16 parameter `ShapeDtypeStruct`s to `float32` for params-only Orbax restore."""
  if not isinstance(abstract_state, nnx.State) or "optimizer" not in abstract_state:
    return model_params_state
  opt = abstract_state["optimizer"]
  if not isinstance(opt, (dict, nnx.State)) or "opt_state" not in opt:
    return model_params_state
  opt_state = opt["opt_state"]
  if not is_residual16_opt_state_dict(opt_state) or "master_residuals" not in opt_state:
    return model_params_state

  res_state = opt_state["master_residuals"]
  res_pure = res_state.to_pure_dict() if hasattr(res_state, "to_pure_dict") else res_state
  flat_res = _flatten_leaves_by_key_tuple(res_pure)
  if not flat_res:
    return model_params_state

  def _promote_var(jax_path, var):
    key_tuple = _path_to_key_tuple(jax_path)
    if key_tuple not in flat_res or not isinstance(var, nnx.Variable):
      return var
    val = var.get_value()
    if isinstance(val, jax.ShapeDtypeStruct):
      return var.replace(value=jax.ShapeDtypeStruct(val.shape, jnp.float32, sharding=getattr(val, "sharding", None)))
    return var

  return jax.tree_util.tree_map_with_path(
      _promote_var,
      model_params_state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )


def overlay_raw_params_with_residual16(state: Any, raw_model_params: Any) -> bool:
  """Overlays params-only restored weights onto `state`, splitting `(bf16, int16)` for residual16."""
  if not isinstance(state, (nnx.State, dict)) or "model" not in state or "optimizer" not in state:
    return False
  opt = state["optimizer"]
  if not isinstance(opt, (nnx.State, dict)) or "opt_state" not in opt:
    return False
  opt_state = opt["opt_state"]
  if not is_residual16_opt_state_dict(opt_state):
    return False

  target_model = state["model"]
  target_pure = target_model.to_pure_dict() if hasattr(target_model, "to_pure_dict") else target_model
  res_target = opt_state["master_residuals"] if "master_residuals" in opt_state else {}
  res_pure = res_target.to_pure_dict() if hasattr(res_target, "to_pure_dict") else res_target

  target_flat = _flatten_leaves_by_key_tuple(target_pure)
  res_target_flat = _flatten_leaves_by_key_tuple(res_pure)
  raw_flat = _flatten_leaves_by_key_tuple(raw_model_params)

  updated_model_flat: dict[tuple[Any, ...], Any] = {}
  updated_res_flat: dict[tuple[Any, ...], Any] = dict(res_target_flat)
  for k, target_val in target_flat.items():
    if k in raw_flat and not isinstance(raw_flat[k], jax.ShapeDtypeStruct):
      raw_val = raw_flat[k]
      if isinstance(raw_val, nnx.Variable):
        raw_val = raw_val.get_value()
      if k in res_target_flat:
        w_bf16, res_i16 = _split_f32_concrete_for_restore(
            raw_val,
            getattr(target_val, "sharding", None),
            getattr(res_target_flat[k], "sharding", None),
        )
        updated_model_flat[k] = w_bf16
        updated_res_flat[k] = res_i16
      else:
        cast_val = raw_val.astype(target_val.dtype) if hasattr(target_val, "dtype") else raw_val
        if hasattr(target_val, "sharding") and target_val.sharding is not None:
          updated_model_flat[k] = jax.device_put(cast_val, target_val.sharding)
        else:
          updated_model_flat[k] = cast_val
    else:
      updated_model_flat[k] = target_val

  nnx.replace_by_pure_dict(target_model, nnx.traversals.unflatten_mapping(updated_model_flat))
  if res_target_flat and isinstance(res_target, nnx.State):
    nnx.replace_by_pure_dict(res_target, nnx.traversals.unflatten_mapping(updated_res_flat))
  return True


def _strip_mu_nu_params(state):
  """Inverse of `_wrap_mu_nu_with_params`: removes the inner 'params' wrap from mu/nu."""
  if not isinstance(state, dict):
    return state
  return {
      k: (v["params"] if k in ("mu", "nu") and isinstance(v, dict) and "params" in v else v) for k, v in state.items()
  }


def _opt_state_from_linen(opt_state):
  """Inverse of `_opt_state_to_linen`: Linen layout -> NNX opt_state.

  A chain (optax.adamw) is a list -> int-keyed dict. An un-chained state (adam_pax:
  `{count, mu, nu}`) stays flat -- re-wrapping it under a `0` index would not line
  up with the model's opt_state.
  """
  if isinstance(opt_state, list):
    return {i: _strip_mu_nu_params(e) for i, e in enumerate(opt_state) if isinstance(e, dict)}
  if not isinstance(opt_state, dict):
    return opt_state
  return _strip_mu_nu_params(opt_state)


def from_linen_checkpoint_dict(linen_pure_dict):
  """Inverse of `to_linen_checkpoint_dict`: Linen on-disk layout -> NNX layout.

  Doesn't restore NNX-only rngs/dropout (absent from Linen); callers fill those.
  """
  if not isinstance(linen_pure_dict, dict):
    return linen_pure_dict
  result = {}
  params = linen_pure_dict.get("params")
  if isinstance(params, dict) and "params" in params:
    result["model"] = params["params"]
  elif params is not None:
    result["model"] = params
  optimizer = {}
  if "step" in linen_pure_dict:
    # Linen stores step as int32; NNX uses uint32.
    optimizer["step"] = _cast_step(linen_pure_dict["step"], jnp.uint32)
  if "opt_state" in linen_pure_dict:
    optimizer["opt_state"] = _opt_state_from_linen(linen_pure_dict["opt_state"])
  if optimizer:
    result["optimizer"] = optimizer
  return result


def split_for_checkpoint(state: nnx.State):
  """Partitions an nnx.State by its on-disk destination.

  Named by on-disk destination. Returns `(linen_state, aux, ephemeral)`:
    linen_state: trainable weights (nnx.Param) and the optimizer -> Linen params/opt_state/step.
      Its model side is pure nnx.Param, so it maps to the Linen `params` collection cleanly.
    aux:         rngs/dropout, batch stats, and any other persistent model variable (e.g. a
      frozen routing table) -> nnx_aux.
    ephemeral:   caches and intermediates, which are recomputed and so are not checkpointed.

  The type-based catch-all `rest` mixes the optimizer (under "optimizer") with any custom
  persistent variable (under "model"); they route to different places, so they're split apart
  here. Grouping by the ephemeral types rather than whitelisting nnx.Param means a new Variable
  subclass persists (via `aux`) instead of being silently dropped. Save and restore must
  partition identically, so both go through here.
  """
  params, rng_state, batch_stats, caches, intermediates, rest = nnx.split_state(
      state, nnx.Param, nnx.RngState, nnx.BatchStat, nnx.Cache, nnx.Intermediate, ...
  )
  optimizer = nnx.State({"optimizer": rest["optimizer"]}) if "optimizer" in rest else nnx.State({})
  custom = nnx.State({"model": rest["model"]}) if "model" in rest else nnx.State({})
  linen_state = nnx.merge_state(params, optimizer)
  aux = nnx.merge_state(rng_state, batch_stats, custom)
  ephemeral = nnx.merge_state(caches, intermediates)
  return linen_state, aux, ephemeral


def to_checkpoint_dict(state: nnx.State | nnx.Module):
  """Reshapes an nnx.State into the on-disk checkpoint layout.

  Weights (nnx.Param) map to the Linen `params` collection and the optimizer to
  opt_state/step, so NNX and Linen checkpoints stay interchangeable. Everything else that
  must persist -- rngs/dropout, batch stats, and any custom variable -- goes under an `nnx_aux`
  subtree. Works on a concrete state (save) or an abstract state (restore target).
  """
  if isinstance(state, nnx.Module):
    state = nnx.state(state)
  linen_state, aux_state, _ = split_for_checkpoint(state)
  pure = linen_state.to_pure_dict()
  linen_dict = to_linen_checkpoint_dict({"model": pure.get("model", {}), "optimizer": pure.get("optimizer", {})})
  aux = aux_state.to_pure_dict()
  if aux:
    linen_dict["nnx_aux"] = aux
  return linen_dict


def apply_checkpoint_aux(aux_state: nnx.State, nnx_aux: dict[str, Any]) -> None:
  """Fills `aux_state` from a checkpoint's `nnx_aux`, skipping entries it no longer has.

  `split_for_checkpoint` routes every persistent non-Param variable to `nnx_aux`, so a
  checkpoint carries whatever RNG streams, batch stats and custom variables the model held
  when it was written. A model that has since stopped holding one of them -- a module that
  no longer keeps an `nnx.Rngs`, say -- would otherwise fail the resume outright with
  "key in pure_dict not available in state".

  Restore is already tolerant in the other direction: an entry the checkpoint lacks keeps
  its fresh init value. This makes it tolerant both ways. Only `nnx_aux` is filtered;
  weights still go through `replace_by_pure_dict` directly, so a genuinely missing weight
  keeps raising.
  """
  present = {tuple(path) for path, _ in nnx.to_flat_state(aux_state)}
  flat = nnx.traversals.flatten_mapping(nnx_aux)
  kept = {path: value for path, value in flat.items() if path in present}
  dropped = [path for path in flat if path not in present]
  if dropped:
    # Log the paths, not just a count: skipping an RNG stream a module stopped holding
    # is routine, but skipping a batch stat or a custom variable means the model and the
    # checkpoint have genuinely diverged, and that should be visible in the log.
    shown = ", ".join("/".join(str(k) for k in path) for path in sorted(dropped)[:20])
    more = f" (+{len(dropped) - 20} more)" if len(dropped) > 20 else ""
    max_logging.log(f"Skipping {len(dropped)} nnx_aux entries absent from this model: {shown}{more}")
  if kept:
    nnx.replace_by_pure_dict(aux_state, nnx.traversals.unflatten_mapping(kept))
