# Copyright 2026 Google LLC
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

"""Path-transparent pinned-host offload and runtime TPU HBM telemetry for m3.

`HostOffloadedOptimizer` parks `opt_state` (such as FP32 AdamW first and second
moments) in `"pinned_host"` memory between steps and transfers them to `"device"`
inside `update`, preserving identical `nnx.state(model, nnx.Param)` paths and
step-for-step numerical trajectories when compiled with buffer donation
(`@nnx.jit(donate_argnums=(0, 1))`). `maybe_offload_parameters` optionally moves
`nnx.Param` state to `"pinned_host"` — either all parameters
(`parameter_memory_host_offload=True`) or only parameters whose `"/"`-joined
paths match `host_offload_param_patterns` (`selective_host_offload=True`),
leaving all other parameters and optimizer state on device.

For selectively offloaded `nnx.Embed` tables, `HostOffloadedEmbed` replaces the
module path-transparently with a static `nnx.Embed` subclass that gathers only
the needed rows per step from `pinned_host` to device via
`jax.experimental.compute_on("device_host")` (shard-locally inside
`jax.shard_map` + ICI `psum` when row-sharded). Under `value_and_grad` /
`pure_value_and_grad`, backward produces a row-sparse `SparseHostGrad` without
materializing a dense full-table gradient in HBM. Exact dense AdamW updates
(`host_compute` or `streamed_chunks`) remain the default; `lazy_sparse` is an
opt-in approximation that updates only rows touched in the current step and
skips weight/moment decay on untouched rows.

Inside traced regions, `to_memory` lowers transfers via
`jax.device_put(x, jax.memory.Space.*)` so XLA schedules host-device DMA copies
within the compiled program; outside tracing, concrete sharded arrays use
`x.sharding.with_memory_kind(kind)`.

Cluster trade-off: optimizer host offload is a memory-capacity mechanism rather
than a throughput optimization. On 64-chip TPU7x MoE runs, offloading FP32 AdamW
moments saves ~23.6-26.8 GiB/device of HBM (enabling fit within the 94.7 GiB HBM
ceiling), but incurs ~1.0-1.83 s/step (~21.8% step time) of PCIe transfer cost
when device HBM already accommodates resident moments.

`get_device_hbm_stats` queries live `device.memory_stats()` and reports both
`peak_bytes_in_use` and `allocated_bytes = bytes_in_use + bytes_reserved`,
capturing XLA HLO temporary workspace held in `bytes_reserved`.
"""

from collections.abc import Callable, Sequence
import contextlib
import functools
import re
import threading
from typing import Any

from flax import nnx
import jax
from jax.experimental import compute_on
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import optax

from maxtext.m3.infra import named_scopes

_GIB = float(2**30)
_HOST_KINDS = frozenset({"pinned_host", "unpinned_host"})
_SPACES = {
    "pinned_host": jax.memory.Space.Host,
    "device": jax.memory.Space.Device,
}
_OPTIMIZER_MODES = frozenset({"host_compute", "streamed_chunks", "lazy_sparse"})
_SELECTIVE_HOST_PATH_KEY = "_m3_selective_host_path"
_OFFLOADED_MODEL_CLASSES: dict[type[nnx.Module], type[nnx.Module]] = {}
_PREGATHERED_STATE = threading.local()
_HOST_SQNORM_CAPTURE = threading.local()
_ACTIVE_PARAM_SHARDINGS = threading.local()


@contextlib.contextmanager
def active_param_shardings(param_shardings: Any):
  """Provides physical parameter shardings during `Auto`-mode `jit` traces (e.g. `MaxTextTrainingEngine`)."""
  prev = getattr(_ACTIVE_PARAM_SHARDINGS, "map", None)
  if param_shardings is None:
    yield
    return
  spec_map: dict[tuple[Any, ...], Any] = {}
  for path, leaf in jax.tree_util.tree_leaves_with_path(
      param_shardings, is_leaf=lambda x: isinstance(x, (nnx.Variable, jax.sharding.Sharding, P))
  ):
    if isinstance(leaf, nnx.Variable):
      leaf = leaf.get_raw_value()
    spec = getattr(leaf, "spec", leaf if isinstance(leaf, P) else None)
    if spec is not None:
      spec_map[tuple(_key_value(k) for k in path)] = spec
  _ACTIVE_PARAM_SHARDINGS.map = spec_map
  try:
    yield
  finally:
    _ACTIVE_PARAM_SHARDINGS.map = prev


def _call_shard_map(fn: Any, *, in_specs: Any, out_specs: Any, args: tuple[Any, ...]) -> Any:
  """Runs `jax.shard_map` under `jax.jit` so eager and `jax.checkpoint` callers can use `Space.Host` transfers."""
  smap = jax.shard_map(fn, in_specs=in_specs, out_specs=out_specs, check_vma=False)
  return jax.jit(smap)(*args)


def _apply_memory_kind_hook(var: nnx.Variable, value: Any) -> Any:
  """Preserves `memory_kind` metadata when `nnx.eval_shape` assigns abstract shardings."""
  kind = var.get_metadata("memory_kind", default=None)
  if not kind:
    return value
  if isinstance(value, jax.ShapeDtypeStruct):
    sharding = getattr(value, "sharding", None)
    if sharding is not None and getattr(sharding, "memory_kind", None) != kind:
      return jax.ShapeDtypeStruct(
          value.shape,
          value.dtype,
          sharding=sharding.with_memory_kind(kind),
      )
  return value


def _attach_memory_kind_metadata(var: nnx.Variable, kind: str) -> None:
  """Records `memory_kind` on `var` and registers `_apply_memory_kind_hook` for `nnx.eval_shape`."""
  var.set_metadata("memory_kind", kind)
  hooks = tuple(var.get_metadata("on_set_value", default=()) or ())
  if _apply_memory_kind_hook not in hooks:
    var.set_metadata("on_set_value", (*hooks, _apply_memory_kind_hook))


def _is_host_leaf(x: Any) -> bool:
  """Returns True when `x` (or its wrapped value) resides in host memory."""
  val = x.get_raw_value() if isinstance(x, nnx.Variable) else x
  if isinstance(val, jax.core.Tracer):
    aval = getattr(val, "aval", None)
    return getattr(aval, "memory_space", None) == jax.memory.Space.Host
  sharding = getattr(val, "sharding", None)
  if getattr(sharding, "memory_kind", None) in _HOST_KINDS:
    return True
  if isinstance(x, nnx.Variable) and x.get_metadata("memory_kind", default=None) in _HOST_KINDS:
    return True
  return False


def _has_host_leaves(tree: Any) -> bool:
  """Returns True if any array or variable leaf in `tree` is in host memory."""
  leaves = jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, (nnx.Variable, SparseHostGrad)))
  return any(isinstance(leaf, SparseHostGrad) or _is_host_leaf(leaf) for leaf in leaves)


def _validate_kind(kind: str) -> Any:
  if kind not in _SPACES:
    raise ValueError(f"Unsupported memory kind {kind!r}; expected one of {sorted(_SPACES)}.")
  return _SPACES[kind]


def _move_raw_value(x: Any, kind: str, target_space: Any) -> Any:
  """Places a single raw leaf value in `kind` while leaving scalars/`None` unchanged."""
  if x is None or isinstance(x, (bool, int, float, complex, str, bytes)):
    return x
  if isinstance(x, jax.core.Tracer):
    return jax.device_put(x, target_space)
  if isinstance(x, jax.ShapeDtypeStruct):
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding.with_memory_kind(kind))
    return x
  if isinstance(x, jax.Array):
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      return jax.device_put(x, sharding.with_memory_kind(kind))
    return jax.device_put(x, target_space)
  if isinstance(x, np.ndarray):
    return jax.device_put(x, target_space)
  return x


def to_memory(tree: Any, kind: str) -> Any:
  """Returns `tree` with array and `nnx.Variable` leaves placed in `kind`.

  Args:
    tree: Arbitrary PyTree containing `nnx.Variable`, traced/concrete JAX
      arrays, `jax.ShapeDtypeStruct`, NumPy arrays, or Python scalars/`None`.
    kind: Target memory kind (`"pinned_host"` or `"device"`).
  """
  target_space = _validate_kind(kind)

  def move(x: Any) -> Any:
    if isinstance(x, nnx.Variable):
      moved = x.replace(_move_raw_value(x.get_raw_value(), kind, target_space))
      _attach_memory_kind_metadata(moved, kind)
      return moved
    return _move_raw_value(x, kind, target_space)

  return jax.tree.map(move, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


def _move_variables_in_place(tree: Any, kind: str) -> None:
  """Moves `nnx.Variable` values in `tree` to `kind` without replacing Variable instances."""
  target_space = _validate_kind(kind)
  for leaf in jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, nnx.Variable)):
    if isinstance(leaf, nnx.Variable):
      _attach_memory_kind_metadata(leaf, kind)
      leaf.set_raw_value(_move_raw_value(leaf.get_raw_value(), kind, target_space))


def _key_value(key: Any) -> Any:
  """Extracts the raw Python key from a JAX PyTree key entry."""
  if hasattr(key, "key"):
    return key.key
  if hasattr(key, "name"):
    return key.name
  if hasattr(key, "idx"):
    return key.idx
  return key


def _format_param_path(path: Sequence[Any]) -> str:
  return "/".join(str(k) for k in path)


def is_selective_host_offload_enabled(config: Any) -> bool:
  """Returns True when selective path-pattern host offload is enabled in `config`."""
  return bool(config.selective_host_offload) if hasattr(config, "selective_host_offload") else False


def get_host_offload_param_patterns(config: Any) -> tuple[str, ...]:
  """Returns normalized regex patterns selecting parameters for host offload."""
  raw = (config.host_offload_param_patterns if hasattr(config, "host_offload_param_patterns") else None) or ()
  if isinstance(raw, str):
    return (raw,) if raw else ()
  return tuple(str(p) for p in raw if str(p))


def _get_optimizer_mode(config: Any) -> str:
  """Returns validated selective host-offload optimizer mode from `config`."""
  mode = (
      str(config.selective_host_offload_optimizer_mode)
      if hasattr(config, "selective_host_offload_optimizer_mode")
      else "host_compute"
  )
  if mode not in _OPTIMIZER_MODES:
    raise ValueError(
        f"Unsupported selective_host_offload_optimizer_mode {mode!r}; expected one of {sorted(_OPTIMIZER_MODES)}."
    )
  return mode


def _get_num_chunks(config: Any) -> int:
  chunks = int(config.selective_host_offload_num_chunks) if hasattr(config, "selective_host_offload_num_chunks") else 16
  if chunks < 1:
    raise ValueError(f"selective_host_offload_num_chunks must be >= 1, got {chunks}.")
  return chunks


def _match_param_paths(params_state: Any, patterns: Sequence[str]) -> frozenset[tuple[Any, ...]]:
  compiled = [re.compile(p) for p in patterns]
  matched: list[tuple[Any, ...]] = []
  for path, _ in nnx.to_flat_state(params_state):
    path_str = _format_param_path(path)
    if any(rx.search(path_str) for rx in compiled):
      matched.append(tuple(path))
  return frozenset(matched)


def selected_host_param_paths(model_or_state: Any) -> frozenset[tuple[Any, ...]]:
  """Returns canonical paths of `nnx.Param` variables marked for selective host offload."""
  state = nnx.state(model_or_state, nnx.Param) if isinstance(model_or_state, nnx.Module) else model_or_state
  paths: list[tuple[Any, ...]] = []
  for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
    if isinstance(leaf, nnx.Variable):
      marked = leaf.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      if marked is not None:
        paths.append(tuple(marked))
  return frozenset(paths)


def has_selective_host_offload(model_or_state: Any) -> bool:
  """Returns True when `model_or_state` has selectively host-offloaded parameters."""
  return bool(selected_host_param_paths(model_or_state))


def _active_mesh() -> Any | None:
  mesh = jax.sharding.get_abstract_mesh()
  if mesh is None or mesh.empty or mesh.size <= 1:
    return None
  return mesh


def _is_explicit_mesh(mesh: Any | None) -> bool:
  return mesh is not None and all(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types)


def _normalize_axis_entry(entry: Any, mesh: Any) -> tuple[str, ...]:
  if entry is None:
    return ()
  names = (entry,) if isinstance(entry, str) else tuple(entry)
  return tuple(ax for ax in names if ax in mesh.shape and mesh.shape[ax] > 1)


def _resolve_2d_axes(
    var_or_arr: Any,
    mesh: Any | None,
    sharding_hint: Any = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
  """Returns active mesh `(row_axes, col_axes)` for a 2D table variable or array."""
  if mesh is None:
    return (), ()
  spec = None
  if sharding_hint is not None:
    hint = sharding_hint.get_raw_value() if isinstance(sharding_hint, nnx.Variable) else sharding_hint
    spec = getattr(hint, "spec", hint if isinstance(hint, P) else None)
  if isinstance(var_or_arr, nnx.Variable):
    raw = var_or_arr.get_raw_value()
    if spec is None:
      active_map = getattr(_ACTIVE_PARAM_SHARDINGS, "map", None)
      sel_path = var_or_arr.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      if active_map and sel_path is not None and sel_path in active_map:
        spec = active_map[sel_path]
    if spec is None:
      spec = var_or_arr.get_metadata("out_sharding", default=None)
  else:
    raw = var_or_arr
  if spec is None:
    sh = getattr(raw, "sharding", None)
    if sh is None and isinstance(raw, jax.core.Tracer):
      sh = getattr(jax.typeof(raw), "sharding", None)
    spec = getattr(sh, "spec", None)
  if spec is None:
    return (), ()
  row_axes = _normalize_axis_entry(spec[0], mesh) if len(spec) >= 1 else ()
  col_axes = _normalize_axis_entry(spec[1], mesh) if len(spec) >= 2 else ()
  return row_axes, col_axes


def _pspec_dim(axes: tuple[str, ...]) -> Any:
  if not axes:
    return None
  return axes[0] if len(axes) == 1 else axes


def _maybe_reshard(x: jax.Array, spec: P, mesh: Any | None) -> jax.Array:
  if _is_explicit_mesh(mesh):
    return jax.sharding.reshard(x, spec)
  return x


def _can_move_vars(vars_list: Sequence[nnx.Variable]) -> bool:
  raw_leaves = [v.get_raw_value() for v in vars_list]
  return bool(raw_leaves) and not any(
      isinstance(v, jax.core.Tracer) or (isinstance(v, jax.ShapeDtypeStruct) and getattr(v, "sharding", None) is None)
      for v in raw_leaves
  )


# ---------------------------------------------------------------------------
# Row-sparse host gather, host scatter-add, and SparseHostGrad container
# ---------------------------------------------------------------------------


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Device)
def _replicated_host_gather(table_h: jax.Array, ids_h: jax.Array) -> jax.Array:
  return table_h[ids_h]


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _replicated_host_scatter_add(table_like_h: jax.Array, ids_h: jax.Array, d_out_h: jax.Array) -> jax.Array:
  zeros_h = jax.device_put(jnp.zeros(table_like_h.shape, dtype=table_like_h.dtype), jax.memory.Space.Host)
  return zeros_h.at[ids_h].add(d_out_h.astype(table_like_h.dtype))


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Device)
def _local_host_gather(local_table_h: jax.Array, ids_h: jax.Array, start_h: jax.Array) -> jax.Array:
  n_local = local_table_h.shape[0]
  rel = ids_h - start_h
  in_range = (rel >= 0) & (rel < n_local)
  zero_i = jax.device_put(jnp.int32(0), jax.memory.Space.Host)
  safe = jnp.where(in_range, rel, zero_i)
  zero_v = jax.device_put(jnp.zeros((), dtype=local_table_h.dtype), jax.memory.Space.Host)
  return jnp.where(in_range[..., None], local_table_h[safe], zero_v)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _local_host_scatter_add(
    local_table_like_h: jax.Array, ids_h: jax.Array, d_out_h: jax.Array, start_h: jax.Array
) -> jax.Array:
  """Scatters `d_out_h` into a zero-initialized local shard on host."""
  n_local = local_table_like_h.shape[0]
  rel = ids_h - start_h
  in_range = (rel >= 0) & (rel < n_local)
  zero_i = jax.device_put(jnp.int32(0), jax.memory.Space.Host)
  safe = jnp.where(in_range, rel, zero_i)
  zero_v = jax.device_put(jnp.zeros((), dtype=local_table_like_h.dtype), jax.memory.Space.Host)
  vals = jnp.where(in_range[..., None], d_out_h.astype(local_table_like_h.dtype), zero_v)
  zeros_h = jax.device_put(jnp.zeros(local_table_like_h.shape, dtype=local_table_like_h.dtype), jax.memory.Space.Host)
  return zeros_h.at[safe].add(vals)


def _host_embed_gather(
    table: jax.Array,
    ids: jax.Array,
    *,
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> jax.Array:
  """Gathers only rows indexed by `ids` from host-resident `table` onto device."""
  mesh = _active_mesh()
  if mesh is not None:
    # Always isolate host compute in shard_map so collectives run on device ICI, not MegaScale host.
    r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    rep_ids_spec = P(*([None] * ids.ndim))
    ids_in = _maybe_reshard(ids, rep_ids_spec, mesh)

    def _shard_gather(local_table: jax.Array, local_ids: jax.Array) -> jax.Array:
      ids_h = jax.device_put(local_ids, jax.memory.Space.Host)
      if row_axes:
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(local_table.shape[0])
        part = _local_host_gather(local_table, ids_h, jax.device_put(start, jax.memory.Space.Host))
        return jax.lax.psum(part, row_axes).astype(dtype)
      return _replicated_host_gather(local_table, ids_h).astype(dtype)

    return _call_shard_map(
        _shard_gather,
        in_specs=(P(r_spec, c_spec), rep_ids_spec),
        out_specs=P(*([None] * ids.ndim), c_spec),
        args=(table, ids_in),
    )

  gathered = _replicated_host_gather(table, jax.device_put(ids, jax.memory.Space.Host))
  return gathered.astype(dtype)


def _host_embed_scatter_add(
    table_like: jax.Array,
    ids: jax.Array,
    d_out: jax.Array,
    *,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> jax.Array:
  """Scatters `d_out` into a host-resident gradient matching `table_like`."""
  mesh = _active_mesh()
  if mesh is not None:
    r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    rep_ids_spec = P(*([None] * ids.ndim))
    out_spec = P(*([None] * ids.ndim), c_spec)
    ids_in = _maybe_reshard(ids, rep_ids_spec, mesh)
    d_out_in = _maybe_reshard(d_out, out_spec, mesh)

    def _shard_scatter(local_tbl: jax.Array, local_ids: jax.Array, local_g: jax.Array) -> jax.Array:
      ids_h = jax.device_put(local_ids, jax.memory.Space.Host)
      g_h = jax.device_put(local_g, jax.memory.Space.Host)
      if row_axes:
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(local_tbl.shape[0])
        return _local_host_scatter_add(local_tbl, ids_h, g_h, jax.device_put(start, jax.memory.Space.Host))
      return _replicated_host_scatter_add(local_tbl, ids_h, g_h)

    return _call_shard_map(
        _shard_scatter,
        in_specs=(P(r_spec, c_spec), rep_ids_spec, out_spec),
        out_specs=P(r_spec, c_spec),
        args=(table_like, ids_in, d_out_in),
    )

  return _replicated_host_scatter_add(
      table_like,
      jax.device_put(ids, jax.memory.Space.Host),
      jax.device_put(d_out, jax.memory.Space.Host),
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _host_embed_lookup_vjp(
    table: jax.Array,
    ids: jax.Array,
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> jax.Array:
  return _host_embed_gather(table, ids, dtype=dtype, row_axes=row_axes, col_axes=col_axes)


def _host_embed_lookup_fwd(
    table: jax.Array,
    ids: jax.Array,
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  out = _host_embed_gather(table, ids, dtype=dtype, row_axes=row_axes, col_axes=col_axes)
  return out, (table, ids)


def _host_embed_lookup_bwd(
    dtype: Any,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    res: tuple[jax.Array, jax.Array],
    g: jax.Array,
) -> tuple[jax.Array, None]:
  """VJP backward scattering device cotangent directly into a host gradient table."""
  del dtype
  table, ids = res
  d_table = _host_embed_scatter_add(table, ids, g, row_axes=row_axes, col_axes=col_axes)
  return (d_table, None)


_host_embed_lookup_vjp.defvjp(_host_embed_lookup_fwd, _host_embed_lookup_bwd)


@jax.tree_util.register_pytree_node_class
class SparseHostGrad:
  """Row-sparse gradient for a host-offloaded embedding table.

  Stores only the accessed row indices `ids` and device cotangent `d_gathered`
  without materializing a dense `(num_embeddings, features)` gradient in HBM.
  """

  def __init__(
      self,
      ids: jax.Array,
      d_gathered: jax.Array,
      *,
      table_shape: tuple[int, int],
      table_dtype: Any,
      row_axes: tuple[str, ...] = (),
      col_axes: tuple[str, ...] = (),
      grad_dtype: Any = None,
  ):
    self.ids = ids
    self.d_gathered = d_gathered
    self.table_shape = tuple(int(d) for d in table_shape)
    self.table_dtype = jnp.dtype(table_dtype)
    self.row_axes = tuple(row_axes)
    self.col_axes = tuple(col_axes)
    self.grad_dtype = jnp.dtype(grad_dtype) if grad_dtype is not None else self.table_dtype

  def with_d_gathered(self, new_d_gathered: jax.Array, *, grad_dtype: Any = None) -> "SparseHostGrad":
    return SparseHostGrad(
        self.ids,
        new_d_gathered,
        table_shape=self.table_shape,
        table_dtype=self.table_dtype,
        row_axes=self.row_axes,
        col_axes=self.col_axes,
        grad_dtype=grad_dtype if grad_dtype is not None else self.grad_dtype,
    )

  def tree_flatten(self):
    children = (self.ids, self.d_gathered)
    aux = (self.table_shape, self.table_dtype, self.row_axes, self.col_axes, self.grad_dtype)
    return children, aux

  @classmethod
  def tree_unflatten(cls, aux, children):
    ids, d_gathered = children
    table_shape, table_dtype, row_axes, col_axes, grad_dtype = aux
    return cls(
        ids,
        d_gathered,
        table_shape=table_shape,
        table_dtype=table_dtype,
        row_axes=row_axes,
        col_axes=col_axes,
        grad_dtype=grad_dtype,
    )


def materialize_sparse_grad(grad: SparseHostGrad) -> jax.Array:
  """Scatters a `SparseHostGrad` into a dense `(num_embeddings, features)` array."""
  flat_ids = grad.ids.reshape(-1)
  flat_dg = grad.d_gathered.reshape(-1, grad.table_shape[1]).astype(grad.grad_dtype)
  return jnp.zeros(grad.table_shape, dtype=grad.grad_dtype).at[flat_ids].add(flat_dg)


@contextlib.contextmanager
def _pregathered_context(mapping: dict[tuple[Any, ...], tuple[jax.Array, jax.Array]]):
  prev = getattr(_PREGATHERED_STATE, "mapping", None)
  _PREGATHERED_STATE.mapping = mapping
  try:
    yield
  finally:
    _PREGATHERED_STATE.mapping = prev


class HostOffloadedEmbed(nnx.Embed):
  """Path-transparent `nnx.Embed` subclass performing row-sparse host gathers."""

  def __init__(self, embed: nnx.Embed, *, scope: str | None = None):  # pylint: disable=super-init-not-called
    self.num_embeddings = embed.num_embeddings
    self.features = embed.features
    self.dtype = embed.dtype
    self.param_dtype = embed.param_dtype
    self.promote_dtype = embed.promote_dtype
    self.embedding = embed.embedding
    self._m3_embed_scope = scope

  @classmethod
  def from_embed(cls, embed: nnx.Embed, *, scope: str | None = None) -> "HostOffloadedEmbed":
    return cls(embed, scope=scope)

  def __call__(self, inputs: jax.Array, out_sharding: Any = None) -> jax.Array:
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError("Input type must be an integer or unsigned integer.")
    scope = getattr(self, "_m3_embed_scope", None)
    scope_ctx = jax.named_scope(scope) if scope else contextlib.nullcontext()
    with scope_ctx:
      param_path = self.embedding.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      active_map = getattr(_PREGATHERED_STATE, "mapping", None)
      if param_path is not None and active_map is not None and param_path in active_map:
        pre_ids, pre_gathered = active_map[param_path]
        if inputs.shape != pre_ids.shape:
          raise ValueError(
              f"HostOffloadedEmbed pre-gather shape mismatch for {param_path}: "
              f"received inputs.shape={inputs.shape}, expected {pre_ids.shape}."
          )
        out = pre_gathered
        if out_sharding is not None:
          out = jax.lax.with_sharding_constraint(out, out_sharding)
        return out
      if _is_host_leaf(self.embedding):
        table = self.embedding.get_raw_value()
        target_dtype = jnp.dtype(self.dtype) if self.dtype is not None else table.dtype
        mesh = _active_mesh()
        row_axes, col_axes = _resolve_2d_axes(self.embedding, mesh)
        out = _host_embed_lookup_vjp(table, inputs, target_dtype, row_axes, col_axes)
        if out_sharding is not None:
          out = jax.lax.with_sharding_constraint(out, out_sharding)
        return out
      return super().__call__(inputs, out_sharding=out_sharding)


class _HostOffloadedParametersMeta(type):
  """Metaclass recognizing single-inheritance parameter-offloaded model subclasses."""

  def __instancecheck__(cls, instance: Any) -> bool:
    return bool(getattr(instance, "_is_host_offloaded_params", False))

  def __subclasscheck__(cls, subclass: type) -> bool:
    return bool(getattr(subclass, "_is_host_offloaded_params", False))


class HostOffloadedParameters(metaclass=_HostOffloadedParametersMeta):
  """Marker type for models whose `nnx.Param`s are parked in `pinned_host`."""


def _offloaded_model_class(cls: type[nnx.Module]) -> type[nnx.Module]:
  """Returns a cached path-transparent single-inheritance subclass of `cls`."""
  if getattr(cls, "_is_host_offloaded_params", False):
    return cls
  subcls = _OFFLOADED_MODEL_CLASSES.get(cls)
  if subcls is None:
    parent_call = cls.__call__
    parent_vocab_tiling = getattr(cls, "logits_from_hidden_states_for_vocab_tiling", None)

    def _with_device_params(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
      # Never pull selectively offloaded `HostOffloadedEmbed` tables into HBM here.
      embed_var_ids = {
          id(node.embedding)
          for _, node in nnx.iter_graph(self)
          if isinstance(node, HostOffloadedEmbed) and isinstance(node.embedding, nnx.Variable)
      }
      param_vars = [
          v
          for v in jax.tree.leaves(nnx.state(self, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
          if isinstance(v, nnx.Variable) and id(v) not in embed_var_ids and _is_host_leaf(v)
      ]
      if not param_vars:
        return fn(self, *args, **kwargs)
      saved_raw = [v.get_raw_value() for v in param_vars]
      for v in param_vars:
        v.set_raw_value(_move_raw_value(v.get_raw_value(), "device", jax.memory.Space.Device))
      try:
        return fn(self, *args, **kwargs)
      finally:
        for v, raw in zip(param_vars, saved_raw):
          v.set_raw_value(raw)

    def _call(self: Any, *args: Any, **kwargs: Any) -> Any:
      return _with_device_params(self, parent_call, *args, **kwargs)

    def _vocab_tiling(self: Any, *args: Any, **kwargs: Any) -> Any:
      if parent_vocab_tiling is None:
        raise AttributeError(f"{cls.__name__} has no 'logits_from_hidden_states_for_vocab_tiling' method.")
      return _with_device_params(self, parent_vocab_tiling, *args, **kwargs)

    namespace: dict[str, Any] = {
        "__slots__": (),
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "_is_host_offloaded_params": True,
        "__call__": _call,
    }
    if parent_vocab_tiling is not None:
      namespace["logits_from_hidden_states_for_vocab_tiling"] = _vocab_tiling

    subcls = type(cls)(
        cls.__name__,
        (cls,),
        namespace,
    )
    _OFFLOADED_MODEL_CLASSES[cls] = subcls
  return subcls


def _replace_child(parent: Any, key: Any, new_child: Any) -> None:
  if isinstance(parent, (list, nnx.List)):
    parent[int(key)] = new_child
  elif isinstance(parent, dict):
    parent[key] = new_child
  else:
    setattr(parent, str(key), new_child)


def _apply_selective_parameter_offload(model: nnx.Module, config: Any) -> tuple[nnx.Module, bool]:
  """Offloads pattern-matched parameters to `pinned_host` and wraps matched `nnx.Embed` modules."""
  patterns = get_host_offload_param_patterns(config)
  if not patterns:
    raise ValueError("selective_host_offload=True requires non-empty host_offload_param_patterns.")
  params = nnx.state(model, nnx.Param)
  selected_paths = _match_param_paths(params, patterns)
  if not selected_paths:
    raise ValueError(
        f"selective_host_offload=True matched 0 parameters for host_offload_param_patterns={list(patterns)!r}."
    )

  selected_vars: list[nnx.Variable] = []
  for path, var in nnx.to_flat_state(params):
    path_tuple = tuple(path)
    if path_tuple in selected_paths and isinstance(var, nnx.Variable):
      var.set_metadata(_SELECTIVE_HOST_PATH_KEY, path_tuple)
      selected_vars.append(var)

  if _can_move_vars(selected_vars):
    _move_variables_in_place(selected_vars, "pinned_host")
  else:
    for v in selected_vars:
      _attach_memory_kind_metadata(v, "pinned_host")

  # Replace matching `nnx.Embed` modules path-transparently with `HostOffloadedEmbed`.
  nodes = dict(nnx.iter_graph(model))
  embed_paths_covered: set[tuple[Any, ...]] = set()
  for mod_path, node in list(nodes.items()):
    if not mod_path or not isinstance(node, nnx.Embed):
      continue
    emb_param_path = (*mod_path, "embedding")
    if emb_param_path not in selected_paths:
      continue
    embed_paths_covered.add(emb_param_path)
    if not isinstance(node, HostOffloadedEmbed):
      parent = nodes.get(mod_path[:-1])
      if parent is None:
        raise ValueError(f"Could not resolve parent module for selected embedding at {mod_path}.")
      scope = named_scopes.scope_of(node)
      _replace_child(parent, mod_path[-1], HostOffloadedEmbed.from_embed(node, scope=scope))

  non_embed_selected = selected_paths - embed_paths_covered
  return model, bool(non_embed_selected)


def maybe_offload_parameters(model: nnx.Module, config: Any) -> nnx.Module:
  """Moves all or pattern-selected `nnx.Param` state to `pinned_host` per `config`."""
  full_offload = bool(config.parameter_memory_host_offload) if hasattr(config, "parameter_memory_host_offload") else False
  selective_offload = is_selective_host_offload_enabled(config)
  if full_offload and selective_offload:
    raise ValueError("selective_host_offload=True cannot be combined with parameter_memory_host_offload=True.")
  if selective_offload:
    model, needs_full_wrap = _apply_selective_parameter_offload(model, config)
    if not needs_full_wrap:
      return model
  elif not full_offload:
    return model
  else:
    params = nnx.state(model, nnx.Param)
    param_leaves = [
        v for v in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, nnx.Variable)) if isinstance(v, nnx.Variable)
    ]
    if _can_move_vars(param_leaves):
      _move_variables_in_place(params, "pinned_host")
    else:
      for v in param_leaves:
        _attach_memory_kind_metadata(v, "pinned_host")
  if not isinstance(model, HostOffloadedParameters):
    object.__setattr__(model, "__class__", _offloaded_model_class(type(model)))
  return model


def has_host_leaves(tree: Any) -> bool:
  """Returns True if any array or variable leaf in `tree` resides in host memory."""
  return _has_host_leaves(tree)


def apply_selective_memory_kind_to_shardings(abs_var_state: Any, sharding_state: Any) -> Any:
  """Propagates `pinned_host` metadata from traced abstract vars onto `NamedSharding` trees."""

  def _is_leaf(x: Any) -> bool:
    return isinstance(x, (nnx.Variable, jax.sharding.Sharding, jax.ShapeDtypeStruct))

  def _apply(a_leaf: Any, s_leaf: Any) -> Any:
    if s_leaf is None:
      return s_leaf
    kind = a_leaf.get_metadata("memory_kind", default=None) if isinstance(a_leaf, nnx.Variable) else None
    if kind not in _HOST_KINDS:
      raw = a_leaf.get_raw_value() if isinstance(a_leaf, nnx.Variable) else a_leaf
      sh = getattr(raw, "sharding", None)
      if getattr(sh, "memory_kind", None) in _HOST_KINDS:
        kind = sh.memory_kind
    if kind not in _HOST_KINDS:
      return s_leaf

    def _to_host_sharding(sh: Any) -> Any:
      if isinstance(sh, jax.sharding.NamedSharding) and sh.memory_kind != kind:
        return sh.with_memory_kind(kind)
      if isinstance(sh, jax.ShapeDtypeStruct):
        inner = getattr(sh, "sharding", None)
        if isinstance(inner, jax.sharding.NamedSharding) and inner.memory_kind != kind:
          return jax.ShapeDtypeStruct(sh.shape, sh.dtype, sharding=inner.with_memory_kind(kind))
      return sh

    def _is_sh_leaf(x: Any) -> bool:
      return isinstance(x, (jax.sharding.Sharding, jax.ShapeDtypeStruct))

    if isinstance(s_leaf, nnx.Variable):
      updated = jax.tree.map(_to_host_sharding, s_leaf.get_raw_value(), is_leaf=_is_sh_leaf)
      return s_leaf.replace(updated)
    return jax.tree.map(_to_host_sharding, s_leaf, is_leaf=_is_sh_leaf)

  return jax.tree.map(_apply, abs_var_state, sharding_state, is_leaf=_is_leaf)


# ---------------------------------------------------------------------------
# Sparse value_and_grad (row-sparse backward with zero dense HBM gradient)
# ---------------------------------------------------------------------------


def _collect_host_embeds(model: nnx.Module) -> list[tuple[tuple[Any, ...], HostOffloadedEmbed, Any]]:
  """Returns `(param_path, embed_module, parent_module)` for each `HostOffloadedEmbed`."""
  nodes = dict(nnx.iter_graph(model))
  specs: list[tuple[tuple[Any, ...], HostOffloadedEmbed, Any]] = []
  for mod_path, node in nodes.items():
    if isinstance(node, HostOffloadedEmbed) and isinstance(node.embedding, nnx.Variable):
      param_path = node.embedding.get_metadata(_SELECTIVE_HOST_PATH_KEY, default=None)
      if param_path is None:
        param_path = (*mod_path, "embedding")
      parent = nodes.get(mod_path[:-1]) if mod_path else None
      specs.append((tuple(param_path), node, parent))
  return specs


def _extract_tokens_and_segments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    input_tokens: jax.Array | None,
    segment_ids: jax.Array | None,
) -> tuple[jax.Array | None, jax.Array | None]:
  """Extracts token and optional segment ID arrays from training-step batch arguments."""
  if input_tokens is not None:
    return input_tokens, segment_ids
  for candidate in (*args, *kwargs.values()):
    if isinstance(candidate, dict):
      toks = candidate.get("inputs")
      if toks is None:
        toks = candidate.get("decoder_input_tokens")
      if toks is not None:
        segs = segment_ids
        if segs is None:
          segs = candidate.get("inputs_segmentation")
        if segs is None:
          segs = candidate.get("decoder_segment_ids")
        if segs is None:
          segs = candidate.get("segment_ids")
        return toks, segs
  return None, segment_ids


def _prepare_host_embed_gathers(
    model: nnx.Module,
    input_tokens: jax.Array,
    segment_ids: jax.Array | None,
) -> tuple[dict[tuple[Any, ...], jax.Array], dict[tuple[Any, ...], dict[str, Any]]]:
  """Pre-gathers rows for every `HostOffloadedEmbed` in `model`."""
  mesh = _active_mesh()
  gathered_map: dict[tuple[Any, ...], jax.Array] = {}
  meta_map: dict[tuple[Any, ...], dict[str, Any]] = {}
  for param_path, embed_mod, parent_mod in _collect_host_embeds(model):
    ids_fn = getattr(parent_mod, "embed_ids_from_tokens", None) if parent_mod is not None else None
    if callable(ids_fn):
      ids = ids_fn(input_tokens, segment_ids=segment_ids)
    else:
      ids = input_tokens
    table = embed_mod.embedding.get_raw_value()
    target_dtype = jnp.dtype(embed_mod.dtype) if embed_mod.dtype is not None else table.dtype
    row_axes, col_axes = _resolve_2d_axes(embed_mod.embedding, mesh)
    gathered = _host_embed_gather(table, ids, dtype=target_dtype, row_axes=row_axes, col_axes=col_axes)
    gathered_map[param_path] = gathered
    meta_map[param_path] = {
        "ids": ids,
        "table_shape": table.shape,
        "table_dtype": table.dtype,
        "row_axes": row_axes,
        "col_axes": col_axes,
    }
  return gathered_map, meta_map


def _partition_state_by_paths(
    state: nnx.State, excluded_paths: frozenset[tuple[Any, ...]]
) -> tuple[nnx.State, nnx.State]:
  """Splits `state` into `(kept_state, excluded_state)` by canonical leaf path."""
  kept_items = []
  excl_items = []
  for path, val in nnx.to_flat_state(state):
    if tuple(path) in excluded_paths:
      excl_items.append((path, val))
    else:
      kept_items.append((path, val))
  return nnx.from_flat_state(kept_items), nnx.from_flat_state(excl_items)


def _assemble_grads_with_sparse(
    dev_grads: nnx.State,
    d_gathered_map: dict[tuple[Any, ...], jax.Array],
    meta_map: dict[tuple[Any, ...], dict[str, Any]],
    *,
    grad_dtype: Any = None,
) -> nnx.State:
  """Merges device gradients with `SparseHostGrad` leaves into a full parameter-shaped `nnx.State`."""
  items = list(nnx.to_flat_state(dev_grads))
  for param_path, dg in d_gathered_map.items():
    meta = meta_map[param_path]
    t_dtype = meta["table_dtype"]
    eff_grad_dtype = (
        jnp.dtype(grad_dtype) if grad_dtype is not None and jnp.dtype(t_dtype) == jnp.float32 else jnp.dtype(t_dtype)
    )
    items.append(
        (
            param_path,
            SparseHostGrad(
                meta["ids"],
                dg,
                table_shape=meta["table_shape"],
                table_dtype=t_dtype,
                row_axes=meta["row_axes"],
                col_axes=meta["col_axes"],
                grad_dtype=eff_grad_dtype,
            ),
        )
    )
  return nnx.from_flat_state(items)


def value_and_grad(
    loss_fn: Callable[..., Any],
    model: nnx.Module,
    *args: Any,
    has_aux: bool = False,
    input_tokens: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
  """Computes `(value, grads)` with row-sparse gradients for `HostOffloadedEmbed` tables."""
  if not has_selective_host_offload(model) or not _collect_host_embeds(model):
    return nnx.value_and_grad(loss_fn, has_aux=has_aux)(model, *args, **kwargs)
  tokens, segs = _extract_tokens_and_segments(args, kwargs, input_tokens, segment_ids)
  if tokens is None:
    raise ValueError(
        "m3_offload.value_and_grad requires token IDs (via `input_tokens` or a batch dict with "
        "'inputs'/'input_tokens') when HostOffloadedEmbed modules are present; refusing dense HBM fallback."
    )

  gathered_map, meta_map = _prepare_host_embed_gathers(model, tokens, segs)
  if not gathered_map:
    raise ValueError(
        "m3_offload.value_and_grad found HostOffloadedEmbed modules but pre-gather produced no active gathers."
    )

  host_embed_paths = frozenset(gathered_map.keys())
  graphdef, params, rest = nnx.split(model, nnx.Param, ...)
  dev_params, host_params = _partition_state_by_paths(params, host_embed_paths)

  def _diff_fn(dev_p: nnx.State, g_map: dict[tuple[Any, ...], jax.Array]) -> Any:
    full_p = nnx.merge_state(dev_p, host_params) if jax.tree.leaves(host_params) else dev_p
    local_model = nnx.merge(graphdef, full_p, rest, copy=True)
    ctx_map = {p: (meta_map[p]["ids"], g_map[p]) for p in g_map}
    with _pregathered_context(ctx_map):
      return loss_fn(local_model, *args, **kwargs)

  val_out, (dev_grads, d_gathered_map) = jax.value_and_grad(_diff_fn, argnums=(0, 1), has_aux=has_aux)(
      dev_params, gathered_map
  )
  grads = _assemble_grads_with_sparse(dev_grads, d_gathered_map, meta_map)
  return val_out, grads


def pure_value_and_grad(
    diff_wrapper: Callable[..., Any],
    model_graphdef: Any,
    curr_params: nnx.State,
    *,
    custom_params: nnx.State,
    rest: nnx.State,
    config: Any,
    data: dict[str, Any],
    has_aux: bool = True,
) -> tuple[Any, tuple[nnx.State, nnx.State]]:
  """Pure-state `value_and_grad` for `train.py` with row-sparse host-embed gradients."""
  probe_model = nnx.merge(model_graphdef, curr_params, custom_params, rest, copy=True)
  if not _collect_host_embeds(probe_model):
    return jax.value_and_grad(diff_wrapper, argnums=(0, 1), has_aux=has_aux)(
        curr_params, custom_params, rest, config, data
    )
  tokens, segs = _extract_tokens_and_segments((data,), {}, None, None)
  if tokens is None:
    raise ValueError(
        "m3_offload.pure_value_and_grad requires token IDs in `data` ('inputs'/'input_tokens') "
        "when HostOffloadedEmbed modules are present; refusing dense HBM fallback."
    )
  gathered_map, meta_map = _prepare_host_embed_gathers(probe_model, tokens, segs)
  if not gathered_map:
    raise ValueError(
        "m3_offload.pure_value_and_grad found HostOffloadedEmbed modules but pre-gather produced no active gathers."
    )

  host_embed_paths = frozenset(gathered_map.keys())
  dev_params, host_params = _partition_state_by_paths(curr_params, host_embed_paths)

  def _wrapped(
      dev_p: nnx.State,
      cust_p: nnx.State,
      g_map: dict[tuple[Any, ...], jax.Array],
  ) -> Any:
    full_p = nnx.merge_state(dev_p, host_params) if jax.tree.leaves(host_params) else dev_p
    ctx_map = {p: (meta_map[p]["ids"], g_map[p]) for p in g_map}
    with _pregathered_context(ctx_map):
      return diff_wrapper(full_p, cust_p, rest, config, data)

  val_out, (dev_grads, custom_grads, d_gathered_map) = jax.value_and_grad(_wrapped, argnums=(0, 1, 2), has_aux=has_aux)(
      dev_params, custom_params, gathered_map
  )
  raw_grads = _assemble_grads_with_sparse(
      dev_grads,
      d_gathered_map,
      meta_map,
      grad_dtype=config.grad_dtype if hasattr(config, "grad_dtype") else None,
  )
  return val_out, (raw_grads, custom_grads)


def engine_value_and_grad(
    diff_wrapper: Callable[..., Any],
    model_graphdef: Any,
    params: nnx.State,
    rest: nnx.State,
    batch: Any,
    *,
    param_shardings: Any = None,
) -> tuple[Any, nnx.State]:
  """Engine `fwd_bwd` differentiation avoiding `<host>` cotangents across scan/cond boundaries."""
  with active_param_shardings(param_shardings):
    full_model = nnx.merge(model_graphdef, params, rest, copy=True)
    if not _collect_host_embeds(full_model):
      return jax.value_and_grad(diff_wrapper, argnums=0, has_aux=True)(params, rest, batch)
    tokens, segs = _extract_tokens_and_segments((batch,), {}, None, None)
    if tokens is None:
      raise ValueError(
          "m3_offload.engine_value_and_grad requires token IDs in `batch` ('inputs'/'input_tokens') "
          "when HostOffloadedEmbed modules are present; refusing dense HBM fallback."
      )
    gathered_map, meta_map = _prepare_host_embed_gathers(full_model, tokens, segs)
    if not gathered_map:
      raise ValueError(
          "m3_offload.engine_value_and_grad found HostOffloadedEmbed modules but pre-gather produced no active gathers."
      )

    host_embed_paths = frozenset(gathered_map.keys())
    dev_params, host_params = _partition_state_by_paths(params, host_embed_paths)
    host_param_map = {tuple(p): v for p, v in nnx.to_flat_state(host_params)}

    def _wrapped(dev_p: nnx.State, g_map: dict[tuple[Any, ...], jax.Array]) -> Any:
      full_p = nnx.merge_state(dev_p, host_params) if jax.tree.leaves(host_params) else dev_p
      ctx_map = {p: (meta_map[p]["ids"], g_map[p]) for p in g_map}
      with _pregathered_context(ctx_map):
        return diff_wrapper(full_p, rest, batch)

    val_out, (dev_grads, d_gathered_map) = jax.value_and_grad(_wrapped, argnums=(0, 1), has_aux=True)(
        dev_params, gathered_map
    )
    grad_items = list(nnx.to_flat_state(dev_grads))
    for param_path, dg in d_gathered_map.items():
      meta = meta_map[param_path]
      p_leaf = host_param_map[param_path]
      table = p_leaf.get_raw_value() if isinstance(p_leaf, nnx.Variable) else p_leaf
      d_table_h = _host_embed_scatter_add(table, meta["ids"], dg, row_axes=meta["row_axes"], col_axes=meta["col_axes"])
      grad_leaf = p_leaf.replace(d_table_h) if isinstance(p_leaf, nnx.Variable) else d_table_h
      grad_items.append((param_path, grad_leaf))
    return val_out, nnx.from_flat_state(grad_items)


# ---------------------------------------------------------------------------
# Exact global L2 norm, clipping, and host-safe gradient ops
# ---------------------------------------------------------------------------


def _sparse_grad_sqnorm(grad: SparseHostGrad) -> jax.Array:
  """Exact squared L2 norm `||scatter_add(0, ids, d_gathered)||_2^2` in O(M*D) memory."""
  mesh = _active_mesh()

  def _local_sqnorm(ids_arr: jax.Array, dg_arr: jax.Array, start: jax.Array, n_local: int) -> jax.Array:
    flat_ids = ids_arr.reshape(-1)
    flat_dg = dg_arr.reshape(-1, dg_arr.shape[-1]).astype(jnp.float32)
    m = flat_ids.shape[0]
    rel = flat_ids - start
    in_range = (rel >= 0) & (rel < n_local)
    # Send out-of-shard entries to a unique sentinel so they stay masked out cleanly.
    sentinel = jnp.int32(n_local)
    key = jnp.where(in_range, rel, sentinel)
    vals = jnp.where(in_range[:, None], flat_dg, jnp.zeros((), dtype=jnp.float32))
    order = jnp.argsort(key)
    sorted_key = key[order]
    sorted_vals = vals[order]
    is_new = jnp.concatenate([jnp.array([True]), sorted_key[1:] != sorted_key[:-1]])
    seg_ids = jnp.cumsum(is_new.astype(jnp.int32)) - 1
    seg_sums = jax.ops.segment_sum(sorted_vals, seg_ids, num_segments=m)
    # Match exact grad_dtype rounding before squaring.
    seg_sums = seg_sums.astype(grad.grad_dtype).astype(jnp.float32)
    return jnp.sum(seg_sums * seg_sums)

  if mesh is not None:
    c_spec = _pspec_dim(grad.col_axes)
    rep_ids_spec = P(*([None] * grad.ids.ndim))
    dg_spec = P(*([None] * grad.ids.ndim), c_spec)
    ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
    dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)
    n_shards = 1
    for ax in grad.row_axes:
      n_shards *= mesh.shape[ax]
    n_local = grad.table_shape[0] // n_shards
    sum_axes = (*grad.row_axes, *grad.col_axes)

    def _body(local_ids: jax.Array, local_dg: jax.Array) -> jax.Array:
      start = jax.lax.axis_index(grad.row_axes).astype(jnp.int32) * jnp.int32(n_local) if grad.row_axes else jnp.int32(0)
      local_sq = _local_sqnorm(local_ids, local_dg, start, n_local)
      return jax.lax.psum(local_sq, sum_axes) if sum_axes else local_sq

    return _call_shard_map(_body, in_specs=(rep_ids_spec, dg_spec), out_specs=P(), args=(ids_in, dg_in))

  return _local_sqnorm(grad.ids, grad.d_gathered, jnp.int32(0), grad.table_shape[0])


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Device)
def _host_array_sqnorm_raw(x_h: jax.Array) -> jax.Array:
  x32 = x_h.astype(jnp.float32)
  return jnp.sum(x32 * x32)


def _host_array_sqnorm(x: jax.Array, var_or_arr: Any = None, *, sharding_hint: Any = None) -> jax.Array:
  """Computes squared L2 norm of a host-resident array on host without moving it to HBM."""
  mesh = _active_mesh()
  row_axes, col_axes = _resolve_2d_axes(var_or_arr if var_or_arr is not None else x, mesh, sharding_hint=sharding_hint)
  if mesh is not None and x.ndim == 2:
    r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    sum_axes = (*row_axes, *col_axes)

    def _body(local_x: jax.Array) -> jax.Array:
      local_sq = _host_array_sqnorm_raw(local_x)
      return jax.lax.psum(local_sq, sum_axes) if sum_axes else local_sq

    return _call_shard_map(_body, in_specs=(P(r_spec, c_spec),), out_specs=P(), args=(x,))
  return _host_array_sqnorm_raw(x)


def _apply_host_elementwise(
    host_fn: Any,
    ref_leaf: Any,
    host_arrays: tuple[jax.Array, ...],
    device_scalars: tuple[jax.Array, ...] = (),
    *,
    sharding_hint: Any = None,
) -> jax.Array:
  """Isolates an elementwise host-compute op inside `shard_map` when a mesh is active."""
  mesh = _active_mesh()
  arr0 = host_arrays[0]
  if mesh is not None and arr0.ndim == 2:
    row_axes, col_axes = _resolve_2d_axes(ref_leaf, mesh, sharding_hint=sharding_hint)
    tbl_spec = P(_pspec_dim(row_axes), _pspec_dim(col_axes))
    in_specs = (tbl_spec,) * len(host_arrays) + (P(),) * len(device_scalars)
    n_arr = len(host_arrays)

    def _body(*local_args: jax.Array) -> jax.Array:
      loc_arrs = local_args[:n_arr]
      loc_scalars = tuple(jax.device_put(s, jax.memory.Space.Host) for s in local_args[n_arr:])
      return host_fn(*loc_arrs, *loc_scalars)

    return _call_shard_map(_body, in_specs=in_specs, out_specs=tbl_spec, args=(*host_arrays, *device_scalars))
  scalars_h = tuple(jax.device_put(s, jax.memory.Space.Host) for s in device_scalars)
  return host_fn(*host_arrays, *scalars_h)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _host_cast_array(x_h: jax.Array, dtype_holder_h: jax.Array) -> jax.Array:
  return x_h.astype(dtype_holder_h.dtype)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _host_add_arrays(a_h: jax.Array, b_h: jax.Array) -> jax.Array:
  return a_h + b_h.astype(a_h.dtype)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _host_scale_array(x_h: jax.Array, scale_h: jax.Array) -> jax.Array:
  return x_h * scale_h.astype(x_h.dtype)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _host_clip_array(x_h: jax.Array, trigger_h: jax.Array, inv_norm_h: jax.Array) -> jax.Array:
  scaled = x_h * inv_norm_h.astype(x_h.dtype)
  return jnp.where(trigger_h, x_h, scaled)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _host_where_arrays(true_h: jax.Array, false_h: jax.Array, cond_h: jax.Array) -> jax.Array:
  return jnp.where(cond_h, true_h, false_h)


@compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
def _host_zero_if(arr_h: jax.Array, cond_h: jax.Array) -> jax.Array:
  return jnp.where(cond_h, jax.device_put(jnp.zeros((), dtype=arr_h.dtype), jax.memory.Space.Host), arr_h)


def _is_tree_leaf(x: Any) -> bool:
  return isinstance(x, (nnx.Variable, SparseHostGrad, jax.sharding.Sharding, P))


def select_tree_if(cond: jax.Array, true_tree: Any, false_tree: Any, *, shardings: Any = None) -> Any:
  """Elementwise `jnp.where(cond, true_tree, false_tree)` keeping host leaves on host."""

  def _sel(t_leaf: Any, f_leaf: Any, sh_leaf: Any = None) -> Any:
    if isinstance(f_leaf, SparseHostGrad):
      t_dg = t_leaf.d_gathered if isinstance(t_leaf, SparseHostGrad) else jnp.zeros_like(f_leaf.d_gathered)
      return f_leaf.with_d_gathered(jnp.where(cond, t_dg, f_leaf.d_gathered))
    t_val = t_leaf.get_raw_value() if isinstance(t_leaf, nnx.Variable) else t_leaf
    f_val = f_leaf.get_raw_value() if isinstance(f_leaf, nnx.Variable) else f_leaf
    if not hasattr(f_val, "shape") or not hasattr(f_val, "dtype"):
      return f_leaf
    if _is_host_leaf(t_leaf) or _is_host_leaf(f_leaf):
      t_h = t_val if _is_host_leaf(t_leaf) else jax.device_put(t_val, jax.memory.Space.Host)
      f_h = f_val if _is_host_leaf(f_leaf) else jax.device_put(f_val, jax.memory.Space.Host)
      out = _apply_host_elementwise(_host_where_arrays, f_leaf, (t_h, f_h), (cond,), sharding_hint=sh_leaf)
      return f_leaf.replace(out) if isinstance(f_leaf, nnx.Variable) else out
    out = jnp.where(cond, t_val, f_val)
    return f_leaf.replace(out) if isinstance(f_leaf, nnx.Variable) else out

  if shardings is not None:
    return jax.tree.map(_sel, true_tree, false_tree, shardings, is_leaf=_is_tree_leaf)
  return jax.tree.map(_sel, true_tree, false_tree, is_leaf=_is_tree_leaf)


def zero_tree_if(cond: jax.Array, tree: Any, *, shardings: Any = None) -> Any:
  """Zeros leaves when `cond` is True without copying host leaves onto HBM."""

  def _zero(leaf: Any, sh_leaf: Any = None) -> Any:
    if isinstance(leaf, SparseHostGrad):
      return leaf.with_d_gathered(jnp.where(cond, jnp.zeros_like(leaf.d_gathered), leaf.d_gathered))
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
      return leaf
    if _is_host_leaf(leaf):
      out = _apply_host_elementwise(_host_zero_if, leaf, (val,), (cond,), sharding_hint=sh_leaf)
      return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out
    out = jnp.where(cond, jnp.zeros_like(val), val)
    return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out

  if shardings is not None:
    return jax.tree.map(_zero, tree, shardings, is_leaf=_is_tree_leaf)
  return jax.tree.map(_zero, tree, is_leaf=_is_tree_leaf)


def cast_grad_leaf(leaf: Any, grad_dtype: Any, *, sharding: Any = None) -> Any:
  """Casts a gradient leaf (`device`, `pinned_host`, or `SparseHostGrad`) without moving host arrays to HBM."""
  target = jnp.dtype(grad_dtype)
  if isinstance(leaf, SparseHostGrad):
    if leaf.table_dtype == jnp.float32 and leaf.grad_dtype != target:
      return leaf.with_d_gathered(leaf.d_gathered, grad_dtype=target)
    return leaf
  val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
  if not hasattr(val, "dtype") or val.dtype == target:
    return leaf
  if _is_host_leaf(leaf):
    casted = _apply_host_elementwise(
        _host_cast_array, leaf, (val,), (jnp.zeros((), dtype=target),), sharding_hint=sharding
    )
    return leaf.replace(casted) if isinstance(leaf, nnx.Variable) else casted
  casted = val.astype(target)
  return leaf.replace(casted) if isinstance(leaf, nnx.Variable) else casted


def cast_grad_tree(grads: Any, grad_dtype: Any) -> Any:
  """Casts float32 gradient leaves to `grad_dtype`, keeping host leaves on host."""
  target = jnp.dtype(grad_dtype)

  def _cast(leaf: Any) -> Any:
    if isinstance(leaf, SparseHostGrad):
      return cast_grad_leaf(leaf, target)
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if hasattr(val, "dtype") and val.dtype == jnp.float32 and target != jnp.float32:
      return cast_grad_leaf(leaf, target)
    return leaf

  return jax.tree.map(_cast, grads, is_leaf=lambda x: isinstance(x, (nnx.Variable, SparseHostGrad)))


def accumulate_grad_leaves(acc: Any, g: Any, sharding: Any = None) -> Any:
  """Adds gradient leaf `g` into `acc` on host when both reside in host memory."""
  acc_val = acc.get_raw_value() if isinstance(acc, nnx.Variable) else acc
  g_val = g.get_raw_value() if isinstance(g, nnx.Variable) else g
  if _is_host_leaf(acc) or _is_host_leaf(g):
    acc_h = jax.device_put(acc_val, jax.memory.Space.Host) if not _is_host_leaf(acc) else acc_val
    g_h = jax.device_put(g_val, jax.memory.Space.Host) if not _is_host_leaf(g) else g_val
    out = _apply_host_elementwise(_host_add_arrays, acc, (acc_h, g_h), sharding_hint=sharding)
    return acc.replace(out) if isinstance(acc, nnx.Variable) else out
  out = acc_val + g_val.astype(acc_val.dtype)
  return acc.replace(out) if isinstance(acc, nnx.Variable) else out


def scale_grad_leaf(leaf: Any, factor: jax.Array, *, sharding: Any = None) -> Any:
  """Multiplies a gradient leaf by scalar `factor`, preserving host placement."""
  if isinstance(leaf, SparseHostGrad):
    return leaf.with_d_gathered(leaf.d_gathered * factor.astype(leaf.d_gathered.dtype))
  val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
  if _is_host_leaf(leaf):
    out = _apply_host_elementwise(_host_scale_array, leaf, (val,), (factor,), sharding_hint=sharding)
    return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out
  out = val * factor.astype(val.dtype)
  return leaf.replace(out) if isinstance(leaf, nnx.Variable) else out


def l2norm_pytree(tree: Any, *, shardings: Any = None) -> jax.Array:
  """Exact global L2 norm across device arrays, `SparseHostGrad`, and `pinned_host` leaves."""
  leaves = jax.tree.leaves(tree, is_leaf=_is_tree_leaf)
  sh_leaves = jax.tree.leaves(shardings, is_leaf=_is_tree_leaf) if shardings is not None else [None] * len(leaves)
  sq_terms: list[jax.Array] = []
  for leaf, sh_leaf in zip(leaves, sh_leaves):
    if leaf is None:
      continue
    if isinstance(leaf, SparseHostGrad):
      sq_terms.append(_sparse_grad_sqnorm(leaf))
      continue
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not hasattr(val, "shape"):
      continue
    if _is_host_leaf(leaf):
      sq_terms.append(_host_array_sqnorm(val, leaf, sharding_hint=sh_leaf))
    else:
      v32 = val.astype(jnp.float32)
      sq_terms.append(jnp.sum(v32 * v32))
  if not sq_terms:
    return jnp.array(0.0, dtype=jnp.float32)
  return jnp.sqrt(functools.reduce(jnp.add, sq_terms))


def clip_grads_and_norms(grads: Any, max_norm: float, *, shardings: Any = None) -> tuple[Any, jax.Array, jax.Array]:
  """Clips a mixed gradient tree in one pass, returning `(clipped, raw_norm, clipped_norm)`."""
  g_norm = l2norm_pytree(grads, shardings=shardings)
  if max_norm <= 0:
    return grads, g_norm, g_norm
  max_norm_arr = jnp.asarray(max_norm, dtype=jnp.float32)
  trigger = jnp.squeeze(g_norm < max_norm_arr)
  safe_norm = jnp.where(trigger, jnp.ones_like(g_norm), g_norm)
  scale_f32 = max_norm_arr / safe_norm

  def _clip(leaf: Any, sh_leaf: Any = None) -> Any:
    if leaf is None:
      return None
    if isinstance(leaf, SparseHostGrad):
      # Clip in `grad_dtype` (e.g. fp32 when `weight_dtype=float32, dtype=bfloat16`) to avoid bf16 pre-rounding.
      dg = leaf.d_gathered.astype(leaf.grad_dtype)
      clipped_dg = jax.lax.select(trigger, dg, (dg / safe_norm.astype(dg.dtype)) * max_norm_arr.astype(dg.dtype))
      return leaf.with_d_gathered(clipped_dg)
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not hasattr(val, "shape"):
      return leaf
    if _is_host_leaf(leaf):
      clipped = _apply_host_elementwise(_host_clip_array, leaf, (val,), (trigger, scale_f32), sharding_hint=sh_leaf)
      return leaf.replace(clipped) if isinstance(leaf, nnx.Variable) else clipped
    clipped = jax.lax.select(trigger, val, (val / safe_norm.astype(val.dtype)) * max_norm_arr.astype(val.dtype))
    return leaf.replace(clipped) if isinstance(leaf, nnx.Variable) else clipped

  if shardings is not None:
    clipped = jax.tree.map(_clip, grads, shardings, is_leaf=_is_tree_leaf)
  else:
    clipped = jax.tree.map(_clip, grads, is_leaf=_is_tree_leaf)
  clipped_norm = jnp.where(trigger, g_norm, jnp.where(jnp.isfinite(g_norm), max_norm_arr, g_norm))
  return clipped, g_norm, clipped_norm


def clip_grads_by_global_norm(grads: Any, max_norm: float, *, shardings: Any = None) -> Any:
  """Clips a mixed device/`SparseHostGrad`/`pinned_host` gradient tree by global L2 norm."""
  clipped, _, _ = clip_grads_and_norms(grads, max_norm, shardings=shardings)
  return clipped


@contextlib.contextmanager
def capture_host_param_sqnorms():
  """Captures post-update squared L2 norms of host-offloaded params inside the optimizer host pass."""
  prev = getattr(_HOST_SQNORM_CAPTURE, "sqnorms", None)
  captured: dict[tuple[Any, ...], jax.Array] = {}
  _HOST_SQNORM_CAPTURE.sqnorms = captured
  try:
    yield captured
  finally:
    _HOST_SQNORM_CAPTURE.sqnorms = prev


def l2norm_pytree_with_overrides(tree: Any, sqnorm_overrides: dict[tuple[Any, ...], jax.Array]) -> jax.Array:
  """Exact global L2 norm substituting pre-computed squared norms for captured host parameters."""
  if not sqnorm_overrides:
    return l2norm_pytree(tree)
  sq_terms: list[jax.Array] = []
  for path, leaf in nnx.to_flat_state(tree):
    path_tuple = tuple(path)
    if path_tuple in sqnorm_overrides:
      sq_terms.append(sqnorm_overrides[path_tuple])
      continue
    if leaf is None:
      continue
    if isinstance(leaf, SparseHostGrad):
      sq_terms.append(_sparse_grad_sqnorm(leaf))
      continue
    val = leaf.get_raw_value() if isinstance(leaf, nnx.Variable) else leaf
    if not hasattr(val, "dtype") or not hasattr(val, "shape"):
      continue
    if _is_host_leaf(leaf):
      sq_terms.append(_host_array_sqnorm(val, leaf))
    else:
      v32 = val.astype(jnp.float32)
      sq_terms.append(jnp.sum(v32 * v32))
  if not sq_terms:
    return jnp.array(0.0, dtype=jnp.float32)
  return jnp.sqrt(functools.reduce(jnp.add, sq_terms))


# ---------------------------------------------------------------------------
# Selective optimizer state slicing & host-compute / streamed-chunks / lazy-sparse updates
# ---------------------------------------------------------------------------


def _nest_path(path: tuple[Any, ...], value: Any) -> dict[Any, Any]:
  node: Any = value
  for key in reversed(path):
    node = {key: node}
  return node


def _unnest_path(path: tuple[Any, ...], tree: Any) -> Any:
  node = tree
  for key in path:
    node = node[key]
  return node


def _filter_mapping_by_paths(
    mapping: dict[Any, Any],
    keep_fn: Callable[[tuple[Any, ...]], bool],
    prefix: tuple[Any, ...] = (),
) -> dict[Any, Any]:
  """Filters a nested parameter/state dict keeping only leaves where `keep_fn(path)` is True."""
  out: dict[Any, Any] = {}
  for k, v in mapping.items():
    sub_path = (*prefix, k)
    if isinstance(v, (dict, nnx.State)):
      sub_dict = dict(v) if isinstance(v, nnx.State) else v
      filtered = _filter_mapping_by_paths(sub_dict, keep_fn, sub_path)
      if filtered:
        out[k] = filtered
    elif keep_fn(sub_path):
      out[k] = v
  return out


def _flat_dict_paths(mapping: dict[Any, Any], prefix=()) -> set[tuple[Any, ...]]:
  paths: set[tuple[Any, ...]] = set()
  for k, v in mapping.items():
    sub_path = (*prefix, k)
    if isinstance(v, (dict, nnx.State)):
      sub_dict = dict(v) if isinstance(v, nnx.State) else v
      paths |= _flat_dict_paths(sub_dict, sub_path)
    else:
      paths.add(sub_path)
  return paths


def _slice_opt_state(
    opt_state: Any,
    all_param_paths: frozenset[tuple[Any, ...]],
    keep_fn: Callable[[tuple[Any, ...]], bool],
) -> Any:
  """Filters parameter-shaped subtrees inside `opt_state` while keeping scalars intact."""
  if isinstance(opt_state, (dict, nnx.State)):
    as_dict = dict(opt_state) if isinstance(opt_state, nnx.State) else opt_state
    leaf_paths = _flat_dict_paths(as_dict)
    if leaf_paths and leaf_paths.issubset(all_param_paths):
      return _filter_mapping_by_paths(as_dict, keep_fn)
    return {k: _slice_opt_state(v, all_param_paths, keep_fn) for k, v in as_dict.items()}
  if isinstance(opt_state, tuple) and hasattr(opt_state, "_fields"):
    return type(opt_state)(
        *[_slice_opt_state(getattr(opt_state, f), all_param_paths, keep_fn) for f in opt_state._fields]
    )
  if isinstance(opt_state, tuple):
    return tuple(_slice_opt_state(v, all_param_paths, keep_fn) for v in opt_state)
  if isinstance(opt_state, list):
    return [_slice_opt_state(v, all_param_paths, keep_fn) for v in opt_state]
  return opt_state


def _lookup_path_in_tree(tree: Any, key_tuple: tuple[Any, ...]) -> Any:
  """Traverses a nested opt_state PyTree along `key_tuple`."""
  node = tree
  for k in key_tuple:
    if isinstance(node, (dict, nnx.State)):
      node = node[k]
    elif isinstance(node, (tuple, list)) and isinstance(k, int):
      node = node[k]
    elif hasattr(node, str(k)):
      node = getattr(node, str(k))
    else:
      node = node[k]
  return node


def _is_matching_opt_leaf(
    key_tuple: tuple[Any, ...], selected_paths: frozenset[tuple[Any, ...]]
) -> tuple[Any, ...] | None:
  for p_path in selected_paths:
    n = len(p_path)
    if len(key_tuple) >= n and key_tuple[-n:] == p_path:
      return p_path
  return None


def _run_tx_on_single_leaf(
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    grad_arr: jax.Array,
    opt_state_sub: Any,
    param_arr: jax.Array,
    **kwargs: Any,
) -> tuple[jax.Array, Any]:
  """Runs `tx.update` + `apply_updates` on a single parameter wrapped at its exact path."""
  g_nested = _nest_path(p_path, grad_arr)
  p_nested = _nest_path(p_path, param_arr)
  updates_nested, new_st = tx.update(g_nested, opt_state_sub, p_nested, **kwargs)
  upd_arr = _unnest_path(p_path, updates_nested)
  new_p = optax.apply_updates(param_arr, upd_arr)
  return new_p, new_st


def _selective_host_compute_update(
    param: jax.Array,
    grad: Any,
    opt_state_sub: Any,
    *,
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    kwargs_arrays: dict[str, Any],
) -> tuple[jax.Array, Any]:
  """Runs exact dense optimizer update on host CPU with zero HBM full-table traffic."""
  mesh = _active_mesh()
  is_sparse = isinstance(grad, SparseHostGrad)

  capture_map = getattr(_HOST_SQNORM_CAPTURE, "sqnorms", None)
  capture_sqnorm = capture_map is not None

  @compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
  def host_sparse_fn(p_loc, st_loc_h, sparse_h, start_h, kw_h):
    ids_h, dg_h = sparse_h
    n_local = p_loc.shape[0]
    g_dtype = grad.grad_dtype
    if row_axes:
      rel = ids_h - start_h
      in_range = (rel >= 0) & (rel < n_local)
      zero_i = jax.device_put(jnp.int32(0), jax.memory.Space.Host)
      safe = jnp.where(in_range, rel, zero_i)
      zero_v = jax.device_put(jnp.zeros((), dtype=g_dtype), jax.memory.Space.Host)
      vals = jnp.where(in_range[..., None], dg_h.astype(g_dtype), zero_v)
      z_buf = jax.device_put(jnp.zeros(p_loc.shape, dtype=g_dtype), jax.memory.Space.Host)
      g_loc = z_buf.at[safe].add(vals)
    else:
      z_buf = jax.device_put(jnp.zeros(p_loc.shape, dtype=g_dtype), jax.memory.Space.Host)
      g_loc = z_buf.at[ids_h].add(dg_h.astype(g_dtype))
    p_new, st_new = _run_tx_on_single_leaf(tx, p_path, g_loc, st_loc_h, p_loc, **kw_h)
    if capture_sqnorm:
      p32 = p_new.astype(jnp.float32)
      return p_new, st_new, jnp.sum(p32 * p32)
    return p_new, st_new

  @compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Host)
  def host_dense_fn(p_loc, st_loc_h, g_loc, kw_h):
    p_new, st_new = _run_tx_on_single_leaf(tx, p_path, g_loc, st_loc_h, p_loc, **kw_h)
    if capture_sqnorm:
      p32 = p_new.astype(jnp.float32)
      return p_new, st_new, jnp.sum(p32 * p32)
    return p_new, st_new

  def _move_scalars(tree: Any, space: Any) -> Any:
    return jax.tree.map(lambda x: jax.device_put(x, space) if hasattr(x, "ndim") and x.ndim == 0 else x, tree)

  if mesh is not None:
    r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    tbl_spec = P(r_spec, c_spec)
    st_spec = jax.tree.map(lambda x: tbl_spec if hasattr(x, "ndim") and x.ndim == 2 else P(), opt_state_sub)
    kw_spec = jax.tree.map(lambda _: P(), kwargs_arrays)
    sum_axes = (*row_axes, *col_axes)
    out_specs = (tbl_spec, st_spec, P()) if capture_sqnorm else (tbl_spec, st_spec)
    if is_sparse:
      rep_ids_spec = P(*([None] * grad.ids.ndim))
      dg_spec = P(*([None] * grad.ids.ndim), c_spec)
      ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
      dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)

      def _smap_sparse(p_loc, st_loc, ids_loc, dg_loc, kw_loc):
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
        res = host_sparse_fn(
            p_loc,
            _move_scalars(st_loc, jax.memory.Space.Host),
            (jax.device_put(ids_loc, jax.memory.Space.Host), jax.device_put(dg_loc, jax.memory.Space.Host)),
            jax.device_put(start, jax.memory.Space.Host),
            _move_scalars(kw_loc, jax.memory.Space.Host),
        )
        p_new, st_new_h = res[0], res[1]
        if capture_sqnorm:
          sq_dev = jax.device_put(res[2], jax.memory.Space.Device)
          sq_tot = jax.lax.psum(sq_dev, sum_axes) if sum_axes else sq_dev
          return p_new, _move_scalars(st_new_h, jax.memory.Space.Device), sq_tot
        return p_new, _move_scalars(st_new_h, jax.memory.Space.Device)

      out = _call_shard_map(
          _smap_sparse,
          in_specs=(tbl_spec, st_spec, rep_ids_spec, dg_spec, kw_spec),
          out_specs=out_specs,
          args=(param, opt_state_sub, ids_in, dg_in, kwargs_arrays),
      )
      if capture_sqnorm:
        capture_map[p_path] = out[2]
        return out[0], out[1]
      return out

    def _smap_dense(p_loc, st_loc, g_loc, kw_loc):
      res = host_dense_fn(
          p_loc,
          _move_scalars(st_loc, jax.memory.Space.Host),
          g_loc,
          _move_scalars(kw_loc, jax.memory.Space.Host),
      )
      p_new, st_new_h = res[0], res[1]
      if capture_sqnorm:
        sq_dev = jax.device_put(res[2], jax.memory.Space.Device)
        sq_tot = jax.lax.psum(sq_dev, sum_axes) if sum_axes else sq_dev
        return p_new, _move_scalars(st_new_h, jax.memory.Space.Device), sq_tot
      return p_new, _move_scalars(st_new_h, jax.memory.Space.Device)

    out = _call_shard_map(
        _smap_dense,
        in_specs=(tbl_spec, st_spec, tbl_spec, kw_spec),
        out_specs=out_specs,
        args=(param, opt_state_sub, grad, kwargs_arrays),
    )
    if capture_sqnorm:
      capture_map[p_path] = out[2]
      return out[0], out[1]
    return out

  st_h = _move_scalars(opt_state_sub, jax.memory.Space.Host)
  kw_h = _move_scalars(kwargs_arrays, jax.memory.Space.Host)
  if is_sparse:
    res = host_sparse_fn(
        param,
        st_h,
        (jax.device_put(grad.ids, jax.memory.Space.Host), jax.device_put(grad.d_gathered, jax.memory.Space.Host)),
        jax.device_put(jnp.int32(0), jax.memory.Space.Host),
        kw_h,
    )
  else:
    res = host_dense_fn(param, st_h, grad, kw_h)
  p_new, st_new_h = res[0], res[1]
  if capture_sqnorm:
    capture_map[p_path] = jax.device_put(res[2], jax.memory.Space.Device)
  return p_new, _move_scalars(st_new_h, jax.memory.Space.Device)


def _selective_streamed_chunks_update(
    param: jax.Array,
    grad: Any,
    opt_state_sub: Any,
    *,
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    num_chunks: int,
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    kwargs_arrays: dict[str, Any],
) -> tuple[jax.Array, Any]:
  """Runs exact dense optimizer update by streaming `num_chunks` row slices through device."""
  mesh = _active_mesh()
  is_sparse = isinstance(grad, SparseHostGrad)

  def _run_local_stream(p_loc, st_loc, grad_arg, shard_start, kw_loc):
    n_local, width = p_loc.shape
    k = min(num_chunks, n_local)
    cs = (n_local + k - 1) // k
    k_eff = (n_local + cs - 1) // cs

    st_leaves, st_treedef = jax.tree_util.tree_flatten(st_loc)
    arr_indices = [
        idx for idx, lf in enumerate(st_leaves) if hasattr(lf, "shape") and tuple(lf.shape) == (n_local, width)
    ]
    host_arrs_init = (p_loc, *[st_leaves[idx] for idx in arr_indices])

    @compute_on.compute_on(
        compute_type="device_host",
        out_memory_spaces=(jax.memory.Space.Device,) * len(host_arrs_init),
    )
    def _read_chunk(arrs_h, off_h):
      return tuple(jax.lax.dynamic_slice(a, (off_h, 0), (cs, width)) for a in arrs_h)

    @compute_on.compute_on(
        compute_type="device_host",
        out_memory_spaces=(jax.memory.Space.Host,) * len(host_arrs_init),
    )
    def _write_chunk(arrs_h, chunks_h, off_h):
      return tuple(jax.lax.dynamic_update_slice(a, c, (off_h, 0)) for a, c in zip(arrs_h, chunks_h))

    @compute_on.compute_on(compute_type="device_host", out_memory_spaces=jax.memory.Space.Device)
    def _read_grad_chunk(g_h, off_h):
      return jax.lax.dynamic_slice(g_h, (off_h, 0), (cs, width))

    def _scan_body(carry, i):
      arrs_h = carry
      raw_off = i * jnp.int32(cs)
      slice_off = jnp.minimum(raw_off, jnp.int32(n_local - cs))
      off_h = jax.device_put(slice_off, jax.memory.Space.Host)
      chunks_dev = _read_chunk(arrs_h, off_h)
      p_c = chunks_dev[0]
      st_arr_c = chunks_dev[1:]
      if is_sparse:
        ids_loc, dg_loc = grad_arg
        g_dtype = grad.grad_dtype
        rel = ids_loc - (shard_start + slice_off)
        in_chunk = (rel >= 0) & (rel < cs)
        safe = jnp.where(in_chunk, rel, jnp.int32(0))
        vals = jnp.where(in_chunk[..., None], dg_loc.astype(g_dtype), jnp.zeros((), dtype=g_dtype))
        g_c = jnp.zeros((cs, width), dtype=g_dtype).at[safe].add(vals)
      else:
        g_c = _read_grad_chunk(grad_arg, off_h)

      cur_leaves = list(st_leaves)
      for pos, idx in enumerate(arr_indices):
        cur_leaves[idx] = st_arr_c[pos]
      st_c = jax.tree_util.tree_unflatten(st_treedef, cur_leaves)
      p_new_c, st_new_c = _run_tx_on_single_leaf(tx, p_path, g_c, st_c, p_c, **kw_loc)
      new_leaves, _ = jax.tree_util.tree_flatten(st_new_c)
      new_arr_c = (p_new_c, *[new_leaves[idx] for idx in arr_indices])

      # Mask overlapping rows on the final chunk so every row updates exactly once.
      active = ((slice_off + jnp.arange(cs, dtype=jnp.int32)) >= raw_off)[:, None]
      masked_chunks = tuple(jnp.where(active, new_c, old_c) for new_c, old_c in zip(new_arr_c, chunks_dev))
      chunks_h = tuple(jax.device_put(c, jax.memory.Space.Host) for c in masked_chunks)
      arrs_h = _write_chunk(arrs_h, chunks_h, off_h)
      scalar_out = tuple(lf for idx, lf in enumerate(new_leaves) if idx not in arr_indices)
      return arrs_h, scalar_out

    final_arrs_h, scalar_hist = jax.lax.scan(_scan_body, host_arrs_init, jnp.arange(k_eff, dtype=jnp.int32))
    final_scalars = [s[-1] for s in scalar_hist]
    out_leaves = list(st_leaves)
    arr_set = set(arr_indices)
    sc_iter = iter(final_scalars)
    arr_iter = iter(final_arrs_h[1:])
    for idx, _ in enumerate(out_leaves):
      out_leaves[idx] = next(arr_iter) if idx in arr_set else next(sc_iter)
    return final_arrs_h[0], jax.tree_util.tree_unflatten(st_treedef, out_leaves)

  if mesh is not None:
    r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    tbl_spec = P(r_spec, c_spec)
    st_spec = jax.tree.map(lambda x: tbl_spec if hasattr(x, "ndim") and x.ndim == 2 else P(), opt_state_sub)
    kw_spec = jax.tree.map(lambda _: P(), kwargs_arrays)
    if is_sparse:
      rep_ids_spec = P(*([None] * grad.ids.ndim))
      dg_spec = P(*([None] * grad.ids.ndim), c_spec)
      ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
      dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)

      def _smap_sparse(p_loc, st_loc, ids_loc, dg_loc, kw_loc):
        start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
        return _run_local_stream(p_loc, st_loc, (ids_loc, dg_loc), start, kw_loc)

      return _call_shard_map(
          _smap_sparse,
          in_specs=(tbl_spec, st_spec, rep_ids_spec, dg_spec, kw_spec),
          out_specs=(tbl_spec, st_spec),
          args=(param, opt_state_sub, ids_in, dg_in, kwargs_arrays),
      )

    def _smap_dense(p_loc, st_loc, g_loc, kw_loc):
      start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
      return _run_local_stream(p_loc, st_loc, g_loc, start, kw_loc)

    return _call_shard_map(
        _smap_dense,
        in_specs=(tbl_spec, st_spec, tbl_spec, kw_spec),
        out_specs=(tbl_spec, st_spec),
        args=(param, opt_state_sub, grad, kwargs_arrays),
    )

  grad_arg = (grad.ids, grad.d_gathered) if is_sparse else grad
  return _run_local_stream(param, opt_state_sub, grad_arg, jnp.int32(0), kwargs_arrays)


def _selective_lazy_sparse_update(
    param: jax.Array,
    grad: SparseHostGrad,
    opt_state_sub: Any,
    *,
    tx: optax.GradientTransformation,
    p_path: tuple[Any, ...],
    row_axes: tuple[str, ...],
    col_axes: tuple[str, ...],
    kwargs_arrays: dict[str, Any],
) -> tuple[jax.Array, Any]:
  """Opt-in sparse-row AdamW approximation updating only touched rows (skips decay on untouched rows)."""
  mesh = _active_mesh()

  def _run_local_sparse(p_loc, st_loc, sparse_loc, shard_start, kw_loc):
    ids_loc, dg_loc = sparse_loc
    n_local, width = p_loc.shape
    g_dtype = grad.grad_dtype
    flat_ids = ids_loc.reshape(-1)
    flat_dg = dg_loc.reshape(-1, width).astype(g_dtype)
    m = flat_ids.shape[0]
    rel = flat_ids - shard_start
    in_range = (rel >= 0) & (rel < n_local)
    sentinel = jnp.int32(n_local)
    key = jnp.where(in_range, rel, sentinel)
    vals = jnp.where(in_range[:, None], flat_dg, jnp.zeros((), dtype=g_dtype))
    order = jnp.argsort(key)
    sorted_key = key[order]
    sorted_vals = vals[order]
    is_new = jnp.concatenate([jnp.array([True]), sorted_key[1:] != sorted_key[:-1]])
    seg_ids = jnp.cumsum(is_new.astype(jnp.int32)) - 1
    unique_grads = jax.ops.segment_sum(sorted_vals, seg_ids, num_segments=m)
    unique_ids = jnp.full((m,), sentinel, dtype=jnp.int32).at[seg_ids].min(sorted_key)
    valid_row = unique_ids < sentinel
    safe_ids = jnp.where(valid_row, unique_ids, jnp.int32(0))

    st_leaves, st_treedef = jax.tree_util.tree_flatten(st_loc)
    arr_indices = [
        idx for idx, lf in enumerate(st_leaves) if hasattr(lf, "shape") and tuple(lf.shape) == (n_local, width)
    ]
    host_arrs_init = (p_loc, *[st_leaves[idx] for idx in arr_indices])

    @compute_on.compute_on(
        compute_type="device_host",
        out_memory_spaces=(jax.memory.Space.Device,) * len(host_arrs_init),
    )
    def _gather_rows(arrs_h, idx_h):
      return tuple(a[idx_h] for a in arrs_h)

    @compute_on.compute_on(
        compute_type="device_host",
        out_memory_spaces=(jax.memory.Space.Host,) * len(host_arrs_init),
    )
    def _scatter_rows(arrs_h, rows_h, write_idx_h):
      # Out-of-bounds index `n_local` with `mode="drop"` avoids clobbering row 0.
      return tuple(a.at[write_idx_h].set(r, mode="drop") for a, r in zip(arrs_h, rows_h))

    rows_dev = _gather_rows(host_arrs_init, jax.device_put(safe_ids, jax.memory.Space.Host))
    cur_leaves = list(st_leaves)
    for pos, idx in enumerate(arr_indices):
      cur_leaves[idx] = rows_dev[pos + 1]
    st_rows = jax.tree_util.tree_unflatten(st_treedef, cur_leaves)
    p_rows_new, st_rows_new = _run_tx_on_single_leaf(tx, p_path, unique_grads, st_rows, rows_dev[0], **kw_loc)
    new_leaves, _ = jax.tree_util.tree_flatten(st_rows_new)
    new_rows_h = tuple(
        jax.device_put(r, jax.memory.Space.Host) for r in (p_rows_new, *[new_leaves[idx] for idx in arr_indices])
    )
    write_idx = jnp.where(valid_row, unique_ids, sentinel)
    updated_arrs_h = _scatter_rows(host_arrs_init, new_rows_h, jax.device_put(write_idx, jax.memory.Space.Host))

    out_leaves = list(new_leaves)
    for pos, idx in enumerate(arr_indices):
      out_leaves[idx] = updated_arrs_h[pos + 1]
    return updated_arrs_h[0], jax.tree_util.tree_unflatten(st_treedef, out_leaves)

  if mesh is not None:
    r_spec, c_spec = _pspec_dim(row_axes), _pspec_dim(col_axes)
    tbl_spec = P(r_spec, c_spec)
    st_spec = jax.tree.map(lambda x: tbl_spec if hasattr(x, "ndim") and x.ndim == 2 else P(), opt_state_sub)
    rep_ids_spec = P(*([None] * grad.ids.ndim))
    dg_spec = P(*([None] * grad.ids.ndim), c_spec)
    kw_spec = jax.tree.map(lambda _: P(), kwargs_arrays)
    ids_in = _maybe_reshard(grad.ids, rep_ids_spec, mesh)
    dg_in = _maybe_reshard(grad.d_gathered, dg_spec, mesh)

    def _smap(p_loc, st_loc, ids_loc, dg_loc, kw_loc):
      start = jax.lax.axis_index(row_axes).astype(jnp.int32) * jnp.int32(p_loc.shape[0]) if row_axes else jnp.int32(0)
      return _run_local_sparse(p_loc, st_loc, (ids_loc, dg_loc), start, kw_loc)

    return _call_shard_map(
        _smap,
        in_specs=(tbl_spec, st_spec, rep_ids_spec, dg_spec, kw_spec),
        out_specs=(tbl_spec, st_spec),
        args=(param, opt_state_sub, ids_in, dg_in, kwargs_arrays),
    )

  return _run_local_sparse(param, opt_state_sub, (grad.ids, grad.d_gathered), jnp.int32(0), kwargs_arrays)


class HostOffloadedOptimizer(nnx.Optimizer):
  """An `nnx.Optimizer` parking all or selected `opt_state`/params in `pinned_host` between steps."""

  def __init__(
      self,
      model: nnx.Module,
      tx: optax.GradientTransformation,
      wrt: Any = nnx.Param,
      *,
      offload_opt_state: bool = True,
      selected_param_paths: frozenset[tuple[Any, ...]] = frozenset(),
      optimizer_mode: str = "host_compute",
      num_chunks: int = 16,
      clip_max_norm: float = 0.0,
  ):
    if optimizer_mode not in _OPTIMIZER_MODES:
      raise ValueError(f"Unsupported optimizer_mode {optimizer_mode!r}; expected one of {sorted(_OPTIMIZER_MODES)}.")
    self._offload_opt_state = offload_opt_state
    self._selected_param_paths = frozenset(tuple(p) for p in selected_param_paths)
    self._optimizer_mode = optimizer_mode
    self._num_chunks = int(num_chunks)
    self._clip_max_norm = float(clip_max_norm)

    if self._selected_param_paths:
      # Selective mode: leave selected host params in `pinned_host` during `tx.init` so
      # `zeros_like` allocates their moments directly in host memory without an HBM spike.
      super().__init__(model, tx, wrt=wrt)
      selected_opt_vars: list[nnx.Variable] = []
      other_opt_vars: list[nnx.Variable] = []
      for path, leaf in jax.tree_util.tree_leaves_with_path(
          self.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      ):
        if not isinstance(leaf, nnx.Variable):
          continue
        raw = leaf.get_raw_value()
        if not hasattr(raw, "ndim") or raw.ndim == 0:
          continue
        key_tuple = tuple(_key_value(k) for k in path)
        if _is_matching_opt_leaf(key_tuple, self._selected_param_paths) is not None:
          selected_opt_vars.append(leaf)
        else:
          other_opt_vars.append(leaf)
      if _can_move_vars(selected_opt_vars):
        _move_variables_in_place(selected_opt_vars, "pinned_host")
      else:
        for v in selected_opt_vars:
          _attach_memory_kind_metadata(v, "pinned_host")
          raw = v.get_raw_value()
          if isinstance(raw, jax.core.Tracer) and not _is_host_leaf(v):
            v.set_raw_value(jax.device_put(raw, jax.memory.Space.Host))
      if self._offload_opt_state and other_opt_vars:
        if _can_move_vars(other_opt_vars):
          _move_variables_in_place(other_opt_vars, "pinned_host")
        else:
          for v in other_opt_vars:
            _attach_memory_kind_metadata(v, "pinned_host")
            raw = v.get_raw_value()
            if isinstance(raw, jax.core.Tracer) and not _is_host_leaf(v):
              v.set_raw_value(jax.device_put(raw, jax.memory.Space.Host))
      return

    param_vars = [
        v
        for v in jax.tree.leaves(nnx.state(model, wrt), is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ]
    restore_host_params = any(_is_host_leaf(v) for v in param_vars)
    saved_raw = [v.get_raw_value() for v in param_vars] if restore_host_params else []
    if restore_host_params:
      _move_variables_in_place(param_vars, "device")
    try:
      super().__init__(model, tx, wrt=wrt)
    finally:
      if restore_host_params:
        for v, raw in zip(param_vars, saved_raw):
          _attach_memory_kind_metadata(v, "pinned_host")
          v.set_raw_value(raw)
    if self._offload_opt_state:
      _move_variables_in_place(self.opt_state, "pinned_host")

  def _update_selective(self, model: nnx.Module, grads: Any, **kwargs: Any) -> None:
    """Updates device and selectively host-offloaded parameters in place."""
    if self._clip_max_norm > 0.0:
      grads = clip_grads_by_global_norm(grads, self._clip_max_norm)

    params_state = nnx.state(model, self.wrt, graph=self.graph)
    param_var_map = {tuple(p): v for p, v in nnx.to_flat_state(params_state)}
    all_param_paths = frozenset(param_var_map.keys())
    selected = self._selected_param_paths & all_param_paths

    grads_state = nnx.state(grads, self.wrt, graph=self.graph) if isinstance(grads, nnx.Module) else grads
    grad_val_map = {
        tuple(p): (v.get_raw_value() if isinstance(v, nnx.Variable) else v) for p, v in nnx.to_flat_state(grads_state)
    }
    opt_state_pure = nnx.as_pure(self.opt_state)
    kwargs_arrays = nnx.as_pure(kwargs)

    # 1. Update non-selected parameters on device.
    dev_paths = all_param_paths - selected
    updated_dev_opt_state: Any = None
    if dev_paths:
      dev_params_pure = _filter_mapping_by_paths(nnx.as_pure(params_state), lambda p: p in dev_paths)
      dev_grads_pure = _filter_mapping_by_paths(nnx.as_pure(grads_state), lambda p: p in dev_paths)
      dev_opt_in = _slice_opt_state(opt_state_pure, all_param_paths, lambda p: p in dev_paths)
      if self._offload_opt_state:
        dev_opt_in = to_memory(dev_opt_in, "device")
      dev_updates, updated_dev_opt_state = self.tx.update(dev_grads_pure, dev_opt_in, dev_params_pure, **kwargs_arrays)
      new_dev_params = optax.apply_updates(dev_params_pure, dev_updates)
      for p_path in dev_paths:
        param_var_map[p_path].set_raw_value(_unnest_path(p_path, new_dev_params))
      if self._offload_opt_state:
        updated_dev_opt_state = to_memory(updated_dev_opt_state, "pinned_host")

    # 2. Update each selectively host-offloaded parameter without materializing dense HBM gradients.
    mesh = _active_mesh()
    updated_host_opt_states: dict[tuple[Any, ...], Any] = {}
    for p_path in sorted(selected):
      p_var = param_var_map[p_path]
      p_val = p_var.get_raw_value()
      g_val = grad_val_map[p_path]
      st_sub = _slice_opt_state(opt_state_pure, all_param_paths, lambda p, target=p_path: p == target)
      row_axes, col_axes = _resolve_2d_axes(p_var, mesh)

      if self._optimizer_mode == "streamed_chunks":
        p_new, st_new = _selective_streamed_chunks_update(
            p_val,
            g_val,
            st_sub,
            tx=self.tx,
            p_path=p_path,
            num_chunks=self._num_chunks,
            row_axes=row_axes,
            col_axes=col_axes,
            kwargs_arrays=kwargs_arrays,
        )
      elif self._optimizer_mode == "lazy_sparse":
        if not isinstance(g_val, SparseHostGrad):
          raise ValueError(
              f"selective_host_offload_optimizer_mode='lazy_sparse' requires SparseHostGrad for {p_path}, "
              f"got dense {type(g_val).__name__}. Use 'host_compute' or 'streamed_chunks' with dense host "
              "gradients (such as MaxTextTrainingEngine)."
          )
        p_new, st_new = _selective_lazy_sparse_update(
            p_val,
            g_val,
            st_sub,
            tx=self.tx,
            p_path=p_path,
            row_axes=row_axes,
            col_axes=col_axes,
            kwargs_arrays=kwargs_arrays,
        )
      else:
        p_new, st_new = _selective_host_compute_update(
            p_val,
            g_val,
            st_sub,
            tx=self.tx,
            p_path=p_path,
            row_axes=row_axes,
            col_axes=col_axes,
            kwargs_arrays=kwargs_arrays,
        )
      p_var.set_raw_value(p_new)
      updated_host_opt_states[p_path] = st_new

    # 3. Write updated optimizer state back into existing Variables in place.
    fallback_scalar_state = (
        updated_dev_opt_state if updated_dev_opt_state is not None else next(iter(updated_host_opt_states.values()))
    )
    for path, leaf in jax.tree_util.tree_leaves_with_path(self.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)):
      if not isinstance(leaf, nnx.Variable):
        continue
      key_tuple = tuple(_key_value(k) for k in path)
      matched_p = _is_matching_opt_leaf(key_tuple, selected)
      if matched_p is not None:
        leaf.set_raw_value(_lookup_path_in_tree(updated_host_opt_states[matched_p], key_tuple))
      else:
        leaf.set_raw_value(_lookup_path_in_tree(fallback_scalar_state, key_tuple))
    self.step[...] += 1

  def update(self, model: nnx.Module, grads: Any, **kwargs: Any) -> Any:  # pylint: disable=arguments-differ
    """Performs an optimizer step, handling full or selective pinned-host offload."""
    if self._selected_param_paths:
      return self._update_selective(model, grads, **kwargs)
    if self._offload_opt_state:
      _move_variables_in_place(self.opt_state, "device")
    params = nnx.state(model, self.wrt)
    offload_params = _has_host_leaves(params)
    if offload_params:
      _move_variables_in_place(params, "device")
    if offload_params or _has_host_leaves(grads):
      grads = to_memory(grads, "device")
    super().update(model, grads, **kwargs)
    if offload_params:
      _move_variables_in_place(nnx.state(model, self.wrt), "pinned_host")
    if self._offload_opt_state:
      _move_variables_in_place(self.opt_state, "pinned_host")
    return None


def make_optimizer(
    model: nnx.Module,
    tx: optax.GradientTransformation,
    config: Any,
    wrt: Any = nnx.Param,
    *,
    clip_max_norm: float = 0.0,
    dense_host_grads: bool = False,
) -> nnx.Optimizer:
  """Builds the optimizer, using `HostOffloadedOptimizer` when any host offload mode is active."""
  offload_opt = bool(getattr(config, "optimizer_memory_host_offload", False))
  selective = is_selective_host_offload_enabled(config) or has_selective_host_offload(model)
  if selective:
    mode = _get_optimizer_mode(config)
    if dense_host_grads and mode == "lazy_sparse":
      raise ValueError(
          "selective_host_offload_optimizer_mode='lazy_sparse' is incompatible with dense pinned_host "
          "gradients (MaxTextTrainingEngine); use 'host_compute' or 'streamed_chunks'."
      )
    selected_paths = selected_host_param_paths(model)
    if not selected_paths:
      patterns = get_host_offload_param_patterns(config)
      selected_paths = _match_param_paths(nnx.state(model, wrt), patterns)
    return HostOffloadedOptimizer(
        model,
        tx,
        wrt=wrt,
        offload_opt_state=offload_opt,
        selected_param_paths=selected_paths,
        optimizer_mode=mode,
        num_chunks=_get_num_chunks(config),
        clip_max_norm=clip_max_norm,
    )
  offload_params = bool(getattr(config, "parameter_memory_host_offload", False)) or _has_host_leaves(
      nnx.state(model, wrt)
  )
  if offload_opt or offload_params:
    return HostOffloadedOptimizer(model, tx, wrt=wrt, offload_opt_state=offload_opt)
  return nnx.Optimizer(model, tx, wrt=wrt)


def get_device_hbm_stats(device: Any = None) -> dict[str, float] | None:
  """Queries live `device.memory_stats()` including reserved XLA HLO workspace.

  Returns `None` on backends (such as CPU or non-addressable Pathways IFRT devices)
  where `memory_stats()` returns `None`, raises `RuntimeError`, or omits
  `bytes_in_use` / `peak_bytes_in_use`.
  """
  if device is None:
    local_devices = jax.local_devices()
    if not local_devices:
      return None
    device = local_devices[0]
  stats_fn = getattr(device, "memory_stats", None)
  if not callable(stats_fn):
    return None
  try:
    stats = stats_fn()  # pylint: disable=not-callable
  except RuntimeError:
    return None
  if not stats or "bytes_in_use" not in stats or "peak_bytes_in_use" not in stats:
    return None
  bytes_in_use = int(stats["bytes_in_use"])
  bytes_reserved = int(stats.get("bytes_reserved", 0))
  allocated_bytes = bytes_in_use + bytes_reserved
  peak_bytes_in_use = int(stats["peak_bytes_in_use"])
  peak_allocated_bytes = max(allocated_bytes, peak_bytes_in_use)
  return {
      "bytes_in_use": float(bytes_in_use),
      "bytes_reserved": float(bytes_reserved),
      "allocated_bytes": float(allocated_bytes),
      "peak_bytes_in_use": float(peak_bytes_in_use),
      "peak_allocated_bytes": float(peak_allocated_bytes),
      "hbm_allocated_gib": allocated_bytes / _GIB,
      "hbm_peak_in_use_gib": peak_bytes_in_use / _GIB,
  }
