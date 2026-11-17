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

"""Unit and compiled-HLO tests for m3 FSDP weight prefetching (`m3/infra/fsdp_prefetch.py`)."""

import gc
import re
from types import SimpleNamespace

from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import fsdp_prefetch, remat, scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.train import loop as m3_loop
from maxtext.utils import model_creation_utils


def _tiny_qwen3_config(
    *,
    num_layers=6,
    scan_layers=True,
    param_scan_axis=1,
    remat_policy="full",
    prefetch_fsdp_weights=False,
    prefetch_fsdp_defer_grad_reduce_scatter=False,
    shard_mode="explicit",
):
  """Minimal Qwen3 config exercised by CPU unit tests."""
  return SimpleNamespace(
      use_m3_model=True,
      model_name="qwen3-0.6b",
      scan_layers=scan_layers,
      param_scan_axis=param_scan_axis,
      remat_policy=remat_policy,
      prefetch_fsdp_weights=prefetch_fsdp_weights,
      prefetch_fsdp_defer_grad_reduce_scatter=prefetch_fsdp_defer_grad_reduce_scatter,
      shard_mode=shard_mode,
      attention="dot_product",
      quantization="",
      parameter_memory_host_offload=False,
      decoder_layer_input="device",
      num_decoder_layers=num_layers,
      emb_dim=16,
      mlp_dim=32,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=4,
      vocab_size=32,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      rope_max_timescale=1000000,
      float32_logits=True,
      logits_dot_in_fp32=True,
      normalize_embedding_logits=False,
      final_logits_soft_cap=0,
      cast_logits_to_fp32=True,
      max_target_length=4,
      global_batch_size_to_train_on=4,
      micro_batch_size_to_train_on=4,
      per_device_batch_size=1,
      init_weights_seed=0,
      learning_rate=1e-3,
      gradient_clipping_threshold=0.0,
      opt_type="adamw",
      adam_b1=0.9,
      adam_b2=0.999,
      adam_eps=1e-8,
      adam_eps_root=0.0,
      adam_weight_decay=0.0,
      adamw_mask=[],
      trainable_parameters_mask=[],
      skip_step_on_spikes=False,
      mu_dtype=None,
      use_iota_embed=False,
      logical_axis_rules=(),
  )


def _flat_params(mod):
  return {
      tuple(k.key for k in path): np.asarray(v[...])
      for path, v in jax.tree_util.tree_flatten_with_path(
          nnx.state(mod, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)
      )[0]
  }


def _make_batch(mesh, batch_axes="fsdp", seq_len=4):
  """Creates deterministic sharded token and position batches."""
  spec = P(batch_axes, None) if batch_axes is not None else P(None, None)
  sharding = NamedSharding(mesh, spec)
  tokens = jax.device_put(
      jnp.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8], [8, 7, 6, 5]], dtype=jnp.int32)[:, :seq_len],
      sharding,
  )
  positions = jax.device_put(
      jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (4, seq_len)),
      sharding,
  )
  return tokens, positions


def _build_qwen3(cfg, mesh):
  model = create_qwen3_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))
  model = scan.maybe_apply_scan(model, cfg)
  model = remat.maybe_apply_remat(model, cfg)
  model = fsdp_prefetch.maybe_apply_fsdp_prefetch(model, cfg, mesh=mesh)
  return model


def _loss_and_grad_fn(explicit: bool, batch_axes="fsdp"):
  """Returns a `(loss, grads)` callable compatible with explicit and auto meshes."""
  out_spec = P(batch_axes, None, None)

  def _run(model, tokens, positions):
    def loss_fn(m):
      logits = m(tokens, positions)
      if explicit:
        log_probs = jax.sharding.auto_axes(jax.nn.log_softmax, out_sharding=out_spec)(logits)
      else:
        log_probs = jax.nn.log_softmax(logits)
      return -jnp.mean(log_probs[..., 0])

    return nnx.value_and_grad(loss_fn)(model)

  return _run


def _parse_hlo_computations(hlo_text: str) -> dict[str, list[str]]:
  """Splits compiled XLA HLO text into `{computation_name: [instruction_lines]}`."""
  comps: dict[str, list[str]] = {}
  cur_name = None
  for line in hlo_text.splitlines():
    m = re.match(r"^(?:ENTRY\s+)?(%[\w.\-]+)\s*\(", line)
    if m and line.rstrip().endswith("{"):
      cur_name = m.group(1)
      comps[cur_name] = []
    elif line.strip() == "}" and cur_name is not None:
      cur_name = None
    elif cur_name is not None and "=" in line:
      comps[cur_name].append(line.strip())
  return comps


_HLO_DEBUG_SECTIONS = frozenset({"FileNames", "FunctionNames", "FileLocations", "StackFrames"})
_HLO_COMP_DEF_RE = re.compile(r"^(ENTRY\s+)?(%[\w.\-]+)\s*\((.*)\)\s*->\s*(.*\{\s*)$")
_HLO_PARAM_DECL_RE = re.compile(r"(?:^|(?<=,\s)|(?<=\*/))([A-Za-z_][\w.\-]*)(\s*:)")
_HLO_STR_OR_IDENT_RE = re.compile(r'"[^"]*"|%([A-Za-z_][\w.\-]*)')


def _hlo_comp_stem(name: str) -> str:
  stem = re.sub(r"\.\d+(?=\.|_spmd\b|$)", "", name)
  return re.sub(r"\bregion_\d+", "region", stem)


def _hlo_ssa_stem(name: str) -> str:
  return re.sub(r"\.\d+(?=\.|$)", "", name)


def _normalize_hlo(hlo_text: str) -> str:
  """Canonicalizes debug tables and monotonic XLA SSA/subcomputation counters while preserving graph wiring."""
  lines = hlo_text.splitlines()
  comp_map: dict[str, str] = {}
  comp_counts: dict[str, int] = {}
  for line in lines:
    m = _HLO_COMP_DEF_RE.match(line.strip())
    if m:
      raw_comp = m.group(2)[1:]
      if raw_comp not in comp_map:
        stem = _hlo_comp_stem(raw_comp)
        idx = comp_counts.get(stem, 0)
        comp_counts[stem] = idx + 1
        comp_map[raw_comp] = f"{stem}#{idx}"

  out_lines: list[str] = []
  in_debug_section = False
  in_comp = False
  local_map: dict[str, str] = {}
  local_counts: dict[str, int] = {}

  def _map_local(raw_name: str) -> str:
    if raw_name in comp_map:
      return comp_map[raw_name]
    if raw_name not in local_map:
      stem = _hlo_ssa_stem(raw_name)
      idx = local_counts.get(stem, 0)
      local_counts[stem] = idx + 1
      local_map[raw_name] = f"{stem}#{idx}"
    return local_map[raw_name]

  def _replace_param(pm: re.Match[str]) -> str:
    return f"{_map_local(pm.group(1))}{pm.group(2)}"

  def _replace_ident(im: re.Match[str]) -> str:
    return f"%{_map_local(im.group(1))}" if im.group(1) is not None else im.group(0)

  for line in lines:
    stripped = line.strip()
    if not in_comp:
      if stripped in _HLO_DEBUG_SECTIONS:
        in_debug_section = True
        continue
      if in_debug_section:
        if not stripped or re.match(r"^\d+\s+", stripped):
          continue
        in_debug_section = False
      if stripped.startswith("HloModule "):
        line = re.sub(r"^(\s*HloModule\s+)\S+?,", r"\1<module>,", line, count=1)
        out_lines.append(line)
        continue

    m = _HLO_COMP_DEF_RE.match(stripped)
    if m:
      in_comp = True
      local_map.clear()
      local_counts.clear()
      entry_prefix = m.group(1) or ""
      comp_canon = comp_map[m.group(2)[1:]]
      params_str = _HLO_PARAM_DECL_RE.sub(_replace_param, m.group(3))
      out_lines.append(f"{entry_prefix}%{comp_canon} ({params_str}) -> {m.group(4).rstrip()}")
      continue

    if in_comp and (stripped == "}" or stripped.startswith("},")):
      in_comp = False
      out_lines.append(line)
      continue

    if in_comp:
      line = re.sub(r"\s*\bstack_frame_id=\d+", "", line)
      line = re.sub(r",?\s*metadata=\{\s*\}", "", line)
      line = _HLO_STR_OR_IDENT_RE.sub(_replace_ident, line)
    if stripped or (out_lines and out_lines[-1] != ""):
      out_lines.append(line)

  return "\n".join(out_lines)


class _TupleCarryStatefulLayer(nnx.Module):
  """Toy layer with tuple carry `(hidden, aux_sum)` and mutable non-Param state."""

  def __init__(self, dim: int, mesh, *, rngs: nnx.Rngs):
    spec = P("fsdp", None)
    w = jax.random.normal(rngs.params(), (dim, dim), dtype=jnp.float32) * 0.2
    w = jax.device_put(w, NamedSharding(mesh, spec))
    self.weight = nnx.Param(w, out_sharding=spec, eager_sharding=True)
    self.calls = nnx.Variable(jnp.array(0, dtype=jnp.int32), out_sharding=P())

  def __call__(self, carry, *, scale: float = 1.0):
    x, aux = carry
    self.calls[...] = self.calls[...] + 1
    y = jnp.tanh(jnp.einsum("bd,df->bf", x, self.weight[...])) * scale
    return (x + y, aux + jnp.mean(y**2))


class FsdpPrefetchHelpersTest(absltest.TestCase):
  """Tests for pure-Python FSDP axis and SparseCore helpers."""

  def test_fsdp_axes_and_spec_removal(self):
    self.assertEqual(fsdp_prefetch.fsdp_axes_for_dim(("fsdp", "expert")), ("fsdp", "expert"))
    self.assertEqual(fsdp_prefetch.fsdp_axes_for_dim(("expert", "fsdp")), ("fsdp",))
    self.assertEqual(fsdp_prefetch.fsdp_axes_for_dim("expert"), ())
    self.assertEqual(
        fsdp_prefetch.remove_fsdp_from_spec(P(("fsdp", "fsdp_transpose", "context", "expert"), "tensor")),
        P(None, "tensor"),
    )
    self.assertEqual(
        fsdp_prefetch.remove_fsdp_from_spec(P("expert", ("fsdp", "context"), "tensor")),
        P("expert", None, "tensor"),
    )

  def test_has_multi_trip_loop_threshold(self):
    for n in range(1, 6):
      self.assertFalse(fsdp_prefetch._has_multi_trip_loop(n))  # pylint: disable=protected-access
    for n in (6, 8, 28):
      self.assertTrue(fsdp_prefetch._has_multi_trip_loop(n))  # pylint: disable=protected-access

  def test_merge_offload_residual_layout_fails_closed_on_count_mismatch(self):
    layout_cls = fsdp_prefetch._ResidualLayout  # pylint: disable=protected-access
    merge_fn = fsdp_prefetch._merge_offload_residual_layout  # pylint: disable=protected-access
    raw = layout_cls(treedef=None, src=(None, None), saved_avals=("a", "b"), saved_offload_mask=(True, False))
    dev_ok = layout_cls(treedef=None, src=(None, None), saved_avals=("a_dev", "b_dev"), saved_offload_mask=(False, False))
    merged = merge_fn(raw, dev_ok)
    self.assertEqual(merged.saved_avals, ("a_dev", "b_dev"))
    self.assertEqual(merged.saved_offload_mask, (True, False))
    dev_bad = layout_cls(treedef=None, src=(None,), saved_avals=("a_dev",), saved_offload_mask=(False,))
    with self.assertRaisesRegex(ValueError, "saved residual leaf count"):
      merge_fn(raw, dev_bad)

  def test_default_sparse_core_id_targets_tpu7x_only(self):
    fake_tpu7x = SimpleNamespace(devices=np.array([SimpleNamespace(device_kind="TPU7x")]))
    fake_v5p = SimpleNamespace(devices=np.array([SimpleNamespace(device_kind="TPU v5")]))
    fake_cpu = SimpleNamespace(devices=np.array([SimpleNamespace(device_kind="cpu")]))
    self.assertEqual(fsdp_prefetch._default_sparse_core_id(fake_tpu7x), 1)  # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch._default_sparse_core_id(fake_v5p))  # pylint: disable=protected-access
    self.assertIsNone(fsdp_prefetch._default_sparse_core_id(fake_cpu))  # pylint: disable=protected-access

  def test_normalize_hlo_canonicalizes_counters_and_detects_real_diffs(self):
    hlo_a = """HloModule jit__step, is_scheduled=true, num_partitions=4

FileNames
1 "/a/model.py"

FunctionNames
1 "fn_a"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10 end_line=10 column=1 end_column=1}

StackFrames
1 {file_location_id=1 parent_frame_id=0}

%region_1.8_spmd.clone (Arg_0.6: f32[], Arg_1.7: f32[]) -> f32[] {
  %Arg_0.6 = f32[] parameter(0)
  %Arg_1.7 = f32[] parameter(1)
  ROOT %add.8 = f32[] add(%Arg_0.6, %Arg_1.7), metadata={op_name="jit(_step)/add" stack_frame_id=1}
}

%bitcast_copy_fusion_computation (param_0: f32[4]) -> f32[4] {
  %param_0 = f32[4]{0} parameter(0)
  ROOT %bitcast = f32[4]{0} bitcast(%param_0), metadata={stack_frame_id=1}
}

ENTRY %main.37_spmd (Arg_0.1: f32[4], /*index=5*/Arg_5.2: f32[4]) -> f32[4] {
  %Arg_0.1 = f32[4]{0} parameter(0)
  %Arg_5.2 = f32[4]{0} parameter(1)
  %ws = f32[4]{0} fusion(%Arg_0.1), kind=kLoop, calls=%bitcast_copy_fusion_computation
  ROOT %ar = f32[4]{0} all-reduce(%ws), to_apply=%region_1.8_spmd.clone, metadata={op_name="ar" stack_frame_id=1}
}
"""
    hlo_b = """HloModule jit__step.42, is_scheduled=true, num_partitions=4

FileNames
1 "/x/other.py"
2 "/y/extra.py"

FunctionNames
1 "other_fn"

FileLocations
1 {file_name_id=2 function_name_id=1 line=99 end_line=99 column=4 end_column=4}

StackFrames
1 {file_location_id=1 parent_frame_id=0}
2 {file_location_id=1 parent_frame_id=1}

%region_18.408_spmd.clone (Arg_0.906: f32[], Arg_1.907: f32[]) -> f32[] {
  %Arg_0.906 = f32[] parameter(0)
  %Arg_1.907 = f32[] parameter(1)
  ROOT %add.908 = f32[] add(%Arg_0.906, %Arg_1.907), metadata={op_name="jit(_step)/add" stack_frame_id=2}
}

%bitcast_copy_fusion_computation.77 (param_0.312: f32[4]) -> f32[4] {
  %param_0.312 = f32[4]{0} parameter(0)
  ROOT %bitcast.88 = f32[4]{0} bitcast(%param_0.312)
}

ENTRY %main.537_spmd (Arg_0.501: f32[4], /*index=5*/Arg_5.502: f32[4]) -> f32[4] {
  %Arg_0.501 = f32[4]{0} parameter(0)
  %Arg_5.502 = f32[4]{0} parameter(1)
  %ws.19 = f32[4]{0} fusion(%Arg_0.501), kind=kLoop, calls=%bitcast_copy_fusion_computation.77
  ROOT %ar.70 = f32[4]{0} all-reduce(%ws.19), to_apply=%region_18.408_spmd.clone, metadata={op_name="ar" stack_frame_id=2}
}
"""
    self.assertEqual(_normalize_hlo(hlo_a), _normalize_hlo(hlo_b))
    # Positive controls: rewiring operands or inserting an extra barrier must fail equality.
    hlo_rewired = hlo_b.replace("add(%Arg_0.906, %Arg_1.907)", "add(%Arg_1.907, %Arg_0.906)")
    self.assertNotEqual(_normalize_hlo(hlo_a), _normalize_hlo(hlo_rewired))
    hlo_barrier = hlo_b.replace(
        "ROOT %ar.70 = f32[4]{0} all-reduce(%ws.19)",
        "%opt-barrier.1 = f32[4]{0} opt-barrier(%ws.19)\n" "  ROOT %ar.70 = f32[4]{0} all-reduce(%opt-barrier.1)",
    )
    self.assertNotEqual(_normalize_hlo(hlo_a), _normalize_hlo(hlo_barrier))


class FsdpPrefetchCorrectnessTest(parameterized.TestCase):
  """Numerical parity, path transparency, and edge-case tests on 4-device CPU."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4).")

  @parameterized.named_parameters(
      ("explicit_full_pf", "explicit", "full", False, 1, 6),
      ("explicit_full_pf_drs", "explicit", "full", True, 1, 6),
      ("auto_full_pf", "auto", "full", False, 1, 6),
      ("explicit_minimal_pf", "explicit", "minimal", False, 1, 6),
      ("explicit_minimal_pf_drs", "explicit", "minimal", True, 1, 6),
      ("explicit_axis0_pf_drs", "explicit", "full", True, 0, 6),
      ("explicit_2layers_pf_drs", "explicit", "full", True, 1, 2),
      ("explicit_3layers_pf_drs", "explicit", "full", True, 1, 3),
      ("explicit_1layer_full", "explicit", "full", True, 1, 1),
      ("explicit_1layer_minoff", "explicit", "minimal_offloaded", True, 1, 1),
  )
  def test_qwen3_prefetch_matches_baseline_loss_grads_and_paths(
      self, mode, remat_policy, defer_rs, scan_axis, num_layers
  ):
    explicit = mode == "explicit"
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=explicit)
    self.enter_context(jax.set_mesh(mesh))
    tokens, positions = _make_batch(mesh)

    cfg_base = _tiny_qwen3_config(
        num_layers=num_layers,
        param_scan_axis=scan_axis,
        remat_policy=remat_policy,
        prefetch_fsdp_weights=False,
        shard_mode=mode,
    )
    cfg_pf = _tiny_qwen3_config(
        num_layers=num_layers,
        param_scan_axis=scan_axis,
        remat_policy=remat_policy,
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=defer_rs,
        shard_mode=mode,
    )
    m_base = _build_qwen3(cfg_base, mesh)
    m_pf = _build_qwen3(cfg_pf, mesh)

    # Path-transparency check: identical parameter keys, shapes, and dtypes.
    p_base, p_pf = _flat_params(m_base), _flat_params(m_pf)
    self.assertEqual(set(p_base.keys()), set(p_pf.keys()))
    for k in p_base:
      self.assertEqual(p_base[k].shape, p_pf[k].shape, msg=str(k))
      self.assertEqual(p_base[k].dtype, p_pf[k].dtype, msg=str(k))

    step = nnx.jit(_loss_and_grad_fn(explicit))
    loss_base, grad_base = step(m_base, tokens, positions)
    loss_pf, grad_pf = step(m_pf, tokens, positions)

    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    gb, gp = _flat_params(grad_base), _flat_params(grad_pf)
    self.assertEqual(set(gb.keys()), set(gp.keys()))
    for k in gb:
      np.testing.assert_allclose(gp[k], gb[k], rtol=1e-5, atol=1e-5, err_msg=str(k))

    # Negative control: perturbing one scanned weight must break loss parity.
    perturbed = _build_qwen3(cfg_pf, mesh)
    perturbed.decoder.layers.mlp.wi_0.kernel[...] = perturbed.decoder.layers.mlp.wi_0.kernel[...] + 0.5
    loss_perturbed, _ = step(perturbed, tokens, positions)
    self.assertGreater(abs(float(loss_perturbed) - float(loss_base)), 1e-4)

  @parameterized.named_parameters(
      ("fsdp2_tp2", {"fsdp": 2, "tensor": 2}, "fsdp"),
      ("fsdp2_fsdp_t2", {"fsdp": 2, "fsdp_transpose": 2}, ("fsdp", "fsdp_transpose")),
      ("fsdp2_dp2", {"data": 2, "fsdp": 2}, ("data", "fsdp")),
  )
  def test_multiaxis_sharding_matches_baseline(self, ici, batch_axes):
    mesh = create_mesh(devices=jax.devices()[:4], ici=ici, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    tokens, positions = _make_batch(mesh, batch_axes=batch_axes)

    cfg_base = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(
        num_layers=4,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    m_base = _build_qwen3(cfg_base, mesh)
    m_pf = _build_qwen3(cfg_pf, mesh)
    step = nnx.jit(_loss_and_grad_fn(True, batch_axes=batch_axes))
    loss_base, grad_base = step(m_base, tokens, positions)
    loss_pf, grad_pf = step(m_pf, tokens, positions)
    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    gb, gp = _flat_params(grad_base), _flat_params(grad_pf)
    for k in gb:
      np.testing.assert_allclose(gp[k], gb[k], rtol=1e-5, atol=1e-5, err_msg=str(k))

  def test_tuple_carry_and_stateful_rest_match_regular_scan(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    def make_stack(enable_prefetch: bool):
      seq = nnx.Sequential(*[_TupleCarryStatefulLayer(8, mesh, rngs=nnx.Rngs(i)) for i in range(6)])
      scanned = scan.scan_layers(seq, param_scan_axis=1)
      scanned.remat = remat.Remat(policy=jax.checkpoint_policies.nothing_saveable)
      if enable_prefetch:
        fsdp_prefetch.apply_fsdp_prefetch(scanned, defer_grad_reduce_scatter=True, mesh=mesh)
      return scanned

    s_base = make_stack(False)
    s_pf = make_stack(True)
    x = jax.device_put(jnp.ones((4, 8), dtype=jnp.float32) * 0.3, NamedSharding(mesh, P("fsdp", None)))
    aux0 = jax.device_put(jnp.array(0.0, dtype=jnp.float32), NamedSharding(mesh, P()))

    def loss_fn(mod):
      y, aux = mod((x, aux0), scale=0.5)
      return jnp.mean(y) + aux

    loss_base, grad_base = nnx.value_and_grad(loss_fn)(s_base)
    loss_pf, grad_pf = nnx.value_and_grad(loss_fn)(s_pf)
    np.testing.assert_allclose(loss_pf, loss_base, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_pf.weight[...], grad_base.weight[...], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(s_pf.calls[...], s_base.calls[...])

  def test_remat_none_fails_closed(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    cfg = _tiny_qwen3_config(num_layers=4, remat_policy="none", prefetch_fsdp_weights=True)
    model = _build_qwen3(cfg, mesh)
    tokens, positions = _make_batch(mesh)
    with self.assertRaisesRegex(ValueError, "remat_policy='none'"):
      model(tokens, positions)

  def test_non_fsdp_mesh_and_disabled_modes_fall_back_cleanly(self):
    mesh_dp = create_mesh(devices=jax.devices()[:4], ici={"data": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh_dp))
    cfg_base = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=True)
    m_base = _build_qwen3(cfg_base, mesh_dp)
    m_pf = _build_qwen3(cfg_pf, mesh_dp)
    tokens, positions = _make_batch(mesh_dp, batch_axes="data")
    step = nnx.jit(_loss_and_grad_fn(True, batch_axes="data"))
    l_base, _ = step(m_base, tokens, positions)
    l_pf, _ = step(m_pf, tokens, positions)
    np.testing.assert_allclose(l_pf, l_base, rtol=1e-6, atol=1e-6)

    # Decode mode and host parameter offload leave model untouched.
    m_dec = fsdp_prefetch.maybe_apply_fsdp_prefetch(m_base, cfg_pf, mesh=mesh_dp, model_mode="decode")
    self.assertIsNone(m_dec.decoder.layers.fsdp_prefetch)
    cfg_offload = _tiny_qwen3_config(num_layers=4, remat_policy="full", prefetch_fsdp_weights=True)
    cfg_offload.parameter_memory_host_offload = True
    m_off = fsdp_prefetch.maybe_apply_fsdp_prefetch(m_base, cfg_offload, mesh=mesh_dp)
    self.assertIsNone(m_off.decoder.layers.fsdp_prefetch)

  def test_transform_order_invariance_and_dual_driver_wiring(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    cfg = _tiny_qwen3_config(
        num_layers=4,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    # Apply prefetch before scan and verify ScannedLayers inherits it.
    m_before = create_qwen3_model(cfg, mesh=mesh, rngs=nnx.Rngs(0))
    m_before = fsdp_prefetch.maybe_apply_fsdp_prefetch(m_before, cfg, mesh=mesh)
    m_before = scan.maybe_apply_scan(m_before, cfg)
    m_before = remat.maybe_apply_remat(m_before, cfg)
    self.assertIsInstance(m_before.decoder.layers.fsdp_prefetch, fsdp_prefetch.FsdpPrefetchRunner)

    # Both production driver (`model_creation_utils.from_config`) and `m3/train/loop.py:setup_model` wire it.
    m_mcu = model_creation_utils.from_config(cfg, mesh=mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(m_mcu.decoder.layers.fsdp_prefetch, fsdp_prefetch.FsdpPrefetchRunner)
    self.assertTrue(m_mcu.decoder.layers.fsdp_prefetch.defer_grad_reduce_scatter)

    m_loop, _ = m3_loop.setup_model(cfg, mesh)
    self.assertIsInstance(m_loop.decoder.layers.fsdp_prefetch, fsdp_prefetch.FsdpPrefetchRunner)
    self.assertTrue(m_loop.decoder.layers.fsdp_prefetch.defer_grad_reduce_scatter)

  def test_minimal_offloaded_remat_six_layers_parity(self):
    """6-layer `minimal_offloaded` exercises multi-trip `unroll=2` host-stack residual offload."""
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))
    cfg_base = _tiny_qwen3_config(num_layers=6, remat_policy="minimal_offloaded", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(num_layers=6, remat_policy="minimal_offloaded", prefetch_fsdp_weights=True)
    cfg_drs = _tiny_qwen3_config(
        num_layers=6,
        remat_policy="minimal_offloaded",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )
    m_base = _build_qwen3(cfg_base, mesh)
    m_pf = _build_qwen3(cfg_pf, mesh)
    m_drs = _build_qwen3(cfg_drs, mesh)
    tokens, positions = _make_batch(mesh)
    step = nnx.jit(_loss_and_grad_fn(True))
    loss_base, grads_base = step(m_base, tokens, positions)
    for m in (m_pf, m_drs):
      loss_m, grads_m = step(m, tokens, positions)
      np.testing.assert_allclose(loss_m, loss_base, rtol=1e-6, atol=1e-6)
      for k, ref_g in _flat_params(grads_base).items():
        np.testing.assert_allclose(_flat_params(grads_m)[k], ref_g, rtol=1e-5, atol=1e-5, err_msg=f"grad mismatch {k}")


class FsdpPrefetchHloStructureTest(absltest.TestCase):
  """Compiled-HLO structural and byte-identity tests (with positive & negative controls)."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4).")

  def _compile_model_hlo(self, model, cfg, mesh, logits_transform=None):
    """Compiles one train step and returns post-optimization HLO text."""
    tokens, positions = _make_batch(mesh)
    graphdef, state = nnx.split(model)
    explicit = cfg.shard_mode == "explicit"
    out_spec = P("fsdp", None, None)

    def _step(st, tok, pos):
      def loss_fn(s):
        logits = nnx.merge(graphdef, s)(tok, pos)
        if logits_transform is not None:
          logits = logits_transform(logits)
        if explicit:
          lp = jax.sharding.auto_axes(jax.nn.log_softmax, out_sharding=out_spec)(logits)
        else:
          lp = jax.nn.log_softmax(logits)
        return -jnp.mean(lp[..., 0])

      return jax.value_and_grad(loss_fn)(st)

    compiled = jax.jit(_step).lower(state, tokens, positions).compile()
    return compiled.as_text()

  def _compile_train_step_hlo(self, cfg, mesh):
    return self._compile_model_hlo(_build_qwen3(cfg, mesh), cfg, mesh)

  def test_disabled_hlo_is_byte_identical_to_untransformed_baseline(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    cfg_off = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=False)
    # Build without calling `maybe_apply_fsdp_prefetch` at all.
    m_plain = create_qwen3_model(cfg_off, mesh=mesh, rngs=nnx.Rngs(0))
    m_plain = scan.maybe_apply_scan(m_plain, cfg_off)
    m_plain = remat.maybe_apply_remat(m_plain, cfg_off)
    m_disabled = _build_qwen3(cfg_off, mesh)
    hlo_untransformed, hlo_disabled = [self._compile_model_hlo(m, cfg_off, mesh) for m in (m_plain, m_disabled)]
    norm_untransformed = _normalize_hlo(hlo_untransformed)
    norm_disabled = _normalize_hlo(hlo_disabled)
    self.assertEqual(norm_disabled, norm_untransformed)

    # Also verify stability across inner trace-cache eviction (reproduces xdist memory-pressure frame shifts).
    jax.clear_caches()
    gc.collect()
    hlo_disabled_evicted = self._compile_model_hlo(_build_qwen3(cfg_off, mesh), cfg_off, mesh)
    self.assertEqual(_normalize_hlo(hlo_disabled_evicted), norm_untransformed)

    # Positive controls: enabling prefetch, inserting a single barrier, or inserting an extra collective
    # must each fail `_normalize_hlo` equality.
    cfg_on = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=True)
    hlo_enabled = self._compile_model_hlo(_build_qwen3(cfg_on, mesh), cfg_on, mesh)
    self.assertNotEqual(_normalize_hlo(hlo_enabled), norm_disabled)

    hlo_with_barrier = self._compile_model_hlo(m_plain, cfg_off, mesh, logits_transform=jax.lax.optimization_barrier)
    self.assertNotEqual(_normalize_hlo(hlo_with_barrier), norm_untransformed)

    hlo_with_extra_ag = self._compile_model_hlo(
        m_plain,
        cfg_off,
        mesh,
        logits_transform=lambda x: jax.lax.with_sharding_constraint(x, P(None, None, None)),
    )
    self.assertNotEqual(_normalize_hlo(hlo_with_extra_ag), norm_untransformed)

  def test_compiled_hlo_prefetches_next_layer_weight_before_current_layer_compute_ends(self):
    mesh = create_mesh(devices=jax.devices()[:4], ici={"fsdp": 4}, explicit=True)
    self.enter_context(jax.set_mesh(mesh))

    cfg_off = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=False)
    cfg_pf = _tiny_qwen3_config(num_layers=6, remat_policy="full", prefetch_fsdp_weights=True)
    cfg_drs = _tiny_qwen3_config(
        num_layers=6,
        remat_policy="full",
        prefetch_fsdp_weights=True,
        prefetch_fsdp_defer_grad_reduce_scatter=True,
    )

    hlo_off = self._compile_train_step_hlo(cfg_off, mesh)
    hlo_pf = self._compile_train_step_hlo(cfg_pf, mesh)
    hlo_drs = self._compile_train_step_hlo(cfg_drs, mesh)

    prefetch_scope_re = re.compile(r'op_name="[^"]*fsdp_prefetch')
    # Negative control: disabled HLO has no `fsdp_prefetch` op scope.
    self.assertIsNone(prefetch_scope_re.search(hlo_off))
    # Positive control: enabled HLO carries `fsdp_prefetch` op scope and explicit collectives.
    self.assertIsNotNone(prefetch_scope_re.search(hlo_pf))
    self.assertIn("all-gather(", hlo_pf)
    self.assertIn("reduce-scatter(", hlo_pf)

    # Find the forward `while` loop body inside `hlo_pf`: with `num_layers=6` and `unroll=2`,
    # layers 1..4 execute in a 2-trip unrolled loop (2 layers/iter), carrying prefetched weights
    # and issuing layer k+1's all-gather BEFORE layer k's compute finishes.
    comps_pf = _parse_hlo_computations(hlo_pf)
    fwd_bodies = [
        (name, lines)
        for name, lines in comps_pf.items()
        if any(prefetch_scope_re.search(ln) for ln in lines)
        and any("all-gather(" in ln for ln in lines)
        and not any("reduce-scatter(" in ln for ln in lines)
    ]
    self.assertNotEmpty(fwd_bodies, "Expected a compiled forward prefetch while-loop body with all-gather ops.")
    dot_comps = {name for name, lines in comps_pf.items() if any(" dot(" in ln or "= dot(" in ln for ln in lines)}

    def _is_dot_op(line: str) -> bool:
      if " dot(" in line or "= dot(" in line:
        return True
      m = re.search(r"\bcalls=(%[\w.\-]+)", line)
      return bool(m and m.group(1) in dot_comps)

    _, fwd_lines = fwd_bodies[0]
    ag_indices = [i for i, ln in enumerate(fwd_lines) if "all-gather(" in ln]
    dot_indices = [i for i, ln in enumerate(fwd_lines) if _is_dot_op(ln)]
    self.assertNotEmpty(ag_indices, "Expected weight all-gather instructions in forward prefetch while body.")
    self.assertNotEmpty(dot_indices, "Expected matmul dot instructions in forward prefetch while body.")
    # Prefetched layer k+1's weight all-gather is produced before layer k's compute finishes.
    self.assertLess(min(ag_indices), max(dot_indices))
    # Unroll=2 body has 2 layers of weight all-gathers (7 FSDP weights/layer * 2 = 14) vs 7 in non-prefetch.
    self.assertGreaterEqual(len(ag_indices), 12)

    # Negative control: non-prefetch forward while body has only 1 layer of weight all-gathers (7).
    comps_off = _parse_hlo_computations(hlo_off)
    off_fwd_bodies = [
        lines
        for lines in comps_off.values()
        if any("all-gather(" in ln for ln in lines) and not any("reduce-scatter(" in ln for ln in lines)
    ]
    self.assertNotEmpty(off_fwd_bodies)
    self.assertLess(sum(1 for ln in off_fwd_bodies[0] if "all-gather(" in ln), len(ag_indices))

    # Deferred RS keeps backward reduce-scatters in the backward while body and changes the carry layout.
    self.assertIn("reduce-scatter(", hlo_drs)
    self.assertNotEqual(hlo_drs, hlo_pf)


if __name__ == "__main__":
  absltest.main()
