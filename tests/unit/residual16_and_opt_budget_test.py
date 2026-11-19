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

"""Unit, Orbax round-trip, and budget allocation tests for residual16 master storage and opt budget."""

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax
from optax.contrib._muon import muon as optax_muon
from orbax.checkpoint import v1 as ocp
from pydantic import ValidationError

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.utils import lora_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import qk_clip_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path


class _TinyResidualModel(nnx.Module):
  """Compact 2-layer NNX model exercising 2D kernels, 1D biases, and optional router bias."""

  def __init__(self, rngs: nnx.Rngs, include_gate: bool = False):
    self.embed = nnx.Param(
        jax.random.normal(rngs.params(), (16, 8), dtype=jnp.float32),
        out_sharding=("fsdp", None),
    )
    self.linear1 = nnx.Linear(8, 16, dtype=jnp.bfloat16, rngs=rngs)
    self.linear2 = nnx.Linear(16, 8, dtype=jnp.bfloat16, use_bias=False, rngs=rngs)
    if include_gate:
      self.gate = nnx.Dict({"bias": nnx.Param(jnp.zeros((4,), dtype=jnp.float32))})

  def __call__(self, tokens: jax.Array) -> jax.Array:
    x = self.embed.get_value().astype(jnp.bfloat16)[tokens]
    h = jax.nn.silu(self.linear1(x))
    return self.linear2(h)


class _TinyMLAModel(nnx.Module):
  """Minimal MLA projection stub for verifying QK-Clip updates both bf16 weights and residuals."""

  def __init__(self, rngs: nnx.Rngs):
    self.self_attention = nnx.Dict(
        {
            "wq_b": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (4, 2, 8), dtype=jnp.float32))}),
            "wkv_b": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (4, 2, 8), dtype=jnp.float32))}),
        }
    )


def _ckpt_config(load_parameters_path: str = "", load_full_state_path: str = "", fp32_master_storage: str = "f32"):
  return SimpleNamespace(
      enable_diloco=False,
      enable_checkpointing=True,
      enable_continuous_checkpointing=False,
      enable_emergency_checkpoint=False,
      enable_multi_tier_checkpointing=False,
      enable_autocheckpoint=False,
      save_checkpoint_on_start=True,
      checkpoint_period=1,
      local_checkpoint_period=0,
      async_checkpointing=False,
      dataset_type="tfds",
      lora=None,
      checkpoint_storage_target_data_file_size_bytes=checkpointing.DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE,
      elastic_enabled=False,
      logical_axis_rules=[],
      load_parameters_path=load_parameters_path,
      load_full_state_path=load_full_state_path,
      checkpoint_storage_concurrent_gb=8,
      enable_single_replica_ckpt_restoring=False,
      checkpoint_storage_use_ocdbt=True,
      checkpoint_storage_use_zarr3=True,
      enable_orbax_v1=False,
      checkpoint_conversion_fn=None,
      source_checkpoint_layout="orbax",
      expansion_factor_real_data=-1,
      scan_layers=False,
      optimizer_memory_host_offload=False,
      optimizer_device_resident_gib=0.0,
      parameter_memory_host_offload=False,
      fp32_master_storage=fp32_master_storage,
      weight_dtype=jnp.float32,
      routed_bias=False,
      float32_gate_logits=False,
  )


class TestBitLevelResidual16Encoding(unittest.TestCase):
  """Exhaustive IEEE-754 edge-case and random bit-exact split/reconstruct verification."""

  def test_exact_roundtrip_across_ieee754_edge_cases_and_random_patterns(self):
    edge_u32 = np.array(
        [
            0x00000000,  # +0.0
            0x80000000,  # -0.0
            0x00000001,  # smallest positive subnormal
            0x80000001,  # smallest negative subnormal
            0x00007FFF,  # just below subnormal midpoint
            0x00008000,  # exact tie in subnormal range
            0x00008001,  # just above subnormal midpoint
            0x007FFFFF,  # largest positive subnormal
            0x807FFFFF,  # largest negative subnormal
            0x00800000,  # smallest positive normal
            0x80800000,  # smallest negative normal
            0x3F800000,  # +1.0
            0xBF800000,  # -1.0
            0x3F807FFF,  # 1.0 + (half bf16 ulp - 1 f32 ulp)
            0x3F808000,  # 1.0 + exact half bf16 ulp (even bf16)
            0x3F808001,  # 1.0 + (half bf16 ulp + 1 f32 ulp)
            0x3F818000,  # exact half bf16 ulp (odd bf16)
            0xBF808000,  # negative exact tie
            0x33800000,  # 2^-24
            0x4B800000,  # 2^24
            0x7F7F7FFF,  # near max finite without bf16 overflow
            0x7F7F8000,  # rounds bf16 up to +Inf while residual stays negative!
            0x7F7FFFFF,  # +finfo(float32).max (bf16 rounds to +Inf, residual reconstructs max!)
            0xFF7FFFFF,  # -finfo(float32).max
            0x7F800000,  # +Inf
            0xFF800000,  # -Inf
        ],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(2026)
    rand_u32 = rng.integers(0, 0x100000000, size=200_000, dtype=np.uint32)
    rand_f32 = rand_u32.view(np.float32)
    rand_u32 = rand_u32[~np.isnan(rand_f32)]

    all_u32 = np.concatenate([edge_u32, rand_u32])
    x_f32 = jnp.asarray(all_u32.view(np.float32))
    w_bf16, res_i16 = train_state_nnx.split_f32_to_bf16_residual16(x_f32)
    self.assertEqual(w_bf16.dtype, jnp.bfloat16)
    self.assertEqual(res_i16.dtype, jnp.int16)

    rec_f32 = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
    self.assertEqual(rec_f32.dtype, jnp.float32)
    rec_u32 = np.asarray(jax.lax.bitcast_convert_type(rec_f32, jnp.uint32))
    np.testing.assert_array_equal(rec_u32, all_u32)

  def test_nan_patterns_never_wrap_to_zero_or_finite(self):
    nan_u32 = np.array([0x7FC00000, 0x7FC00001, 0x7FFFFFFF, 0xFFC00001, 0xFFFFFFFF], dtype=np.uint32)
    x_nan = jax.lax.bitcast_convert_type(jnp.asarray(nan_u32), jnp.float32)
    w_bf16, res_i16 = train_state_nnx.split_f32_to_bf16_residual16(x_nan)
    rec_f32 = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
    self.assertTrue(bool(jnp.all(jnp.isnan(w_bf16))))
    self.assertTrue(bool(jnp.all(jnp.isnan(rec_f32))))

  def test_exact_bf16_inputs_produce_zero_residual(self):
    bf16_vals = jnp.array([-3.5, -0.0, 0.0, 0.125, 1.0, 42.0], dtype=jnp.bfloat16)
    w_bf16, res_i16 = train_state_nnx.split_f32_to_bf16_residual16(bf16_vals.astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(res_i16), np.zeros_like(np.asarray(res_i16)))
    rec = train_state_nnx.reconstruct_f32_from_bf16_residual16(w_bf16, res_i16)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(rec, jnp.uint32)),
        np.asarray(jax.lax.bitcast_convert_type(bf16_vals.astype(jnp.float32), jnp.uint32)),
    )


class TestSmallLRConvergenceParity(unittest.TestCase):
  """Verifies residual16 accumulates sub-bf16-ULP updates where pure bf16 weights stall at zero."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def _run_small_lr_comparison(self, tx: optax.GradientTransformation):
    """Runs 40 small-LR steps comparing f32, residual16, and pure bf16 weights."""
    cfg_f32 = SimpleNamespace(fp32_master_storage="f32", weight_dtype=jnp.float32, routed_bias=False)
    cfg_r16 = SimpleNamespace(fp32_master_storage="residual16", weight_dtype=jnp.float32, routed_bias=False)

    m_f32 = _TinyResidualModel(nnx.Rngs(11))
    m_r16 = _TinyResidualModel(nnx.Rngs(11))
    m_bf16_only = _TinyResidualModel(nnx.Rngs(11))
    # Pure bf16 weights without residual16 master weights:
    _, bf16_p, _ = nnx.split(m_bf16_only, nnx.Param, ...)
    nnx.update(
        m_bf16_only,
        jax.tree.map(
            lambda v: v.replace(value=v.get_value().astype(jnp.bfloat16)),
            bf16_p,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        ),
    )

    ts_f32 = train_state_nnx.create_train_state(m_f32, tx, cfg_f32)
    ts_r16 = train_state_nnx.create_train_state(m_r16, tx, cfg_r16)
    ts_bf16 = train_state_nnx.TrainStateNNX(m_bf16_only, nnx.Optimizer(m_bf16_only, tx, wrt=nnx.Param))

    init_bf16_embed = np.array(ts_bf16.model.embed.get_value())
    tokens = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)

    @nnx.jit
    def step_fn(ts):
      def loss_fn(model):
        # Compute loss on bf16 working weights in f32 reference too so forward/backward grads
        # are identical and only master-weight storage precision differs!
        out = model(tokens)
        return jnp.mean(out.astype(jnp.float32) ** 2)

      grads = nnx.grad(loss_fn)(ts.model)
      ts.apply_gradients(grads)

    # Also align m_f32 initial weights to exact bf16-rounded values so step-0 forward is identical:
    _, f32_p, _ = nnx.split(ts_f32.model, nnx.Param, ...)
    nnx.update(
        ts_f32.model,
        jax.tree.map(
            lambda v: v.replace(value=v.get_value().astype(jnp.bfloat16).astype(jnp.float32)),
            f32_p,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        ),
    )
    # Re-initialize ts_r16 from the exact same starting values:
    m_r16_aligned = _TinyResidualModel(nnx.Rngs(11))
    _, r16_p, _ = nnx.split(m_r16_aligned, nnx.Param, ...)
    nnx.update(
        m_r16_aligned,
        jax.tree.map(
            lambda v: v.replace(value=v.get_value().astype(jnp.bfloat16).astype(jnp.float32)),
            r16_p,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        ),
    )
    ts_r16 = train_state_nnx.create_train_state(m_r16_aligned, tx, cfg_r16)

    for _ in range(40):
      step_fn(ts_f32)
      step_fn(ts_r16)
      step_fn(ts_bf16)

    # Pure bf16 stalls completely at lr=1e-5 around ~O(1) weights (bf16 ULP at 1.0 is 2^-7 = 7.8e-3 >> 1e-5):
    np.testing.assert_array_equal(np.asarray(ts_bf16.model.embed.get_value()), init_bf16_embed)

    # Reconstructed residual16 master weights advance and closely track f32 master weights:
    r16_ckpt = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16))
    f32_ckpt = train_state_nnx.to_checkpoint_dict(nnx.state(ts_f32))
    r16_rec_embed = np.asarray(r16_ckpt["params"]["params"]["embed"])
    f32_embed = np.asarray(f32_ckpt["params"]["params"]["embed"])
    self.assertGreater(np.max(np.abs(r16_rec_embed - init_bf16_embed.astype(np.float32))), 2e-4)
    np.testing.assert_allclose(r16_rec_embed, f32_embed, rtol=1e-5, atol=1e-6)

  def test_adamw_small_lr_updates_match_f32_while_bf16_stalls(self):
    self._run_small_lr_comparison(optax.adamw(learning_rate=1e-5, weight_decay=0.01))

  def test_muon_small_lr_updates_match_f32_while_bf16_stalls(self):
    self._run_small_lr_comparison(optax_muon(learning_rate=1e-5, weight_decay=0.01))

  def test_qk_clip_updates_both_working_weights_and_residuals(self):
    nnx.use_eager_sharding(False)
    cfg = SimpleNamespace(
        fp32_master_storage="residual16",
        weight_dtype=jnp.float32,
        routed_bias=False,
        attention_type="mla",
        qk_clip_threshold=10.0,
        qk_nope_head_dim=4,
    )
    m_r16 = _TinyMLAModel(nnx.Rngs(3))
    m_f32 = _TinyMLAModel(nnx.Rngs(3))
    ts_r16 = train_state_nnx.create_train_state(m_r16, optax.adamw(1e-3), cfg)
    ts_f32 = train_state_nnx.TrainStateNNX(m_f32, nnx.Optimizer(m_f32, optax.adamw(1e-3), wrt=nnx.Param))

    intermediates = {"self_attention": {"max_logits": jnp.array([[25.0, 5.0]], dtype=jnp.float32)}}
    qk_clip_utils.apply_qk_clip_nnx(ts_r16, intermediates, cfg)
    qk_clip_utils.apply_qk_clip_nnx(ts_f32, intermediates, cfg)

    r16_rec = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16))["params"]["params"]
    f32_ref = train_state_nnx.to_checkpoint_dict(nnx.state(ts_f32))["params"]["params"]
    np.testing.assert_allclose(
        r16_rec["self_attention"]["wq_b"]["kernel"],
        f32_ref["self_attention"]["wq_b"]["kernel"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        r16_rec["self_attention"]["wkv_b"]["kernel"],
        f32_ref["self_attention"]["wkv_b"]["kernel"],
        rtol=0,
        atol=0,
    )


class TestOrbaxCheckpointCompatibility(unittest.TestCase):
  """Full Orbax save, on-disk schema/dtype inspection, cross-restore, and params-only restore tests."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)
    self._tmpdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)
    self._mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))

  def _make_manager(self, subdir: str):
    return checkpointing.create_orbax_checkpoint_manager(
        os.path.join(self._tmpdir, subdir),
        enable_checkpointing=True,
        use_async=False,
        save_interval_steps=1,
        dataset_type="tfds",
    )

  def _init_fn(self, cfg, seed: int = 0):
    tx = optax.adamw(learning_rate=1e-4, weight_decay=0.01)

    def fn():
      model = _TinyResidualModel(nnx.Rngs(seed), include_gate=True)
      return train_state_nnx.create_train_state(model, tx, cfg)

    return fn

  def _sharded_abstract_and_shardings(self, init_fn):
    repl = NamedSharding(self._mesh, P())
    abstract = nnx.eval_shape(lambda: nnx.state(init_fn()))
    sharded_abs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=repl) if hasattr(x, "shape") else x,
        abstract,
    )
    shardings = jax.tree.map(lambda _: repl, abstract)
    return sharded_abs, shardings

  def _setup_state(self, manager, init_fn, cfg):
    abstract, shardings = self._sharded_abstract_and_shardings(init_fn)
    with mock.patch.object(maxtext_utils, "get_abstract_state", return_value=(abstract, None, shardings)):
      state, *_ = maxtext_utils.setup_initial_state(None, cfg, self._mesh, manager, init_fn, True)
    return state

  def test_residual16_orbax_save_produces_standard_f32_checkpoint_and_cross_restores_bit_exact(self):
    cfg_r16 = _ckpt_config(fp32_master_storage="residual16")
    cfg_r16.routed_bias = True
    cfg_f32 = _ckpt_config(fp32_master_storage="f32")
    cfg_f32.routed_bias = True

    mgr = self._make_manager("r16_ckpt")
    init_r16 = self._init_fn(cfg_r16, seed=10)
    ts_r16 = init_r16()
    tokens = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)

    for _ in range(3):
      grads = nnx.grad(lambda m: jnp.mean(m(tokens).astype(jnp.float32) ** 2))(ts_r16.model)
      ts_r16.apply_gradients(grads)

    # Save checkpoint at step 3 from residual16 state:
    checkpointing.save_checkpoint(mgr, 3, ts_r16, config=cfg_r16, force=True)
    checkpointing.wait_until_finished(mgr)

    # 1. Inspect raw on-disk Orbax checkpoint metadata & values: must be 100% standard float32 MaxText layout
    # with ZERO `master_residuals` or `"inner"` keys leaked!
    step_path = os.path.join(self._tmpdir, "r16_ckpt", "3")
    raw_loaded = ocp.load(step_path, checkpointable_name="items")
    self.assertIn("params", raw_loaded)
    self.assertIn("opt_state", raw_loaded)
    self.assertNotIn("master_residuals", str(raw_loaded.keys()))
    if isinstance(raw_loaded["opt_state"], dict):
      self.assertNotIn("master_residuals", raw_loaded["opt_state"])
      self.assertNotIn("inner", raw_loaded["opt_state"])
    for leaf in jax.tree.leaves(raw_loaded["params"]["params"]):
      self.assertEqual(leaf.dtype, jnp.float32)

    # Advance original ts_r16 one more step for bit-exact continuation reference:
    next_tokens = jnp.array([[2, 4, 6, 8], [1, 3, 5, 7]], dtype=jnp.int32)
    grads_ref = nnx.grad(lambda m: jnp.mean(m(next_tokens).astype(jnp.float32) ** 2))(ts_r16.model)
    ts_r16.apply_gradients(grads_ref)
    expected_after_step4 = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16))["params"]["params"]

    # 2. Restore into residual16 (with a different init seed!) and verify bit-exact continued step:
    init_r16_fresh = self._init_fn(cfg_r16, seed=999)
    restored_r16_state = self._setup_state(mgr, init_r16_fresh, cfg_r16)
    graphdef_r16 = nnx.graphdef(init_r16_fresh())
    ts_r16_resumed = nnx.merge(graphdef_r16, restored_r16_state)
    self.assertEqual(ts_r16_resumed.model.embed.get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_r16_resumed.model.gate["bias"].get_value().dtype, jnp.float32)
    grads_resumed = nnx.grad(lambda m: jnp.mean(m(next_tokens).astype(jnp.float32) ** 2))(ts_r16_resumed.model)
    ts_r16_resumed.apply_gradients(grads_resumed)
    actual_r16_step4 = train_state_nnx.to_checkpoint_dict(nnx.state(ts_r16_resumed))["params"]["params"]
    for p_exp, p_got in zip(jax.tree.leaves(expected_after_step4), jax.tree.leaves(actual_r16_step4)):
      np.testing.assert_array_equal(
          np.asarray(jax.lax.bitcast_convert_type(p_got, jnp.uint32)),
          np.asarray(jax.lax.bitcast_convert_type(p_exp, jnp.uint32)),
      )

    # 3. Cross-restore residual16 checkpoint into f32 train state:
    init_f32_fresh = self._init_fn(cfg_f32, seed=777)
    restored_f32_state = self._setup_state(mgr, init_f32_fresh, cfg_f32)
    graphdef_f32 = nnx.graphdef(init_f32_fresh())
    ts_f32_resumed = nnx.merge(graphdef_f32, restored_f32_state)
    self.assertEqual(ts_f32_resumed.model.embed.get_value().dtype, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(ts_f32_resumed.model.embed.get_value(), jnp.uint32)),
        np.asarray(jax.lax.bitcast_convert_type(raw_loaded["params"]["params"]["embed"], jnp.uint32)),
    )

    # 4. Reverse cross-restore: save from f32 -> restore into residual16 (both full state and load_parameters_path):
    mgr_f32 = self._make_manager("f32_ckpt")
    checkpointing.save_checkpoint(mgr_f32, 5, ts_f32_resumed, config=cfg_f32, force=True)
    checkpointing.wait_until_finished(mgr_f32)

    restored_from_f32 = self._setup_state(mgr_f32, self._init_fn(cfg_r16, seed=555), cfg_r16)
    rec_from_f32 = train_state_nnx.to_checkpoint_dict(restored_from_f32)["params"]["params"]
    for p_exp, p_got in zip(
        jax.tree.leaves(train_state_nnx.to_checkpoint_dict(nnx.state(ts_f32_resumed))["params"]["params"]),
        jax.tree.leaves(rec_from_f32),
    ):
      np.testing.assert_array_equal(
          np.asarray(jax.lax.bitcast_convert_type(p_got, jnp.uint32)),
          np.asarray(jax.lax.bitcast_convert_type(p_exp, jnp.uint32)),
      )

    # 5. Params-only load (`load_parameters_path`) into residual16 preserves full float32 precision in residuals:
    f32_step5_path = os.path.join(self._tmpdir, "f32_ckpt", "5")
    cfg_params_only = _ckpt_config(load_parameters_path=f32_step5_path, fp32_master_storage="residual16")
    cfg_params_only.routed_bias = True
    params_only_state = self._setup_state(None, self._init_fn(cfg_params_only, seed=333), cfg_params_only)
    self.assertEqual(int(params_only_state["optimizer"]["step"].get_value()), 0)
    rec_params_only = train_state_nnx.to_checkpoint_dict(params_only_state)["params"]["params"]
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(rec_params_only["embed"], jnp.uint32)),
        np.asarray(jax.lax.bitcast_convert_type(ts_f32_resumed.model.embed.get_value(), jnp.uint32)),
    )


class TestOptimizerDeviceResidentBudgetAndConfig(unittest.TestCase):
  """Tests optimizer_device_resident_gib budget allocation, ZeRO-1 sharding interaction, and config validation."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def test_budget_zero_partial_and_overflow_and_deterministic_ordering(self):
    devices = np.array(jax.devices()[:2])
    mesh = Mesh(devices, ("fsdp",))
    sharded = NamedSharding(mesh, P("fsdp", None))
    replicated = NamedSharding(mesh, P())

    # Leaf sizes per device on 2-device fsdp mesh:
    # - big_mu: (2048, 256) float32 sharded on dim 0 -> (1024, 256) * 4 = 1,048,576 B (1.0 MiB)
    # - mid_res: (2048, 256) int16 sharded on dim 0 -> (1024, 256) * 2 = 524,288 B (0.5 MiB)
    # - small_bias: (256,) float32 replicated -> 256 * 4 = 1,024 B
    # - step: () uint32 scalar -> 0 B (always offloaded unless budget covers non-scalars? scalars stay pinned_host)
    opt_abs = {
        "step": jax.ShapeDtypeStruct((), jnp.uint32, sharding=replicated),
        "opt_state": {
            "inner": {
                "mu": {
                    "big": jax.ShapeDtypeStruct((2048, 256), jnp.float32, sharding=sharded),
                    "bias": jax.ShapeDtypeStruct((256,), jnp.float32, sharding=replicated),
                }
            },
            "master_residuals": {
                "big": jax.ShapeDtypeStruct((2048, 256), jnp.int16, sharding=sharded),
            },
        },
    }
    opt_shard = {
        "step": replicated,
        "opt_state": {
            "inner": {"mu": {"big": sharded, "bias": replicated}},
            "master_residuals": {"big": sharded},
        },
    }

    # 1. Budget = 0.0 -> 0 resident leaves
    res0, b_res0, b_off0 = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs, opt_shard, 0.0)
    self.assertEqual(len(res0), 0)
    self.assertEqual(b_res0, 0)
    self.assertEqual(b_off0, 1_048_576 + 524_288 + 1_024)

    # 2. Partial budget = 1.25 MiB -> picks largest 1.0 MiB leaf ("big" mu), skips 0.5 MiB leaf (would hit 1.5 MiB),
    # and packs 1 KiB "bias" leaf (1.0 MiB + 1 KiB <= 1.25 MiB)!
    budget_gib = (1_048_576 + 262_144) / float(1 << 30)
    res_part, b_res_part, b_off_part = train_state_nnx.pick_device_resident_optimizer_leaves(
        opt_abs, opt_shard, budget_gib
    )
    self.assertEqual(
        res_part,
        {
            ("opt_state", "inner", "mu", "big"),
            ("opt_state", "inner", "mu", "bias"),
        },
    )
    self.assertEqual(b_res_part, 1_048_576 + 1_024)
    self.assertEqual(b_off_part, 524_288)

    # 3. Budget larger than total state (1.0 GiB) -> all non-scalar leaves resident
    res_all, b_res_all, b_off_all = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs, opt_shard, 1.0)
    self.assertEqual(len(res_all), 3)
    self.assertEqual(b_off_all, 0)
    self.assertEqual(b_res_all, 1_048_576 + 524_288 + 1_024)

    # 4. Deterministic tie-breaking regardless of dict insertion order:
    opt_abs_tie_a = {
        "a": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
        "b": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
    }
    opt_abs_tie_b = {
        "b": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
        "a": jax.ShapeDtypeStruct((256, 256), jnp.float32, sharding=replicated),
    }
    opt_shard_tie_a = {"a": replicated, "b": replicated}
    opt_shard_tie_b = {"b": replicated, "a": replicated}
    one_leaf_gib = (256 * 256 * 4) / float(1 << 30)
    pick_a, _, _ = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs_tie_a, opt_shard_tie_a, one_leaf_gib)
    pick_b, _, _ = train_state_nnx.pick_device_resident_optimizer_leaves(opt_abs_tie_b, opt_shard_tie_b, one_leaf_gib)
    self.assertEqual(pick_a, {("a",)})
    self.assertEqual(pick_b, {("a",)})

  def test_zero1_sharding_preserves_model_memory_kind_under_partial_budget(self):
    devices = np.array(jax.devices()[:2]).reshape(2, 1)
    mesh = Mesh(devices, ("data", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    cfg = SimpleNamespace(
        shard_optimizer_over_data=True,
        optimizer_memory_host_offload=True,
        optimizer_device_resident_gib=1e-4,
        parameter_memory_host_offload=False,
        fp32_master_storage="residual16",
        weight_dtype=jnp.float32,
        routed_bias=False,
        logical_axis_rules=[("fsdp", "fsdp")],
        use_m3_model=False,
    )

    def init_fn():
      m = _TinyResidualModel(nnx.Rngs(0))
      return train_state_nnx.create_train_state(m, optax.adamw(1e-3), cfg)

    _, _, state_mesh_shardings = maxtext_utils.get_abstract_state_nnx(cfg, mesh, init_fn, is_training=True)
    _, updated_shardings = sharding.maybe_update_params_sharding_with_opt(cfg, state_mesh_shardings)
    model_kinds = {
        s.memory_kind for s in jax.tree.leaves(updated_shardings.model, is_leaf=lambda x: isinstance(x, NamedSharding))
    }
    self.assertEqual(model_kinds, {"device"})

  def test_pyconfig_and_pydantic_validation(self):
    cfg = pyconfig.initialize(
        [
            "",
            get_test_config_path(),
            "run_name=test_res16",
            "fp32_master_storage=residual16",
            "optimizer_memory_host_offload=True",
            "optimizer_device_resident_gib=0.25",
        ]
    )
    self.assertEqual(cfg.fp32_master_storage, "residual16")
    self.assertAlmostEqual(cfg.optimizer_device_resident_gib, 0.25)

    # Rejects optimizer_device_resident_gib > 0 when optimizer_memory_host_offload is False:
    with self.assertRaises(ValidationError):
      pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "run_name=test_bad_budget",
              "optimizer_memory_host_offload=False",
              "optimizer_device_resident_gib=1.0",
          ]
      )

    # Rejects negative optimizer_device_resident_gib:
    with self.assertRaises(ValidationError):
      pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "run_name=test_neg_budget",
              "optimizer_memory_host_offload=True",
              "optimizer_device_resident_gib=-0.5",
          ]
      )

    # Rejects invalid fp32_master_storage literal:
    with self.assertRaises(ValidationError):
      pyconfig.initialize(
          [
              "",
              get_test_config_path(),
              "run_name=test_bad_storage",
              "fp32_master_storage=fp16",
          ]
      )

    # RLConfig also inherits both fields and validators cleanly, and rejects residual16:
    self.assertIn("fp32_master_storage", config_types.RLConfig.model_fields)
    self.assertIn("optimizer_device_resident_gib", config_types.RLConfig.model_fields)
    rl_yml_path = os.path.join(os.path.dirname(get_test_config_path()), "post_train", "rl.yml")
    with self.assertRaisesRegex(ValidationError, "fp32_master_storage='residual16' is not supported"):
      pyconfig.initialize_pydantic(
          ["", rl_yml_path, "model_name=qwen3-0.6b", "fp32_master_storage=residual16"],
          config_class=config_types.RLConfig,
      )
    with self.assertRaises(ValidationError):
      pyconfig.initialize_pydantic(
          ["", rl_yml_path, "model_name=qwen3-0.6b", "optimizer_device_resident_gib=0.25"],
          config_class=config_types.RLConfig,
      )

    # Rejects unsupported trainer combinations on MaxTextConfig (LoRA, DiLoCo, SFT/DPO/GRPO):
    for extra_args, match_pat in (
        (["lora.enable_lora=True", "fp32_master_storage=residual16"], "LoRA"),
        (["lora.enable_lora=True", "optimizer_memory_host_offload=True", "optimizer_device_resident_gib=0.25"], "LoRA"),
        (["enable_diloco=True", "fp32_master_storage=residual16"], "DiLoCo"),
        (
            ["enable_diloco=True", "optimizer_memory_host_offload=True", "optimizer_device_resident_gib=0.25"],
            "DiLoCo",
        ),
        (["use_sft=True", "fp32_master_storage=residual16"], "use_sft"),
        (["use_dpo=True", "fp32_master_storage=residual16"], "use_dpo"),
        (["use_grpo=True", "fp32_master_storage=residual16"], "use_grpo"),
    ):
      with self.subTest(extra_args=extra_args):
        with self.assertRaisesRegex(ValidationError, match_pat):
          pyconfig.initialize(["", get_test_config_path(), "run_name=test_unsupported"] + extra_args)


class TestFp32ParamExemptionsAndSaveMemoryKind(unittest.TestCase):
  """Regression tests for shared_expert_gate / A_log / dt_bias / keep_in_fp32_modules and pinned_host save."""

  def setUp(self):
    super().setUp()
    prev = nnx.using_eager_sharding()
    nnx.use_eager_sharding(False)
    self.addCleanup(nnx.use_eager_sharding, prev)

  def test_shared_expert_gate_a_log_dt_bias_and_keep_in_fp32_modules_preserve_f32(self):
    class _HybridGdnMoeBlock(nnx.Module):
      """Minimal Qwen3-Next-style hybrid GatedDeltaNet + shared-expert MoE block."""

      def __init__(self, rngs: nnx.Rngs):
        # GatedDeltaNet recurrence parameters initialized in float32:
        self.linear_attn = nnx.Dict(
            {
                "A_log": nnx.Param(jnp.linspace(-2.0, 2.5, 8, dtype=jnp.float32) + jnp.float32(1e-6)),
                "dt_bias": nnx.Param(jnp.linspace(0.5, 1.5, 8, dtype=jnp.float32) + jnp.float32(1e-6)),
                "out_proj": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32))}),
            }
        )
        # MoE router gate + shared_expert_gate + expert projection:
        self.mlp = nnx.Dict(
            {
                "routed_experts": nnx.Dict(
                    {"gate": nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32))})}
                ),
                "shared_expert_gate": nnx.Dict(
                    {"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 1), dtype=jnp.float32))}
                ),
                "shared_expert": nnx.Dict(
                    {"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 8), dtype=jnp.float32))}
                ),
            }
        )
        self.custom_head = nnx.Dict({"kernel": nnx.Param(jax.random.normal(rngs.params(), (8, 4), dtype=jnp.float32))})

      def forward_gates(self, x: jax.Array, a: jax.Array):
        a_log = jnp.asarray(self.linear_attn["A_log"].get_value(), dtype=jnp.float32)
        dt_bias = jnp.asarray(self.linear_attn["dt_bias"].get_value(), dtype=jnp.float32)
        g = -jnp.exp(a_log) * jax.nn.softplus(a.astype(jnp.float32) + dt_bias)
        sh_gate = jax.nn.sigmoid(
            x.astype(jnp.float32) @ jnp.asarray(self.mlp["shared_expert_gate"]["kernel"].get_value(), dtype=jnp.float32)
        )
        return g, sh_gate

    cfg_f32 = SimpleNamespace(
        fp32_master_storage="f32",
        weight_dtype=jnp.float32,
        float32_gate_logits=True,
        routed_bias=False,
        keep_in_fp32_modules=["custom_head"],
    )
    cfg_r16 = SimpleNamespace(
        fp32_master_storage="residual16",
        weight_dtype=jnp.float32,
        float32_gate_logits=True,
        routed_bias=False,
        keep_in_fp32_modules=["custom_head"],
    )

    m_f32 = _HybridGdnMoeBlock(nnx.Rngs(7))
    m_r16 = _HybridGdnMoeBlock(nnx.Rngs(7))
    ts_f32 = train_state_nnx.create_train_state(m_f32, optax.adamw(1e-3), cfg_f32)
    ts_r16 = train_state_nnx.create_train_state(m_r16, optax.adamw(1e-3), cfg_r16)

    # Exempt parameters remain float32 in state.model:
    self.assertEqual(ts_r16.model.linear_attn["A_log"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.linear_attn["dt_bias"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.mlp["routed_experts"]["gate"]["kernel"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.mlp["shared_expert_gate"]["kernel"].get_value().dtype, jnp.float32)
    self.assertEqual(ts_r16.model.custom_head["kernel"].get_value().dtype, jnp.float32)

    # Standard projection weights use bfloat16 working storage + int16 master residuals:
    self.assertEqual(ts_r16.model.linear_attn["out_proj"]["kernel"].get_value().dtype, jnp.bfloat16)
    self.assertEqual(ts_r16.model.mlp["shared_expert"]["kernel"].get_value().dtype, jnp.bfloat16)

    # Step-0 forward recurrence decay `g` and `shared_expert_gate` output match f32 bit-for-bit:
    x = jax.random.normal(jax.random.PRNGKey(19), (2, 8), dtype=jnp.bfloat16)
    a = jax.random.normal(jax.random.PRNGKey(23), (2, 8), dtype=jnp.bfloat16)
    g_f32, sh_f32 = ts_f32.model.forward_gates(x, a)
    g_r16, sh_r16 = ts_r16.model.forward_gates(x, a)
    np.testing.assert_array_equal(np.asarray(g_r16), np.asarray(g_f32))
    np.testing.assert_array_equal(np.asarray(sh_r16), np.asarray(sh_f32))

  def test_to_linen_checkpoint_dict_places_reconstructed_f32_on_pinned_host_with_fallback(self):
    devices = np.array(jax.devices()[:2])
    mesh = Mesh(devices, ("fsdp",))
    cfg = SimpleNamespace(
        fp32_master_storage="residual16",
        weight_dtype=jnp.float32,
        parameter_memory_host_offload=False,
        routed_bias=False,
        float32_gate_logits=False,
    )
    with jax.set_mesh(mesh):
      model = _TinyResidualModel(nnx.Rngs(31))
      sharded_spec = NamedSharding(mesh, P("fsdp", None), memory_kind="device")
      model.embed.set_value(jax.device_put(model.embed.get_value(), sharded_spec))
      ts = train_state_nnx.create_train_state(model, optax.adamw(1e-3), cfg)

    self.assertEqual(ts.model.embed.get_value().sharding.memory_kind, "device")
    ckpt_dict = train_state_nnx.to_checkpoint_dict(ts)
    saved_embed = ckpt_dict["params"]["params"]["embed"]
    self.assertEqual(saved_embed.dtype, jnp.float32)
    self.assertEqual(saved_embed.sharding.memory_kind, "pinned_host")

    # Verify fallback to `device` when a backend raises ValueError on `with_memory_kind("pinned_host")`:
    orig_with_mem = NamedSharding.with_memory_kind

    def _reject_pinned_host(self_sharding, kind):
      if kind == "pinned_host":
        raise ValueError("pinned_host unsupported on mock backend")
      return orig_with_mem(self_sharding, kind)

    with mock.patch.object(NamedSharding, "with_memory_kind", _reject_pinned_host):
      fallback_embed = train_state_nnx._reconstruct_f32_concrete_for_save(  # pylint: disable=protected-access
          ts.model.embed.get_value(),
          ts.optimizer.opt_state["master_residuals"]["embed"].get_value(),
      )
    self.assertEqual(fallback_embed.dtype, jnp.float32)
    self.assertEqual(fallback_embed.sharding.memory_kind, "device")
    np.testing.assert_array_equal(np.asarray(fallback_embed), np.asarray(saved_embed))

  def test_unsupported_entrypoints_fail_fast(self):
    # 1. lora_utils.setup_initial_lora_state rejects residual16 and optimizer_device_resident_gib > 0:
    res16_ns = SimpleNamespace(fp32_master_storage="residual16", optimizer_device_resident_gib=0.0)
    budget_ns = SimpleNamespace(fp32_master_storage="f32", optimizer_device_resident_gib=0.5)
    with self.assertRaisesRegex(ValueError, "LoRA does not support fp32_master_storage"):
      lora_utils.setup_initial_lora_state(None, optax.sgd(1e-3), res16_ns, None, None, "/tmp/fake_lora")
    with self.assertRaisesRegex(ValueError, "LoRA does not support optimizer_device_resident_gib"):
      lora_utils.setup_initial_lora_state(None, optax.sgd(1e-3), budget_ns, None, None, "/tmp/fake_lora")

    # 2. Distillation rejects residual16 and optimizer_device_resident_gib > 0 on student or teacher:
    if importlib.util.find_spec("tunix") is not None:
      from maxtext.trainers.post_train.distillation import train_distill  # pylint: disable=import-outside-toplevel

      ok_ns = SimpleNamespace(fp32_master_storage="f32", optimizer_device_resident_gib=0.0, vocab_size=32)
      bad_student = SimpleNamespace(fp32_master_storage="residual16", optimizer_device_resident_gib=0.0, vocab_size=32)
      bad_teacher = SimpleNamespace(fp32_master_storage="f32", optimizer_device_resident_gib=0.5, vocab_size=32)
      with self.assertRaisesRegex(ValueError, "Distillation \\(student_config\\) does not support fp32_master_storage"):
        train_distill.train_distill(bad_student, ok_ns)
      with self.assertRaisesRegex(
          ValueError, "Distillation \\(teacher_config\\) does not support optimizer_device_resident_gib"
      ):
        train_distill.train_distill(ok_ns, bad_teacher)

    # 3. MaxTextTrainingEngine.__init__ fails fast with NotImplementedError (using lightweight stub if tunix.dist_rl absent):
    need_stub = importlib.util.find_spec("tunix") is None or importlib.util.find_spec("tunix.dist_rl") is None
    stub_modules = {}
    if need_stub:
      tunix_mod = sys.modules.get("tunix") or types.ModuleType("tunix")
      dist_rl_mod = types.ModuleType("tunix.dist_rl")
      data_types_mod = types.ModuleType("tunix.dist_rl.data_types")
      sft_mod = sys.modules.get("tunix.sft") or types.ModuleType("tunix.sft")
      sft_utils_mod = sys.modules.get("tunix.sft.utils") or types.ModuleType("tunix.sft.utils")
      rl_mod = sys.modules.get("tunix.rl") or types.ModuleType("tunix.rl")
      rl_common_mod = types.ModuleType("tunix.rl.common")
      for cls_name in ("TrainerPayload", "RLTrainerPayload"):
        setattr(data_types_mod, cls_name, type(cls_name, (), {}))
      for cls_name in ("LossOutput", "WeightedMetric"):
        if not hasattr(sft_utils_mod, cls_name):
          setattr(sft_utils_mod, cls_name, type(cls_name, (), {}))
      setattr(rl_common_mod, "RolloutOutput", type("RolloutOutput", (), {}))
      stub_modules = {
          "tunix": tunix_mod,
          "tunix.dist_rl": dist_rl_mod,
          "tunix.dist_rl.data_types": data_types_mod,
          "tunix.sft": sft_mod,
          "tunix.sft.utils": sft_utils_mod,
          "tunix.rl": rl_mod,
          "tunix.rl.common": rl_common_mod,
      }
    with mock.patch.dict(sys.modules, stub_modules):
      try:
        maxtext_engine = importlib.import_module("maxtext.training_engine.maxtext_engine")
      except ModuleNotFoundError:
        maxtext_engine = None
      if maxtext_engine is not None:
        cfg_res16 = pyconfig.initialize(
            ["", get_test_config_path(), "run_name=test_eng_res16", "fp32_master_storage=residual16"]
        )
        with self.assertRaisesRegex(NotImplementedError, "fp32_master_storage"):
          maxtext_engine.MaxTextTrainingEngine(cfg_res16)
        cfg_budget = pyconfig.initialize(
            [
                "",
                get_test_config_path(),
                "run_name=test_eng_budget",
                "optimizer_memory_host_offload=True",
                "optimizer_device_resident_gib=0.25",
            ]
        )
        with self.assertRaisesRegex(NotImplementedError, "optimizer_device_resident_gib"):
          maxtext_engine.MaxTextTrainingEngine(cfg_budget)


if __name__ == "__main__":
  unittest.main()
