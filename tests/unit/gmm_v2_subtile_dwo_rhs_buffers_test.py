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

"""Unit and TPU hardware tests for gmm_v2 subtile-M, transposed dW_o^T, and wi forward RHS buffer count."""

import functools
import os
import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest
import qwix

from maxtext.configs import pyconfig
from maxtext.configs import types as config_types
from maxtext.kernels.megablox import ops as megablox_ops
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from tests.utils.test_helpers import get_post_train_test_config_path
from tests.utils.test_helpers import get_test_config_path


def _tpu_target(device_kind: str = "TPU7x"):
  """Context targeting an abstract TPU device for host-side AOT lowering."""
  device = jax.sharding.AbstractDevice(device_kind=device_kind, num_cores=1, platform="tpu")
  return jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh((), (), abstract_device=device))


def _kernel_names_and_mlir(fn, *args, device_kind: str = "TPU7x") -> tuple[list[str], str]:
  with _tpu_target(device_kind):
    mlir = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",)).as_text()
  return re.findall(r'kernel_name = "([^"]+)"', mlir), mlir


def _same_bits(a, b) -> bool:
  a_np, b_np = np.asarray(a), np.asarray(b)
  if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:
    return False
  unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a_np.dtype.itemsize]
  return bool(np.array_equal(a_np.view(unsigned), b_np.view(unsigned)))


def _poison_tpu_memory():
  """Fills TPU VMEM/SMEM scratchpad with NaN bit patterns to catch out-of-bounds reads."""
  tpu_info = pltpu.get_tpu_info()
  vmem_size = (4 * 1024 * 1024) // 4
  smem_size = (tpu_info.smem_capacity_bytes // 4) - 8192

  def poison_kernel(in_ref, out_ref, v_scratch, s_scratch):
    del in_ref, out_ref
    v_scratch[...] = jnp.full_like(v_scratch, jnp.nan)
    for i in range(s_scratch.shape[0]):
      s_scratch[i] = 0x7FC00000

  pl.pallas_call(
      poison_kernel,
      out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
      grid=(1,),
      scratch_shapes=[
          pltpu.VMEM((vmem_size // 128, 128), jnp.float32),
          pltpu.SMEM((smem_size,), jnp.int32),
      ],
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
  )(jnp.zeros((1,), dtype=jnp.float32))


def _quantize_blockwise(x: jax.Array, qdtype: jnp.dtype, block_size: int = 128) -> tuple[jax.Array, jax.Array]:
  """Quantizes [G, K, N] along K (axis 1) into blocks, returning (q, [G, num_blocks, 1, N] f32 scale)."""
  if jnp.issubdtype(qdtype, jnp.integer):
    max_val = float(jnp.iinfo(qdtype).max)
    min_val = float(jnp.iinfo(qdtype).min)
  else:
    max_val = float(jnp.finfo(qdtype).max)
    min_val = float(jnp.finfo(qdtype).min)
  g, k, n = x.shape
  num_blocks = k // block_size
  x_blk = x.astype(jnp.float32).reshape(g, num_blocks, block_size, n)
  absmax = jnp.max(jnp.abs(x_blk), axis=2, keepdims=True)
  scale = jnp.where(absmax == 0, 1.0, absmax / max_val).astype(jnp.float32)
  q = jnp.clip(x_blk / scale, min_val, max_val).astype(qdtype).reshape(g, k, n)
  return q, scale


def _reference_ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    group_offset: int = 0,
    transpose_rhs: bool = False,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    partial_sum: jax.Array | None = None,
    fuse_act: str | None = None,
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
  """Float32 reference grouped matmul with zeroing of unvisited rows when partial_sum is None."""
  w = rhs.astype(jnp.float32)
  if transpose_rhs:
    w = jnp.swapaxes(w, 1, 2)
  if rhs_scale is not None:
    g, num_blocks, _, n = rhs_scale.shape
    block_size = w.shape[1] // num_blocks
    w = (w.reshape(g, num_blocks, block_size, n) * rhs_scale.astype(jnp.float32)).reshape(g, -1, n)
  local_gs = group_sizes[group_offset : group_offset + w.shape[0]]
  x_f32 = lhs.astype(jnp.float32)
  m = x_f32.shape[0]
  n_out = w.shape[2] // 2 if fuse_act is not None else w.shape[2]
  if partial_sum is not None:
    out = partial_sum.astype(jnp.float32).copy()
  else:
    out = jnp.zeros((m, n_out), dtype=jnp.float32)
  # Prefix of lhs before group_offset:
  row_cursor = int(np.sum(np.asarray(group_sizes)[:group_offset]))
  for g_idx in range(w.shape[0]):
    cnt = int(np.asarray(local_gs)[g_idx])
    if cnt > 0:
      seg = x_f32[row_cursor : row_cursor + cnt] @ w[g_idx]
      if rhs_bias is not None:
        seg = seg + rhs_bias[g_idx].astype(jnp.float32)
      if fuse_act == "silu":
        half = seg.shape[1] // 2
        seg = jax.nn.silu(seg[:, :half]) * seg[:, half:]
      if partial_sum is not None:
        seg = seg + partial_sum[row_cursor : row_cursor + cnt].astype(jnp.float32)
      out = out.at[row_cursor : row_cursor + cnt].set(seg)
    row_cursor += cnt
  return out.astype(out_dtype)


def _gmm_v2_standin(
    lhs,
    rhs,
    group_sizes,
    rhs_scale=None,
    partial_sum=None,
    group_offset=None,
    *,
    transpose_rhs=False,
    preferred_element_type=None,
    **kwargs,
):
  """CPU JAX stand-in for gmm_v2 used to test ops.gmm and RoutedMoE wiring."""
  del rhs_scale, kwargs
  out_dtype = preferred_element_type or lhs.dtype
  w = rhs.swapaxes(1, 2) if transpose_rhs else rhs
  off = 0 if group_offset is None else jnp.reshape(group_offset, ())
  gs = jax.lax.dynamic_slice_in_dim(group_sizes, off, w.shape[0])
  out = jax.lax.ragged_dot(lhs, w, gs, preferred_element_type=jnp.float32).astype(out_dtype)
  if partial_sum is not None:
    out = out + partial_sum[: out.shape[0]]
  return out


def _tgmm_v2_standin(
    lhs,
    rhs,
    group_sizes,
    num_actual_groups,
    rhs_scale=None,
    partial_sum=None,
    group_offset=None,
    *,
    transpose_out=False,
    preferred_element_type=None,
    **kwargs,
):
  """CPU JAX stand-in for tgmm_v2 used to test ops.gmm and RoutedMoE wiring."""
  del rhs_scale, kwargs
  out_dtype = preferred_element_type or lhs.dtype
  off = 0 if group_offset is None else jnp.reshape(group_offset, ())
  gs = jax.lax.dynamic_slice_in_dim(group_sizes, off, num_actual_groups)
  dim_nums = jax.lax.RaggedDotDimensionNumbers(
      dot_dimension_numbers=(([0], [0]), ([], [])),
      lhs_ragged_dimensions=[0],
      rhs_group_dimensions=[],
  )
  out = jax.lax.ragged_dot_general(lhs, rhs, gs, dim_nums, preferred_element_type=jnp.float32).astype(out_dtype)
  if transpose_out:
    out = jnp.swapaxes(out, 1, 2)
  if partial_sum is not None:
    out = out + partial_sum
  return out


class ConfigValidationTest(parameterized.TestCase):
  """Tests Tier-2 MoEKernels fields and validators on MaxTextConfig and RLConfig."""

  def _init_cfg(self, **overrides):
    """Initializes a MaxTextConfig with Tokamax gmm_v2 enabled."""
    base = {
        "skip_jax_distributed_system": True,
        "model_name": "deepseek3-test",
        "sparse_matmul": True,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
    }
    base.update(overrides)
    return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **base)

  def test_defaults_on_maxtext_and_rl_configs(self):
    cfg = pyconfig.initialize(
        [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()],
        skip_jax_distributed_system=True,
    )
    self.assertFalse(cfg.enable_subtile_m)
    self.assertFalse(cfg.moe_wo_rhs_transpose)
    self.assertIsNone(cfg.wi_tile_fwd_rhs_buffer_count)

    # Verify MoEKernels is in RLConfig MRO and carries identical defaults/validators.
    self.assertIn(config_types.MoEKernels, config_types.RLConfig.__mro__)
    rl_cfg = pyconfig.initialize(
        ["maxtext.trainers.post_train.rl.train_rl", get_post_train_test_config_path("rl")],
        config_class=config_types.RLConfig,
        skip_jax_distributed_system=True,
    )
    self.assertFalse(rl_cfg.enable_subtile_m)
    self.assertFalse(rl_cfg.moe_wo_rhs_transpose)
    self.assertIsNone(rl_cfg.wi_tile_fwd_rhs_buffer_count)

  @parameterized.parameters(None, 2, 3, 4)
  def test_valid_options_accepted(self, rhs_buffers):
    cfg = self._init_cfg(
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        wi_tile_fwd_rhs_buffer_count=rhs_buffers,
    )
    self.assertTrue(cfg.enable_subtile_m)
    self.assertTrue(cfg.moe_wo_rhs_transpose)
    self.assertEqual(cfg.wi_tile_fwd_rhs_buffer_count, rhs_buffers)

  @parameterized.named_parameters(
      ("rhs_buf_1", {"wi_tile_fwd_rhs_buffer_count": 1}, "wi_tile_fwd_rhs_buffer_count >= 2"),
      ("rhs_buf_0", {"wi_tile_fwd_rhs_buffer_count": 0}, "wi_tile_fwd_rhs_buffer_count >= 2"),
      ("rhs_buf_neg", {"wi_tile_fwd_rhs_buffer_count": -2}, "wi_tile_fwd_rhs_buffer_count >= 2"),
      ("rhs_buf_bool_true", {"wi_tile_fwd_rhs_buffer_count": True}, "must be None or an int >= 2"),
      ("rhs_buf_bool_false", {"wi_tile_fwd_rhs_buffer_count": False}, "must be None or an int >= 2"),
      ("rhs_buf_float", {"wi_tile_fwd_rhs_buffer_count": 2.0}, "must be None or an int >= 2"),
      ("rhs_buf_no_gmm_v2", {"wi_tile_fwd_rhs_buffer_count": 2, "use_gmm_v2": False}, "use_gmm_v2=True"),
      ("subtile_no_gmm_v2", {"enable_subtile_m": True, "use_gmm_v2": False}, "use_gmm_v2=True"),
      ("subtile_no_tokamax", {"enable_subtile_m": True, "use_tokamax_gmm": False}, "use_tokamax_gmm=True"),
      ("subtile_dense_moe", {"enable_subtile_m": True, "sparse_matmul": False}, "sparse_matmul=True"),
      ("wo_transpose_no_gmm_v2", {"moe_wo_rhs_transpose": True, "use_gmm_v2": False}, "use_gmm_v2=True"),
      (
          "wo_transpose_flat_fsdp",
          {"moe_wo_rhs_transpose": True, "moe_flat_fsdp_weights": True},
          "moe_flat_fsdp_weights=False",
      ),
      (
          "wo_transpose_2d_fsdp",
          {"moe_wo_rhs_transpose": True, "use_2d_fsdp_sharding": True},
          "use_2d_fsdp_sharding=False",
      ),
      (
          "wo_transpose_two_stage_ag",
          {"moe_wo_rhs_transpose": True, "moe_fsdp_use_two_stage_all_gather": True},
          "moe_fsdp_use_two_stage_all_gather=False",
      ),
  )
  def test_invalid_options_rejected(self, overrides, expected_msg):
    with self.assertRaisesRegex(Exception, re.escape(expected_msg)):
      self._init_cfg(**overrides)


class KernelMetadataAndLoweringTest(parameterized.TestCase):
  """Host-side config, BlockSpec, metadata, and TPU7x AOT lowering checks."""

  def _make_cfgs(self, *, enable_subtile_m: bool = False, rhs_buffer_count=None):
    """Builds GmmConfigs and returns (cfgs, weight_buffer_count) under TPU7x target."""
    lhs = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16)
    gs = jnp.array([64, 0, 128, 64], dtype=jnp.int32)
    off = jnp.array([0], dtype=jnp.int32)
    extra = {}
    if enable_subtile_m:
      extra["enable_subtile_m"] = True
    if rhs_buffer_count is not None:
      extra["rhs_buffer_count"] = rhs_buffer_count
    with _tpu_target():
      cfgs = gmm_v2.make_gmm_configs(
          lhs,
          rhs,
          None,
          None,
          None,
          gs,
          off,
          tile_info=gmm_v2.TileSizes(128, 256, 256),
          vmem_limit_bytes=None,
          out_dtype=jnp.bfloat16,
          acc_dtype=None,
          maybe_quantize_lhs=True,
          zero_initialize=True,
          **extra,
      )
      (_, weights_spec, _), _ = gmm_v2.generate_block_specs(None, cfgs, num_gm=4)
    return cfgs, weights_spec.weight.pipeline_mode.buffer_count

  def test_gmm_v2_metadata_scope_and_rhs_buffer_spec(self):
    base_cfgs, base_buf = self._make_cfgs()
    self.assertFalse(base_cfgs.enable_subtile_m)
    self.assertIsNone(base_cfgs.rhs_buffer_count)
    self.assertEqual(base_buf, 3)
    base_meta = gmm_v2.get_metadata(base_cfgs)
    self.assertNotIn("enable_subtile_m", base_meta)
    self.assertNotIn("rhs_buffer_count", base_meta)
    base_scope = gmm_v2.get_scope_name(base_cfgs)
    self.assertNotIn("subtile_m", base_scope)
    self.assertNotIn("rhsbuf", base_scope)

    opt_cfgs, opt_buf = self._make_cfgs(enable_subtile_m=True, rhs_buffer_count=2)
    self.assertTrue(opt_cfgs.enable_subtile_m)
    self.assertEqual(opt_cfgs.rhs_buffer_count, 2)
    self.assertEqual(opt_buf, 2)
    opt_meta = gmm_v2.get_metadata(opt_cfgs)
    self.assertEqual(opt_meta.pop("enable_subtile_m"), 1)
    self.assertEqual(opt_meta.pop("rhs_buffer_count"), 2)
    self.assertEqual(opt_meta, base_meta)
    self.assertEqual(gmm_v2.get_scope_name(opt_cfgs), base_scope + "-subtile_m-rhsbuf_2")

  @parameterized.parameters(1, 0, -1, True, False, 2.0)
  def test_gmm_v2_rejects_invalid_rhs_buffer_count(self, invalid_depth):
    with self.assertRaisesRegex(ValueError, "rhs_buffer_count must be None or an int >= 2"):
      self._make_cfgs(rhs_buffer_count=invalid_depth)

  def test_ops_gmm_zero_hlo_drift_at_defaults_and_lowering_flags(self):
    lhs = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)
    rhs = jax.ShapeDtypeStruct((4, 256, 512), jnp.bfloat16)
    rhs_t = jax.ShapeDtypeStruct((4, 512, 256), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    tiling = (128, 256, 256, 128, 256, 256, 128, 256, 256)

    def run_ops_gmm(l, r, g, **extra_kwargs):
      def f(x, w):
        return megablox_ops.gmm(
            x,
            w,
            g,
            preferred_element_type=jnp.bfloat16,
            tiling=tiling,
            use_tokamax_backend=True,
            use_gmm_v2=True,
            use_dlhs_transpose_rhs=True,
            **extra_kwargs,
        )

      out, vjp_fn = jax.vjp(f, l, r)
      dlhs, drhs = vjp_fn(jnp.ones_like(out))
      return out, dlhs, drhs

    names_implicit, mlir_implicit = _kernel_names_and_mlir(run_ops_gmm, lhs, rhs, gs)
    names_explicit_default, mlir_explicit_default = _kernel_names_and_mlir(
        functools.partial(run_ops_gmm, enable_subtile_m=False, rhs_transpose=False, fwd_rhs_buffer_count=None),
        lhs,
        rhs,
        gs,
    )
    self.assertEqual(names_implicit, names_explicit_default)
    self.assertEqual(mlir_implicit, mlir_explicit_default)

    names_sub_buf, _ = _kernel_names_and_mlir(
        functools.partial(run_ops_gmm, enable_subtile_m=True, fwd_rhs_buffer_count=2),
        lhs,
        rhs,
        gs,
    )
    self.assertLen(names_sub_buf, 3)
    # Forward kernel gets both -subtile_m and -rhsbuf_2; DLHS gets -subtile_m ONLY; TGMM gets neither.
    self.assertTrue(names_sub_buf[0].endswith("-subtile_m-rhsbuf_2"), names_sub_buf)
    self.assertTrue(names_sub_buf[1].endswith("-subtile_m") and "rhsbuf" not in names_sub_buf[1], names_sub_buf)
    self.assertTrue(names_sub_buf[2].startswith("tgmm_v2-") and "rhsbuf" not in names_sub_buf[2], names_sub_buf)

    names_rhs_t, mlir_rhs_t = _kernel_names_and_mlir(
        functools.partial(run_ops_gmm, enable_subtile_m=True, rhs_transpose=True),
        lhs,
        rhs_t,
        gs,
    )
    self.assertLen(names_rhs_t, 3)
    self.assertIn("-transpose_rhs-subtile_m", names_rhs_t[0])
    # DLHS reads rhs_t [G, N, K] directly without transpose_rhs or swapaxes!
    self.assertNotIn("transpose_rhs", names_rhs_t[1])
    self.assertIn("-subtile_m", names_rhs_t[1])
    self.assertTrue(names_rhs_t[2].startswith("tgmm_v2-") and names_rhs_t[2].endswith("-transpose_out"), names_rhs_t)
    self.assertNotIn("stablehlo.transpose", mlir_rhs_t)

  @parameterized.parameters(
      ({"enable_subtile_m": True}, "enable_subtile_m=True requires"),
      ({"rhs_transpose": True}, "rhs_transpose=True requires"),
      ({"fwd_rhs_buffer_count": 2}, "fwd_rhs_buffer_count requires"),
  )
  def test_ops_gmm_rejects_v2_only_options_without_gmm_v2(self, bad_kwarg, msg):
    lhs = jnp.zeros((16, 16), jnp.bfloat16)
    rhs = jnp.zeros((2, 16, 16), jnp.bfloat16)
    gs = jnp.array([8, 8], jnp.int32)
    with self.assertRaisesRegex(ValueError, msg):
      megablox_ops.gmm(lhs, rhs, gs, use_tokamax_backend=True, use_gmm_v2=False, **bad_kwarg)

  def test_unaligned_m_and_oversized_tile_m_lowering_shapes(self):
    """Unaligned M (m % sublane != 0) and tile_m > m lower cleanly on TPU7x target and preserve shapes."""
    lhs = jax.ShapeDtypeStruct((19, 256), jnp.bfloat16)
    rhs_t = jax.ShapeDtypeStruct((4, 384, 256), jnp.bfloat16)
    dout = jax.ShapeDtypeStruct((19, 384), jnp.bfloat16)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    tiling = (64, 128, 128, 64, 128, 128, 64, 128, 128)

    def fwd_bwd(x, w_t, dy, g):
      def loss(x_in, w_in):
        y = megablox_ops.gmm(
            x_in,
            w_in,
            g,
            preferred_element_type=jnp.bfloat16,
            tiling=tiling,
            use_tokamax_backend=True,
            use_gmm_v2=True,
            use_dlhs_transpose_rhs=True,
            enable_subtile_m=True,
            rhs_transpose=True,
            fwd_rhs_buffer_count=2,
        )
        return jnp.sum(y.astype(jnp.float32) * dy.astype(jnp.float32)), y

      (_, y), (dlhs, drhs_t) = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(x, w_t)
      return y, dlhs, drhs_t

    with _tpu_target():
      out_shapes = jax.eval_shape(fwd_bwd, lhs, rhs_t, dout, gs)
    self.assertEqual(out_shapes[0].shape, (19, 384))
    self.assertEqual(out_shapes[1].shape, (19, 256))
    self.assertEqual(out_shapes[2].shape, (4, 384, 256))
    names, _ = _kernel_names_and_mlir(fwd_bwd, lhs, rhs_t, dout, gs)
    self.assertLen(names, 3)
    self.assertIn("-m_32-", names[0])
    self.assertIn("-tm_32-", names[0])
    self.assertTrue(names[2].endswith("-transpose_out"), names[2])

  @parameterized.parameters(
      (jnp.float8_e4m3fn, jnp.bfloat16, 128, False),
      (jnp.float8_e4m3fn, jnp.bfloat16, 128, True),
      (jnp.bfloat16, jnp.float32, 128, False),
      (jnp.bfloat16, jnp.float32, 19, True),
  )
  def test_tgmm_v2_rejects_mixed_sublane_dtypes_cleanly(self, lhs_dtype, rhs_dtype, m_rows, transpose_out):
    """Mixed-sublane tgmm_v2 operands must raise a clean AssertionError rather than crashing Mosaic."""
    lhs = jax.ShapeDtypeStruct((m_rows, 256), lhs_dtype)
    rhs = jax.ShapeDtypeStruct((m_rows, 256), rhs_dtype)
    gs = jax.ShapeDtypeStruct((4,), jnp.int32)
    with _tpu_target():
      with self.assertRaisesRegex(AssertionError, "size_lhs_sublane should be the same as size_rhs_sublane"):
        jax.eval_shape(
            functools.partial(
                tgmm_v2.tgmm_v2,
                num_actual_groups=4,
                tile_info=gmm_v2.TileSizes(128, 128, 128),
                preferred_element_type=jnp.bfloat16,
                transpose_out=transpose_out,
            ),
            lhs,
            rhs,
            gs,
        )


class RoutedMoELayerParityAndZeroDriftTest(parameterized.TestCase):
  """End-to-end RoutedMoE parity, kernel-argument routing, and zero HLO drift checks."""

  def _moe_cfg(self, num_devices: int, **overrides):
    """Initializes a compact DeepSeek3 MoE config for multi-device testing."""
    ep = 2 if num_devices >= 4 else 1
    fsdp = num_devices // ep
    base = {
        "skip_jax_distributed_system": True,
        "model_name": "deepseek3-test",
        "override_model_config": True,
        "base_emb_dim": 256,
        "base_moe_mlp_dim": 128,
        "base_mlp_dim": 256,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "per_device_batch_size": 1.0,
        "max_target_length": 64,
        "ici_expert_parallelism": ep,
        "ici_fsdp_parallelism": fsdp,
        "sparse_matmul": True,
        "megablox": True,
        "use_tokamax_gmm": True,
        "use_gmm_v2": True,
        "use_ring_of_experts": ep > 1,
        "use_ragged_sort": False,
        "wi_tile_fwd_batch_seq": 128,
        "wi_tile_fwd_embed_dim": 256,
        "wi_tile_fwd_mlp_dim": 128,
        "wi_tile_dlhs_batch_seq": 128,
        "wi_tile_dlhs_mlp_dim": 128,
        "wi_tile_dlhs_embed_dim": 256,
        "wi_tile_drhs_batch_seq": 128,
        "wi_tile_drhs_embed_dim": 256,
        "wi_tile_drhs_mlp_dim": 128,
        "wo_tile_fwd_batch_seq": 128,
        "wo_tile_fwd_mlp_dim": 128,
        "wo_tile_fwd_embed_dim": 256,
        "wo_tile_dlhs_batch_seq": 128,
        "wo_tile_dlhs_embed_dim": 256,
        "wo_tile_dlhs_mlp_dim": 128,
        "wo_tile_drhs_batch_seq": 128,
        "wo_tile_drhs_mlp_dim": 128,
        "wo_tile_drhs_embed_dim": 256,
        "dtype": "bfloat16",
        "weight_dtype": "bfloat16",
        "hardware": "tpu",
    }
    base.update(overrides)
    return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **base)

  def _build_moe(self, cfg, mesh):
    """Instantiates RoutedMoE and shards its parameters onto mesh."""
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      module = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          intermediate_dim=cfg.base_moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=None,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      gdef, params, rest = nnx.split(module, nnx.Param, ...)
      param_shardings = nnx.get_named_sharding(params, mesh)
      params = jax.tree.map(jax.device_put, params, param_shardings)
      return gdef, params, rest, param_shardings

  @parameterized.named_parameters(
      ("split_wi_buf2", {"wi_tile_fwd_rhs_buffer_count": 2, "prefuse_moe_weights": False}),
      ("prefused_wi_buf3", {"wi_tile_fwd_rhs_buffer_count": 3, "prefuse_moe_weights": True}),
      (
          "chunked_accum_wgrad_buf2",
          {
              "wi_tile_fwd_rhs_buffer_count": 2,
              "num_moe_token_chunks": 2,
              "moe_accumulate_chunk_wgrad": True,
          },
      ),
  )
  def test_routed_moe_fwd_bwd_bit_exact_and_call_routing(self, extra_overrides):
    num_dev = jax.device_count()
    cfg_ref = self._moe_cfg(
        num_dev,
        prefuse_moe_weights=extra_overrides.get("prefuse_moe_weights", False),
        num_moe_token_chunks=extra_overrides.get("num_moe_token_chunks", 1),
        moe_accumulate_chunk_wgrad=extra_overrides.get("moe_accumulate_chunk_wgrad", False),
    )
    cfg_opt = self._moe_cfg(
        num_dev,
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        **extra_overrides,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    gdef_ref, params, rest_ref, param_shardings = self._build_moe(cfg_ref, mesh)
    gdef_opt, _, rest_opt, _ = self._build_moe(cfg_opt, mesh)

    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(11), (batch, 64, 256), dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.key(12), (batch, 64, 256), dtype=jnp.bfloat16)

    def loss_fn(gdef, rest, p, inp):
      mod = nnx.merge(gdef, p, rest)
      out = mod(inp)[0]
      return jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32)), out

    recorded_gmm = []
    recorded_tgmm = []

    def rec_gmm(lhs, rhs, *args, **kwargs):
      recorded_gmm.append(
          (
              rhs.shape,
              kwargs.get("enable_subtile_m", False),
              kwargs.get("transpose_rhs", False),
              kwargs.get("rhs_buffer_count", None),
          )
      )
      return _gmm_v2_standin(lhs, rhs, *args, **kwargs)

    def rec_tgmm(lhs, rhs, *args, **kwargs):
      recorded_tgmm.append(((lhs.shape[-1], rhs.shape[-1]), kwargs.get("transpose_out", False)))
      return _tgmm_v2_standin(lhs, rhs, *args, **kwargs)

    aux_shardings = (
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None, None)),
    )

    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_tgmm_v2_standin),
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_ref.logical_axis_rules),
    ):
      (loss_ref, out_ref), grads_ref = jax.jit(
          jax.value_and_grad(lambda p: loss_fn(gdef_ref, rest_ref, p, x), has_aux=True),
          out_shardings=(aux_shardings, param_shardings),
      )(params)

    with (
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=rec_gmm),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=rec_tgmm),
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(cfg_opt.logical_axis_rules),
    ):
      (loss_opt, out_opt), grads_opt = jax.jit(
          jax.value_and_grad(lambda p: loss_fn(gdef_opt, rest_opt, p, x), has_aux=True),
          out_shardings=(aux_shardings, param_shardings),
      )(params)

    self.assertTrue(_same_bits(loss_opt, loss_ref))
    self.assertTrue(_same_bits(out_opt, out_ref))
    opt_leaves = jax.tree_util.tree_leaves_with_path(grads_opt)
    ref_leaves = jax.tree_util.tree_leaves_with_path(grads_ref)
    self.assertLen(opt_leaves, len(ref_leaves))
    for (path, g_opt_leaf), (_, g_ref_leaf) in zip(opt_leaves, ref_leaves):
      self.assertEqual(g_opt_leaf.sharding, g_ref_leaf.sharding, msg=f"sharding mismatch on {path}")
      self.assertTrue(_same_bits(g_opt_leaf, g_ref_leaf), msg=f"grad mismatch on {path}")

    # Every gmm_v2 call had enable_subtile_m=True.
    self.assertTrue(all(sub for _, sub, _, _ in recorded_gmm), recorded_gmm)
    # rhs_buffer_count was passed ONLY to wi forward calls (never wo, never DLHS).
    expected_buf = extra_overrides["wi_tile_fwd_rhs_buffer_count"]
    with_buf = [entry for entry in recorded_gmm if entry[3] is not None]
    self.assertNotEmpty(with_buf)
    self.assertTrue(all(entry[3] == expected_buf and entry[0][1] == 256 and not entry[2] for entry in with_buf))
    # Only wo's TGMM call had transpose_out=True.
    transposed_tgmms = [t for t in recorded_tgmm if t[1]]
    self.assertNotEmpty(transposed_tgmms)
    self.assertTrue(all(dims == (128, 256) for dims, _ in transposed_tgmms), recorded_tgmm)
    self.assertTrue(any(not is_trans for _, is_trans in recorded_tgmm), recorded_tgmm)

  def test_routed_moe_cpu_ragged_dot_fallback_parity(self):
    if jax.default_backend() != "cpu":
      self.skipTest("CPU ragged_dot fallback test only runs on CPU backend.")
    num_dev = jax.device_count()
    cfg_ref = self._moe_cfg(num_dev, hardware="cpu", megablox=False, use_tokamax_gmm=False, use_gmm_v2=False)
    cfg_opt = self._moe_cfg(num_dev, hardware="cpu", moe_wo_rhs_transpose=True)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)
    gdef_ref, params, rest_ref, _ = self._build_moe(cfg_ref, mesh)
    gdef_opt, _, rest_opt, _ = self._build_moe(cfg_opt, mesh)
    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(21), (batch, 64, 256), dtype=jnp.bfloat16)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      loss_ref, grads_ref = jax.jit(jax.value_and_grad(lambda p: jnp.sum(nnx.merge(gdef_ref, p, rest_ref)(x)[0])))(params)
      loss_opt, grads_opt = jax.jit(jax.value_and_grad(lambda p: jnp.sum(nnx.merge(gdef_opt, p, rest_opt)(x)[0])))(params)
    np.testing.assert_allclose(float(loss_opt), float(loss_ref), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(
        np.asarray(grads_opt["wo"][...], np.float32),
        np.asarray(grads_ref["wo"][...], np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

  def test_routed_moe_hlo_zero_drift_and_transposed_dwo_hlo(self):
    num_dev = jax.device_count()
    cfg_implicit = self._moe_cfg(num_dev)
    cfg_explicit = self._moe_cfg(
        num_dev,
        enable_subtile_m=False,
        moe_wo_rhs_transpose=False,
        wi_tile_fwd_rhs_buffer_count=None,
    )
    cfg_dwo = self._moe_cfg(
        num_dev,
        enable_subtile_m=True,
        moe_wo_rhs_transpose=True,
        wi_tile_fwd_rhs_buffer_count=2,
    )
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_implicit), cfg_implicit.mesh_axes)
    gdef_imp, params, rest_imp, param_shardings = self._build_moe(cfg_implicit, mesh)
    gdef_exp, _, rest_exp, _ = self._build_moe(cfg_explicit, mesh)
    gdef_dwo, _, rest_dwo, _ = self._build_moe(cfg_dwo, mesh)

    batch = max(1, num_dev)
    x = jax.random.normal(jax.random.key(31), (batch, 64, 256), dtype=jnp.bfloat16)
    loss_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def compile_moe_hlo(cfg, gdef, rest):
      with (
          mock.patch.object(gmm_v2, "gmm_v2", side_effect=_gmm_v2_standin),
          mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=_tgmm_v2_standin),
          jax.set_mesh(mesh),
          nn_partitioning.axis_rules(cfg.logical_axis_rules),
      ):

        def step(p, inp):
          mod = nnx.merge(gdef, p, rest)
          out = mod(inp)[0]
          return jnp.sum(out.astype(jnp.float32))

        lowered = jax.jit(jax.value_and_grad(step), out_shardings=(loss_sharding, param_shardings)).lower(params, x)
        return lowered.as_text()

    mlir_imp = compile_moe_hlo(cfg_implicit, gdef_imp, rest_imp)
    mlir_exp = compile_moe_hlo(cfg_explicit, gdef_exp, rest_exp)
    mlir_dwo = compile_moe_hlo(cfg_dwo, gdef_dwo, rest_dwo)

    def _strip_mod(s: str) -> str:
      return re.sub(r"module @jit_\w+", "module @jit_step", s)

    # Strip module name header to verify 100% zero StableHLO drift at default config.
    self.assertEqual(_strip_mod(mlir_imp), _strip_mod(mlir_exp))

    if num_dev >= 4:
      # In Shardy StableHLO, _gather_transposed_wo constrains [4, 256, 128] to [{"expert"}, {"fsdp"}, {}]
      # in both forward (before all-gather) and backward (reduce-scatter along axis 1 before shard transpose),
      # whereas baseline mlir_exp has zero transposed [{"expert"}, {"fsdp"}, {}] constraints on [4, 256, 128].
      pat = r'sdy\.sharding_constraint[^\n]*\[\{"expert"\}, \{"fsdp"\}, \{\}\][^\n]*tensor<4x256x128xbf16>'
      self.assertLen(re.findall(pat, mlir_dwo), 2, mlir_dwo)
      self.assertEmpty(re.findall(pat, mlir_exp), mlir_exp)


@pytest.mark.tpu_only
class SubtileMDynamicTpuTest(parameterized.TestCase):
  """Real TPU execution tests for gmm_v2 enable_subtile_m=True across skewed/edge group distributions."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU backend.")

  @parameterized.named_parameters(
      ("skewed_all_counts_tm128_fp32", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 320, 128, jnp.float32),
      ("skewed_all_counts_tm128_bf16", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 320, 128, jnp.bfloat16),
      ("skewed_all_counts_tm512_fp32", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 520, 512, jnp.float32),
      ("skewed_all_counts_tm512_bf16", [0, 1, 7, 8, 9, 15, 16, 31, 33, 64, 129], 520, 512, jnp.bfloat16),
      ("all_zero_groups", [0, 0, 0, 0], 128, 128, jnp.float32),
      ("last_expert_unaligned_m", [0, 0, 0, 15], 19, 64, jnp.float32),
      ("tail_groups_unaligned_m", [64, 0, 7, 1], 77, 32, jnp.float32),
      ("tiny_tm16_sub8", [0, 7, 8, 9, 16], 45, 16, jnp.float32),
      ("buckets_64_256", [8, 16, 32, 64, 128, 200], 460, 256, jnp.float32),
  )
  def test_subtile_m_skewed_groups_and_accumulators(self, group_sizes_list, m_rows, tile_m, acc_dtype):
    _poison_tpu_memory()
    num_groups = len(group_sizes_list)
    k0, k1 = jax.random.split(jax.random.key(101))
    lhs = jax.random.normal(k0, (m_rows, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (num_groups, 256, 256), jnp.float32) * 0.1).astype(jnp.bfloat16)
    gs = jnp.array(group_sizes_list, dtype=jnp.int32)

    out_sub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(tile_m, 128, 128),
        preferred_element_type=jnp.bfloat16,
        acc_dtype=acc_dtype,
        enable_subtile_m=True,
    )
    out_nosub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(tile_m, 128, 128),
        preferred_element_type=jnp.bfloat16,
        acc_dtype=acc_dtype,
        enable_subtile_m=False,
    )
    ref = _reference_ragged_dot(lhs, rhs, gs, out_dtype=jnp.bfloat16)
    self.assertFalse(bool(jnp.any(jnp.isnan(out_sub))))
    atol = 3e-2 if acc_dtype == jnp.bfloat16 else 1e-2
    rtol = 3e-2 if acc_dtype == jnp.bfloat16 else 1e-2
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(ref, np.float32), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(out_nosub, np.float32), rtol=rtol, atol=atol)

  def test_subtile_m_group_offset_partial_k_bias_partial_sum_and_silu(self):
    _poison_tpu_memory()
    k0, k1, k2, k3 = jax.random.split(jax.random.key(202), 4)
    # 6 global groups, local shard handles 4 experts starting at group_offset=1; size_k=384 with tile_k=256 (valid_k=128).
    gs = jnp.array([19, 0, 7, 15, 65, 11], dtype=jnp.int32)
    m_rows = 140
    lhs = jax.random.normal(k0, (m_rows, 384), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (4, 384, 256), jnp.float32) * 0.1).astype(jnp.bfloat16)
    bias = (jax.random.normal(k2, (4, 1, 256), jnp.float32) * 0.05).astype(jnp.bfloat16)
    ps = (jax.random.normal(k3, (m_rows, 256), jnp.float32) * 0.1).astype(jnp.bfloat16)
    offset = jnp.array([1], dtype=jnp.int32)

    out_sub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        rhs_bias=bias,
        partial_sum=ps,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.bfloat16,
        zero_initialize=False,
        enable_subtile_m=True,
    )
    out_nosub = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        rhs_bias=bias,
        partial_sum=ps,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.bfloat16,
        zero_initialize=False,
        enable_subtile_m=False,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(out_sub))))
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(out_nosub, np.float32), rtol=1e-2, atol=1e-2)

    # Fused SiLU with unaligned M (m_rows=140):
    rhs_fused = (jax.random.normal(k1, (4, 256, 512), jnp.float32) * 0.1).astype(jnp.bfloat16)
    lhs_256 = lhs[:, :256]
    act_sub = gmm_v2.gmm_v2(
        lhs_256,
        rhs_fused,
        gs,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.bfloat16,
        fuse_act="silu",
        enable_subtile_m=True,
    )
    act_ref = gmm_v2.gmm_v2(
        lhs_256,
        rhs_fused,
        gs,
        group_offset=offset,
        tile_info=gmm_v2.TileSizes(128, 256, 128),
        preferred_element_type=jnp.bfloat16,
        fuse_act="silu",
        enable_subtile_m=False,
    )
    self.assertEqual(act_sub.shape, (m_rows, 256))
    np.testing.assert_allclose(np.asarray(act_sub, np.float32), np.asarray(act_ref, np.float32), rtol=1e-2, atol=1e-2)

  @parameterized.parameters(jnp.float8_e4m3fn, jnp.int8)
  def test_subtile_m_quantized_fp8_and_int8(self, qdtype):
    _poison_tpu_memory()
    k0, k1 = jax.random.split(jax.random.key(303))
    gs = jnp.array([0, 7, 15, 33, 64], dtype=jnp.int32)
    lhs = jax.random.normal(k0, (128, 256), jnp.float32).astype(jnp.bfloat16)
    rhs_f = jax.random.normal(k1, (5, 256, 256), jnp.float32) * 0.1
    rhs_q, rhs_scale = _quantize_blockwise(rhs_f, qdtype, block_size=128)

    out_sub = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        gs,
        rhs_scale=rhs_scale,
        tile_info=gmm_v2.TileSizes(128, 128, 128),
        preferred_element_type=jnp.bfloat16,
        enable_subtile_m=True,
    )
    out_nosub = gmm_v2.gmm_v2(
        lhs,
        rhs_q,
        gs,
        rhs_scale=rhs_scale,
        tile_info=gmm_v2.TileSizes(128, 128, 128),
        preferred_element_type=jnp.bfloat16,
        enable_subtile_m=False,
    )
    self.assertFalse(bool(jnp.any(jnp.isnan(out_sub))))
    np.testing.assert_allclose(np.asarray(out_sub, np.float32), np.asarray(out_nosub, np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.tpu_only
class TransposedDwoAndRhsBufferTpuTest(parameterized.TestCase):
  """Real TPU execution tests for tgmm_v2(transpose_out=True), ops.gmm(rhs_transpose=True), and RoutedMoE."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU backend.")

  def test_tgmm_v2_transpose_out_exact_parity_with_scale_and_partial_sum(self):
    _poison_tpu_memory()
    k0, k1, k2, k3 = jax.random.split(jax.random.key(404), 4)
    gs = jnp.array([0, 1, 7, 15, 33, 129], dtype=jnp.int32)
    m_rows = 200
    lhs = jax.random.normal(k0, (m_rows, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = jax.random.normal(k1, (m_rows, 384), jnp.float32).astype(jnp.bfloat16)
    scale = jnp.abs(jax.random.normal(k2, (1, 1, 384), jnp.float32)) * 0.1 + 0.05
    ps = (jax.random.normal(k3, (6, 256, 384), jnp.float32) * 0.05).astype(jnp.bfloat16)

    for use_scale in (False, True):
      for use_ps in (False, True):
        s_arg = scale if use_scale else None
        ps_arg = ps if use_ps else None
        ps_t_arg = jnp.swapaxes(ps, 1, 2) if use_ps else None
        ref = tgmm_v2.tgmm_v2(
            lhs,
            rhs,
            gs,
            6,
            rhs_scale=s_arg,
            partial_sum=ps_arg,
            tile_info=gmm_v2.TileSizes(128, 128, 256),
            preferred_element_type=jnp.bfloat16,
            transpose_out=False,
        )
        got_t = tgmm_v2.tgmm_v2(
            lhs,
            rhs,
            gs,
            6,
            rhs_scale=s_arg,
            partial_sum=ps_t_arg,
            tile_info=gmm_v2.TileSizes(128, 128, 256),
            preferred_element_type=jnp.bfloat16,
            transpose_out=True,
        )
        self.assertEqual(got_t.shape, (6, 384, 256))
        np.testing.assert_allclose(
            np.asarray(got_t, np.float32),
            np.asarray(jnp.swapaxes(ref, 1, 2), np.float32),
            rtol=1e-2,
            atol=1e-2,
        )

  @parameterized.parameters(False, True)
  def test_ops_gmm_fwd_bwd_subtile_rhs_transpose_and_buffers_on_tpu(self, quantized):
    _poison_tpu_memory()
    k0, k1, k2 = jax.random.split(jax.random.key(505), 3)
    gs = jnp.array([0, 1, 7, 8, 15, 31, 33, 129], dtype=jnp.int32)
    lhs = jax.random.normal(k0, (256, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (8, 256, 384), jnp.float32) * 0.05).astype(jnp.bfloat16)
    rhs_t = jnp.swapaxes(rhs, 1, 2)
    dout = jax.random.normal(k2, (256, 384), jnp.float32).astype(jnp.bfloat16)
    tiling = (128, 256, 128, 128, 128, 256, 128, 256, 128)
    qrule = (
        qwix.QtRule(
            weight_qtype=jnp.float8_e4m3fn,
            weight_calibration_method="absmax",
            act_qtype=jnp.float8_e4m3fn,
            act_calibration_method="absmax",
            bwd_qtype=jnp.float8_e5m2,
            bwd_calibration_method="absmax",
        )
        if quantized
        else None
    )

    def loss_base(x, w):
      y = megablox_ops.gmm(
          x,
          w,
          gs,
          preferred_element_type=jnp.bfloat16,
          tiling=tiling,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          use_qwix_quantization=quantized,
          qwix_rule=qrule,
          use_dlhs_transpose_rhs=True,
      )
      return jnp.sum(y.astype(jnp.float32) * dout.astype(jnp.float32)), y

    def loss_opt(x, w_t):
      y = megablox_ops.gmm(
          x,
          w_t,
          gs,
          preferred_element_type=jnp.bfloat16,
          tiling=tiling,
          use_tokamax_backend=True,
          use_gmm_v2=True,
          use_qwix_quantization=quantized,
          qwix_rule=qrule,
          use_dlhs_transpose_rhs=True,
          enable_subtile_m=True,
          rhs_transpose=True,
          fwd_rhs_buffer_count=2,
      )
      return jnp.sum(y.astype(jnp.float32) * dout.astype(jnp.float32)), y

    (_, y_base), (dlhs_base, drhs_base) = jax.jit(jax.value_and_grad(loss_base, argnums=(0, 1), has_aux=True))(lhs, rhs)
    (_, y_opt), (dlhs_opt, drhs_opt_t) = jax.jit(jax.value_and_grad(loss_opt, argnums=(0, 1), has_aux=True))(lhs, rhs_t)

    tol = 5e-2 if quantized else 1e-2
    np.testing.assert_allclose(np.asarray(y_opt, np.float32), np.asarray(y_base, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(dlhs_opt, np.float32), np.asarray(dlhs_base, np.float32), rtol=tol, atol=tol)
    np.testing.assert_allclose(
        np.asarray(jnp.swapaxes(drhs_opt_t, 1, 2), np.float32),
        np.asarray(drhs_base, np.float32),
        rtol=tol,
        atol=tol,
    )

  @parameterized.parameters(2, 3)
  def test_rhs_buffer_count_bit_exact_on_tpu(self, buf_count):
    k0, k1 = jax.random.split(jax.random.key(606))
    gs = jnp.array([31, 0, 129, 64], dtype=jnp.int32)
    lhs = jax.random.normal(k0, (256, 256), jnp.float32).astype(jnp.bfloat16)
    rhs = (jax.random.normal(k1, (4, 256, 512), jnp.float32) * 0.05).astype(jnp.bfloat16)
    ref = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(128, 256, 256),
        preferred_element_type=jnp.bfloat16,
    )
    got = gmm_v2.gmm_v2(
        lhs,
        rhs,
        gs,
        tile_info=gmm_v2.TileSizes(128, 256, 256),
        preferred_element_type=jnp.bfloat16,
        rhs_buffer_count=buf_count,
    )
    self.assertTrue(_same_bits(got, ref))

  def test_tgmm_v2_rejects_mixed_sublane_on_tpu(self):
    gs = jnp.array([32, 32, 32, 32], dtype=jnp.int32)
    lhs_fp8 = jnp.zeros((128, 256), dtype=jnp.float8_e4m3fn)
    rhs_bf16 = jnp.zeros((128, 256), dtype=jnp.bfloat16)
    rhs_f32 = jnp.zeros((128, 256), dtype=jnp.float32)
    for l_arr, r_arr, t_out in ((lhs_fp8, rhs_bf16, False), (lhs_fp8, rhs_bf16, True), (rhs_bf16, rhs_f32, False)):
      with self.assertRaisesRegex(AssertionError, "size_lhs_sublane should be the same as size_rhs_sublane"):
        tgmm_v2.tgmm_v2(
            l_arr,
            r_arr,
            gs,
            4,
            tile_info=gmm_v2.TileSizes(128, 128, 128),
            preferred_element_type=jnp.bfloat16,
            transpose_out=t_out,
        )

  def test_routed_moe_multi_device_tpu_parity_and_compiled_hlo(self):
    num_dev = jax.device_count()
    ep = 2 if num_dev >= 4 else 1
    fsdp = num_dev // ep

    def make_tpu_cfg(**extra):
      kw = {
          "skip_jax_distributed_system": True,
          "model_name": "deepseek3-test",
          "override_model_config": True,
          "base_emb_dim": 2048,
          "base_moe_mlp_dim": 1024,
          "base_mlp_dim": 2048,
          "num_experts": 16,
          "num_experts_per_tok": 2,
          "per_device_batch_size": 1.0,
          "max_target_length": 128,
          "ici_expert_parallelism": ep,
          "ici_fsdp_parallelism": fsdp,
          "sparse_matmul": True,
          "megablox": True,
          "use_tokamax_gmm": True,
          "use_gmm_v2": True,
          "use_ring_of_experts": ep > 1,
          "use_ragged_sort": ep > 1,
          "wi_tile_fwd_batch_seq": 256,
          "wi_tile_fwd_embed_dim": 2048,
          "wi_tile_fwd_mlp_dim": 1024,
          "wi_tile_dlhs_batch_seq": 256,
          "wi_tile_dlhs_mlp_dim": 1024,
          "wi_tile_dlhs_embed_dim": 2048,
          "wi_tile_drhs_batch_seq": 256,
          "wi_tile_drhs_embed_dim": 2048,
          "wi_tile_drhs_mlp_dim": 1024,
          "wo_tile_fwd_batch_seq": 256,
          "wo_tile_fwd_mlp_dim": 1024,
          "wo_tile_fwd_embed_dim": 2048,
          "wo_tile_dlhs_batch_seq": 256,
          "wo_tile_dlhs_embed_dim": 2048,
          "wo_tile_dlhs_mlp_dim": 1024,
          "wo_tile_drhs_batch_seq": 256,
          "wo_tile_drhs_mlp_dim": 1024,
          "wo_tile_drhs_embed_dim": 2048,
          "dtype": "bfloat16",
          "weight_dtype": "bfloat16",
          "hardware": "tpu",
      }
      kw.update(extra)
      return pyconfig.initialize([os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()], **kw)

    cfg_ref = make_tpu_cfg()
    cfg_opt2 = make_tpu_cfg(enable_subtile_m=True, moe_wo_rhs_transpose=True, wi_tile_fwd_rhs_buffer_count=2)
    cfg_opt3 = make_tpu_cfg(enable_subtile_m=True, moe_wo_rhs_transpose=True, wi_tile_fwd_rhs_buffer_count=3)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg_ref), cfg_ref.mesh_axes)

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      m_ref = moe.RoutedMoE(
          config=cfg_ref,
          num_experts=cfg_ref.num_experts,
          num_experts_per_tok=cfg_ref.num_experts_per_tok,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", None),
          intermediate_dim=cfg_ref.base_moe_mlp_dim,
          dtype=cfg_ref.dtype,
          weight_dtype=cfg_ref.weight_dtype,
          quant=None,
          rngs=nnx.Rngs(params=0, dropout=1),
      )
      gdef_ref, params, rest_ref = nnx.split(m_ref, nnx.Param, ...)
      param_sharding = nnx.get_named_sharding(params, mesh)
      params = jax.tree.map(jax.device_put, params, param_sharding)
      gdef_opt2, _, rest_opt2 = nnx.split(
          moe.RoutedMoE(
              config=cfg_opt2,
              num_experts=cfg_opt2.num_experts,
              num_experts_per_tok=cfg_opt2.num_experts_per_tok,
              mesh=mesh,
              kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
              kernel_axes=("embed", None),
              intermediate_dim=cfg_opt2.base_moe_mlp_dim,
              dtype=cfg_opt2.dtype,
              weight_dtype=cfg_opt2.weight_dtype,
              quant=None,
              rngs=nnx.Rngs(params=0, dropout=1),
          ),
          nnx.Param,
          ...,
      )
      gdef_opt3, _, rest_opt3 = nnx.split(
          moe.RoutedMoE(
              config=cfg_opt3,
              num_experts=cfg_opt3.num_experts,
              num_experts_per_tok=cfg_opt3.num_experts_per_tok,
              mesh=mesh,
              kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
              kernel_axes=("embed", None),
              intermediate_dim=cfg_opt3.base_moe_mlp_dim,
              dtype=cfg_opt3.dtype,
              weight_dtype=cfg_opt3.weight_dtype,
              quant=None,
              rngs=nnx.Rngs(params=0, dropout=1),
          ),
          nnx.Param,
          ...,
      )

    x_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None, None))
    x = jax.device_put(jax.random.normal(jax.random.key(707), (num_dev, 128, 2048), dtype=jnp.bfloat16), x_sharding)
    dy = jax.device_put(jax.random.normal(jax.random.key(708), (num_dev, 128, 2048), dtype=jnp.bfloat16), x_sharding)
    aux_sharding = (jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), x_sharding)

    def make_step(gdef, rest):
      def step(p, inp, grad_out):
        def loss(weights):
          mod = nnx.merge(gdef, weights, rest)
          out = mod(inp)[0]
          return jnp.sum(out.astype(jnp.float32) * grad_out.astype(jnp.float32)), out

        return jax.value_and_grad(loss, has_aux=True)(p)

      return jax.jit(step, out_shardings=(aux_sharding, param_sharding))

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg_ref.logical_axis_rules):
      step_ref = make_step(gdef_ref, rest_ref)
      step_opt2 = make_step(gdef_opt2, rest_opt2)
      step_opt3 = make_step(gdef_opt3, rest_opt3)
      (_, out_ref), grads_ref = step_ref(params, x, dy)
      (_, out_opt2), grads_opt2 = step_opt2(params, x, dy)
      (_, out_opt3), grads_opt3 = step_opt3(params, x, dy)

      if fsdp > 1:
        hlo_ref = step_ref.lower(params, x, dy).compile().as_text()
        hlo_opt2 = step_opt2.lower(params, x, dy).compile().as_text()
        # Match actual full-shape HLO transpose / physical-relayout copy opcodes
        # (`= bf16[E_local, ...]{layout} (transpose|copy)(` — XLA canonicalizes 3D layout
        # swaps to physical-layout `copy(` + `bitcast(`), never metadata strings.
        e_local = cfg_ref.num_experts // ep
        full_wo_relayout_pat = rf"=\s*bf16\[{e_local},(?:1024,2048|2048,1024)\]\S*\s+(?:transpose|copy)\("
        ref_count = len(re.findall(full_wo_relayout_pat, hlo_ref))
        opt2_count = len(re.findall(full_wo_relayout_pat, hlo_opt2))
        self.assertGreater(ref_count, 0)
        self.assertLess(opt2_count, ref_count)
        self.assertEqual(opt2_count, 0)
        minor_rs_pat = r"reduce-scatter(?:-start)?\([^\n]*dimensions=\{2\}"
        self.assertGreater(len(re.findall(minor_rs_pat, hlo_ref)), 0)
        self.assertEqual(len(re.findall(minor_rs_pat, hlo_opt2)), 0)
        self.assertNotIn("-transpose_out", hlo_ref)
        self.assertIn("-transpose_out", hlo_opt2)

    np.testing.assert_allclose(np.asarray(out_opt2, np.float32), np.asarray(out_ref, np.float32), rtol=2e-2, atol=2e-2)
    self.assertTrue(_same_bits(out_opt3, out_opt2))
    for k in ("gate", "wi_0", "wi_1", "wo"):
      g_opt2_leaf = jax.tree_util.tree_leaves(grads_opt2[k])[0]
      g_opt3_leaf = jax.tree_util.tree_leaves(grads_opt3[k])[0]
      g_ref_leaf = jax.tree_util.tree_leaves(grads_ref[k])[0]
      np.testing.assert_allclose(
          np.asarray(g_opt2_leaf, np.float32),
          np.asarray(g_ref_leaf, np.float32),
          rtol=2e-2,
          atol=2e-2,
      )
      self.assertTrue(_same_bits(g_opt3_leaf, g_opt2_leaf))


if __name__ == "__main__":
  absltest.main()
