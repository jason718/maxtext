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

"""Unit and 4-CPU multi-device tests for `forced_routed_experts` router replay in `maxtext.m3.core.moe`."""

import functools
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
import numpy as np

from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
# pylint: disable=too-many-positional-arguments
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core import moe as m3_moe
from maxtext.m3.core.sharding import DefaultSharding, create_mesh


def _make_expert_weights(
    num_experts: int = 4,
    embed: int = 16,
    mlp: int = 12,
    *,
    dtype=jnp.float32,
    seed: int = 10,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Builds deterministic `(wi_0, wi_1, wo)` expert weight tensors."""
  k0, k1, k2 = jax.random.split(jax.random.key(seed), 3)
  scale = 0.15
  wi_0 = (jax.random.normal(k0, (num_experts, embed, mlp), dtype=jnp.float32) * scale).astype(dtype)
  wi_1 = (jax.random.normal(k1, (num_experts, embed, mlp), dtype=jnp.float32) * scale).astype(dtype)
  wo = (jax.random.normal(k2, (num_experts, mlp, embed), dtype=jnp.float32) * scale).astype(dtype)
  return wi_0, wi_1, wo


def _cpu_ragged_all_to_all_ref(
    operand: jax.Array,
    output: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    axis_name: str,
) -> jax.Array:
  """Pure-JAX collective reference matching `jax.lax.ragged_all_to_all` semantics on CPU."""
  del output_offsets
  ep_size = send_sizes.shape[0]
  in_len = operand.shape[0]
  out_len = output.shape[0]
  cap = max(1, min(int(in_len), int(out_len)))
  col = jnp.arange(cap, dtype=jnp.int32)[None, :]
  gather_idx = input_offsets[:, None] + col
  valid_send = col < send_sizes[:, None]
  safe_gather = jnp.clip(gather_idx, 0, max(int(in_len) - 1, 0))
  staged = jnp.where(valid_send[:, :, None], operand[safe_gather], jnp.zeros((), dtype=operand.dtype))
  recvd = jax.lax.all_to_all(staged, axis_name, split_axis=0, concat_axis=0)

  recv_ends = jnp.cumsum(recv_sizes)
  recv_starts = recv_ends - recv_sizes
  row_idx = jnp.arange(out_len, dtype=jnp.int32)
  src_idx = jnp.searchsorted(recv_ends, row_idx, side="right")
  valid_row = row_idx < recv_ends[-1]
  safe_src = jnp.minimum(src_idx, ep_size - 1)
  pos_in_src = jnp.clip(row_idx - recv_starts[safe_src], 0, cap - 1)
  return jnp.where(valid_row[:, None], recvd[safe_src, pos_in_src], output)


class NoneReplayZeroOverheadTest(parameterized.TestCase):
  """Verifies `forced_routed_experts=None` is a zero-overhead no-op with identical HLO."""

  @parameterized.named_parameters(
      ("dense", "dense"),
      ("ragged_dot", "ragged_dot"),
      ("tokamax", "tokamax"),
  )
  def test_none_forced_routed_experts_matches_base_hlo_and_grads(self, kind):
    mesh = create_mesh(
        ici={"fsdp": 1, "expert": 2, "tensor": 2} if jax.device_count() >= 4 else None,
        devices=jax.devices()[:4] if jax.device_count() >= 4 else jax.devices()[:1],
    )
    x = jax.random.normal(jax.random.key(0), (4, 4, 16), dtype=jnp.float32)
    logits = jax.random.normal(jax.random.key(1), (4, 4, 4), dtype=jnp.float32)
    weights, indices, _ = m3_moe.route_top_k(logits, 2, scoring="softmax", norm_topk_prob=True)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12)
    w_gu = jnp.stack([wi_0, wi_1], axis=2) if kind == "tokamax" else jnp.concatenate([wi_0, wi_1], axis=-1)

    executor = m3_moe.MoEExecutor(
        kind=kind,
        batch=("data", "fsdp") if kind == "tokamax" else None,
        experts="expert" if kind == "tokamax" else None,
        mlp="tensor" if kind == "tokamax" else None,
        length="context" if kind == "tokamax" else None,
    )

    with jax.set_mesh(mesh):
      fn_default = jax.jit(executor)
      fn_none = jax.jit(functools.partial(executor, forced_routed_experts=None))
      hlo_default = fn_default.lower(x, weights, indices, wi_0, wi_1, wo).as_text("hlo")
      hlo_none = fn_none.lower(x, weights, indices, wi_0, wi_1, wo).as_text("hlo")
      # Strip module header name before comparing lowered HLO instructions.
      body_default = "\n".join(hlo_default.splitlines()[1:])
      body_none = "\n".join(hlo_none.splitlines()[1:])
      self.assertEqual(body_default, body_none)

      out_default = fn_default(x, weights, indices, wi_0, wi_1, wo)
      out_none = fn_none(x, weights, indices, wi_0, wi_1, wo)
      self.assertEqual(float(jnp.max(jnp.abs(out_default - out_none))), 0.0)

      f_default = executor.fused(x, weights, indices, w_gu, wo)
      f_none = executor.fused(x, weights, indices, w_gu, wo, forced_routed_experts=None)
      self.assertEqual(float(jnp.max(jnp.abs(f_default - f_none))), 0.0)


class NaturalTopKExactEquivalenceTest(parameterized.TestCase):
  """Verifies 0.0 max abs diff forward & backward when `forced_routed_experts == natural_top_k`."""

  @parameterized.named_parameters(
      ("softmax_unnorm", "softmax", False, 1.0, False, 0.0),
      ("softmax_norm", "softmax", True, 1.5, False, 0.0),
      ("topk_softmax_unnorm", "topk_softmax", False, 1.0, False, 0.0),
      ("topk_softmax_norm", "topk_softmax", True, 1.25, False, 0.0),
      ("sigmoid_norm_bias_groups", "sigmoid", True, 2.5, True, 1e-20),
      ("sigmoid_unnorm_plain", "sigmoid", False, 1.0, False, 0.0),
  )
  def test_route_top_k_natural_replay_exact_zero_diff_fwd_and_bwd(
      self, scoring, norm_topk_prob, scale, use_bias_groups, eps
  ):
    k0, k1, k2 = jax.random.split(jax.random.key(42), 3)
    num_experts = 8
    top_k = 2
    logits = jax.random.normal(k0, (3, 4, num_experts), dtype=jnp.float32)
    # Add small spacing so top-k order is strictly tie-free.
    logits = logits + jnp.arange(num_experts, dtype=jnp.float32) * 1e-2
    bias = jax.random.normal(k1, (num_experts,), dtype=jnp.float32) * 0.1 if use_bias_groups else None
    num_group = 4 if use_bias_groups else 1
    topk_group = 2 if use_bias_groups else 1
    cot_w = jax.random.normal(k2, (3, 4, top_k), dtype=jnp.float32)

    nat_w, nat_idx, nat_probs = m3_moe.route_top_k(
        logits,
        top_k,
        scoring=scoring,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=scale,
        e_score_correction_bias=bias,
        num_group=num_group,
        topk_group=topk_group,
        eps=eps,
        forced_routed_experts=None,
    )
    rep_w, rep_idx, rep_probs = m3_moe.route_top_k(
        logits,
        top_k,
        scoring=scoring,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=scale,
        e_score_correction_bias=bias,
        num_group=num_group,
        topk_group=topk_group,
        eps=eps,
        forced_routed_experts=nat_idx,
    )

    self.assertEqual(float(jnp.max(jnp.abs(rep_w - nat_w))), 0.0)
    self.assertTrue(bool(jnp.array_equal(rep_idx, nat_idx)))
    self.assertEqual(float(jnp.max(jnp.abs(rep_probs - nat_probs))), 0.0)

    def loss_nat(l_in):
      w, _, _ = m3_moe.route_top_k(
          l_in,
          top_k,
          scoring=scoring,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=scale,
          e_score_correction_bias=bias,
          num_group=num_group,
          topk_group=topk_group,
          eps=eps,
          forced_routed_experts=None,
      )
      return jnp.sum(w * cot_w)

    def loss_rep(l_in):
      w, _, _ = m3_moe.route_top_k(
          l_in,
          top_k,
          scoring=scoring,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=scale,
          e_score_correction_bias=bias,
          num_group=num_group,
          topk_group=topk_group,
          eps=eps,
          forced_routed_experts=nat_idx,
      )
      return jnp.sum(w * cot_w)

    grad_nat = jax.jit(jax.grad(loss_nat))(logits)
    grad_rep = jax.jit(jax.grad(loss_rep))(logits)
    self.assertEqual(float(jnp.max(jnp.abs(grad_rep - grad_nat))), 0.0)

  @parameterized.named_parameters(
      ("dense_sep", "dense", False),
      ("dense_fused3d", "dense", True),
      ("ragged_dot_sep", "ragged_dot", False),
      ("ragged_dot_fused4d", "ragged_dot", True),
      ("tokamax_ep1_sep", "tokamax_ep1", False),
      ("tokamax_ep1_fused3d", "tokamax_ep1", True),
      ("tokamax_ep2_tp2_sep", "tokamax_ep2", False),
      ("tokamax_ep2_tp2_fused4d", "tokamax_ep2", True),
      ("tokamax_ep4_overflow_sep", "tokamax_ep4_ov", False),
      ("tokamax_ep4_overflow_fused3d", "tokamax_ep4_ov", True),
  )
  def test_executor_natural_top_k_exact_zero_diff_fwd_and_bwd(self, mode, fused):
    if mode in ("tokamax_ep2", "tokamax_ep4_ov") and jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 CPU devices; got {jax.device_count()}.")

    num_experts = 8
    top_k = 2
    batch, length, embed, mlp = 4, 4, 16, 12
    k0, k1, k2 = jax.random.split(jax.random.key(101), 3)
    x = jax.random.normal(k0, (batch, length, embed), dtype=jnp.float32)
    logits = jax.random.normal(k1, (batch, length, num_experts), dtype=jnp.float32)
    logits = logits + jnp.arange(num_experts, dtype=jnp.float32) * 1e-2
    cotangent = jax.random.normal(k2, x.shape, dtype=jnp.float32)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=num_experts, embed=embed, mlp=mlp, seed=17)
    w_gu = (
        jnp.stack([wi_0, wi_1], axis=2)
        if mode in ("ragged_dot", "tokamax_ep2")
        else jnp.concatenate([wi_0, wi_1], axis=-1)
    )

    if mode == "dense":
      mesh = create_mesh(devices=jax.devices()[:1])
      executor = m3_moe.DENSE
    elif mode == "ragged_dot":
      mesh = create_mesh(devices=jax.devices()[:1])
      executor = m3_moe.RAGGED_DOT
    elif mode == "tokamax_ep1":
      mesh = create_mesh(devices=jax.devices()[:1])
      executor = m3_moe.MoEExecutor(kind="tokamax", bwd_wgrad_overlap=True)
    elif mode == "tokamax_ep2":
      mesh = create_mesh(ici={"fsdp": 1, "expert": 2, "tensor": 2}, devices=jax.devices()[:4])
      executor = m3_moe.MoEExecutor(
          kind="tokamax",
          batch=("data", "fsdp"),
          experts="expert",
          mlp="tensor",
          length="context",
          bwd_wgrad_overlap=True,
      )
    else:
      mesh = create_mesh(ici={"fsdp": 1, "expert": 4, "tensor": 1}, devices=jax.devices()[:4])
      executor = m3_moe.MoEExecutor(
          kind="tokamax",
          batch=("data", "fsdp", "expert"),
          experts="expert",
          mlp="tensor",
          dropless_buffer_factor=1.25,
          bwd_wgrad_overlap=True,
      )

    nat_w, nat_idx, _ = m3_moe.route_top_k(
        logits, top_k, scoring="softmax", norm_topk_prob=True, routed_scaling_factor=1.5
    )

    with jax.set_mesh(mesh):
      if not fused:

        def nat_fwd(x_in, l_in, w0, w1, w_out):
          w, idx, _ = m3_moe.route_top_k(l_in, top_k, scoring="softmax", norm_topk_prob=True, routed_scaling_factor=1.5)
          return executor(x_in, w, idx, w0, w1, w_out)

        def rep_router_fwd(x_in, l_in, fi_in, w0, w1, w_out):
          w, idx, _ = m3_moe.route_top_k(
              l_in,
              top_k,
              scoring="softmax",
              norm_topk_prob=True,
              routed_scaling_factor=1.5,
              forced_routed_experts=fi_in,
          )
          return executor(x_in, w, idx, w0, w1, w_out)

        def rep_exec_fwd(x_in, l_in, fi_in, w0, w1, w_out):
          probs = jax.nn.softmax(l_in, axis=-1)
          return executor(
              x_in,
              nat_w,
              fi_in,
              w0,
              w1,
              w_out,
              forced_routed_experts=fi_in,
              router_probs=probs,
              norm_topk_prob=True,
              routed_scaling_factor=1.5,
          )

        out_nat = jax.jit(nat_fwd)(x, logits, wi_0, wi_1, wo)
        out_rep_r = jax.jit(rep_router_fwd)(x, logits, nat_idx, wi_0, wi_1, wo)
        out_rep_e = jax.jit(rep_exec_fwd)(x, logits, nat_idx, wi_0, wi_1, wo)
        grads_nat = jax.jit(
            jax.grad(
                lambda x_in, l_in, w0, w1, w_out: jnp.sum(nat_fwd(x_in, l_in, w0, w1, w_out) * cotangent),
                argnums=(0, 1, 2, 3, 4),
            )
        )(x, logits, wi_0, wi_1, wo)
        grads_rep_r = jax.jit(
            jax.grad(
                lambda x_in, l_in, fi_in, w0, w1, w_out: jnp.sum(
                    rep_router_fwd(x_in, l_in, fi_in, w0, w1, w_out) * cotangent
                ),
                argnums=(0, 1, 3, 4, 5),
            )
        )(x, logits, nat_idx, wi_0, wi_1, wo)
        grads_rep_e = jax.jit(
            jax.grad(
                lambda x_in, l_in, fi_in, w0, w1, w_out: jnp.sum(
                    rep_exec_fwd(x_in, l_in, fi_in, w0, w1, w_out) * cotangent
                ),
                argnums=(0, 1, 3, 4, 5),
            )
        )(x, logits, nat_idx, wi_0, wi_1, wo)
      else:

        def nat_fwd_f(x_in, l_in, gu, dn):
          w, idx, _ = m3_moe.route_top_k(l_in, top_k, scoring="softmax", norm_topk_prob=True, routed_scaling_factor=1.5)
          return executor.fused(x_in, w, idx, gu, dn)

        def rep_router_fwd_f(x_in, l_in, fi_in, gu, dn):
          w, idx, _ = m3_moe.route_top_k(
              l_in,
              top_k,
              scoring="softmax",
              norm_topk_prob=True,
              routed_scaling_factor=1.5,
              forced_routed_experts=fi_in,
          )
          return executor.fused(x_in, w, idx, gu, dn)

        def rep_exec_fwd_f(x_in, l_in, fi_in, gu, dn):
          probs = jax.nn.softmax(l_in, axis=-1)
          return executor.fused(
              x_in,
              probs,
              None,
              gu,
              dn,
              forced_routed_experts=fi_in,
              norm_topk_prob=True,
              routed_scaling_factor=1.5,
          )

        out_nat = jax.jit(nat_fwd_f)(x, logits, w_gu, wo)
        out_rep_r = jax.jit(rep_router_fwd_f)(x, logits, nat_idx, w_gu, wo)
        out_rep_e = jax.jit(rep_exec_fwd_f)(x, logits, nat_idx, w_gu, wo)
        grads_nat = jax.jit(
            jax.grad(lambda x_in, l_in, gu, dn: jnp.sum(nat_fwd_f(x_in, l_in, gu, dn) * cotangent), argnums=(0, 1, 2, 3))
        )(x, logits, w_gu, wo)
        grads_rep_r = jax.jit(
            jax.grad(
                lambda x_in, l_in, fi_in, gu, dn: jnp.sum(rep_router_fwd_f(x_in, l_in, fi_in, gu, dn) * cotangent),
                argnums=(0, 1, 3, 4),
            )
        )(x, logits, nat_idx, w_gu, wo)
        grads_rep_e = jax.jit(
            jax.grad(
                lambda x_in, l_in, fi_in, gu, dn: jnp.sum(rep_exec_fwd_f(x_in, l_in, fi_in, gu, dn) * cotangent),
                argnums=(0, 1, 3, 4),
            )
        )(x, logits, nat_idx, w_gu, wo)

      self.assertEqual(float(jnp.max(jnp.abs(out_rep_r - out_nat))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(out_rep_e - out_nat))), 0.0)
      for g_r, g_e, g_n in zip(grads_rep_r, grads_rep_e, grads_nat):
        self.assertEqual(float(jnp.max(jnp.abs(g_r - g_n))), 0.0)
        # Multi-device mesh parallel reductions introduce up to 1 ULP (~6e-8) float32 roundoff.
        self.assertLessEqual(float(jnp.max(jnp.abs(g_e - g_n))), 1e-6)

  def test_explicit_mesh_and_sequence_sharding_partition_forced_routed_experts(self):
    if jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 CPU devices; got {jax.device_count()}.")
    ici = {"context": 2, "expert": 2}
    mesh_exp = create_mesh(ici=ici, devices=jax.devices()[:4], explicit=True)
    sharding = DefaultSharding()
    cfg = SimpleNamespace(num_experts=4, sparse_matmul=True, moe_bwd_wgrad_overlap=True)
    exec_exp = m3_moe.moe_from_config(cfg, sharding)
    self.assertEqual(exec_exp.length, "context")

    x = jax.random.normal(jax.random.key(7), (4, 4, 16), dtype=jnp.float32)
    logits = jax.random.normal(jax.random.key(8), (4, 4, 4), dtype=jnp.float32)
    logits = logits + jnp.arange(4, dtype=jnp.float32) * 1e-2
    cotangent = jax.random.normal(jax.random.key(9), x.shape, dtype=jnp.float32)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=16, mlp=12, seed=21)
    nat_w, nat_idx, nat_probs = m3_moe.route_top_k(logits, 2, scoring="softmax", norm_topk_prob=True)

    with jax.set_mesh(mesh_exp):
      tok_s = NamedSharding(mesh_exp, sharding("tokens", ("batch", "length", "embed"), m3_moe.TensorType.ACTIVATION))
      w0_s = NamedSharding(mesh_exp, sharding("wi_0", ("experts", "expert_embed", "expert_mlp")))
      wo_s = NamedSharding(mesh_exp, sharding("wo", ("experts", "expert_mlp", "expert_embed")))

      x_d = jax.device_put(x, tok_s)
      w_d = jax.device_put(nat_w, tok_s)
      idx_d = jax.device_put(nat_idx, tok_s)
      probs_d = jax.device_put(nat_probs, tok_s)
      cot_d = jax.device_put(cotangent, tok_s)
      wi0_d = jax.device_put(wi_0, w0_s)
      wi1_d = jax.device_put(wi_1, w0_s)
      wo_d = jax.device_put(wo, wo_s)

      def nat_exp_fwd(x_in, p_in, fi_in, w0, w1, w_out):
        w_cur, idx_cur = m3_moe.apply_forced_routed_experts(p_in, fi_in, norm_topk_prob=True)
        return exec_exp(x_in, w_cur, idx_cur, w0, w1, w_out)

      def rep_exp_fwd(x_in, p_in, fi_in, w0, w1, w_out):
        return exec_exp(
            x_in,
            w_d,
            fi_in,
            w0,
            w1,
            w_out,
            forced_routed_experts=fi_in,
            router_probs=p_in,
            norm_topk_prob=True,
        )

      out_nat = jax.jit(nat_exp_fwd)(x_d, probs_d, idx_d, wi0_d, wi1_d, wo_d)
      out_rep = jax.jit(rep_exp_fwd)(x_d, probs_d, idx_d, wi0_d, wi1_d, wo_d)
      grads_nat = jax.jit(
          jax.grad(
              lambda x_in, p_in, fi_in, w0, w1, w_out: jnp.sum(nat_exp_fwd(x_in, p_in, fi_in, w0, w1, w_out) * cot_d),
              argnums=(0, 1, 3, 4, 5),
          )
      )(x_d, probs_d, idx_d, wi0_d, wi1_d, wo_d)
      grads_rep = jax.jit(
          jax.grad(
              lambda x_in, p_in, fi_in, w0, w1, w_out: jnp.sum(rep_exp_fwd(x_in, p_in, fi_in, w0, w1, w_out) * cot_d),
              argnums=(0, 1, 3, 4, 5),
          )
      )(x_d, probs_d, idx_d, wi0_d, wi1_d, wo_d)

      self.assertEqual(float(jnp.max(jnp.abs(np.asarray(out_rep) - np.asarray(out_nat)))), 0.0)
      for g_r, g_n in zip(grads_rep, grads_nat):
        # Explicit mesh sharding parallel reductions introduce up to 1 ULP (~6e-8) float32 roundoff.
        self.assertLessEqual(float(jnp.max(jnp.abs(np.asarray(g_r) - np.asarray(g_n)))), 1e-6)


class ForcedExpertOverrideAndNegativeControlTest(parameterized.TestCase):
  """Verifies non-natural forced expert override matches manual expert math and shifts away from natural top-k."""

  @parameterized.named_parameters(
      ("softmax_norm", "softmax", True, 1.4),
      ("topk_softmax", "topk_softmax", False, 1.0),
      ("sigmoid_norm", "sigmoid", True, 2.5),
  )
  def test_non_natural_forced_override_matches_manual_and_differs_from_natural(self, scoring, norm_topk_prob, scale):
    if jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 CPU devices; got {jax.device_count()}.")
    num_experts = 4
    top_k = 2
    batch, length, embed, mlp = 4, 3, 16, 12
    k0, k1, k2 = jax.random.split(jax.random.key(303), 3)
    x = jax.random.normal(k0, (batch, length, embed), dtype=jnp.float32)
    logits = jax.random.normal(k1, (batch, length, num_experts), dtype=jnp.float32)
    cotangent = jax.random.normal(k2, x.shape, dtype=jnp.float32)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=num_experts, embed=embed, mlp=mlp, seed=31)
    w_gu = jnp.concatenate([wi_0, wi_1], axis=-1)

    nat_w, nat_idx, _ = m3_moe.route_top_k(
        logits, top_k, scoring=scoring, norm_topk_prob=norm_topk_prob, routed_scaling_factor=scale
    )
    # Force every token to select the bottom-k experts ((nat_idx + 2) % num_experts), disjoint from natural top-2.
    forced_idx = (nat_idx + 2) % num_experts

    def manual_reference(x_in, l_in, w0, w1, w_out):
      l_f32 = l_in.astype(jnp.float32)
      if scoring == "softmax":
        probs = jax.nn.softmax(l_f32, axis=-1)
        w = jnp.take_along_axis(probs, forced_idx, axis=-1)
      elif scoring == "topk_softmax":
        top_l = jnp.take_along_axis(l_f32, forced_idx, axis=-1)
        w = jax.nn.softmax(top_l, axis=-1)
      else:
        probs = jax.nn.sigmoid(l_f32)
        w = jnp.take_along_axis(probs, forced_idx, axis=-1)
      if norm_topk_prob:
        w = w / jnp.sum(w, axis=-1, keepdims=True)
      w = w * scale
      # Manual per-slot expert MLP evaluation.
      gate = jnp.einsum("bsd,edm->bsem", x_in, w0)
      up = jnp.einsum("bsd,edm->bsem", x_in, w1)
      hid = jax.nn.silu(gate) * up
      exp_out = jnp.einsum("bsem,emd->bsed", hid, w_out)
      chosen_out = jnp.take_along_axis(exp_out, forced_idx[..., None], axis=2)
      return jnp.sum(chosen_out * w[..., None], axis=2)

    ref_out, ref_grads = jax.jit(
        jax.value_and_grad(
            lambda x_in, l_in, w0, w1, w_out: (
                jnp.sum(manual_reference(x_in, l_in, w0, w1, w_out) * cotangent),
                manual_reference(x_in, l_in, w0, w1, w_out),
            ),
            argnums=(0, 1, 2, 3, 4),
            has_aux=True,
        )
    )(x, logits, wi_0, wi_1, wo)
    ref_y = ref_out[1]

    mesh = create_mesh(ici={"fsdp": 1, "expert": 2, "tensor": 2}, devices=jax.devices()[:4])
    exec_ep = m3_moe.MoEExecutor(
        kind="tokamax",
        batch=("data", "fsdp"),
        experts="expert",
        mlp="tensor",
        dropless_buffer_factor=1.25,
        bwd_wgrad_overlap=True,
    )

    with jax.set_mesh(mesh):
      nat_y = exec_ep(x, nat_w, nat_idx, wi_0, wi_1, wo)
      # Negative control: forced non-natural routing MUST differ from natural top-k output.
      self.assertGreater(float(jnp.max(jnp.abs(ref_y - nat_y))), 1e-2)

      for exec_cur in (m3_moe.DENSE, m3_moe.RAGGED_DOT, exec_ep):
        (_, got_y), got_grads = jax.jit(
            jax.value_and_grad(
                lambda x_in, l_in, w0, w1, w_out, ex=exec_cur: (
                    jnp.sum(
                        ex(
                            x_in,
                            *m3_moe.route_top_k(
                                l_in,
                                top_k,
                                scoring=scoring,
                                norm_topk_prob=norm_topk_prob,
                                routed_scaling_factor=scale,
                                forced_routed_experts=forced_idx,
                            )[:2],
                            w0,
                            w1,
                            w_out,
                        )
                        * cotangent
                    ),
                    ex(
                        x_in,
                        *m3_moe.route_top_k(
                            l_in,
                            top_k,
                            scoring=scoring,
                            norm_topk_prob=norm_topk_prob,
                            routed_scaling_factor=scale,
                            forced_routed_experts=forced_idx,
                        )[:2],
                        w0,
                        w1,
                        w_out,
                    ),
                ),
                argnums=(0, 1, 2, 3, 4),
                has_aux=True,
            )
        )(x, logits, wi_0, wi_1, wo)

        np.testing.assert_allclose(np.asarray(got_y), np.asarray(ref_y), rtol=1e-5, atol=1e-5)
        for g_got, g_ref in zip(got_grads, ref_grads):
          np.testing.assert_allclose(np.asarray(g_got), np.asarray(g_ref), rtol=1e-5, atol=1e-5)

      # Also verify unsharded fused fallback with forced_routed_experts.
      w_forced, _ = m3_moe.route_top_k(
          logits,
          top_k,
          scoring=scoring,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=scale,
          forced_routed_experts=forced_idx,
      )[:2]
      unf_y = m3_moe.grouped_unsharded_fused(x, w_forced, nat_idx, w_gu, wo, forced_routed_experts=forced_idx)
      np.testing.assert_allclose(np.asarray(unf_y), np.asarray(ref_y), rtol=1e-5, atol=1e-5)


class SentinelPaddingAndZeroGradLeakageTest(parameterized.TestCase):
  """Verifies `-1` partial-slot and all-`-1` full-row padding produce 0.0 output/grads and zero expert-0 leakage."""

  @parameterized.named_parameters(
      ("softmax_norm", "softmax", True, 1.5, 0.0),
      ("softmax_unnorm", "softmax", False, 1.0, 0.0),
      ("topk_softmax_unnorm", "topk_softmax", False, 1.0, 0.0),
      ("topk_softmax_norm", "topk_softmax", True, 1.2, 0.0),
      ("sigmoid_norm_eps", "sigmoid", True, 2.5, 1e-20),
      ("sigmoid_unnorm", "sigmoid", False, 1.0, 0.0),
  )
  def test_partial_and_all_minus_one_padding_zero_output_zero_nan_and_zero_expert0_leakage(
      self, scoring, norm_topk_prob, scale, eps
  ):
    if jax.device_count() < 4:
      self.skipTest(f"Requires >= 4 CPU devices; got {jax.device_count()}.")
    num_experts = 4
    top_k = 2
    batch, length, embed, mlp = 4, 3, 16, 12
    k0, k1, k2 = jax.random.split(jax.random.key(505), 3)
    x = jax.random.normal(k0, (batch, length, embed), dtype=jnp.float32)
    logits = jax.random.normal(k1, (batch, length, num_experts), dtype=jnp.float32)
    cotangent = jax.random.normal(k2, x.shape, dtype=jnp.float32)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=num_experts, embed=embed, mlp=mlp, seed=41)
    w_gu = jnp.stack([wi_0, wi_1], axis=2)

    # Construct forced_routed_experts where:
    # - Expert 0 is NEVER selected by any active (>= 0) slot (active IDs are only in {1, 2, 3}).
    # - Token (0, 0) and full batch row 2 have all slots = -1 (all-`-1` full-row padding).
    # - Token (0, 1) and token (1, 2) have partial-slot padding `[2, -1]` and `[-1, 3]`.
    forced_idx = jnp.array(
        [
            [[-1, -1], [2, -1], [1, 3]],
            [[3, 1], [2, 3], [-1, 3]],
            [[-1, -1], [-1, -1], [-1, -1]],
            [[1, 2], [-1, -1], [3, -1]],
        ],
        dtype=jnp.int32,
    )
    self.assertFalse(bool(jnp.any(forced_idx == 0)))

    # Verify compute_ep_overflow_stats does not count -1 slots into expert 0 / EP rank 0.
    stats = m3_moe.compute_ep_overflow_stats(forced_idx, num_experts=4, ep_size=2, dropless_buffer_factor=1.25)
    # Active slots: expert 1 appears 3 times, expert 2 appears 3 times, expert 3 appears 5 times, expert 0 appears 0 times.
    # Rank 0 owns experts {0, 1} -> count = 0 + 3 = 3; Rank 1 owns experts {2, 3} -> count = 3 + 5 = 8.
    np.testing.assert_array_equal(np.asarray(stats["dst_counts"]), np.array([3, 8], dtype=np.int32))

    # Negative control: if -1 were naively clamped to 0, Rank 0 count would be 3 + 13 = 16 instead of 3!
    naive_clamped = jnp.maximum(forced_idx, 0)
    naive_stats = m3_moe.compute_ep_overflow_stats(naive_clamped, num_experts=4, ep_size=2)
    self.assertEqual(int(naive_stats["dst_counts"][0]), 16)

    # Check partial-slot weight normalization on token (0, 1) where forced_idx = [2, -1]:
    w_routed, idx_routed, probs_routed = m3_moe.route_top_k(
        logits,
        top_k,
        scoring=scoring,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=scale,
        eps=eps,
        forced_routed_experts=forced_idx,
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(w_routed))))
    self.assertTrue(bool(jnp.array_equal(idx_routed, forced_idx)))
    # All -1 slots must have exact 0.0 weight.
    self.assertEqual(float(jnp.max(jnp.abs(jnp.where(forced_idx < 0, w_routed, 0.0)))), 0.0)
    # On partial-padded token (0, 1) with [2, -1], when norm_topk_prob=True or topk_softmax, active slot weight == scale.
    if norm_topk_prob or scoring == "topk_softmax":
      np.testing.assert_allclose(float(w_routed[0, 1, 0]), scale, rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(float(w_routed[1, 2, 1]), scale, rtol=1e-6, atol=1e-6)
    self.assertEqual(float(w_routed[0, 1, 1]), 0.0)
    self.assertEqual(float(w_routed[1, 2, 0]), 0.0)

    mesh_ep1 = create_mesh(devices=jax.devices()[:1])
    mesh_ep2 = create_mesh(ici={"fsdp": 1, "expert": 2, "tensor": 2}, devices=jax.devices()[:4])
    exec_ep1 = m3_moe.MoEExecutor(kind="tokamax", bwd_wgrad_overlap=True)
    exec_ep2 = m3_moe.MoEExecutor(
        kind="tokamax",
        batch=("data", "fsdp"),
        experts="expert",
        mlp="tensor",
        dropless_buffer_factor=1.25,
        bwd_wgrad_overlap=True,
    )

    calls = {"gmm_group0": []}

    def _record_group0(g0):
      calls["gmm_group0"].append(int(np.asarray(g0)))

    def fake_gmm_v2(
        *, lhs, rhs, group_sizes, rhs_scale, tile_info, preferred_element_type, maybe_quantize_lhs, transpose_rhs=False
    ):
      del rhs_scale, tile_info, maybe_quantize_lhs
      jax.debug.callback(_record_group0, group_sizes[0])
      rhs_eff = rhs.swapaxes(1, 2) if transpose_rhs else rhs
      return jax.lax.ragged_dot(lhs, rhs_eff, group_sizes, preferred_element_type=jnp.float32).astype(
          preferred_element_type
      )

    def fake_tgmm_v2(
        *, lhs, rhs, group_sizes, num_actual_groups, rhs_scale, precision, preferred_element_type, tile_info
    ):
      del rhs_scale, precision, tile_info
      jax.debug.callback(_record_group0, group_sizes[0])
      _, vjp_fn = jax.vjp(
          lambda r_w: jax.lax.ragged_dot(lhs, r_w, group_sizes, preferred_element_type=jnp.float32),
          jnp.zeros((num_actual_groups, lhs.shape[1], rhs.shape[1]), dtype=lhs.dtype),
      )
      (drhs,) = vjp_fn(rhs.astype(jnp.float32))
      return drhs.astype(preferred_element_type)

    # Check EP=1 mocked TPU gmm_v2/tgmm_v2 group_sizes[0] == 0 and exact 0.0 expert-0 grads.
    with (
        mock.patch.object(m3_kernels, "is_tpu_mesh", return_value=True),
        mock.patch.object(gmm_v2, "gmm_v2", side_effect=fake_gmm_v2),
        mock.patch.object(tgmm_v2, "tgmm_v2", side_effect=fake_tgmm_v2),
        jax.set_mesh(mesh_ep1),
    ):
      (_, out_ep1), grads_ep1 = jax.jit(
          jax.value_and_grad(
              lambda x_in, l_in, w0, w1, w_out: (
                  jnp.sum(
                      exec_ep1(
                          x_in,
                          *m3_moe.route_top_k(
                              l_in,
                              top_k,
                              scoring=scoring,
                              norm_topk_prob=norm_topk_prob,
                              routed_scaling_factor=scale,
                              eps=eps,
                              forced_routed_experts=forced_idx,
                          )[:2],
                          w0,
                          w1,
                          w_out,
                      )
                      * cotangent
                  ),
                  exec_ep1(
                      x_in,
                      *m3_moe.route_top_k(
                          l_in,
                          top_k,
                          scoring=scoring,
                          norm_topk_prob=norm_topk_prob,
                          routed_scaling_factor=scale,
                          eps=eps,
                          forced_routed_experts=forced_idx,
                      )[:2],
                      w0,
                      w1,
                      w_out,
                  ),
              ),
              argnums=(0, 1, 2, 3, 4),
              has_aux=True,
          )
      )(x, logits, wi_0, wi_1, wo)

    # Every single gmm_v2 / tgmm_v2 call on EP=1 must see group_sizes[0] == 0!
    self.assertNotEmpty(calls["gmm_group0"])
    self.assertEqual(max(calls["gmm_group0"]), 0)

    # Verify across DENSE, RAGGED_DOT, EP=1, and 4-device EP=2 (with overflow rounds) for both separate and fused:
    with jax.set_mesh(mesh_ep2):
      (_, out_dense), grads_dense = jax.jit(
          jax.value_and_grad(
              lambda x_in, l_in, w0, w1, w_out: (
                  jnp.sum(
                      m3_moe.DENSE(
                          x_in,
                          *m3_moe.route_top_k(
                              l_in,
                              top_k,
                              scoring=scoring,
                              norm_topk_prob=norm_topk_prob,
                              routed_scaling_factor=scale,
                              eps=eps,
                              forced_routed_experts=forced_idx,
                          )[:2],
                          w0,
                          w1,
                          w_out,
                      )
                      * cotangent
                  ),
                  m3_moe.DENSE(
                      x_in,
                      *m3_moe.route_top_k(
                          l_in,
                          top_k,
                          scoring=scoring,
                          norm_topk_prob=norm_topk_prob,
                          routed_scaling_factor=scale,
                          eps=eps,
                          forced_routed_experts=forced_idx,
                      )[:2],
                      w0,
                      w1,
                      w_out,
                  ),
              ),
              argnums=(0, 1, 2, 3, 4),
              has_aux=True,
          )
      )(x, logits, wi_0, wi_1, wo)

      (_, out_ep2), grads_ep2 = jax.jit(
          jax.value_and_grad(
              lambda x_in, p_in, w0, w1, w_out: (
                  jnp.sum(
                      exec_ep2(
                          x_in,
                          p_in,
                          None,
                          w0,
                          w1,
                          w_out,
                          forced_routed_experts=forced_idx,
                          norm_topk_prob=norm_topk_prob,
                          routed_scaling_factor=scale,
                          scores_are_logits=(scoring == "topk_softmax"),
                          eps=eps,
                      )
                      * cotangent
                  ),
                  exec_ep2(
                      x_in,
                      p_in,
                      None,
                      w0,
                      w1,
                      w_out,
                      forced_routed_experts=forced_idx,
                      norm_topk_prob=norm_topk_prob,
                      routed_scaling_factor=scale,
                      scores_are_logits=(scoring == "topk_softmax"),
                      eps=eps,
                  ),
              ),
              argnums=(0, 1, 2, 3, 4),
              has_aux=True,
          )
      )(x, logits if scoring == "topk_softmax" else probs_routed, wi_0, wi_1, wo)

      (_, out_ep2_f), grads_ep2_f = jax.jit(
          jax.value_and_grad(
              lambda x_in, p_in, gu, dn: (
                  jnp.sum(
                      exec_ep2.fused(
                          x_in,
                          p_in,
                          None,
                          gu,
                          dn,
                          forced_routed_experts=forced_idx,
                          norm_topk_prob=norm_topk_prob,
                          routed_scaling_factor=scale,
                          scores_are_logits=(scoring == "topk_softmax"),
                          eps=eps,
                      )
                      * cotangent
                  ),
                  exec_ep2.fused(
                      x_in,
                      p_in,
                      None,
                      gu,
                      dn,
                      forced_routed_experts=forced_idx,
                      norm_topk_prob=norm_topk_prob,
                      routed_scaling_factor=scale,
                      scores_are_logits=(scoring == "topk_softmax"),
                      eps=eps,
                  ),
              ),
              argnums=(0, 1, 2, 3),
              has_aux=True,
          )
      )(x, logits if scoring == "topk_softmax" else probs_routed, w_gu, wo)

    for out_cur in (out_ep1, out_dense, out_ep2, out_ep2_f):
      self.assertTrue(bool(jnp.all(jnp.isfinite(out_cur))))
      # All-`-1` padded tokens ((0, 0), all of batch 2, and (3, 1)) must have exact 0.0 output!
      self.assertEqual(float(jnp.max(jnp.abs(out_cur[0, 0]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(out_cur[2]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(out_cur[3, 1]))), 0.0)
      np.testing.assert_allclose(np.asarray(out_cur), np.asarray(out_dense), rtol=1e-5, atol=1e-5)

    for grads_cur in (grads_ep1, grads_dense, grads_ep2):
      dx, dlogits_or_probs, dw0, dw1, dwo = grads_cur
      for g_leaf in grads_cur:
        self.assertTrue(bool(jnp.all(jnp.isfinite(g_leaf))))
      # All-`-1` padded tokens must have exact 0.0 dx and 0.0 router gradient!
      self.assertEqual(float(jnp.max(jnp.abs(dx[0, 0]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dx[2]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dx[3, 1]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dlogits_or_probs[0, 0]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dlogits_or_probs[2]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dlogits_or_probs[3, 1]))), 0.0)
      # Expert 0 was never selected by any valid (>= 0) slot, so its weight gradients must be exact 0.0!
      self.assertEqual(float(jnp.max(jnp.abs(dw0[0]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dw1[0]))), 0.0)
      self.assertEqual(float(jnp.max(jnp.abs(dwo[0]))), 0.0)

    # Also check fused expert 0 weight gradients and router_probs[:, :, 0] gradient on EP=2:
    dx_f, dp_f, dgu_f, ddn_f = grads_ep2_f
    self.assertEqual(float(jnp.max(jnp.abs(dx_f[2]))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(dp_f[:, :, 0]))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(dgu_f[0]))), 0.0)
    self.assertEqual(float(jnp.max(jnp.abs(ddn_f[0]))), 0.0)

    np.testing.assert_allclose(np.asarray(grads_ep1[0]), np.asarray(grads_dense[0]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(grads_ep1[1]), np.asarray(grads_dense[1]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(grads_ep2[0]), np.asarray(grads_dense[0]), rtol=1e-5, atol=1e-5)


class ValidationFailClosedTest(parameterized.TestCase):
  """Verifies fail-closed validation on invalid shape, rank, dtype, and out-of-range expert IDs."""

  def test_fail_closed_on_invalid_shape_rank_dtype_and_out_of_range_ids(self):
    x = jnp.ones((2, 3, 8), dtype=jnp.float32)
    probs = jnp.full((2, 3, 4), 0.25, dtype=jnp.float32)
    weights = jnp.full((2, 3, 2), 0.5, dtype=jnp.float32)
    indices = jnp.zeros((2, 3, 2), dtype=jnp.int32)
    wi_0, wi_1, wo = _make_expert_weights(num_experts=4, embed=8, mlp=6)
    w_gu = jnp.concatenate([wi_0, wi_1], axis=-1)

    # Wrong rank (2D instead of 3D).
    with self.assertRaisesRegex(ValueError, "rank|Expected 3D forced_routed_experts"):
      m3_moe.apply_forced_routed_experts(probs, jnp.zeros((6, 2), dtype=jnp.int32))
    with self.assertRaisesRegex(ValueError, "Expected 3D forced_routed_experts"):
      m3_moe.routed_experts(x, weights, indices, wi_0, wi_1, wo, forced_routed_experts=jnp.zeros((6, 2), dtype=jnp.int32))

    # Mismatched token prefix shape.
    with self.assertRaisesRegex(ValueError, "forced_routed_experts shape"):
      m3_moe.route_top_k(probs, 2, forced_routed_experts=jnp.zeros((2, 4, 2), dtype=jnp.int32))
    with self.assertRaisesRegex(ValueError, "forced_routed_experts shape"):
      m3_moe.routed_experts_fused(
          x, weights, indices, w_gu, wo, forced_routed_experts=jnp.zeros((2, 4, 2), dtype=jnp.int32)
      )

    # Mismatched top_k in route_top_k or indices vs forced_routed_experts.
    with self.assertRaisesRegex(ValueError, "forced_routed_experts shape"):
      m3_moe.route_top_k(probs, 2, forced_routed_experts=jnp.zeros((2, 3, 3), dtype=jnp.int32))
    with self.assertRaisesRegex(ValueError, "indices shape .* must match forced_routed_experts shape"):
      m3_moe.routed_experts(
          x, weights, indices, wi_0, wi_1, wo, forced_routed_experts=jnp.zeros((2, 3, 3), dtype=jnp.int32)
      )

    # Float or bool dtype rejected.
    with self.assertRaisesRegex(ValueError, "integer dtype"):
      m3_moe.apply_forced_routed_experts(probs, jnp.zeros((2, 3, 2), dtype=jnp.float32))
    with self.assertRaisesRegex(ValueError, "integer dtype"):
      m3_moe.apply_forced_routed_experts(probs, jnp.zeros((2, 3, 2), dtype=jnp.bool_))

    # Out-of-range ID < -1 (e.g. -2) rejected.
    bad_neg = jnp.zeros((2, 3, 2), dtype=jnp.int32).at[0, 1, 0].set(-2)
    with self.assertRaisesRegex(ValueError, r"must be in \[-1, 4\)"):
      m3_moe.route_top_k(probs, 2, forced_routed_experts=bad_neg)
    with self.assertRaisesRegex(ValueError, r"must be in \[-1, 4\)"):
      m3_moe.routed_experts(x, weights, indices, wi_0, wi_1, wo, forced_routed_experts=bad_neg)

    # Out-of-range ID >= num_experts (e.g. 4 when num_experts == 4) rejected.
    bad_hi = jnp.zeros((2, 3, 2), dtype=jnp.int32).at[1, 0, 1].set(4)
    with self.assertRaisesRegex(ValueError, r"must be in \[-1, 4\)"):
      m3_moe.apply_forced_routed_experts(probs, bad_hi)
    with self.assertRaisesRegex(ValueError, r"must be in \[-1, 4\)"):
      m3_moe.routed_experts_fused(x, weights, indices, w_gu, wo, forced_routed_experts=bad_hi)

    # Missing indices when forced_routed_experts is None.
    with self.assertRaisesRegex(ValueError, "indices must be provided when forced_routed_experts is None"):
      m3_moe.routed_experts(x, weights, None, wi_0, wi_1, wo)


if __name__ == "__main__":
  absltest.main()
