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

"""Unit and virtual-topology tests for compound MoE sharding and scale-tuned `Sharding` presets."""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from maxtext.m3.core import moe as moe_lib
from maxtext.m3.core import sharding as sharding_lib
from maxtext.m3.core.sharding import CPasEP
from maxtext.m3.core.sharding import DefaultSharding
from maxtext.m3.core.sharding import EPasCP
from maxtext.m3.core.sharding import EPasDP
from maxtext.m3.core.sharding import FSDPasDPForAttn
from maxtext.m3.core.sharding import FSDPxEP
from maxtext.m3.core.sharding import FSDPxEP_2D
from maxtext.m3.core.sharding import MESH_AXIS_NAMES
from maxtext.m3.core.sharding import SHARDING_PRESETS
from maxtext.m3.core.sharding import TPasEP
from maxtext.m3.core.sharding import TPxEP
from maxtext.m3.core.sharding import TensorType
from maxtext.m3.core.sharding import compound_moe_param_metadata
from maxtext.m3.core.sharding import compound_moe_spec
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.core.sharding import fsdp_x_ep_axes
from maxtext.m3.core.sharding import param_metadata
from maxtext.m3.core.sharding import project_spec_to_mesh
from maxtext.m3.core.sharding import sharding_from_config
from maxtext.m3.core.sharding import sharding_preset_from_name
from maxtext.m3.core.sharding import tp_x_ep_axes


class CompoundMoEHelpersTest(parameterized.TestCase):
  """Tests `FSDPxEP`, `TPxEP`, `compound_moe_spec`, and `compound_moe_param_metadata`."""

  def test_axis_constants_and_helpers(self):
    self.assertEqual(FSDPxEP, ("fsdp", "expert"))
    self.assertEqual(FSDPxEP_2D, ("fsdp", "fsdp_transpose", "expert"))
    self.assertEqual(TPxEP, ("expert", "tensor"))
    self.assertEqual(tp_x_ep_axes(), ("expert", "tensor"))
    self.assertEqual(fsdp_x_ep_axes(two_d_fsdp=False), ("fsdp", "expert"))
    self.assertEqual(fsdp_x_ep_axes(two_d_fsdp=True), ("fsdp", "fsdp_transpose", "expert"))

    with jax.sharding.set_mesh(None):
      self.assertEqual(fsdp_x_ep_axes(), ("fsdp", "expert"))

    abs_1d = jax.sharding.AbstractMesh((1, 1, 16, 1, 1, 1, 1, 8), MESH_AXIS_NAMES)
    with jax.sharding.use_abstract_mesh(abs_1d):
      self.assertEqual(fsdp_x_ep_axes(), ("fsdp", "expert"))

    abs_2d = jax.sharding.AbstractMesh((1, 1, 8, 2, 1, 1, 1, 8), MESH_AXIS_NAMES)
    with jax.sharding.use_abstract_mesh(abs_2d):
      self.assertEqual(fsdp_x_ep_axes(), ("fsdp", "fsdp_transpose", "expert"))

  def test_compound_moe_spec_fsdp_x_ep_and_tp_x_ep(self):
    with jax.sharding.set_mesh(None):
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep"),
          P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", two_d_fsdp=True),
          P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(
              None, "act", ("experts", "expert_embed", "expert_mlp"), mode="fsdp_x_ep", tensor_type=TensorType.ACTIVATION
          ),
          P("expert", None, ("tensor", "tensor_sequence")),
      )
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="tp_x_ep"),
          P(("expert", "tensor"), None, None),
      )
      self.assertEqual(
          compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="tp_x_ep", muon_friendly=False),
          P(("expert", "tensor"), ("fsdp", "fsdp_transpose", "context"), None),
      )
      with self.assertRaisesRegex(ValueError, "Unknown compound MoE mode"):
        compound_moe_spec(None, "wi_0", ("experts", "expert_embed", "expert_mlp"), mode="bad_mode")


class ShardingStrategyPresetsTest(parameterized.TestCase):
  """Tests `EPasDP`, `EPasCP`, `CPasEP`, `TPasEP`, `FSDPasDPForAttn`, and `sharding_preset_from_name`."""

  def test_ep_as_dp_rules(self):
    std = EPasDP()
    self.assertEqual(
        std("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(std("q_lora", ("q_lora", "q_lora_up_proj")), P(("fsdp", "fsdp_transpose", "expert"), None))
    self.assertEqual(std("router", ("embed_router", None)), P(("fsdp", "fsdp_transpose"), None))
    self.assertEqual(
        std("act", ("batch", "length", "embed"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context", None),
    )

    muon = EPasDP(compound_fsdp_ep=True)
    self.assertEqual(
        muon("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        muon("wo", ("experts", "expert_mlp", "expert_embed")), P(("fsdp", "expert"), ("tensor", "tensor_sequence"), None)
    )
    self.assertEqual(
        muon("act", ("experts", "expert_embed", "expert_mlp"), TensorType.ACTIVATION),
        P("expert", None, ("tensor", "tensor_sequence")),
    )

    muon_2d = EPasDP(compound_fsdp_ep=True, two_d_fsdp=True)
    self.assertEqual(
        muon_2d("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
    )

  def test_ep_as_cp_rules(self):
    s = EPasCP()
    self.assertEqual(
        s("attn_act", ("batch", "length", "heads", "head_dim"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose"), ("context", "expert"), ("tensor", "tensor_sequence"), None),
    )
    self.assertEqual(
        s("norm_act", ("batch", "norm_length", "embed"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose"), ("context", "expert", "tensor_sequence"), None),
    )
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        EPasCP(compound_fsdp_ep=True)("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
    )

  def test_cp_as_ep_rules(self):
    s = CPasEP()
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("context", "expert"), ("fsdp", "fsdp_transpose"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        s("gate", ("embed", "mlp")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
    )
    self.assertEqual(
        s("q_lora", ("q_lora", "q_lora_up_proj")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), None),
    )
    self.assertEqual(
        s("attn_act", ("batch", "length", "heads", "head_dim"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context", ("tensor", "tensor_sequence"), None),
    )
    self.assertEqual(
        s("act", ("experts", "expert_embed", "expert_mlp"), TensorType.ACTIVATION),
        P(("context", "expert"), None, ("tensor", "tensor_sequence")),
    )
    s_comp = CPasEP(compound_fsdp_ep=True)
    self.assertEqual(
        s_comp("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "context", "expert"), None, ("tensor", "tensor_sequence")),
    )

    class _BrokenCPasEP(DefaultSharding):
      """Negative control: claiming `context` on `experts` without excluding it from `expert_embed`."""

      def map_axis(self, axis: str, tensor_name: str, tensor_type: TensorType):
        if axis == "experts":
          return ("context", "expert")
        return super().map_axis(axis, tensor_name, tensor_type)

    with self.assertRaisesRegex(ValueError, "maps both 'experts' and 'expert_embed' onto mesh axis 'context'"):
      _BrokenCPasEP()("wi_0", ("experts", "expert_embed", "expert_mlp"))

  def test_tp_as_ep_rules(self):
    s = TPasEP()
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("expert", "tensor"), ("fsdp", "fsdp_transpose", "context"), None),
    )
    self.assertEqual(
        s("q_proj", ("embed", "heads", "head_dim")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence"), None),
    )
    s_muon = TPasEP(muon_friendly=True)
    self.assertEqual(
        s_muon("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("expert", "tensor"), None, None),
    )
    s_comp = TPasEP(compound_fsdp_ep=True)
    self.assertEqual(
        s_comp("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert", "tensor"), None, None),
    )

  def test_fsdp_as_dp_for_attn_rules(self):
    s = FSDPasDPForAttn()
    # MLA compressed projections shard over "expert" and replicate over FSDP.
    self.assertEqual(s("q_lora", ("q_lora", "q_lora_up_proj")), P("expert", None))
    self.assertEqual(s("kv_lora", ("kv_lora", "kv_lora_up_proj")), P("expert", None))
    self.assertEqual(s("q_proj", ("embed_attn", "heads", "head_dim")), P("expert", ("tensor", "tensor_sequence"), None))
    self.assertEqual(s("wq_a", ("embed", None)), P("expert", None))
    self.assertEqual(s("wkv_a", ("embed", None)), P("expert", None))
    # Dense MLP and token embeddings still shard over FSDP x CP x EP.
    self.assertEqual(
        s("mlp_gate", ("embed", "mlp")),
        P(("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
    )
    # Attention activations still shard batch across data x fsdp x fsdp_transpose x expert.
    self.assertEqual(
        s("mla_q_act", ("batch", "length", "q_lora"), TensorType.ACTIVATION),
        P(("data", "fsdp", "fsdp_transpose", "expert"), "context", None),
    )
    # Routed experts stay EP + FSDP x CP sharded.
    self.assertEqual(
        s("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
    )
    # Compound FSDPxEP mode shards experts over ("fsdp", "expert") with expert_embed=None.
    s_comp = FSDPasDPForAttn(compound_fsdp_ep=True)
    self.assertEqual(
        s_comp("wi_0", ("experts", "expert_embed", "expert_mlp")),
        P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")),
    )
    # Router weight shards on FSDP by default (-rf) and replicates when shard_router_on_fsdp=False.
    self.assertEqual(s("router", ("embed_router", None)), P(("fsdp", "fsdp_transpose"), None))
    self.assertEqual(s("router", ("embed", None)), P(("fsdp", "fsdp_transpose"), None))
    s_no_rf = FSDPasDPForAttn(shard_router_on_fsdp=False)
    self.assertEqual(s_no_rf("router", ("embed_router", None)), P(None, None))
    self.assertEqual(s_no_rf("router", ("embed", None)), P(None, None))

  def test_preset_registry_and_sharding_from_config(self):
    self.assertEqual(sharding_preset_from_name(None), DefaultSharding())
    self.assertEqual(sharding_preset_from_name("default"), DefaultSharding())
    self.assertEqual(sharding_preset_from_name("ep-as-dp"), EPasDP())
    self.assertEqual(sharding_preset_from_name("ep_as_cp"), EPasCP())
    self.assertEqual(sharding_preset_from_name("cp-as-ep"), CPasEP())
    self.assertEqual(sharding_preset_from_name("tp-as-ep"), TPasEP())
    self.assertEqual(sharding_preset_from_name("tp_x_ep"), TPasEP())
    self.assertEqual(sharding_preset_from_name("fsdp-as-dp-for-attn"), FSDPasDPForAttn(shard_router_on_fsdp=True))
    self.assertEqual(sharding_preset_from_name("fsdp-as-dp-for-attn-rf"), FSDPasDPForAttn(shard_router_on_fsdp=True))
    self.assertEqual(sharding_preset_from_name("fsdp-as-dp-for-attn-no-rf"), FSDPasDPForAttn(shard_router_on_fsdp=False))
    self.assertEqual(sharding_preset_from_name("fsdp-x-ep"), EPasDP(compound_fsdp_ep=True))
    self.assertEqual(sharding_preset_from_name("expert-on-fsdp-ep"), EPasDP(compound_fsdp_ep=True))
    self.assertEqual(sharding_preset_from_name("shard-exp-on-fsdp"), EPasDP(compound_fsdp_ep=True))

    preset_instance = FSDPasDPForAttn(compound_fsdp_ep=True)
    self.assertIs(sharding_preset_from_name(preset_instance), preset_instance)
    self.assertEqual(sharding_preset_from_name(EPasDP(), compound_fsdp_ep=True), EPasDP(compound_fsdp_ep=True))
    self.assertEqual(
        sharding_preset_from_name(FSDPasDPForAttn(), compound_fsdp_ep=True, two_d_fsdp=True),
        FSDPasDPForAttn(compound_fsdp_ep=True, two_d_fsdp=True),
    )

    for key in SHARDING_PRESETS:
      self.assertIsInstance(sharding_preset_from_name(key), DefaultSharding)

    with self.assertRaisesRegex(ValueError, "Unknown m3 sharding preset"):
      sharding_preset_from_name("nonexistent_preset")
    with self.assertRaisesRegex(TypeError, "Expected str or None"):
      sharding_preset_from_name(123)  # pytype: disable=wrong-arg-types

    cfg_default = SimpleNamespace()
    self.assertEqual(sharding_from_config(cfg_default), DefaultSharding())

    cfg_mla = SimpleNamespace(
        sharding_preset="fsdp-as-dp-for-attn",
        compound_fsdp_ep=True,
    )
    self.assertEqual(sharding_from_config(cfg_mla), FSDPasDPForAttn(compound_fsdp_ep=True))

    cfg_exp_fsdp = SimpleNamespace(sequence_parallel=False, shard_exp_on_fsdp=True)
    self.assertEqual(sharding_from_config(cfg_exp_fsdp), EPasDP(compound_fsdp_ep=True))

  def test_sharding_from_config_2d_fsdp_outside_mesh_and_default_validation(self):
    with jax.sharding.set_mesh(None):
      # (a) Outside `with jax.set_mesh(mesh):`, `sharding_from_config` extracts 2D FSDP from ICI/DCN config flags.
      s_no_preset = sharding_from_config(SimpleNamespace(ici_fsdp_transpose_parallelism=2, compound_fsdp_ep=True))
      self.assertEqual(s_no_preset, EPasDP(compound_fsdp_ep=True, two_d_fsdp=True))
      self.assertEqual(
          s_no_preset("wi_0", ("experts", "expert_embed", "expert_mlp")),
          P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
      )

      s_dcn = sharding_from_config(SimpleNamespace(dcn_fsdp_transpose_parallelism=2, shard_exp_on_fsdp=True))
      self.assertEqual(s_dcn, EPasDP(compound_fsdp_ep=True, two_d_fsdp=True))

      for preset_name, expected_experts_axis, expected_mlp_axis in (
          ("ep_as_dp", ("fsdp", "fsdp_transpose", "expert"), ("tensor", "tensor_sequence")),
          ("fsdp_as_dp_for_attn", ("fsdp", "fsdp_transpose", "expert"), ("tensor", "tensor_sequence")),
          ("ep_as_cp", ("fsdp", "fsdp_transpose", "expert"), ("tensor", "tensor_sequence")),
          ("cp_as_ep", ("fsdp", "fsdp_transpose", "context", "expert"), ("tensor", "tensor_sequence")),
          ("tp_as_ep", ("fsdp", "fsdp_transpose", "expert", "tensor"), None),
      ):
        s_preset = sharding_from_config(
            SimpleNamespace(
                sharding_preset=preset_name,
                compound_fsdp_ep=True,
                ici_fsdp_transpose_parallelism=2,
            )
        )
        self.assertEqual(
            s_preset("wi_0", ("experts", "expert_embed", "expert_mlp")),
            P(expected_experts_axis, None, expected_mlp_axis),
        )

      s_alias = sharding_from_config(SimpleNamespace(sharding_preset="fsdp-x-ep", ici_fsdp_transpose_parallelism=2))
      self.assertEqual(
          s_alias("wi_0", ("experts", "expert_embed", "expert_mlp")),
          P(("fsdp", "fsdp_transpose", "expert"), None, ("tensor", "tensor_sequence")),
      )

      # (b) `sharding_preset="default"` (or `None` / `DefaultSharding()`) + `ici_fsdp_transpose_parallelism=2`
      # (`compound_fsdp_ep=False`) succeeds cleanly and shards over `fsdp_transpose` natively.
      for default_preset in (None, "", "default", "DEFAULT", "default-sharding", DefaultSharding()):
        s_def = sharding_from_config(SimpleNamespace(sharding_preset=default_preset, ici_fsdp_transpose_parallelism=2))
        self.assertEqual(s_def, DefaultSharding())
        self.assertEqual(
            s_def("wi_0", ("experts", "expert_embed", "expert_mlp")),
            P("expert", ("fsdp", "fsdp_transpose", "context"), ("tensor", "tensor_sequence")),
        )

    # (d) Passing incompatible options to `"default"` or `DefaultSharding()` fails closed with `ValueError`.
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name("default", compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name("", compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name(DefaultSharding(), compound_fsdp_ep=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_preset_from_name(EPasDP(), unknown_flag=True)
    with self.assertRaisesRegex(ValueError, "does not accept options"):
      sharding_from_config(SimpleNamespace(sharding_preset="default", compound_fsdp_ep=True))

  def test_moe_from_config_integrates_with_presets(self):
    cfg = SimpleNamespace(
        num_experts=8,
        sparse_matmul=True,
        use_tokamax_gmm=True,
        capacity_factor=-1.0,
        ragged_buffer_factor=-1.0,
        use_ragged_sort=False,
        use_ring_of_experts=False,
        moe_bwd_wgrad_overlap=False,
    )
    exec_mla = moe_lib.moe_from_config(cfg, FSDPasDPForAttn(compound_fsdp_ep=True))
    self.assertEqual(exec_mla.batch, ("data", "fsdp", "fsdp_transpose", "expert"))
    self.assertEqual(exec_mla.experts, ("fsdp", "expert"))
    self.assertEqual(exec_mla.mlp, ("tensor", "tensor_sequence"))

    exec_ep_cp = moe_lib.moe_from_config(cfg, EPasCP())
    self.assertEqual(exec_ep_cp.batch, ("data", "fsdp", "fsdp_transpose"))
    self.assertEqual(exec_ep_cp.experts, "expert")
    self.assertEqual(exec_ep_cp.mlp, ("tensor", "tensor_sequence"))

    exec_tp_ep = moe_lib.moe_from_config(cfg, TPasEP())
    self.assertEqual(exec_tp_ep.experts, ("expert", "tensor"))
    self.assertIsNone(exec_tp_ep.mlp)


class ShardingPresetsMultiDeviceAndNegativeControlTest(parameterized.TestCase):
  """4-CPU and virtual-topology (`tpu7x-128`) verification with negative controls."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4 or jax.device_count() % 2 != 0:
      self.skipTest(f"Requires >= 4 even devices; found {jax.device_count()}.")
    self.devices = jax.devices()[:4]

  def test_eager_sharding_survives_nnx_wrappers_import_with_negative_control(self):
    """Proves `eager_sharding=True` shards weights after importing `nnx_wrappers`, while omitting it replicates."""
    # Importing legacy `nnx_wrappers` sets `flax_always_shard_variable=False` globally.
    import maxtext.layers.nnx_wrappers as _nnx_wrappers  # pylint: disable=import-outside-toplevel,unused-import

    self.enter_context(nnx.use_eager_sharding(False))
    self.assertFalse(nnx.using_eager_sharding())

    mesh = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    preset = FSDPasDPForAttn(compound_fsdp_ep=True)
    gu_shape = (8, 16, 32)
    q_lora_shape = (16, 8)

    with jax.sharding.set_mesh(mesh):
      # Positive path: `param_metadata` and `compound_moe_param_metadata` inject `eager_sharding=True`.
      gu_meta = compound_moe_param_metadata(preset, "gate_up_proj", ("experts", "expert_embed", "expert_mlp"))
      q_meta = param_metadata(preset, "q_lora", ("q_lora", "q_lora_up_proj"))
      self.assertTrue(gu_meta["eager_sharding"])
      self.assertTrue(q_meta["eager_sharding"])

      gu_param = nnx.Param(jnp.zeros(gu_shape, dtype=jnp.float32), **gu_meta)
      q_param = nnx.Param(jnp.zeros(q_lora_shape, dtype=jnp.float32), **q_meta)

      # Negative control: omitting `eager_sharding=True` leaves parameters 100% replicated when
      # `nnx_wrappers` turns off global `flax_always_shard_variable`.
      gu_broken = nnx.Param(jnp.zeros(gu_shape, dtype=jnp.float32), out_sharding=gu_meta["out_sharding"])
      q_broken = nnx.Param(jnp.zeros(q_lora_shape, dtype=jnp.float32), out_sharding=q_meta["out_sharding"])

    # Positive assertions:
    # 8 experts sharded across fsdp=2 x expert=2 -> 2 complete (16, 32) expert matrices per device (Muon-friendly).
    self.assertEqual(gu_param[...].sharding.shard_shape(gu_shape), (2, 16, 32))
    # q_lora (16, 8) sharded across expert=2 (replicated over fsdp=2) -> (8, 8) per device.
    self.assertEqual(q_param[...].sharding.shard_shape(q_lora_shape), (8, 8))

    # Negative-control assertions: without `eager_sharding=True`, both parameters silently replicate.
    self.assertEqual(gu_broken[...].sharding.shard_shape(gu_shape), gu_shape)
    self.assertEqual(q_broken[...].sharding.shard_shape(q_lora_shape), q_lora_shape)
    self.assertNotEqual(gu_param[...].sharding.shard_shape(gu_shape), gu_broken[...].sharding.shard_shape(gu_shape))

  def test_fsdp_as_dp_for_attn_activation_and_hlo_negative_controls(self):
    """Proves `FSDPasDPForAttn` shards MLA activations over FSDP x EP and avoids expert all-gather in HLO."""
    mesh = create_mesh(ici={"fsdp": 2, "expert": 2}, devices=self.devices)
    preset = FSDPasDPForAttn(compound_fsdp_ep=True)

    batch_size, seq_len, emb_dim, q_lora_dim, mlp_dim, num_experts = 4, 8, 16, 8, 32, 8
    x = jnp.ones((batch_size, seq_len, emb_dim), dtype=jnp.float32)
    wq_a = jnp.ones((emb_dim, q_lora_dim), dtype=jnp.float32)
    wi_0 = jnp.ones((num_experts, emb_dim, mlp_dim), dtype=jnp.float32)

    with jax.sharding.set_mesh(mesh):
      x_shd = NamedSharding(mesh, preset("x", ("batch", "length", "embed"), TensorType.ACTIVATION))
      wq_shd = NamedSharding(mesh, preset("q_lora", ("q_lora", "q_lora_up_proj")))
      wi_shd = NamedSharding(mesh, preset("wi_0", ("experts", "expert_embed", "expert_mlp")))

      x_dev = jax.device_put(x, x_shd)
      wq_dev = jax.device_put(wq_a, wq_shd)
      wi_dev = jax.device_put(wi_0, wi_shd)

      @jax.jit
      def mla_and_expert_step(x_in, wq_in, wi_in):
        q_latent = jnp.einsum("bsd,dq->bsq", x_in, wq_in)
        q_latent = preset.constrain(q_latent, "q_latent", ("batch", "length", "q_lora"))
        # Each of the 4 (fsdp=2, expert=2) shards owns 1 batch row and 2 local experts:
        # local token-to-local-expert contraction requires zero all-gather on `wi_in`.
        return q_latent, wi_in

      q_out, wi_out = mla_and_expert_step(x_dev, wq_dev, wi_dev)

      # Negative control 1: a broken rule that shards attention activations only on "expert"
      # replicates MLA activations across `fsdp=2`, doubling per-device activation memory.
      @jax.jit
      def broken_mla_activation_step(x_in, wq_in):
        q_latent = jnp.einsum("bsd,dq->bsq", x_in, wq_in)
        return jax.lax.with_sharding_constraint(q_latent, P("expert", None, None))

      q_broken = broken_mla_activation_step(x_dev, wq_dev)

      # Negative control 2: a broken rule that leaves `wi_in` replicated in output forces an all-gather in HLO.
      hlo_good = (
          jax.jit(lambda w: w * 2.0, in_shardings=(wi_shd,), out_shardings=wi_shd).lower(wi_dev).compile().as_text()
      )
      replicated_shd = NamedSharding(mesh, P(None, None, None))
      hlo_bad = (
          jax.jit(lambda w: w * 2.0, in_shardings=(wi_shd,), out_shardings=replicated_shd)
          .lower(wi_dev)
          .compile()
          .as_text()
      )

    # Positive assertions:
    # 1. MLA activation `q_out` is sharded across fsdp=2 x expert=2 -> 1 batch row per device `(1, 8, 8)`.
    self.assertEqual(q_out.sharding.shard_shape(q_out.shape), (1, seq_len, q_lora_dim))
    # 2. Expert weights `wi_out` stay sharded across ("fsdp", "expert") -> `(2, 16, 32)` per device, zero all-gather.
    self.assertEqual(wi_out.sharding.shard_shape(wi_out.shape), (2, emb_dim, mlp_dim))
    self.assertNotIn("all-gather", hlo_good)

    # Negative-control assertions:
    # 1. Broken MLA activation rule replicates across `fsdp=2` -> `(2, 8, 8)` per device (2x activation memory).
    self.assertEqual(q_broken.sharding.shard_shape(q_broken.shape), (2, seq_len, q_lora_dim))
    # 2. Broken expert output rule emits an `all-gather` in compiled HLO.
    self.assertIn("all-gather", hlo_bad)

  def test_tpu7x_128_virtual_topology_abstract_sharding_and_hlo(self):
    """Verifies abstract shard shapes and lowered HLO on virtual 128-chip `tpu7x-128` topologies."""
    # `tpu7x-128` (128 chips): fsdp=16, expert=8 (16 * 8 = 128).
    tpu7x_128_ep_mesh = jax.sharding.AbstractMesh(
        (1, 1, 16, 1, 1, 1, 1, 8),
        MESH_AXIS_NAMES,
        axis_types=(jax.sharding.AxisType.Explicit,) * len(MESH_AXIS_NAMES),
    )
    mla_comp = FSDPasDPForAttn(compound_fsdp_ep=True)
    with jax.sharding.use_abstract_mesh(tpu7x_128_ep_mesh):
      q_lora_spec = mla_comp("q_lora", ("q_lora", "q_lora_up_proj"))
      mla_act_spec = mla_comp("q_latent", ("batch", "length", "heads", "head_dim"), TensorType.ACTIVATION)
      wi_comp_spec = mla_comp("wi_0", ("experts", "expert_embed", "expert_mlp"))
      router_spec = mla_comp("router", ("embed_router", None))

    self.assertEqual(q_lora_spec, P("expert", None))
    self.assertEqual(mla_act_spec, P(("fsdp", "expert"), None, None, None))
    self.assertEqual(wi_comp_spec, P(("fsdp", "expert"), None, ("tensor", "tensor_sequence")))
    self.assertEqual(router_spec, P(("fsdp", "fsdp_transpose"), None))

    # Verify per-chip shard shapes at DeepSeek/GLM scale on `tpu7x-128` (256 experts, 7168 emb, 1536 q_lora, 2048 mlp):
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, q_lora_spec).shard_shape((7168, 1536)), (896, 1536))
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, mla_act_spec).shard_shape((128, 4096, 16, 128)), (1, 4096, 16, 128))
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, wi_comp_spec).shard_shape((256, 7168, 2048)), (2, 7168, 2048))
    self.assertEqual(NamedSharding(tpu7x_128_ep_mesh, router_spec).shard_shape((7168, 256)), (448, 256))

    # Also check `EPasCP`, `CPasEP`, and `TPasEP` on a multi-axis 128-chip virtual mesh
    # (`fsdp=8, fsdp_transpose=2, context=2, tensor=2, expert=2` -> 8*2*2*2*2 = 128 chips).
    tpu7x_128_hybrid_mesh = jax.sharding.AbstractMesh(
        (1, 1, 8, 2, 2, 2, 1, 2),
        MESH_AXIS_NAMES,
    )
    ep_cp = EPasCP()
    cp_ep = CPasEP()
    tp_ep = TPasEP()

    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(
                ep_cp("norm_in", ("batch", "norm_length", "embed"), TensorType.ACTIVATION), tpu7x_128_hybrid_mesh
            ),
        ).shard_shape((128, 4096, 7168)),
        (8, 1024, 7168),
    )
    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(cp_ep("wi_0", ("experts", "expert_embed", "expert_mlp")), tpu7x_128_hybrid_mesh),
        ).shard_shape((256, 7168, 2048)),
        (64, 448, 1024),
    )
    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(tp_ep("wi_0", ("experts", "expert_embed", "expert_mlp")), tpu7x_128_hybrid_mesh),
        ).shard_shape((256, 7168, 2048)),
        (64, 224, 2048),
    )
    self.assertEqual(
        NamedSharding(
            tpu7x_128_hybrid_mesh,
            project_spec_to_mesh(
                TPasEP(muon_friendly=True)("wi_0", ("experts", "expert_embed", "expert_mlp")), tpu7x_128_hybrid_mesh
            ),
        ).shard_shape((256, 7168, 2048)),
        (64, 7168, 2048),
    )

    # Verify `sharding_from_config` resolved outside `with mesh:` does not replicate across `fsdp_transpose=2`
    # when placed on a 4-CPU 2D FSDP x EP mesh (`fsdp=1, fsdp_transpose=2, expert=2`).
    mesh_2d = create_mesh(ici={"fsdp": 1, "fsdp_transpose": 2, "expert": 2}, devices=self.devices)
    with jax.sharding.set_mesh(None):
      s_outside = sharding_from_config(
          SimpleNamespace(
              sharding_preset="fsdp_as_dp_for_attn",
              compound_fsdp_ep=True,
              ici_fsdp_transpose_parallelism=2,
          )
      )
      spec_outside = s_outside("wi_0", ("experts", "expert_embed", "expert_mlp"))
    arr_good = jax.device_put(jnp.zeros((8, 16, 32), dtype=jnp.float32), NamedSharding(mesh_2d, spec_outside))
    arr_bad_1d = jax.device_put(
        jnp.zeros((8, 16, 32), dtype=jnp.float32),
        NamedSharding(mesh_2d, P(("fsdp", "expert"), None, ("tensor", "tensor_sequence"))),
    )
    self.assertEqual(arr_good.sharding.shard_shape((8, 16, 32)), (2, 16, 32))
    self.assertEqual(arr_bad_1d.sharding.shard_shape((8, 16, 32)), (4, 16, 32))

    # Confirm sharding module remains free of numpy and flax.linen imports.
    self.assertNotIn("np", vars(sharding_lib))
    self.assertNotIn("numpy", vars(sharding_lib))
    self.assertFalse(any(k == "linen" for k in vars(sharding_lib)))


if __name__ == "__main__":
  absltest.main()
