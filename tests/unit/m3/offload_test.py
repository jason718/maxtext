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

"""Unit tests for `maxtext.m3.infra.offload` and runtime TPU HBM telemetry."""

import re
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.experimental.topologies import get_topology_desc
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.m3.core.sharding import TensorType, mesh_from_config, sharding_from_config
from maxtext.m3.infra import offload as m3_offload
from maxtext.m3.infra import quantization as m3_quant
from maxtext.m3.train import loop
from maxtext.trainers.pre_train import train as legacy_train
from maxtext.training_engine import memory_report
from maxtext.utils import train_utils
from maxtext.utils import accelerator_to_spec_map
from maxtext.utils import max_logging
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path


def _tiny_m3_config(output_dir: str, **overrides):
  """Creates a tiny M3 configuration for testing."""
  args = {
      "model_name": "qwen3-0.6b",
      "use_m3_model": "true",
      "override_model_config": "true",
      "base_emb_dim": "64",
      "base_mlp_dim": "128",
      "base_num_decoder_layers": "2",
      "base_num_query_heads": "4",
      "base_num_kv_heads": "2",
      "head_dim": "16",
      "vocab_size": "256",
      "per_device_batch_size": "2",
      "max_target_length": "16",
      "steps": "4",
      "learning_rate": "1e-3",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_offload_test",
  }
  num_devices = len(jax.devices())
  if num_devices >= 4:
    args["ici_tensor_parallelism"] = "2"
    args["ici_fsdp_parallelism"] = str(num_devices // 2)
  else:
    args["ici_fsdp_parallelism"] = str(num_devices)
  args.update({k: str(v) for k, v in overrides.items()})
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


def _param_paths(model: nnx.Module) -> list[tuple]:
  return [path for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))]


def _param_numpy_map(model: nnx.Module) -> dict[tuple, np.ndarray]:
  return {path: np.asarray(var[...]) for path, var in nnx.to_flat_state(nnx.state(model, nnx.Param))}


class ToMemoryTest(parameterized.TestCase):
  """Tests `to_memory` across NNX variables, sharded arrays, tracers, and scalars."""

  def test_preserves_nnx_variable_subclass_and_metadata(self):
    devices = np.array(jax.devices()).reshape(-1)
    mesh = Mesh(devices, ("fsdp",))
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      var = nnx.Param(
          jnp.ones((len(devices) * 4, 8), dtype=jnp.float32),
          out_sharding=P("fsdp", None),
          eager_sharding=True,
          custom_tag="keep_me",
      )
      host_var = m3_offload.to_memory(var, "pinned_host")
      self.assertIsInstance(host_var, nnx.Param)
      self.assertEqual(host_var.get_metadata("out_sharding"), P("fsdp", None))
      self.assertEqual(host_var.get_metadata("custom_tag"), "keep_me")
      self.assertEqual(host_var[...].sharding.memory_kind, "pinned_host")
      self.assertEqual(host_var[...].sharding.spec, P("fsdp", None))

      dev_var = m3_offload.to_memory(host_var, "device")
      self.assertEqual(dev_var[...].sharding.memory_kind, "device")
      self.assertEqual(dev_var[...].sharding.spec, P("fsdp", None))
      np.testing.assert_array_equal(np.asarray(dev_var[...]), np.ones((len(devices) * 4, 8), dtype=np.float32))

  def test_handles_sharded_arrays_shape_dtype_structs_tracers_and_scalars(self):
    dev0 = jax.local_devices()[0]
    single_sharding = SingleDeviceSharding(dev0)
    sds = jax.ShapeDtypeStruct((4, 4), jnp.float32, sharding=single_sharding)
    tree = {
        "arr": jax.device_put(jnp.arange(8, dtype=jnp.float32), single_sharding),
        "sds": sds,
        "sds_plain": jax.ShapeDtypeStruct((2,), jnp.int32),
        "np_arr": np.array([1.0, 2.0], dtype=np.float32),
        "none": None,
        "int": 7,
        "float": 3.5,
        "str": "meta",
    }
    moved = m3_offload.to_memory(tree, "pinned_host")
    self.assertEqual(moved["arr"].sharding.memory_kind, "pinned_host")
    self.assertEqual(moved["sds"].sharding.memory_kind, "pinned_host")
    self.assertIsNone(getattr(moved["sds_plain"], "sharding", None))
    self.assertEqual(moved["np_arr"].sharding.memory_kind, "pinned_host")
    self.assertIsNone(moved["none"])
    self.assertEqual(moved["int"], 7)
    self.assertEqual(moved["float"], 3.5)
    self.assertEqual(moved["str"], "meta")

    @jax.jit
    def roundtrip_traced(x):
      on_host = m3_offload.to_memory(x, "pinned_host")
      return m3_offload.to_memory(on_host, "device") + 1.0

    out = roundtrip_traced(jnp.ones((4,), dtype=jnp.float32))
    self.assertEqual(out.sharding.memory_kind, "device")
    np.testing.assert_allclose(np.asarray(out), np.full((4,), 2.0, dtype=np.float32))

  def test_rejects_unknown_memory_kind(self):
    """Negative control: invalid memory kind must fail loudly."""
    with self.assertRaisesRegex(ValueError, "Unsupported memory kind"):
      m3_offload.to_memory(jnp.ones((2,)), "hbm_magic")


class HostOffloadParityTest(parameterized.TestCase):
  """Verifies path-transparency and step-for-step loss/weight parity across offload modes."""

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path

  @parameterized.named_parameters(
      ("unscanned", False),
      ("scanned", True),
  )
  def test_all_offload_modes_preserve_paths_and_match_step_trajectory(self, scan_layers: bool):
    base_cfg = _tiny_m3_config(
        self.output_dir,
        scan_layers=scan_layers,
        optimizer_memory_host_offload=False,
        parameter_memory_host_offload=False,
    )
    mesh = mesh_from_config(base_cfg)
    sharding = sharding_from_config(base_cfg)
    batches = [
        loop.make_synthetic_batch(base_cfg, mesh, sharding, jax.random.fold_in(jax.random.key(0), s)) for s in range(4)
    ]

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"), nnx.use_eager_sharding(True):
      base_model, base_opt = loop.setup_model(base_cfg, mesh, sharding=sharding)
      base_cls = type(base_model)
      base_paths = _param_paths(base_model)
      base_losses = [float(loop.train_step(base_model, base_opt, b)) for b in batches]
      base_final_params = _param_numpy_map(base_model)

      for opt_offload, param_offload in ((True, False), (False, True), (True, True)):
        cfg = _tiny_m3_config(
            self.output_dir,
            scan_layers=scan_layers,
            optimizer_memory_host_offload=opt_offload,
            parameter_memory_host_offload=param_offload,
        )
        model, opt = loop.setup_model(cfg, mesh, sharding=sharding)
        self.assertIsInstance(model, base_cls)
        self.assertEqual(_param_paths(model), base_paths)

        param_kinds = {
            v[...].sharding.memory_kind
            for v in jax.tree.leaves(nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
        }
        expected_param_kind = "pinned_host" if param_offload else "device"
        self.assertEqual(param_kinds, {expected_param_kind})

        opt_arrays = [
            v[...] if isinstance(v, nnx.Variable) else v
            for v in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
            if hasattr(v[...] if isinstance(v, nnx.Variable) else v, "sharding")
        ]
        self.assertNotEmpty(opt_arrays)
        expected_opt_kind = "pinned_host" if opt_offload else "device"
        self.assertTrue(all(a.sharding.memory_kind == expected_opt_kind for a in opt_arrays))

        losses = [float(loop.train_step(model, opt, b)) for b in batches]
        np.testing.assert_allclose(losses, base_losses, rtol=1e-6, atol=1e-6)

        # Verify post-step memory kinds remain parked in their configured spaces.
        post_param_kinds = {
            v[...].sharding.memory_kind
            for v in jax.tree.leaves(nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
        }
        self.assertEqual(post_param_kinds, {expected_param_kind})
        post_opt_arrays = [
            v[...] if isinstance(v, nnx.Variable) else v
            for v in jax.tree.leaves(opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable))
            if hasattr(v[...] if isinstance(v, nnx.Variable) else v, "sharding")
        ]
        self.assertTrue(all(a.sharding.memory_kind == expected_opt_kind for a in post_opt_arrays))

        final_params = _param_numpy_map(model)
        for path, base_arr in base_final_params.items():
          np.testing.assert_allclose(final_params[path], base_arr, rtol=1e-6, atol=1e-6)

      # Negative control: perturbing learning_rate must break trajectory parity.
      perturbed_cfg = _tiny_m3_config(
          self.output_dir,
          scan_layers=scan_layers,
          learning_rate="5e-2",
          optimizer_memory_host_offload=True,
          parameter_memory_host_offload=False,
      )
      perturbed_model, perturbed_opt = loop.setup_model(perturbed_cfg, mesh, sharding=sharding)
      perturbed_losses = [float(loop.train_step(perturbed_model, perturbed_opt, b)) for b in batches]
      self.assertFalse(np.allclose(perturbed_losses[1:], base_losses[1:], rtol=1e-4, atol=1e-4))

  def test_model_creation_utils_wires_parameter_offload_for_train_mode_only(self):
    cfg = _tiny_m3_config(self.output_dir, scan_layers=False, parameter_memory_host_offload=True)
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
      train_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      self.assertIsInstance(train_model, m3_offload.HostOffloadedParameters)
      for var in jax.tree.leaves(nnx.state(train_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        self.assertEqual(var[...].sharding.memory_kind, "pinned_host")

      batch_size = max(2, mesh.shape["fsdp"])
      tokens = jnp.ones((batch_size, 8), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32)[None, :], (batch_size, 8))
      logits = train_model(tokens, decoder_positions=positions)
      self.assertEqual(logits.shape, (batch_size, 8, cfg.vocab_size))
      hidden = jnp.ones((batch_size, 8, cfg.emb_dim), dtype=jnp.float32)
      tiled_logits = train_model.logits_from_hidden_states_for_vocab_tiling(hidden, deterministic=True)
      self.assertEqual(tiled_logits.shape, (batch_size, 8, cfg.vocab_size))
      for var in jax.tree.leaves(nnx.state(train_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        self.assertEqual(var[...].sharding.memory_kind, "pinned_host")

      prefill_model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_PREFILL, rngs=nnx.Rngs(0))
      self.assertNotIsInstance(prefill_model, m3_offload.HostOffloadedParameters)
      for var in jax.tree.leaves(nnx.state(prefill_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)):
        self.assertEqual(var[...].sharding.memory_kind, "device")

  def test_vocab_tiling_hook_raises_cleanly_when_missing_on_parent(self):
    class _MinimalModule(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))

      def __call__(self, x):
        return x @ self.w[...]

    class _Cfg:
      parameter_memory_host_offload = True

    mod = m3_offload.maybe_offload_parameters(_MinimalModule(), _Cfg())
    with self.assertRaisesRegex(AttributeError, "logits_from_hidden_states_for_vocab_tiling"):
      mod.logits_from_hidden_states_for_vocab_tiling(jnp.ones((2, 2)))


class RuntimeHbmTelemetryTest(absltest.TestCase):
  """Tests `get_device_hbm_stats` and `loop._log_step` HBM reporting."""

  def test_cpu_and_empty_memory_stats_return_none(self):
    self.assertIsNone(m3_offload.get_device_hbm_stats())

    class _NoneStatsDevice:

      def memory_stats(self):
        return None

    class _IncompleteStatsDevice:

      def memory_stats(self):
        return {"bytes_reserved": 1024}

    class _NonAddressablePathwaysDevice:

      def memory_stats(self):
        raise jax.errors.JaxRuntimeError("INVALID_ARGUMENT: MemoryStats is only supported for addressable PjRt devices.")

    self.assertIsNone(m3_offload.get_device_hbm_stats(_NoneStatsDevice()))
    self.assertIsNone(m3_offload.get_device_hbm_stats(_IncompleteStatsDevice()))
    self.assertIsNone(m3_offload.get_device_hbm_stats(_NonAddressablePathwaysDevice()))

  def test_tpu_memory_stats_captures_reserved_workspace_and_logs_in_step(self):
    gib = 2**30

    # Mirrors DSv4.1 TPU7x runtime observation: `peak_bytes_in_use` reports 48 GiB
    # while XLA holds 46 GiB of HLO temporary workspace in `bytes_reserved` on top
    # of 40 GiB live buffers (`allocated_bytes = 86 GiB`).
    class _MockTpuDevice:

      def memory_stats(self):
        return {
            "bytes_in_use": 40 * gib,
            "bytes_reserved": 46 * gib,
            "peak_bytes_in_use": 48 * gib,
        }

    mock_tpu = _MockTpuDevice()
    stats = m3_offload.get_device_hbm_stats(mock_tpu)
    self.assertIsNotNone(stats)
    assert stats is not None
    self.assertEqual(stats["allocated_bytes"], float(86 * gib))
    self.assertEqual(stats["peak_bytes_in_use"], float(48 * gib))
    self.assertEqual(stats["peak_allocated_bytes"], float(86 * gib))
    self.assertAlmostEqual(stats["hbm_allocated_gib"], 86.0)
    self.assertAlmostEqual(stats["hbm_peak_in_use_gib"], 48.0)
    # Negative control: `peak_bytes_in_use` alone misses 38 GiB of reserved HBM.
    self.assertGreater(stats["hbm_allocated_gib"] - stats["hbm_peak_in_use_gib"], 30.0)

    with mock.patch.object(max_logging, "log") as log_spy:
      loop._log_step(0, jnp.asarray(2.5, dtype=jnp.float32), 0.0, 1024, device=mock_tpu)  # pylint: disable=protected-access
      tpu_msg = log_spy.call_args[0][0]
      self.assertIn("hbm_allocated_gib=86.00", tpu_msg)
      self.assertIn("hbm_peak_in_use_gib=48.00", tpu_msg)

      log_spy.reset_mock()
      loop._log_step(1, jnp.asarray(2.4, dtype=jnp.float32), 0.0, 1024)  # pylint: disable=protected-access
      cpu_msg = log_spy.call_args[0][0]
      self.assertNotIn("hbm_allocated_gib", cpu_msg)

  def test_eval_shape_preserves_pinned_host_sharding_for_aot_compile(self):
    out_dir = self.create_tempdir().full_path
    cfg = _tiny_m3_config(
        out_dir,
        optimizer_memory_host_offload="true",
        parameter_memory_host_offload="true",
    )
    mesh = mesh_from_config(cfg)
    with jax.set_mesh(mesh):
      abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, mesh))

    param_vars = [
        v
        for v in jax.tree.leaves(nnx.state(abs_model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable)
    ]
    opt_vars = [
        v
        for v in jax.tree.leaves(nnx.state(abs_opt, nnx.OptState), is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(v, nnx.Variable) and v.get_raw_value().ndim > 0
    ]
    self.assertNotEmpty(param_vars)
    self.assertNotEmpty(opt_vars)
    for var in param_vars:
      self.assertEqual(var.get_raw_value().sharding.memory_kind, "pinned_host")
    for var in opt_vars:
      self.assertEqual(var.get_raw_value().sharding.memory_kind, "pinned_host")

  def test_tpu7x_aot_compile_reduces_device_argument_and_kernel_device_peak(self):
    out_dir = self.create_tempdir().full_path
    import os  # pylint: disable=import-outside-toplevel

    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    th = accelerator_to_spec_map.get_system_characteristics("tpu7x-8")
    topo = get_topology_desc(
        platform=th.platform,
        topology_name=th.topology_name,
        chip_config_name=th.chip_config_name,
        chips_per_host_bounds=th.chips_per_host_bounds,
        num_slices=1,
        wrap=th.wrap,
    )

    def _compile_step(offload_opt: bool, offload_param: bool = False):
      cfg = _tiny_m3_config(
          out_dir,
          compile_topology="tpu7x-8",
          ici_fsdp_parallelism="4",
          ici_tensor_parallelism="2",
          optimizer_memory_host_offload=str(offload_opt).lower(),
          parameter_memory_host_offload=str(offload_param).lower(),
      )
      tpu_mesh = mesh_from_config(cfg, devices=topo.devices)
      sharding = sharding_from_config(cfg)
      with jax.set_mesh(tpu_mesh):
        data_spec = sharding("tokens", ("batch", "length"), TensorType.ACTIVATION)
        data_sharding = NamedSharding(tpu_mesh, data_spec)
        batch_shape = (int(cfg.per_device_batch_size) * len(topo.devices), int(cfg.max_target_length))
        batch = {
            "inputs": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "targets": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "positions": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
        }
        abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, tpu_mesh))
        lowered = loop.train_step.lower(abs_model, abs_opt, batch)
      compiled = lowered.compile()
      rows = memory_report.memory_report({"update": compiled}, {})
      return compiled.memory_analysis(), rows[0], memory_report.format_memory_report(rows)

    # Negative control: optimizer_memory_host_offload=False keeps all optimizer moments in HBM.
    mem_a, row_a, report_a = _compile_step(offload_opt=False, offload_param=False)
    mem_b, row_b, report_b = _compile_step(offload_opt=True, offload_param=False)
    _, row_c, _ = _compile_step(offload_opt=True, offload_param=True)

    self.assertEqual(mem_a.host_argument_size_in_bytes, 0)
    self.assertEqual(row_a.host_argument, 0)
    self.assertIn(memory_report.PEAK_LABEL, report_a)
    self.assertIn(memory_report.PEAK_LABEL, report_b)

    # Offloading FP32 AdamW moments moves ~2/3 of static state bytes from HBM to S(5) pinned_host
    # and reduces device-resident kernel peak (`KernelMemory.device_total`).
    offloaded_opt_bytes = row_a.argument - row_b.argument
    self.assertGreater(offloaded_opt_bytes, int(0.6 * row_a.argument))
    self.assertEqual(row_b.host_argument, offloaded_opt_bytes)
    self.assertEqual(mem_b.host_argument_size_in_bytes, offloaded_opt_bytes)
    self.assertLess(row_b.argument, row_a.argument)

    # Offloading parameters in addition to optimizer state further reduces device argument and resident bytes.
    self.assertLess(row_c.argument, row_b.argument)
    self.assertGreater(row_c.host_argument, row_b.host_argument)

  def test_legacy_train_step_runtime_keeps_optimizer_in_pinned_host_across_steps(self):
    out_dir = self.create_tempdir().full_path

    def _run_legacy_steps(offload_opt: bool):
      cfg = _tiny_m3_config(
          out_dir,
          dataset_type="synthetic",
          optimizer_memory_host_offload=str(offload_opt).lower(),
      )
      mesh = mesh_from_config(cfg)
      with jax.set_mesh(mesh):
        (
            _,
            _,
            state_mesh_shardings,
            _,
            mesh,
            _,
            data_iterator,
            _,
            _,
            _,
            train_state,
        ) = train_utils.setup_train_loop(cfg, None, mesh=mesh)
        jit_model, state = nnx.split(train_state)
        p_train_step, _ = train_utils.jit_train_and_eval_step(
            cfg,
            jit_model,
            mesh,
            state,
            state_mesh_shardings,
            legacy_train.train_step,
        )
        batch = next(data_iterator)
        init_opt_kinds = {
            x.sharding.memory_kind
            for x in jax.tree.leaves(state.optimizer)
            if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
        }
        state, metrics0 = p_train_step(state, batch)
        step0_opt_kinds = {
            x.sharding.memory_kind
            for x in jax.tree.leaves(state.optimizer)
            if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
        }
        step0_param_kinds = {x.sharding.memory_kind for x in jax.tree.leaves(state.model) if hasattr(x, "sharding")}
        state, metrics1 = p_train_step(state, batch)
      step1_opt_kinds = {
          x.sharding.memory_kind
          for x in jax.tree.leaves(state.optimizer)
          if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0
      }
      losses = [float(metrics0["scalar"]["learning/loss"]), float(metrics1["scalar"]["learning/loss"])]
      return init_opt_kinds, step0_opt_kinds, step1_opt_kinds, step0_param_kinds, losses

    # Negative control: optimizer_memory_host_offload=False keeps optimizer state on device across steps.
    init_dev, step0_dev, step1_dev, param_dev, losses_dev = _run_legacy_steps(offload_opt=False)
    init_host, step0_host, step1_host, param_host, losses_host = _run_legacy_steps(offload_opt=True)

    self.assertEqual(init_dev, {"device"})
    self.assertEqual(step0_dev, {"device"})
    self.assertEqual(step1_dev, {"device"})
    self.assertEqual(param_dev, {"device"})

    # Even though train.py:673 device_puts opt_state to device inside train_step,
    # get_functional_train_with_signature binds out_shardings=(state_mesh_shardings, None),
    # returning updated AdamW moments in pinned_host after step 0 and step 1.
    self.assertEqual(init_host, {"pinned_host"})
    self.assertEqual(step0_host, {"pinned_host"})
    self.assertEqual(step1_host, {"pinned_host"})
    self.assertEqual(param_host, {"device"})
    np.testing.assert_allclose(losses_host, losses_dev, rtol=1e-6, atol=1e-6)

  def test_offloaded_model_class_vocab_tiling_attribute(self):  # pylint: disable=protected-access
    """Verifies `logits_from_hidden_states_for_vocab_tiling` is only present when base class defines it."""

    class _BaseWithoutTiling(nnx.Module):

      def __call__(self, x):
        return x

    class _BaseWithTiling(nnx.Module):

      def __call__(self, x):
        return x

      def logits_from_hidden_states_for_vocab_tiling(self, x):
        return x * 2

    offloaded_cls_without = m3_offload._offloaded_model_class(_BaseWithoutTiling)  # pylint: disable=protected-access
    offloaded_cls_with = m3_offload._offloaded_model_class(_BaseWithTiling)  # pylint: disable=protected-access
    self.assertFalse(hasattr(offloaded_cls_without, "logits_from_hidden_states_for_vocab_tiling"))
    self.assertTrue(hasattr(offloaded_cls_with, "logits_from_hidden_states_for_vocab_tiling"))

  def test_create_model_preserves_qwix_quantization_with_parameter_offload(self):
    """Verifies `model_creation_utils.create_model` keeps Qwix quantization when parameter offload is enabled."""
    toks = jnp.ones((2, 16), dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (2, 16))
    for scan_layers in (True, False):
      for remat_pol in ("none", "full"):
        cfg = _tiny_m3_config(
            self.create_tempdir().full_path,
            scan_layers=str(scan_layers).lower(),
            remat_policy=remat_pol,
            quantization="int8",
            use_qwix_quantization="true",
            parameter_memory_host_offload="true",
            optimizer_memory_host_offload="true",
        )
        mesh = mesh_from_config(cfg)
        with jax.set_mesh(mesh):
          model = model_creation_utils.create_model(cfg, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
          self.assertIsInstance(model, m3_offload.HostOffloadedParameters)
          self.assertTrue(hasattr(type(model), "_unquantized_type"))
          param_kinds = {
              v.get_raw_value().sharding.memory_kind
              for v in jax.tree.leaves(nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable))
              if isinstance(v, nnx.Variable) and hasattr(v.get_raw_value(), "sharding")
          }
          self.assertEqual(param_kinds, {"pinned_host"})
          self.assertGreater(m3_quant.quantized_matmuls(model, jnp.int8, toks, pos), 0)
          out = model(toks, pos)
          self.assertEqual(out.shape, (2, 16, 256))


def _qwen38_mini_config(output_dir: str, **overrides):
  """Creates a compact `qwen3.8-flash-next-mini` M3 configuration for selective offload tests."""
  args = {
      "model_name": "qwen3.8-flash-next-mini",
      "override_model_config": "true",
      "use_m3_model": "true",
      "per_device_batch_size": "1",
      "max_target_length": "16",
      "steps": "4",
      "learning_rate": "1e-3",
      "gradient_clipping_threshold": "1.0",
      "adam_weight_decay": "0.1",
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "enable_checkpointing": "false",
      "base_output_directory": output_dir,
      "run_name": "m3_selective_offload_test",
  }
  num_devices = len(jax.devices())
  if num_devices >= 4:
    args["ici_fsdp_parallelism"] = "2"
    args["ici_expert_parallelism"] = str(num_devices // 2)
  else:
    args["ici_fsdp_parallelism"] = str(num_devices)
  args.update({k: str(v) for k, v in overrides.items()})
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


class SelectiveHostOffloadTest(parameterized.TestCase):
  """Tests pattern-selected pinned-host offload for large embedding tables (e.g. Qwen3.8 PLE)."""

  def test_validation_fails_closed(self):
    out_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(ValueError, "host_offload_param_patterns"):
      _qwen38_mini_config(out_dir, selective_host_offload="true", host_offload_param_patterns="[]")

    with self.assertRaisesRegex(ValueError, "parameter_memory_host_offload"):
      _qwen38_mini_config(
          out_dir,
          selective_host_offload="true",
          parameter_memory_host_offload="true",
      )

    cfg_unmatched = _qwen38_mini_config(
        out_dir,
        selective_host_offload="true",
        host_offload_param_patterns='["nonexistent_param_path_xyz$"]',
    )
    mesh = mesh_from_config(cfg_unmatched)
    with jax.set_mesh(mesh):
      with self.assertRaisesRegex(ValueError, "matched 0 parameters"):
        loop.setup_model(cfg_unmatched, mesh)

    with self.assertRaises(ValueError):
      _qwen38_mini_config(
          out_dir,
          selective_host_offload="true",
          selective_host_offload_optimizer_mode="unknown_mode",
      )

  def test_selective_offload_fails_closed_instead_of_silent_fallback(self):
    """Verifies missing tokens, pre-gather shape mismatch, and dense lazy_sparse raise ValueError."""
    import importlib.util  # pylint: disable=import-outside-toplevel
    import optax  # pylint: disable=import-outside-toplevel

    out_dir = self.create_tempdir().full_path
    cfg = _qwen38_mini_config(out_dir, scan_layers="true", selective_host_offload="true")
    mesh = mesh_from_config(cfg)
    sharding = sharding_from_config(cfg)

    with jax.set_mesh(mesh):
      model, _ = loop.setup_model(cfg, mesh, sharding)
      bad_batch = {"targets": jnp.zeros((4, 16), dtype=jnp.int32)}

      # 1. value_and_grad / pure_value_and_grad / engine_value_and_grad refuse dense fallback when tokens=None.
      with self.assertRaisesRegex(ValueError, "requires token IDs"):
        m3_offload.value_and_grad(loop.loss_fn, model, bad_batch)

      graphdef, params, rest = nnx.split(model, nnx.Param, ...)
      with self.assertRaisesRegex(ValueError, "requires token IDs"):
        m3_offload.pure_value_and_grad(
            lambda p, cp, r, c, d: (jnp.array(0.0), None),
            graphdef,
            params,
            custom_params=nnx.State({}),
            rest=rest,
            config=cfg,
            data=bad_batch,
        )

      with self.assertRaisesRegex(ValueError, "requires token IDs"):
        m3_offload.engine_value_and_grad(
            lambda p, r, b: (jnp.array(0.0), (None, r)),
            graphdef,
            params,
            rest,
            bad_batch,
        )

      # 2. HostOffloadedEmbed refuses pre-gather shape mismatch instead of bypassing pre_gathered.
      ple_embeds = m3_offload._collect_host_embeds(model)  # pylint: disable=protected-access
      self.assertEqual(len(ple_embeds), 1)
      ple_path, ple_embed, _ = ple_embeds[0]
      dummy_pre_ids = jnp.zeros((2, 4), dtype=jnp.int32)
      dummy_pre_vals = jnp.zeros((2, 4, ple_embed.features), dtype=jnp.float32)
      with m3_offload._pregathered_context({ple_path: (dummy_pre_ids, dummy_pre_vals)}):  # pylint: disable=protected-access
        with self.assertRaisesRegex(ValueError, "pre-gather shape mismatch"):
          ple_embed(jnp.zeros((2, 5), dtype=jnp.int32))

      # 3. lazy_sparse rejects dense host gradients both at config/make_optimizer time and in update().
      cfg_ls = _qwen38_mini_config(
          out_dir,
          scan_layers="true",
          selective_host_offload="true",
          selective_host_offload_optimizer_mode="lazy_sparse",
      )
      with self.assertRaisesRegex(ValueError, "incompatible with dense pinned_host gradients"):
        m3_offload.make_optimizer(model, optax.adamw(1e-3), cfg_ls, dense_host_grads=True)

      opt_ls = m3_offload.make_optimizer(model, optax.adamw(1e-3), cfg_ls)
      dense_grads = jax.tree.map(
          lambda v: (
              v.replace(
                  m3_offload.to_memory(jnp.zeros_like(v.get_raw_value()), "pinned_host")
                  if m3_offload._is_host_leaf(v)  # pylint: disable=protected-access
                  else jnp.zeros_like(v.get_raw_value())
              )
              if isinstance(v, nnx.Variable)
              else jnp.zeros_like(v)
          ),
          nnx.state(model, nnx.Param),
          is_leaf=lambda x: isinstance(x, nnx.Variable),
      )
      with self.assertRaisesRegex(ValueError, "requires SparseHostGrad"):
        opt_ls.update(model, dense_grads)

    # 4. AbstractMaxTextEngine fails closed at construction time for lazy_sparse (when tunix is installed).
    if importlib.util.find_spec("tunix") is not None:
      from maxtext.trainers.pre_train import train_compile  # pylint: disable=import-outside-toplevel
      from maxtext.training_engine import maxtext_engine_compile  # pylint: disable=import-outside-toplevel

      cfg_eng_ls = _qwen38_mini_config(
          out_dir,
          compile_topology="tpu7x-8",
          compile_topology_num_slices="1",
          ici_fsdp_parallelism="4",
          ici_expert_parallelism="2",
          scan_layers="true",
          selective_host_offload="true",
          selective_host_offload_optimizer_mode="lazy_sparse",
      )
      tpu_mesh = train_compile.get_topology_mesh(cfg_eng_ls)
      with self.assertRaisesRegex(ValueError, "incompatible with dense pinned_host gradients"):
        maxtext_engine_compile.AbstractMaxTextEngine(cfg_eng_ls, tpu_mesh)

  def test_sharded_2d_embed_gather_sqnorm_clip_and_optimizer_modes(self):
    """Tests 2-axis row-sharded P(('fsdp', 'ep'), None) table with non-divisible n_local and duplicate IDs."""
    devices = np.array(jax.devices())
    if len(devices) >= 4:
      mesh = Mesh(devices[:4].reshape(2, 2), ("fsdp", "ep"))
      row_spec = ("fsdp", "ep")
      n_local = 37  # Odd & non-divisible by num_chunks=5 (mirrors 320,001,920 / 128 = 2,500,015).
      vocab_size = n_local * 4
    else:
      mesh = Mesh(devices.reshape(-1, 1), ("fsdp", "ep"))
      row_spec = ("fsdp",)
      n_local = 37
      vocab_size = n_local * len(devices)
    width = 16

    class _ToyEmbedModel(nnx.Module):
      """Minimal embedding + linear test module."""

      def __init__(self, rngs: nnx.Rngs):
        self.ple_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=width,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            embedding_init=nnx.initializers.normal(0.1),
            rngs=rngs,
        )
        self.proj = nnx.Linear(width, width, use_bias=False, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

      def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.proj(self.ple_embedding(inputs))

    class _ToyCfg:
      parameter_memory_host_offload = False
      optimizer_memory_host_offload = False
      selective_host_offload = True
      host_offload_param_patterns = ("(?:^|/)ple_embedding/embedding$",)
      selective_host_offload_optimizer_mode = "host_compute"
      selective_host_offload_num_chunks = 5

    def _build(selective: bool, opt_mode: str = "host_compute"):
      model = _ToyEmbedModel(nnx.Rngs(0))
      sh_tbl = NamedSharding(mesh, P(row_spec, None))
      model.ple_embedding.embedding.set_metadata("out_sharding", P(row_spec, None))
      model.ple_embedding.embedding.set_raw_value(jax.device_put(model.ple_embedding.embedding.get_raw_value(), sh_tbl))
      model.proj.kernel.set_raw_value(
          jax.device_put(model.proj.kernel.get_raw_value(), NamedSharding(mesh, P(None, None)))
      )
      import optax  # pylint: disable=import-outside-toplevel

      base_tx = optax.adamw(learning_rate=1e-2, weight_decay=0.1)
      if not selective:
        tx = optax.chain(optax.clip_by_global_norm(0.5), base_tx)
        return model, nnx.Optimizer(model, tx, wrt=nnx.Param)
      cfg = _ToyCfg()
      cfg.selective_host_offload_optimizer_mode = opt_mode
      model = m3_offload.maybe_offload_parameters(model, cfg)
      opt = m3_offload.make_optimizer(model, base_tx, cfg, clip_max_norm=0.5)
      return model, opt

    def _loss(m, batch):
      return jnp.mean(m(batch["inputs"]) ** 2)

    @nnx.jit(donate_argnums=(0, 1))
    def _step(m, opt, batch):
      if m3_offload.has_selective_host_offload(m):
        loss, grads = m3_offload.value_and_grad(_loss, m, batch)
      else:
        loss, grads = nnx.value_and_grad(_loss)(m, batch)
      opt.update(m, grads)
      return loss

    with jax.set_mesh(mesh):
      m_ref, opt_ref = _build(False)
      m_hc, opt_hc = _build(True, "host_compute")
      m_sc, opt_sc = _build(True, "streamed_chunks")
      m_ls, opt_ls = _build(True, "lazy_sparse")

      # Construct batches with explicit duplicate row IDs across shards and untouched rows.
      batches = []
      for s in range(3):
        raw_ids = jax.random.randint(jax.random.key(50 + s), (4, 8), 0, vocab_size, dtype=jnp.int32)
        # Force duplicate hits on row 3 and row n_local + 2 across shards.
        raw_ids = raw_ids.at[:, :2].set(jnp.array([3, min(n_local + 2, vocab_size - 1)], dtype=jnp.int32))
        batches.append({"inputs": jax.device_put(raw_ids, NamedSharding(mesh, P(None, None)))})

      # Step-0 exact gradient and global-norm parity (including duplicate-ID accumulation).
      l_ref0, g_ref0 = nnx.value_and_grad(_loss)(m_ref, batches[0])
      l_hc0, g_hc0 = m3_offload.value_and_grad(_loss, m_hc, batches[0])
      np.testing.assert_allclose(float(l_hc0), float(l_ref0), rtol=1e-6, atol=1e-6)
      norm_ref = float(m3_offload.l2norm_pytree(g_ref0))
      norm_hc = float(m3_offload.l2norm_pytree(g_hc0))
      np.testing.assert_allclose(norm_hc, norm_ref, rtol=1e-5, atol=1e-6)

      # Run 3 optimizer steps across all modes.
      losses_ref, losses_hc, losses_sc, losses_ls = [], [], [], []
      for b in batches:
        losses_ref.append(float(_step(m_ref, opt_ref, b)))
        losses_hc.append(float(_step(m_hc, opt_hc, b)))
        losses_sc.append(float(_step(m_sc, opt_sc, b)))
        losses_ls.append(float(_step(m_ls, opt_ls, b)))

      np.testing.assert_allclose(losses_hc, losses_ref, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(losses_sc, losses_ref, rtol=1e-5, atol=1e-5)
      p_ref = _param_numpy_map(m_ref)
      p_hc = _param_numpy_map(m_hc)
      p_sc = _param_numpy_map(m_sc)
      p_ls = _param_numpy_map(m_ls)
      for path in p_ref:
        np.testing.assert_allclose(p_hc[path], p_ref[path], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(p_sc[path], p_ref[path], rtol=1e-5, atol=1e-5)

      # Verify opt-in lazy_sparse intentionally skips weight decay on untouched rows (documenting deviation).
      emb_path = ("ple_embedding", "embedding")
      touched = set(int(x) for b in batches for x in np.asarray(b["inputs"]).reshape(-1))
      untouched = [i for i in range(vocab_size) if i not in touched]
      self.assertNotEmpty(untouched)
      # Dense AdamW decayed untouched rows; lazy_sparse left them untouched, so they must diverge.
      untouched_diff = float(np.max(np.abs(p_ls[emb_path][untouched] - p_ref[emb_path][untouched])))
      self.assertGreater(untouched_diff, 1e-4)

  @parameterized.named_parameters(
      ("scanned", True),
      ("unscanned", False),
  )
  def test_qwen38_mini_loop_parity_and_memory_kind(self, scan_layers: bool):
    out_dir = self.create_tempdir().full_path
    cfg_off = _qwen38_mini_config(
        out_dir,
        scan_layers=str(scan_layers).lower(),
        selective_host_offload="false",
    )
    cfg_on = _qwen38_mini_config(
        out_dir,
        scan_layers=str(scan_layers).lower(),
        selective_host_offload="true",
    )
    mesh = mesh_from_config(cfg_off)
    sharding = sharding_from_config(cfg_off)

    with jax.set_mesh(mesh):
      # Verify eval_shape preserves pinned_host ONLY on selected PLE table + its moments.
      abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg_on, mesh, sharding))
      abs_ple_paths = m3_offload.selected_host_param_paths(abs_model)
      self.assertEqual(len(abs_ple_paths), 1)
      for path, var in nnx.to_flat_state(nnx.state(abs_model, nnx.Param)):
        mk = var.get_raw_value().sharding.memory_kind
        if tuple(path) in abs_ple_paths:
          self.assertEqual(mk, "pinned_host")
        else:
          self.assertIn(mk, (None, "device"))
      abs_host_opt = 0
      abs_dev_opt = 0
      for path, leaf in jax.tree_util.tree_leaves_with_path(
          abs_opt.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      ):
        raw = leaf.get_raw_value()
        if not hasattr(raw, "ndim") or raw.ndim == 0:
          continue
        key_t = tuple(m3_offload._key_value(k) for k in path)  # pylint: disable=protected-access
        is_sel = m3_offload._is_matching_opt_leaf(key_t, abs_ple_paths) is not None  # pylint: disable=protected-access
        if is_sel:
          self.assertEqual(raw.sharding.memory_kind, "pinned_host")
        else:
          self.assertIn(raw.sharding.memory_kind, (None, "device"))
        abs_host_opt += int(is_sel)
        abs_dev_opt += int(not is_sel)
      self.assertEqual(abs_host_opt, 2)
      self.assertGreater(abs_dev_opt, 10)

      m_off, opt_off = loop.setup_model(cfg_off, mesh, sharding)
      m_on, opt_on = loop.setup_model(cfg_on, mesh, sharding)
      self.assertEqual(_param_paths(m_on), _param_paths(m_off))

      # Step-0 loss and full gradient parity (materializing SparseHostGrad for PLE).
      batch0 = loop.make_synthetic_batch(cfg_off, mesh, sharding, jax.random.key(0))
      l0_off, g0_off = nnx.value_and_grad(loop.loss_fn)(m_off, batch0)
      l0_on, g0_on = m3_offload.value_and_grad(loop.loss_fn, m_on, batch0)
      np.testing.assert_allclose(float(l0_on), float(l0_off), rtol=1e-6, atol=1e-6)
      g_off_map = {tuple(p): v.get_raw_value() for p, v in nnx.to_flat_state(g0_off)}
      g_on_map = {
          tuple(p): (v.get_raw_value() if isinstance(v, nnx.Variable) else v) for p, v in nnx.to_flat_state(g0_on)
      }
      for path, go in g_off_map.items():
        gn = g_on_map[path]
        if isinstance(gn, m3_offload.SparseHostGrad):
          gn = m3_offload.materialize_sparse_grad(gn)
        np.testing.assert_allclose(np.asarray(gn), np.asarray(go), rtol=1e-5, atol=1e-5)

      # 3-step exact AdamW trajectory parity (step-0 elementwise <= 1e-6; 3-step Frobenius <= 1e-5).
      losses_off, losses_on = [], []
      for step in range(3):
        batch = loop.make_synthetic_batch(cfg_off, mesh, sharding, jax.random.key(10 + step))
        losses_off.append(float(loop.train_step(m_off, opt_off, batch)))
        losses_on.append(float(loop.train_step(m_on, opt_on, batch)))
        if step == 0:
          p_off0 = _param_numpy_map(m_off)
          p_on0 = _param_numpy_map(m_on)
          for path in p_off0:
            np.testing.assert_allclose(p_on0[path], p_off0[path], rtol=1e-6, atol=1e-6)
      np.testing.assert_allclose(losses_on, losses_off, rtol=1e-5, atol=1e-5)
      p_off = _param_numpy_map(m_off)
      p_on = _param_numpy_map(m_on)
      for path in p_off:
        # Damp denominator at 1.0 for zero-centered `1+scale` norms (`||scale||_2 ~ 0.01`, max_d < 1e-6).
        rel_frob = float(np.linalg.norm(p_on[path] - p_off[path]) / max(float(np.linalg.norm(p_off[path])), 1.0))
        self.assertLess(rel_frob, 1e-5, f"Frobenius drift too large on {path}: {rel_frob:.2e}")
        np.testing.assert_allclose(p_on[path], p_off[path], rtol=1e-4, atol=1e-4)

      # Post-step memory_kind invariants: ONLY PLE weight + its mu/nu remain in pinned_host.
      ple_paths = m3_offload.selected_host_param_paths(m_on)
      for path, var in nnx.to_flat_state(nnx.state(m_on, nnx.Param)):
        mk = var.get_raw_value().sharding.memory_kind
        if tuple(path) in ple_paths:
          self.assertEqual(mk, "pinned_host")
        else:
          self.assertIn(mk, (None, "device"))
      post_host_opt = 0
      for path, leaf in jax.tree_util.tree_leaves_with_path(
          opt_on.opt_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
      ):
        raw = leaf.get_raw_value()
        if not hasattr(raw, "ndim") or raw.ndim == 0:
          continue
        key_t = tuple(m3_offload._key_value(k) for k in path)  # pylint: disable=protected-access
        is_sel = m3_offload._is_matching_opt_leaf(key_t, ple_paths) is not None  # pylint: disable=protected-access
        if is_sel:
          self.assertEqual(raw.sharding.memory_kind, "pinned_host")
          post_host_opt += 1
        else:
          self.assertIn(raw.sharding.memory_kind, (None, "device"))
      self.assertEqual(post_host_opt, 2)

      # Negative control: perturbing the offloaded PLE table must break loss parity.
      ple_path = next(iter(ple_paths))
      ple_var = dict(nnx.to_flat_state(nnx.state(m_on, nnx.Param)))[ple_path]
      ple_var.set_raw_value(m3_offload.to_memory(ple_var.get_raw_value() + 1.0, "pinned_host"))
      perturbed_loss = float(loop.loss_fn(m_on, batch0))
      ref_loss = float(loop.loss_fn(m_off, batch0))
      self.assertGreater(abs(perturbed_loss - ref_loss), 1e-3)

  def test_default_off_hlo_byte_identical(self):
    """Verifies Qwen3.8 default-off (`selective_host_offload=False`) emits byte-identical HLO."""
    out_dir = self.create_tempdir().full_path
    cfg_default = _qwen38_mini_config(out_dir, scan_layers="true", selective_host_offload="false")
    cfg_empty_pat = _qwen38_mini_config(
        out_dir,
        scan_layers="true",
        selective_host_offload="false",
        host_offload_param_patterns="[]",
    )
    self.assertNotEmpty(cfg_default.host_offload_param_patterns)
    mesh = mesh_from_config(cfg_default)
    sharding = sharding_from_config(cfg_default)

    def _strip_debug_metadata(hlo_text: str) -> str:
      # Strip XLA source-location intern tables (FileNames/FunctionNames/FileLocations/StackFrames)
      # and inline `metadata={...}` annotations while keeping every HLO instruction, shape, layout,
      # sharding, and fusion byte-identical.
      body = re.split(r"\n(?:FileNames|FunctionNames|FileLocations|StackFrames)\b", hlo_text, maxsplit=1)[0]
      return re.sub(r",\s*metadata=\{[^}]*\}", "", body)

    def _compile_hlo(cfg):
      with jax.set_mesh(mesh):
        m, opt = loop.setup_model(cfg, mesh, sharding)
        batch = loop.make_synthetic_batch(cfg, mesh, sharding, jax.random.key(0))
        return _strip_debug_metadata(loop.train_step.lower(m, opt, batch).compile().as_text())

    self.assertEqual(_compile_hlo(cfg_default), _compile_hlo(cfg_empty_pat))

  def test_dual_drivers_and_transform_composition(self):
    """Verifies selective offload across `train.py`, `MaxTextTrainingEngine`, and scan+remat+quantize."""
    out_dir = self.create_tempdir().full_path

    # 1. Legacy `train.py` driver (`train_utils.setup_train_loop` + `legacy_train.train_step`).
    def _run_legacy(selective: bool):
      cfg = _qwen38_mini_config(
          out_dir,
          scan_layers="true",
          dataset_type="synthetic",
          selective_host_offload=str(selective).lower(),
      )
      mesh = mesh_from_config(cfg)
      with jax.set_mesh(mesh):
        (
            _,
            _,
            state_mesh_shardings,
            _,
            mesh,
            _,
            data_iter,
            _,
            _,
            _,
            train_state,
        ) = train_utils.setup_train_loop(cfg, None, mesh=mesh)
        jit_model, state = nnx.split(train_state)
        p_step, _ = train_utils.jit_train_and_eval_step(
            cfg, jit_model, mesh, state, state_mesh_shardings, legacy_train.train_step
        )
        losses, gnorms, pnorms = [], [], []
        for _ in range(2):
          batch = next(data_iter)
          state, metrics = p_step(state, batch)
          losses.append(float(metrics["scalar"]["learning/loss"]))
          gnorms.append(float(metrics["scalar"]["learning/grad_norm"]))
          pnorms.append(float(metrics["scalar"]["learning/param_norm"]))
      return state, losses, gnorms, pnorms

    _, l_off, gn_off, pn_off = _run_legacy(False)
    st_on, l_on, gn_on, pn_on = _run_legacy(True)
    np.testing.assert_allclose(l_on, l_off, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(gn_on, gn_off, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pn_on, pn_off, rtol=1e-4, atol=1e-4)
    host_m = [
        x for x in jax.tree.leaves(st_on.model) if hasattr(x, "sharding") and x.sharding.memory_kind == "pinned_host"
    ]
    dev_m = [x for x in jax.tree.leaves(st_on.model) if hasattr(x, "sharding") and x.sharding.memory_kind == "device"]
    host_o = [
        x
        for x in jax.tree.leaves(st_on.optimizer)
        if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0 and x.sharding.memory_kind == "pinned_host"
    ]
    dev_o = [
        x
        for x in jax.tree.leaves(st_on.optimizer)
        if hasattr(x, "sharding") and getattr(x, "ndim", 0) > 0 and x.sharding.memory_kind == "device"
    ]
    self.assertLen(host_m, 1)
    self.assertGreater(len(dev_m), 10)
    self.assertLen(host_o, 2)
    self.assertGreater(len(dev_o), 10)

    # 2. Transform composition: scan + remat + Qwix int8 quantize + named scopes + selective offload.
    cfg_comp = _qwen38_mini_config(
        out_dir,
        scan_layers="true",
        remat_policy="full",
        quantization="int8",
        use_qwix_quantization="true",
        selective_host_offload="true",
    )
    cfg_comp_off = _qwen38_mini_config(
        out_dir,
        scan_layers="true",
        remat_policy="full",
        quantization="int8",
        use_qwix_quantization="true",
        selective_host_offload="false",
    )
    mesh = mesh_from_config(cfg_comp)
    with jax.set_mesh(mesh):
      m_comp_off = model_creation_utils.create_model(cfg_comp_off, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      m_comp_on = model_creation_utils.create_model(cfg_comp, mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      self.assertEqual(_param_paths(m_comp_on), _param_paths(m_comp_off))
      ple_paths = m3_offload.selected_host_param_paths(m_comp_on)
      self.assertLen(ple_paths, 1)
      toks = jnp.ones((cfg_comp.global_batch_size_to_train_on, 16), dtype=jnp.int32)
      pos = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), toks.shape)
      self.assertGreater(m3_quant.quantized_matmuls(m_comp_on, jnp.int8, toks, pos), 0)

    # 3. MaxTextTrainingEngine (when tunix is installed).
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is not None:
      from maxtext.training_engine import maxtext_engine  # pylint: disable=import-outside-toplevel

      def _run_engine(selective: bool):
        cfg_e = _qwen38_mini_config(out_dir, scan_layers="true", selective_host_offload=str(selective).lower())
        eng = maxtext_engine.MaxTextTrainingEngine(cfg_e, mesh=mesh)
        bs = cfg_e.global_batch_size_to_train_on
        seq = cfg_e.max_target_length
        losses = []
        for step in range(2):
          toks = jax.random.randint(jax.random.key(200 + step), (bs, seq), 0, cfg_e.vocab_size, dtype=jnp.int32)
          pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32), (bs, seq))
          seg = jnp.ones((bs, seq), dtype=jnp.int32)
          batch = {
              "inputs": toks,
              "targets": jnp.roll(toks, -1, axis=-1),
              "inputs_position": pos,
              "targets_position": pos,
              "inputs_segmentation": seg,
              "targets_segmentation": seg,
          }
          if step == 0:
            eng.compile(batch)
          eng.fwd_bwd(batch)
          eng.update()
          m = eng.get_metrics()
          lk = next(iter(m.weighted_metrics))
          losses.append(
              float(jnp.sum(m.weighted_metrics[lk].unreduced_sum) / jnp.sum(m.weighted_metrics[lk].denominator))
          )
        jax.block_until_ready(eng._read_state_pure())  # pylint: disable=protected-access
        return losses

      np.testing.assert_allclose(_run_engine(True), _run_engine(False), rtol=1e-5, atol=1e-5)

  def test_tpu7x_aot_selective_ple_offload_reduces_hbm_without_dense_grad(self):
    """Compiles Qwen3.8 mini with an enlarged PLE table on virtual `tpu7x-8` and checks HBM drop."""
    import os  # pylint: disable=import-outside-toplevel

    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "1")
    out_dir = self.create_tempdir().full_path
    th = accelerator_to_spec_map.get_system_characteristics("tpu7x-8")
    topo = get_topology_desc(
        platform=th.platform,
        topology_name=th.topology_name,
        chip_config_name=th.chip_config_name,
        chips_per_host_bounds=th.chips_per_host_bounds,
        num_slices=1,
        wrap=th.wrap,
    )

    def _compile(selective: bool):
      cfg = _qwen38_mini_config(
          out_dir,
          compile_topology="tpu7x-8",
          ici_fsdp_parallelism="4",
          ici_expert_parallelism="2",
          scan_layers="true",
          dtype="bfloat16",
          weight_dtype="bfloat16",
          # Enlarge PLE table (~51.2 MB bf16 + ~102.4 MB bf16 AdamW mu/nu = ~153.6 MB) so HBM drop is unmistakable.
          ngram_vocab_size_base="100000",
          selective_host_offload=str(selective).lower(),
      )
      tpu_mesh = mesh_from_config(cfg, devices=topo.devices)
      sharding = sharding_from_config(cfg)
      with jax.set_mesh(tpu_mesh):
        data_spec = sharding("tokens", ("batch", "length"), TensorType.ACTIVATION)
        data_sharding = NamedSharding(tpu_mesh, data_spec)
        batch_shape = (int(cfg.per_device_batch_size) * len(topo.devices), int(cfg.max_target_length))
        batch = {
            "inputs": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "targets": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
            "positions": jax.ShapeDtypeStruct(batch_shape, jnp.int32, sharding=data_sharding),
        }
        abs_model, abs_opt = nnx.eval_shape(lambda: loop.setup_model(cfg, tpu_mesh))
        compiled = loop.train_step.lower(abs_model, abs_opt, batch).compile()
      rows = memory_report.memory_report({"update": compiled}, {})
      return compiled.memory_analysis(), rows[0], compiled.as_text()

    mem_off, row_off, _ = _compile(False)
    _, row_on, hlo_on = _compile(True)

    self.assertEqual(mem_off.host_argument_size_in_bytes, 0)
    self.assertNotIn("MegaScaleCollective", hlo_on)
    # PLE table + AdamW mu/nu (~153.6 MB in bf16) must move from HBM argument to S(5) host_argument.
    arg_drop = row_off.argument - row_on.argument
    self.assertGreater(arg_drop, 140 * 1024 * 1024)
    self.assertEqual(row_on.host_argument, arg_drop)
    self.assertEqual(row_on.host_alias, row_on.host_argument)
    # Host-compute uses bounded S(5) host scratch (<= 1x offloaded table+moments), never HBM.
    self.assertLessEqual(row_on.host_temp, row_on.host_argument)
    # Device resident peak must drop substantially without materializing a dense PLE gradient in HBM.
    self.assertLess(row_on.device_total, row_off.device_total - 100 * 1024 * 1024)

    # Also verify AbstractMaxTextEngine AOT (when tunix is installed): zero MegaScale & large HBM drop.
    import importlib.util  # pylint: disable=import-outside-toplevel

    if importlib.util.find_spec("tunix") is not None:
      from maxtext.trainers.pre_train import train_compile  # pylint: disable=import-outside-toplevel
      from maxtext.training_engine import maxtext_engine_compile  # pylint: disable=import-outside-toplevel

      def _compile_engine(selective: bool):
        cfg = _qwen38_mini_config(
            out_dir,
            compile_topology="tpu7x-8",
            compile_topology_num_slices="1",
            ici_fsdp_parallelism="4",
            ici_expert_parallelism="2",
            scan_layers="true",
            remat_policy="full",
            dtype="bfloat16",
            weight_dtype="bfloat16",
            ngram_vocab_size_base="100000",
            selective_host_offload=str(selective).lower(),
        )
        tpu_mesh = train_compile.get_topology_mesh(cfg)
        eng = maxtext_engine_compile.AbstractMaxTextEngine(cfg, tpu_mesh)
        compiled = eng.compile_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))
        rows = memory_report.memory_report(compiled, eng.train_state_avals())
        peak = memory_report.device_peak(rows)
        ms_count = sum(c.as_text().count("MegaScaleCollective") for c in compiled.values())
        return peak, ms_count

      peak_off, ms_off = _compile_engine(False)
      peak_on, ms_on = _compile_engine(True)
      self.assertEqual(ms_off, 0)
      self.assertEqual(ms_on, 0)
      self.assertLess(peak_on.device_total, peak_off.device_total - 100 * 1024 * 1024)


if __name__ == "__main__":
  absltest.main()
