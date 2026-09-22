# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for shared-engine extension hooks and optimizer ownership."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPOptimizerConfig

from verl_omni.pipelines.bagel_unigrpo.diffusers_training_adapter import BagelUniGRPO
from verl_omni.pipelines.bagel_unigrpo.training_runtime import BagelUniGRPORuntime
from verl_omni.pipelines.model_base import DiffusionModelBase, DiffusionTrainingRuntime
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.engine.fsdp.training_utils import optimizer_parameters


def _engine(module, runtime=None):
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = module
    engine._training_runtime = runtime
    engine._explicit_fsdp2_units = False
    engine.optimizer_config = FSDPOptimizerConfig(lr=0.1, weight_decay=0.0, clip_grad=100.0)
    engine.optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    return engine


def test_default_adapter_hooks_preserve_existing_path():
    module = torch.nn.Linear(1, 1)
    assert DiffusionModelBase.fsdp2_sharding_units(module) is None
    assert DiffusionModelBase.build_training_runtime(module, None, None) is None
    engine = _engine(module)
    engine._run_forward_backward_batch = Mock(return_value={"default": True})
    data = TensorDict({}, [])
    assert engine.forward_backward_batch(data, None) == {"default": True}
    engine._run_forward_backward_batch.assert_called_once_with(data, None, False, timesteps_key="all_timesteps")
    with pytest.raises(NotImplementedError, match="no actor-side sampling"):
        engine.generate_rollout(data)
    with pytest.raises(NotImplementedError, match="no actor-side evaluation"):
        engine.evaluate_rollout(data)


def test_non_bagel_runtime_accumulates_two_losses_before_one_engine_step():
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.data.fill_(1.0)

    class TwoLossRuntime(DiffusionTrainingRuntime):
        def forward_backward_batch(self, data, loss_function, forward_only=False):
            if not forward_only:
                (module.weight.square().sum()).backward()
                (3 * module.weight.square().sum()).backward()
            return {"loss": [4.0], "metrics": {}, "model_output": {}}

        def generate(self, data):
            return data

        def evaluate(self, data):
            return data

    engine = _engine(module, TwoLossRuntime())
    engine.optimizer.step = Mock(wraps=engine.optimizer.step)
    data = TensorDict({}, [])
    engine.optimizer_zero_grad()
    engine.forward_backward_batch(data, None)
    assert module.weight.item() == 1.0
    assert module.weight.grad.item() == 8.0
    engine.optimizer.step.assert_not_called()
    assert engine.optimizer_step() == pytest.approx(8.0)
    engine.optimizer.step.assert_called_once()
    assert module.weight.item() == pytest.approx(0.2)
    assert engine.generate_rollout(data) is data
    assert engine.evaluate_rollout(data) is data


def test_explicit_optimizer_groups_preserve_options_and_exclude_frozen_parameters():
    module = torch.nn.ModuleDict({"base": torch.nn.Linear(2, 2), "expert": torch.nn.Linear(2, 2)})
    module.base.bias.requires_grad_(False)
    cfg = FSDPOptimizerConfig(
        lr=0.1, betas=(0.8, 0.95), weight_decay=0.02, override_optimizer_config={"foreach": False, "eps": 1e-7}
    )
    cfg.param_group_lrs = {"expert.weight": 0.3, "expert": 0.2}
    engine = _engine(module)
    engine.optimizer_config = cfg
    opt = engine._build_optimizer(module)
    assert [g["lr"] for g in opt.param_groups] == [0.1, 0.3, 0.2]
    actual = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(actual) == {id(p) for p in module.parameters() if p.requires_grad}
    assert len(actual) == len(set(actual))
    assert opt.defaults["betas"] == (0.8, 0.95)
    assert opt.defaults["eps"] == 1e-7
    assert opt.defaults["foreach"] is False
    assert opt.defaults["weight_decay"] == 0.02
    plain = SimpleNamespace(param_group_lrs=None)
    assert list(optimizer_parameters(module, plain)) == list(module.parameters())


@pytest.mark.parametrize("overrides", [{"": 1.0}, {"expert": -1.0}, {"expert": float("nan")}])
def test_invalid_lr_group_is_rejected(overrides):
    with pytest.raises(ValueError, match="Invalid parameter LR group"):
        optimizer_parameters(torch.nn.Linear(1, 1), SimpleNamespace(lr=0.1, param_group_lrs=overrides))


def test_bagel_adapter_selects_functionally_called_leaves():
    module = torch.nn.Module()
    module.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
    module.embed_tokens = torch.nn.Embedding(2, 2)
    module.lm_head = torch.nn.Linear(2, 2)
    units = BagelUniGRPO.fsdp2_sharding_units(module)
    assert units == [*module.layers, module.embed_tokens, module.lm_head]
    assert module not in units
    first = BagelUniGRPO.build_training_runtime(module, None, None)
    second = BagelUniGRPO.build_training_runtime(module, None, None)
    assert isinstance(first, DiffusionTrainingRuntime)
    assert first is not second
    assert first._updater is None and first._replica is None


def test_bagel_runtime_rejects_forward_only_instead_of_returning_fake_loss():
    runtime = BagelUniGRPORuntime(torch.nn.Linear(1, 1), None, None)
    with pytest.raises(NotImplementedError, match="requires backward"):
        runtime.forward_backward_batch(TensorDict({}, []), None, forward_only=True)


def test_actor_loss_config_applies_even_when_rollout_created_updater_first():
    runtime = BagelUniGRPORuntime(torch.nn.Linear(1, 1), None, None)
    runtime._updater = SimpleNamespace(_loss_cfg=SimpleNamespace(diffusion_loss=SimpleNamespace()))
    cfg = SimpleNamespace(mse_weight=0.0, ratio_norm=False, clip_ratio=0.1, adv_clip_max=2.0)
    updater = runtime._get_updater(cfg)
    assert updater.mse_weight == 0.0
    assert updater.ratio_norm is False
    assert vars(updater._loss_cfg.diffusion_loss) == vars(cfg)


def test_worker_dispatch_is_algorithm_independent():
    from verl_omni.workers.engine_workers import ActorRolloutRefWorker

    output = TensorDict({"result": torch.ones(1)}, [1])
    engine = SimpleNamespace(evaluate_rollout=Mock(return_value=output))
    worker = SimpleNamespace(actor=SimpleNamespace(engine=engine))
    request = TensorDict({"input": torch.zeros(1)}, [1])
    assert ActorRolloutRefWorker.evaluate.__wrapped__(worker, request)["result"].item() == 1.0
    engine.evaluate_rollout.assert_called_once_with(request)


def test_bagel_runtime_backward_only_does_not_own_optimizer():
    module = torch.nn.Linear(1, 1, bias=False)
    runtime = BagelUniGRPORuntime(module, None, None)

    def backward(scale, metric):
        loss = scale * module.weight.square().sum()
        loss.backward()
        return {metric: loss.item()}

    runtime._updater = SimpleNamespace(
        _ar_backward=lambda *_: backward(1, "ar/loss"),
        _image_backward=lambda *_: backward(3, "image/loss"),
        _loss_cfg=SimpleNamespace(diffusion_loss=SimpleNamespace()),
    )
    data = tu.get_tensordict({"unigrpo_samples": ["trajectory"], "advantages": torch.tensor([1.0])})
    cfg = SimpleNamespace(
        diffusion_loss=SimpleNamespace(mse_weight=0.0, ratio_norm=True, clip_ratio=1e-6, adv_clip_max=5.0)
    )
    weight = module.weight.detach().clone()
    result = runtime.forward_backward_batch(data, partial(lambda **kw: None, config=cfg))
    torch.testing.assert_close(module.weight, weight)
    torch.testing.assert_close(module.weight.grad, weight * 8)
    assert set(result) == {"loss", "metrics", "model_output"}
    assert not hasattr(runtime, "optimizer")


@pytest.mark.parametrize("rank,fail_export", [(0, False), (1, False), (0, True)])
def test_report_evaluation_syncs_all_ranks_and_limits_export(monkeypatch, tmp_path, rank, fail_export):
    import json

    import transformers

    from verl_omni.pipelines.bagel_unigrpo import pipeline, rollout
    from verl_omni.pipelines.bagel_unigrpo.evaluation import BagelReportEvaluator
    from verl_omni.utils.reward_score import pickscore_reward

    module = torch.nn.Linear(1, 1)
    replica = torch.nn.Linear(1, 1)
    config = SimpleNamespace(local_path="unused", path="unused")
    runtime = BagelUniGRPORuntime(module, config, None)
    sync = Mock()
    generate = Mock(return_value=([1], torch.zeros(3, 8, 8, dtype=torch.uint8)))
    if fail_export:
        generate.side_effect = RuntimeError("export failed")
    monkeypatch.setattr(rollout, "build_replica", lambda *_: replica)
    monkeypatch.setattr(rollout, "sync_replica_from_master", sync)
    monkeypatch.setattr(rollout, "build_unigrpo_pipeline_kwargs", lambda *_: {})
    monkeypatch.setattr(pipeline, "BagelUniPipeline", lambda *_args, **_kwargs: SimpleNamespace(generate_eval=generate))
    monkeypatch.setattr("verl.utils.device.get_device_name", lambda: "cpu")
    monkeypatch.setattr("verl.utils.device.get_device_id", lambda: 0)
    monkeypatch.setattr(
        "verl.utils.device.get_torch_device",
        lambda: SimpleNamespace(empty_cache=lambda: None, manual_seed_all=lambda _: None),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(decode=lambda *_args, **_kwargs: "thinking"),
    )
    monkeypatch.setattr(
        pickscore_reward, "_PickScoreInferencer", lambda **_: SimpleNamespace(score=lambda *_: torch.tensor([0.75]))
    )
    data = tu.get_tensordict({"prompt_token_ids": [[1, 2]], "ground_truth": ["prompt"]})
    tu.assign_non_tensor(data, output_dir=str(tmp_path), step=3, seed=42)
    result = BagelReportEvaluator(runtime).evaluate(data)
    sync.assert_called_once_with(replica, module)
    assert module.training
    assert next(replica.parameters()).device.type == "cpu"
    if rank != 0:
        generate.assert_not_called()
        assert result is None
        assert not list(tmp_path.iterdir())
    elif fail_export:
        assert result is None
    else:
        assert result["scores"].item() == 0.75
        artifact = tmp_path / "report_ff/step_0003/p0"
        assert artifact.with_suffix(".png").exists()
        assert artifact.with_suffix(".txt").read_text() == "thinking"
        assert json.loads(artifact.with_suffix(".json").read_text())["pickscore"] == 0.75


def test_evaluation_preserves_training_rng(monkeypatch):
    runtime = BagelUniGRPORuntime(torch.nn.Linear(1, 1), None, None)

    def evaluate(_):
        torch.manual_seed(123)
        return torch.rand(3)

    runtime._evaluator = SimpleNamespace(evaluate=evaluate)
    fork_rng = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: fork_rng(devices=[]))
    monkeypatch.setattr("verl.utils.device.get_device_id", lambda: 0)
    before = torch.random.get_rng_state().clone()
    runtime.evaluate(TensorDict({}, []))
    assert torch.equal(before, torch.random.get_rng_state())
