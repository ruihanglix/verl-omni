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

"""Fixed-prompt BAGEL evaluation and report export, outside the shared engine/worker."""

import logging

import torch

logger = logging.getLogger(__name__)


class BagelReportEvaluator:
    """Synchronize on all ranks, then generate and export samples on rank zero."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._report_tokenizer = None
        self._report_scorer = None

    def evaluate(self, data):
        """Fixed-prompt report samples at one checkpoint (official-CFG eval on the trainside replica).

        Mirrors the standalone ``dump_report_samples``: every rank first re-syncs the flat bf16
        replica from the FSDP master (an all-gather per trainable ``DTensor``, so ALL ranks must
        reach it), then only rank 0 samples on the local replica (collective-free) and writes each
        prompt's official-CFG image + COMPLETE thinking text + PickScore under
        ``<out_dir>/report_ff/step_<NNNN>/``. Rank-0 work is wrapped so a dump failure logs and
        returns instead of dead-locking the peers at the next collective. Returns rank-0's
        per-prompt PickScores (``None`` off rank 0 / on failure).
        """
        from verl.utils import tensordict_utils as tu

        eval_prompts = tu.get(data, "prompt_token_ids")
        eval_gts = tu.get(data, "ground_truth")
        out_dir = tu.get_non_tensor_data(data, "output_dir", default=None)
        step = tu.get_non_tensor_data(data, "step", default=0)
        seed = tu.get_non_tensor_data(data, "seed", default=1234)
        if not out_dir:
            raise ValueError("Report evaluation requires output_dir")
        runtime = self.runtime
        import os

        import torch.distributed as dist
        from verl.utils.device import get_device_id, get_device_name, get_torch_device

        from verl_omni.pipelines.bagel_unigrpo.pipeline import BagelUniPipeline
        from verl_omni.pipelines.bagel_unigrpo.rollout import (
            build_replica,
            build_unigrpo_pipeline_kwargs,
            sync_replica_from_master,
        )

        device = torch.device(get_device_name(), get_device_id())
        model_path = runtime.model_config.local_path or runtime.model_config.path
        if runtime._replica is None:
            runtime._replica = build_replica(model_path, device)
        else:
            runtime._replica.to(device)

        # Collective (all ranks): refresh the replica from the current FSDP master weights.
        was_training = runtime.module.training
        runtime.module.eval()
        try:
            sync_replica_from_master(runtime._replica, runtime.module)
        finally:
            runtime.module.train(was_training)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            runtime._replica.to("cpu")
            get_torch_device().empty_cache()
            return None

        # Rank-0-only past this point (no collectives): catch + log so a bad dump costs only its
        # samples instead of dead-locking the peers at the next collective.
        try:
            import json

            from PIL import Image

            from verl_omni.utils.reward_score.pickscore_reward import _PickScoreInferencer, _to_pil_hwc

            runtime._replica.eval()
            pipeline = BagelUniPipeline(
                runtime._replica, **build_unigrpo_pipeline_kwargs(runtime.model_config, runtime._replica)
            )
            sdir = os.path.join(out_dir, "report_ff", f"step_{int(step):04d}")
            os.makedirs(sdir, exist_ok=True)

            if self._report_tokenizer is None:
                from transformers import AutoTokenizer

                self._report_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = self._report_tokenizer

            images = []
            for i, pids in enumerate(eval_prompts):
                torch.manual_seed(int(seed))
                get_torch_device().manual_seed_all(int(seed))
                gen = torch.Generator(device=device).manual_seed(int(seed))
                think_ids, image = pipeline.generate_eval([int(t) for t in pids], generator=gen)
                Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(os.path.join(sdir, f"p{i}.png"))
                text = tokenizer.decode(think_ids, skip_special_tokens=True) if tokenizer is not None else ""
                with open(os.path.join(sdir, f"p{i}.txt"), "w") as handle:
                    handle.write(text)
                images.append(image)

            if self._report_scorer is None:
                self._report_scorer = _PickScoreInferencer(device=device)
            scores = self._report_scorer.score(list(eval_gts), [_to_pil_hwc(im) for im in images]).tolist()
            for i, score in enumerate(scores):
                with open(os.path.join(sdir, f"p{i}.json"), "w") as handle:
                    json.dump({"pickscore": float(score), "prompt": eval_gts[i]}, handle)
            logger.info(
                "UniGRPO report dump step %s: pickscore mean=%.4f -> %s",
                step,
                sum(scores) / max(len(scores), 1),
                sdir,
            )
            return tu.get_tensordict({"scores": torch.tensor(scores, dtype=torch.float32)})
        except Exception:
            import traceback

            logger.warning("UniGRPO report dump step %s failed (non-fatal):\n%s", step, traceback.format_exc())
            return None
        finally:
            import gc

            runtime._replica.to("cpu")
            if self._report_scorer is not None:
                del self._report_scorer
                self._report_scorer = None
            gc.collect()
            get_torch_device().empty_cache()
