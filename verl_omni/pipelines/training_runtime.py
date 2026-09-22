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

"""Optional stateful training/sampling hooks constructed by a model adapter.

A runtime supplies model/algorithm work without replacing the engine lifecycle.
It may accumulate gradients, but must never zero gradients, step the optimizer or
advance its scheduler. The engine continues to own those operations and checkpoints.
"""

from abc import ABC, abstractmethod
from typing import Callable

from tensordict import TensorDict


class DiffusionTrainingRuntime(ABC):
    """Extension contract for algorithms that need a custom backward schedule."""

    @abstractmethod
    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only: bool = False) -> dict:
        """Return loss/metrics/model_output; accumulate gradients only when training.

        The result has the same contract as BaseEngine.forward_backward_batch:
        {"loss": list, "metrics": dict, "model_output": dict}. Implementations that
        cannot perform inference must reject forward_only explicitly.
        """
        raise NotImplementedError

    def generate(self, data: TensorDict) -> TensorDict:
        """Generate a local batch; all actor ranks enter this operation together."""
        raise NotImplementedError(f"{type(self).__name__} does not support actor-side generation")

    def evaluate(self, data: TensorDict) -> TensorDict | None:
        """Evaluate a broadcast request on all ranks; non-output ranks may return None.

        Complete collective work before rank-local export. Rank-local export failures
        must not leave peer ranks waiting at subsequent training collectives.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support actor-side evaluation")
