# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RawRewardFn = Callable[..., Any] | None


class RewardManagerBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score: RawRewardFn):
        """Initialize reward manager.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = get_event_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor: # J：将每个样本的 reward score 赋值给 rm_scores 张量的最后一个 Response token
        """Assemble per-sample reward scores into the ``rm_scores`` tensor for a batch.

        Args:
            data: The concatenated batch passed to :meth:`run_single`.
                ``data.batch["prompts"]``, ``data.batch["responses"]`` and
                ``data.batch["attention_mask"]`` are expected to be present
                for the default implementation.
            scores: List of scalar reward scores, one per sample in ``data``.

        Returns:
            torch.Tensor: The ``rm_scores`` tensor with leading dimension equal to
            ``len(data)``.
        """
        prompt_length = data.batch["prompts"].size(1) # J：data.batch["prompts"] 是（batch_size, prompt_length）的张量
        # J：prompt_length 所有样本相同的一个统一的整数，因为数据已经经过 padding 了
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1) # J：这里得到的是每个样本的有效 Response 长度（不包括 prompt），data.batch["attention_mask"] 是（batch_size, total_seq_len）的张量
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32) # J：rm_scores 是（batch_size, response_length）的张量
        rm_scores[torch.arange(rm_scores.size(0), device=rm_scores.device), valid_response_length - 1] = ( # J：将 reward score 赋值给每个样本的最后一个 Response token
            rm_scores.new_tensor(scores) # J：新张量会复制输入数据，并默认继承原张量的 dtype 和 device 属性
        )
        return rm_scores # J：返回 rm_scores 张量，仅每个样本的最后一个 Response token 有 reward score，其他的 token 都是 reward score=0
