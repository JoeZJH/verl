# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"): # J：对奖励添加 KL 散度惩罚
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"] # J：GDPO 场景当前也抽取这个字段吗？
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty( # J：计算 KL 散度，根据 config 中的配置
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld # J：用 score 减去 KL 散度的惩罚，作为奖励

    # J：计算当前 KL 散度的平均值（seq-mean-token-mean），用于更新 KL 系数
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size) # J：根据当前 KL 散度和步数，更新 KL 系数，可能是自适应的（朝预定的 KL 目标值调整），也可能是固定值
    data.batch["token_level_rewards"] = token_level_rewards

    # J: "actor/reward_kl_penalty" 是当前 KL 散度的平均值（seq-mean-token-mean），用于更新 KL 系数，也用于上报指标
    # J: "actor/reward_kl_penalty_coeff" 是当前 KL 系数（AdaptiveKLController 中，这个值会变化）
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto): # J：计算 Response 部分的注意力掩码，读取 data.batch["attention_mask"]
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1) # J：注：responses.size(1) 是 padding 后的最大长度 ，不是每个 response 的真实长度
    attention_mask = data.batch["attention_mask"] # J：获取注意力掩码
    return attention_mask[:, -response_length:] # J：对注意力掩码进行截断，仅返回 Response 部分的注意力掩码


def compute_spec_decode_metrics( # J：计算 speculative decoding 指标，如 accept_rate, accept_length, etc
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage( # J：计算 advantage 估计
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE: # J: GAE 估计 模式
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return( # J：计算 GAE 估计 的 advantage 和 returns
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO: # J: GRPO 估计 模式
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage( # J：计算 GRPO 估计 的 advantage
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"], # J：用于分组计算 GRPO advantage 的 uid
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else: # J： 其他 advantage 估计 模式
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator) # J：获取 advantage 估计函数，其实 GRPO 和 GAE 也可以合并代码以后走这里，只是现在把它们的计算逻辑放到了其他分支中
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"): # J: GDPO 估计 模式
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


@deprecated(
    "main_ppo.py is deprecated, and wil be replaced by main_ppo_sync.py in v0.8.0, please use main_ppo_sync.py instead."
)
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine # J：是否使用混合引擎，Bool 类型，当前为 True
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            # J：如果使用混合引擎，必须包含 ActorRolloutRefWorker（对应 ActorRolloutRef 角色 或 Role.ActorRollout 角色）
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping # J：角色到 Worker 类的映射，键是角色，值是 Worker 类的类型
        self.resource_pool_manager = resource_pool_manager # J：资源池管理器，用于管理 Ray 资源池
        self.use_reference_policy = need_reference_policy(self.config) # J：根据是否需要 kl_in_reward 或 kl_loss（任意一个均需要参考策略），判断是否需要参考策略，Bool 类型，config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
        self.use_teacher_policy = need_teacher_policy(self.config) # J：根据是否开启 distillation，判断是否需要教师模型，Bool 类型, config.distillation

        self.use_rm = need_reward_model(self.config) # J：根据是否开启奖励模型，判断是否需要奖励模型，Bool 类型，config.reward.reward_model.enable

        self.use_critic = need_critic(self.config) # J：根据是否开启 critic，判断是否需要 critic，Bool 类型，config.critic.enable or config.algorithm.adv_estimator == AdvantageEstimator.GAE
        self.ray_worker_group_cls = ray_worker_group_cls # J：RayWorkerGroup 类
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        # J：如果 lora 模式训练，那么 Reference 就是 Actor 的 Base 模型部分，也就是说 ref_in_actor=True
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self._init_dump_executor()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None: # J：如果未指定训练数据集，默认使用默认的 RLHF 数据集类
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None: # J：如果未指定验证数据集，默认使用默认的 RLHF 数据集类
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None: # J：如果未指定训练采样器，默认使用默认的采样器
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None: # J：如果未指定 collate_fn 函数，默认使用默认的 collate_fn 函数
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset, # J：训练数据集，根据 sampler 输出的 index 来采样数据
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size), # J：gen_batch_size(生成的 batch 大小)，若未配置则使用 train_batch_size
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler, # J：训练数据集的采样器，负责输出 index
        )

        # J：优先使用 self.config.data.val_batch_size 配置
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset) # J：默认使用整个验证数据集大小作为 batch 大小

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True), # J：验证时也可以配置打乱数据，默认 True
            drop_last=False,
            collate_fn=collate_fn, # J：与训练数据集的 collate_fn 函数相同
        )

        # J：确保训练数据集和验证数据集的 batch 大小大于 0
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        # J：打印训练数据集和验证数据集的 batch 大小
        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        # J：计算总训练步数，默认等于训练数据集的 batch 大小乘以总轮数
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs # 注意：这里是数据的 epoch

        # J：如果配置了 total_training_steps，则使用配置的值，不需要遍历完所有数据
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps # J：将总训练步数赋值给 self.total_training_steps
        print(f"Total training steps: {self.total_training_steps}") # J：打印总训练步数

        try:
            OmegaConf.set_struct(self.config, True) # J：将配置设置为结构体，方便后续操作，防止创建不存在的字段
            with open_dict(self.config): # J：将配置转换为字典，方便后续操作，此时可继续添加字段
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    # J：如果配置了 actor_rollout_ref.actor.optim，则将 total_training_steps 赋值给 optim.total_training_steps
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    # J：如果配置了 critic.optim，则将 total_training_steps 赋值给 optim.total_training_steps
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps): # J：写出生成样本到 JSONL 文件
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl") # J：生成 JSONL 文件名，格式为 ./${global_steps}.jsonl

        n = len(inputs)
        base_data = { # J：写出内容包含输入、输出、 ground_truth、分数、step 等
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores, # J：每个样本的分数
            "step": [global_steps] * n, # J：这里是生成一个 长度为 n 的列表，每个元素都是 global_steps，保持与输入样本的对应关系，方便后续展开和组合后通过 step 来筛选样本
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}") # J：打印 dump 生成样本的 JSONL 文件名，说明已经完成写入

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path): # J：异步 dump 生成样本到 JSONL 文件
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit( # J：提交 dump 任务到 dump executor
            self._write_generations, # J：写出生成样本到 JSONL 文件
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path, # J：dump 生成样本的路径
            global_steps,
        )
        self._dump_futures.append(future) # J：将 dump 任务添加到 dump_futures 列表中
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures: # J：顺便遍历 dump_futures 列表，检查是否有已完成的任务
            if f.done():
                # J：如果 dump 任务已完成，则检查是否成功
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f) # J：如果 dump 任务未完成，则将其添加到 still_pending 列表中
        self._dump_futures = still_pending # J：将 dump_futures 列表更新为 still_pending 列表，即保留未完成的任务

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self): # J：关闭 dump executor 运行器，确保所有异步 dump 操作完成（正确执行完异步任务后退出）
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True) # J：等待所有 dump 任务完成，确保 dump executor 正常退出（正确执行完异步任务后退出）

    def _log_rollout_data( # J：异步 dump 生成样本到 JSONL 文件
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            # J：sample_gts 存储每个样本的 ground_truth，gts 应该是 ground_truths 的简称？
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = { # J：将 reward_extra_infos_dict 中的每个值转换为列表，方便 dump 到 JSONL 文件
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations( # J：异步 dump 生成样本到 JSONL 文件
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir, # J：dump 生成样本的路径
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto: # J：从 batch 中提取生成样本的 batch
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys() # J：两个 set 取交集，保留 batch 中的 non_tensor_batch 中的 reward 相关字段

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys # J：两个 set 取差集，保留 batch 中的 non_tensor_batch 中的非 reward 相关字段
        # J：这里执行后，batch 中的 non_tensor_batch 中仅包含 reward 相关字段
        gen_batch = batch.pop( # J: gen_batch 是一个 DataProto 对象，仅包含非 reward 相关字段
            batch_keys=batch_keys_to_pop, # J：batch_keys_to_pop 是一个空列表，不删除 batch 中的任何字段
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop), # J：将 non_tensor_batch_keys_to_pop 转换为列表，用于删除 non_tensor_batch 中的字段
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch) # J：此时 batch 中的 non_tensor_batch 中仅包含 reward 相关字段，将它们添加到 gen_batch 中的 non_tensor_batch 中，用于计算 reward score

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor: # J：计算 reward score 并返回包含 rm_scores 张量和 reward_extra_info 字段的 DataProto 对象，注：仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0
        # J：问题：reward_loop_manager.compute_rm_score(batch) 返回的是一个 DataProto 对象，这里写错 为 tuple[torch.Tensor, dict[str, Any]] | torch.Tensor 了
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)  # J：计算 reward score 并返回包含 rm_scores 张量和 reward_extra_info 字段的 DataProto 对象，注：仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0
        print(f"batch_reward: {batch_reward}")
        return batch_reward # 注：仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0

    def _validate(self, merged: bool = False): # J：验证模型在验证集上的性能
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch: # J：如果 test_batch 中没有 uid 字段，就添加一个随机的 uid 字段
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat( # J：重复 test_batch n 次，每个样本都生成 n 个样本
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [ # J：获取 test_batch 中每个样本的 ground_truth 字段
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations( # J：异步 dump 验证时的生成样本到 JSONL 文件（注意：验证时也是要生成样本的，这里是记录验证时的生成样本）
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        # J：创建资源池并申请资源
        # J：每个资源池包含多个 placement group，每个 placement group 对应一个节点
        self.resource_pool_manager.create_resource_pool() # J：创建资源池，按照每个节点一个 placement group （包含多个 bundle）完成资源申请

        # J: resource_pool_dict 是一个字典，键是池名，值是资源池对象
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine: # J：当前为 True
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role) # J：根据 Role 获取对应的资源池对象
            actor_rollout_cls = RayClassWithInitArgs( # J：创建 Ray Actor 类包装器，用于延迟实例化类，将类的构造函数参数存储起来，后续在需要时再实例化 Actor 类
                cls=self.role_worker_mapping[actor_role], # J：根据 Role 获取对应的 Worker 类，这里对应 verl.workers.engine_workers.ActorRolloutRefWorker 类
                config=self.config.actor_rollout_ref, # J：获取 Actor Rollout Ref 的配置
                distillation_config=self.config.get("distillation"), # J：获取蒸馏配置
                role=str(actor_role), # J：将 Role 转换为字符串，比如 Role.ActorRolloutRef 对应 "actor_rollout_ref"
            )
            # J：将 Actor Rollout Ref 类包装器添加到资源池字典中
            # J：键是 [池名]，值是 一个 dict 对象（键 是[str(角色)]，值是 RayClassWithInitArgs 对象）
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError # J：当前仅支持混合引擎

        # create critic
        if self.use_critic:
            # J：Critic 角色的资源池对象不是
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic) # J：根据 Role 获取对应的资源池对象

            from verl.workers.config import CriticConfig

            # J：将 OmegaConf 配置转换为数据 class 对象
            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic) # J：将 OmegaConf 配置转换为数据 class 对象（返回对象为 _target_ 参数指定数据 class 类型）

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig( # J：构造 TrainingWorkerConfig 对象，包含各种配置参数，继承了 BaseConfig，所以可以像字典一样使用
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            # J：Critic 角色对应的类是 verl.workers.engine_workers.TrainingWorker 类
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg) # J：创建 Ray Actor 类包装器，用于延迟实例化类，将类的构造函数参数存储起来，后续在需要时再实例化 Actor 类
            
            # J：将 Critic Worker 类包装器添加到资源池字典中
            # J：键是 [池名]，值是 一个 dict 对象（键 是[str(角色)]，值是 RayClassWithInitArgs 对象）
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # J：创建参考策略 Worker 类（若需要）
        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping: # J：当前一般 Role.RefPolicy 不在 role_worker_mapping 中
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        # J：wg_kwargs 用于设置 RayWorkerGroup 的参数，比如超时时间、设备名称等
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        # J：OmegaConf.select() 用于从 OmegaConf 配置中提取值，返回一个 Optional 对象，这里是检查 ray_wait_register_center_timeout 是否被设置
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys": # J：检查上报工具是否为 nsys
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name # J：设置设备名称，用于指定要使用的 GPU 或 CPU

        for resource_pool, class_dict in self.resource_pool_to_cls.items(): # J：遍历资源池字典，每个资源池包含多个角色的 Ray Actor 类包装器对象
            if not class_dict: # J：class_dict 是一个 dict 对象（键 是[str(角色)]，值是 RayClassWithInitArgs 对象）
                continue
            # J：create_colocated_worker_cls 返回封装了 WorkerDict 类（一个 @ray.remote 封装过的，Worker 的子类）的 RayClassWithInitArgs 类对象
            # J：WorkerDict 内部 worker_dict 属性持有多个角色的 Actor 类的 去除 @ray.remote 装饰器后的类对象（普通对象）
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict) # J：返回 RayClassWithInitArgs 类对象，持有 WorkerDict 类
            
            # J：为每个资源池创建一个 RayWorkerGroup 对象
            wg_dict = self.ray_worker_group_cls( # J：每个资源池创建一个 RayWorkerGroup 对象（ray_worker_group_cls = RayWorkerGroup）
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls, # J：封装了 WorkerDict 类的 RayClassWithInitArgs 类对象，RayWorkerGroup 会根据这个类对象在每个进程上创建 WorkerDict 类 实例
                **wg_kwargs,
            )
            # J：生成一个 dict 对象（键为 prefix（即 str(角色) ），值为 RayWorkerGroup 对象，即每个角色对应的 RayWorkerGroup 实例）
            # J：理解：生成的每个 RayWorkerGroup 实例都共享相同的 Workers 的资源（不会重新创建 Worker），但每个 RayWorkerGroup 实例仅包含指定的 prefix 开头的方法，且将方法名中的前缀替换为原始方法名
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg) # J：all_wg 是一个 dict 对象（键 是[str(角色)]，值是 RayWorkerGroup 对象, 这些 RayWorkerGroup 对象共享相同的 Workers 的资源（不会重新创建 Worker），但每个 RayWorkerGroup 实例仅包含指定的 prefix 开头的方法，且将方法名中的前缀替换为原始方法名）

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)] # J：获取 Critic 角色对应的的 RayWorkerGroup 对象
            self.critic_wg.reset() # J：reset() 方法在 RayWorkerGroup 类中是不存在的，这里调用的是绑定在 RayWorkerGroup 实例上的 reset() 方法
                                   # J：reset() 方法的路径是：
                                   # J： 第一步：TrainingWorker 定义 reset() 方法，verl.workers.engine_workers.TrainingWorker.reset
                                   # J： 第二步：worker 原生对象被 WorkerDict 对象持有并绑定 reset() 方法，调用时会委托给对应的 self.worker_dict[key]（即每个角色的普通类（解除 @ray.remote 装饰）的实例）的 reset() 方法，此时调用需要指定角色名为前缀（一个 WorkerDict 绑定了多个 Role，所以需要添加前缀区分）
                                   # J： 第三步：WorkerDict 实例被 RayWorkerGroup 实例持有并绑定 reset() 方法，调用时会委托给对应的 self.worker_dict[key]（即每个角色的普通类（解除 @ray.remote 装饰）的实例）的 reset() 方法，此时调用依然需要指定前缀
                                   # J： 第四步：spawn() 方法将 RayWorkerGroup 拆开成分 Role 的多个 RayWorkerGroup 实例（dict<role, RayWorkerGroup> 存储），（即每个角色对应的 RayWorkerGroup 实例），此时剔除角色名前缀，直接调用 reset() 方法即可
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg) # J：orig_critic_cfg 是 self.config.critic 初始化的结果（CriticConfig 类型）
            self.critic_wg.set_loss_fn(value_loss_) # J：设置损失函数计算函数（value_loss_），这里 set_loss_fn() 也是绑定得到的方法, verl.workers.engine_workers.TrainingWorker.set_loss_fn
            # J：这里不用初始化模型？

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model() # J：add_ref_policy_worker 函数已经废弃，不可能走到这里
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]
                # J：这里不用初始化，因为后面会为 actor 角色初始化模型

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)] # J：actor_role 是 Role.ActorRolloutRef 或 Role.ActorRollout
        self.actor_rollout_wg.init_model() # J：@ActorRolloutRefWorker 定义 init_model() 方法，verl.workers.engine_workers.ActorRolloutRefWorker.init_model，这里还会完成 loss_fn 绑定等

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager( # J：创建 RewardLoopManager 实例
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy: # J：For 蒸馏训练
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel) # J：获取 TeacherModel 角色对应的的资源池（RayResourcePool）
            self.teacher_model_manager = MultiTeacherModelManager( # J：创建 MultiTeacherModelManager 实例
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        # J：fqn 是 fully qualified name 的简称，例如 "verl.experimental.agent_loop.agent_loop.AgentLoopManager"
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager") # J：根据 fqn 加载自定义的 AgentLoopManager 实现，返回一个类 Type，并复制给 AgentLoopManager
        else:
            from verl.experimental.agent_loop import AgentLoopManager # J：默认使用 AgentLoopManager 实现

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        # J：enable_agent_reward_loop 用于判断是否直接在 Loop 中流式计算奖励，还是通过 RewardLoopManager 来计算奖励
        # J：如果没启用 RM，奖励计算由 Agent Loop 内部直接处理（比如 rule-based reward，通过 Python 函数计算）。这时不存在资源竞争，自然可以直接流式计算，不需要复杂的协调机制
        # J：即使有 RM，但如果它被分配了 独立的 GPU 资源池 ，那么 RM 可以独立运行，一边 Actor 做 rollout，一边 RM 在另一个资源池上并行计算奖励，结果通过流式（streaming）方式返回，互不阻塞
        # J：最后：如果 self.use_rm = True 且 enable_resource_pool = False ，说明 RM 和 Actor/Rollout 共享计算资源 。此时不能直接流式计算，因为奖励计算会占用主流程的资源，需要通过 RewardLoopManager 来协调调度，避免资源冲突
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create( # J：创建 LLMServerManager 实例
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        # J：如果 enable_agent_reward_loop 为 True，直接将 reward_loop_workers 传递给 AgentLoopManager，因为在 AgentLoopManager 中会使用这些 worker 来流式计算奖励，否则传递 None
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create( # J：创建 AgentLoopManager 实例
            config=self.config,
            llm_client=self.llm_server_manager.get_client(), # J：获取 LLMServerManager 实例的 client
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None, # J：For MOPD
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        # J：TODO：CheckpointEngineManager 中许多函数还需要阅读
        self.checkpoint_manager = CheckpointEngineManager( # J：创建 CheckpointEngineManager 实例，负责同步权重等
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self): # J：保存 ckpt
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join( # J: 保存路径格式为 default_local_dir/global_step_{global_steps}
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor") # J: 保存路径格式为 default_local_dir/global_step_{global_steps}/actor

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        # J：注：非常有用的功能，可以配置在保存检查点时删除之前的检查点，避免占用磁盘空间，但需要小心删除掉需要的 ckpt
        # J：以后建议使用 max_actor_ckpt_to_keep 和 max_critic_ckpt_to_keep 来配置保留的 ckpt 数量
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False) # J: 是否在保存检查点时删除之前的检查点
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint( # J：保存 actor ckpt，如果配置了 config.actor_rollout_ref.actor.checkpoint.async_save 为 True，则会立刻返回，否则阻塞直到完成 ckpt 存储（详情见 verl.trainer.config.CheckpointConfig）
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic)) # J: 保存路径格式为 default_local_dir/global_step_{global_steps}/critic
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            # J：特别说明，Critic 和 Actor 可以分开指定同步 或 异步
            self.critic_wg.save_checkpoint( # J：保存 critic ckpt，如果配置了 config.critic.checkpoint.async_save 为 True，则会立刻返回，否则阻塞直到完成 ckpt 存储（详情见 verl.trainer.config.CheckpointConfig）
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        # J：保存 dataloader 状态字典到文件，方便恢复训练
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt") # J: 保存路径格式为 default_local_dir/global_step_{global_steps}/data.pt
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        # J：问题：这里为什么只根据 Actor 是否异步来判断，Critic 不用关注吗？（Critic 和 Actor 是可以分开指定是否异步 save 的）
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return # J: 当 async_save 为 True 时，不写入 latest_checkpointed_iteration.txt 文件
        # J：将最新保存的 global_steps 写入 latest_checkpointed_iteration.txt 文件（当看到这个文件的内容时，就知道这个 global_steps 已经完成 save 了）
        local_latest_checkpointed_iteration = os.path.join( # J: 保存路径格式为 default_local_dir/latest_checkpointed_iteration.txt
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps)) # J: 写入最新保存的 global_steps 到 latest_checkpointed_iteration.txt 文件

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    # J：平衡每个 DP rank 上的 token 数量，当 use_prefix_grouper 为 True 时，根据 uid 进行分组，保证相同 uid 的样本在同一个 DP rank 上，用于 prefix sharing 优化
    # J：该函数用于在单控器上 重排 batch 数据，使每个 DP（数据并行）rank 分到相近数量的 token ，避免因序列长度不均导致某些 rank 计算负载过重而形成等待
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False): # J：平衡每个 DP rank 上的 token 数量
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch: # J：均衡策略1（use_prefix_grouper=True 且含 uid）
                    #                                                                      # J：调用 get_group_balanced_partitions ，保证相同 uid 的样本落在同一 rank，利于 prefix 复用；要求 num_groups % dp_size == 0
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch: # J：均衡策略2（keep_minibatch=True）
                             # J：解耦 DP 均衡与 minibatch：先按 minibatch 切片，每个 minibatch 内部独立做 dp_size 路均衡，再拼接成全局 partition
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else: # J：均衡策略3（默认方法）
              # J：直接对全部样本做 get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size)
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            # J：非 PrefixGrouper 场景下，每个 partition 内按 workload 升序排序，再用 partition[::2] + partition[1::2][::-1] —— 让流水线并行首尾阶段处理较短序列，降低气泡
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x)) # J：按 workload 升序排序
                ordered_partition = partition[::2] + partition[1::2][::-1] # J：把较小 micro-batch 放到首尾，降低气泡（[0,1,2,3,4,5] -> [0,2,4,5,3,1]）
                global_partition_lst[idx] = ordered_partition # J：更新 global_partition_lst 中的 partition 为有序 partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition]) # J：将 global_partition_lst 中的所有 partition 展平并为一个列表
        batch.reorder(global_idx) # J：根据 global_idx 重新排序 batch 中的数据
        global_balance_stats = log_seqlen_unbalance( # J：计算并记录 sequence length 不平衡相关的指标
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats) # J：更新 metrics 中 sequence length 不平衡相关的指标

    def _compute_values(self, batch: DataProto) -> DataProto: # J：执行一次推理，获取更新前的 Critic 预估值（values），返回 batch["values"] 字段的 DataProto 类型
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td) # J：执行一次推理，获取更新前的 Critic 预估值（values）
        output = output.get()
        values = tu.get(output, "values") # J：获取 values 字段
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values) # J：将 values 字段转换为 DataProto 类型
        return values # J：返回 values 字段

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto: # J：计算 "ref" 的 log_probs，回填到 batch["ref_log_prob"] 字段
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True # J：核心参数，用于标记使用 Reference 模型推理而不是 Actor 推理
        # J：关于区分 使用 actor 还是 Reference 模型的理解：
        # J：情况1：不使用 lora，则一定有 ref_in_actor=False，此时不能在 Actor 中找到 Reference
        # J：情况2：使用 LoRA，则有 ref_in_actor=True
        # J：    情况2.1：此时当设置 no_lora_adapter=True 时，表示不加载 LoRA，只使用最早的 Base，即 Reference
        # J：    情况2.2：此时当设置 no_lora_adapter=False 时，表示加载 LoRA，只使用 Base + LoRA，实现得到 Actor（ old 策略）

        tu.assign_non_tensor(batch_td, **metadata) # J：添加元信息，标记不需要计算 entropy，也不计算 loss
        if self.ref_in_actor: # J：只有在开启 lora 时，才会出现 ref_in_actor=True
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()}) # J：回填到 “ref_log_prob” 字段
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto): # J：计算 old_log_probs，返回 old_log_probs(DataProto, 包含 old_log_probs, entropys, routed_experts, sum_pi_squared 等 key)  和 old_log_prob_mfu(torch.Tensor, 是计算 old_log_probs 的 mfu 指标)
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True, # J：标记需要计算 entropy，在后续多层调用后，会根据 logits 计算出熵并存下来
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td) # J：计算 old_log_probs，到这里时，其实 \pi_{old} 等于 \pi_{\theta}，所以直接对 actor 进行前向推理得到的结果就是 old_log_probs
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu # J：返回 old_log_probs(DataProto, 包含 old_log_probs, entropys, routed_experts, sum_pi_squared 等 key)  和 old_log_prob_mfu(torch.Tensor, 是计算 old_log_probs 的 mfu 指标)

    def _update_actor(self, batch: DataProto) -> DataProto: # J：更新 Actor 网络，核心函数
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature # J: 确保和推理时的 temperature 一致，都是从 rollout_config 中读取的（其实在 Rollout 前已经赋值过一次了？这里重复但没错）
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td) # J：更新 Actor 网络（核心函数）
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor() # J：初始化 dump executor，用于异步 dump 生成轨迹样本到 JSONL 文件

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking( # J：初始化 Tracking 类，用于记录训练指标，支持不同的 backend，如 WandB、TensorBoard 等
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger, # J：这里可以是一个 list，用于指定默认 backend，如 WandB、TensorBoard 和 Console 等
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint() # J：加载检查点会修改 global_steps 为检查点中的 global_steps
        self.checkpoint_manager.update_weights(self.global_steps) # J：更新 Actor 网络的权重到 rollout replicas

        # J：current_epoch 和 global_steps 都从 0 开始计数（但这里 global_steps 可能因为续训练而不为 0）
        current_epoch = self.global_steps // len(self.train_dataloader) # J：计算当前训练轮数 = 当前 global_steps / 训练数据集大小

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True): # J：一般训练不会开启这个，因为时间会比较久
            val_metrics = self._validate() # J：验证模型在训练前的指标情况
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor() # J：关闭 dump executor，因为验证模式下不需要 dump 数据，这里的 dump_executor 是异步 dump 生成轨迹样本到 JSONL 文件的
                return

        # J：问题，当前配置参数似乎是 config.actor_rollout_ref.rollout.skip_rollout 了
        # J：这个参数的目的是：是在序列生成过程中加入 跳过/缓存/重复 逻辑，避免重复执行昂贵的 rollout 生成
        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager) # J：初始化 RolloutSkip 类，用于管理跳过/缓存/重复 逻辑
            rollout_skip.wrap_generate_sequences() # J：装饰器工厂函数 ，用于包装 rollout 工作组（ rollout_wg ）的 generate_sequences 方法，目的是在序列生成过程中加入 跳过/缓存/重复 逻辑，避免重复执行昂贵的 rollout 生成

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1 # J：从 1 开始计数上报信息，global_steps=0 表示初始 ckpt 状态
        last_val_metrics = None # J：记录上一次验证的指标情况，用于记录和打印最后一次评估指标
        self.max_steps_duration = 0 # J：记录训练过程中，单个 Step 的最大耗时

        prev_step_profile = False
        curr_step_profile = ( # J：判断当前 step 是否在配置的 profiling step 列表中
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs): # J：遍历训练轮数
            for batch_dict in self.train_dataloader: # J：遍历训练数据集
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    # J：如果 actor_rollout_wg （actor+rollout 的 Ray 工作组）有 async_calls_finalize_fn_exec 方法，就以 非阻塞（blocking=False） 方式调用一次
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False) # 触发对所有 worker 上 尚未完成的异步调用 （async calls queue）做一次 finalize（推进/收尾），通常是把 worker 内 Megatron AsyncCallsQueue 里挂起的异步请求真正执行/清理掉
                    # J：理解，随 PR #4253 "[megatron] fix: megatron async save ckpt fix" 引入（commit 9d772002）
                    # J：只有用 Megatron 后端 + async_save 开启 时，worker 上才会有挂起的异步 ckpt 保存请求，工作组才会被附加上这个 async_calls_finalize_fn_exec 方法。FSDP 等其它后端不会有，因此才需要 hasattr 兜底
                    # J：关于 async_calls_finalize_fn_exec 函数的使用：
                    # J： 第一：每步开头 (blocking=False ) ：清理/推进 上一个 step 遗留的异步保存请求， 不阻塞 当前 step——这样既保证异步 ckpt 不会无限堆积，又不影响训练吞吐。这是这行注释里说的"上一轮遗留的异步调用，非阻塞地执行掉"。
                    # J： 第二：最后一步末尾 (blocking=True ) ：训练结束前 阻塞 等待所有异步保存真正完成，避免进程退出时还有未落盘的 ckpt。这与紧随其后的 _shutdown_dump_executor() 配合，确保干净退出
                metrics = {} # J：重置 metrics 字典
                timing_raw = {} # J：重置 timing_raw 字典

                with marked_timer("start_profile", timing_raw): # J: 这里 start_profile 是记录开启 profiling 所需要 的时间消耗 key，理解：开启 profile 本身就需要时间
                    # J：在指定的训练 step 上，向 actor_rollout / ref_policy / critic 三个 worker group 广播开启性能采集（torch profiler / nsys / NVTX / 内存快照 / 精度调试器等）
                    # J：所以这是一次 Ray RPC 广播，所有 rank 上的 DistProfiler.start() 都会被触发
                    # J：特别说明：训练动辄成千上万 step，全程 profiling 既慢又产文件巨大。verl 用"按 step 精确采样 + 可选连续段"的方式，让用户只对感兴趣的 step（如 warmup 后的几个 step）做性能/内存/精度分析，便于排查训练瓶颈、显存占用、数值精度等问题
                    self._start_profiling( # J：传入参数名为 do_profile，当 do_profile=True 时，向所有相关 worker group 广播"开启 profiling"的指令
                        not prev_step_profile and curr_step_profile # J：只需要对每段开始（start step） 打开 profiling
                        if self.config.global_profiler.profile_continuous_steps # J：一般为 False，但 continuous_steps 为 True 时，会连续采集 profiling 数据，此时开启仅在段首即可
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict) # J：将 batch_dict 转换为 DataProto 类型
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature # J：temperature 采样参数配置，用于接下来采样使用

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array( # J：为每个 prompt 生成一个唯一的随机 uid，用于在 trace 中标识每个 prompt
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch) # J：获取 gen_batch, 包含 prompt, temperature, uid 等信息

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps # J：将当前 step 传递给 trace，用于记录当前 step
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True) # J：重复 gen_batch 以生成多个 rollout

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX: # J：如果使用 REMAX 优势估计器，需要生成一个 greedy baseline
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    # J：__do_sample__ 是一个 临时的内部控制标志位 ，专门用于 REMAX 优势估计器的组合 rollout 场景（注：REMAX 需要为每个 prompt 额外生成一个 greedy baseline 作为参考）
                    # J：- 主生成批次（policy rollout）： __do_sample__ = True （使用随机采样）
                    # J：- 基线批次（REMAX baseline）： __do_sample__ = False （使用贪心解码）
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool) # J：将 policy rollout 的 __do_sample__ 标志位全部设为 True
                    gen_baseline_batch = gen_batch.slice(0, None) # J：完整复制 gen_batch 以生成 baseline，注意 gen_batch 是没有复制过的（gen_batch_output 才是复制过的），数量跟原始 Batch size 一致(不是 Batch size * rollout_n)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool) # J：将 baseline 的 __do_sample__ 标志位全部设为 False
                    # J：组合时，前面是 policy rollout，后面是 baseline rollout
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch]) # J：合并生成请求，得到包含（batch_size * rollout_n + batch_size）大小的 DataProto 对象
                    num_sampled_prompts = len(gen_batch_output) # J：仍只记录 policy rollout 的 prompt 数量，用于后续分离出除 baseline 外的采样样本
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps # J：判断是否是最后一步
                with marked_timer("step", timing_raw): # J：记录 step 时间(这里是整个 step 的时间，包括生成、奖励评估、更新等)， timing_raw 是用于存储 step 时间的字典
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"): # J：开始生成阶段并记录生成时间，时间上报到 timing_raw 字典中 {"gen": 生成时间}
                        if curr_step_profile: # J：如果当前 step 开启了性能采集
                            self.llm_server_manager.start_profile() # J：开启 llm server 性能采集
                        
                        # J：TODO，Generate 的详细细节还需要再看看，尤其涉及到 Agent Loop 的部分
                        combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch) # J：核心函数，执行生成序列操作，这里是阻塞调用，会等待所有 rollout 引擎完成生成，再返回结果
                        self.checkpoint_manager.sleep_replicas() # J：把 rollout 引擎休眠，腾出显存给训练用
                        if curr_step_profile: # J：如果当前 step 开启了性能采集
                            self.llm_server_manager.stop_profile() # J：关闭 llm server 性能采集

                        # J：TODO，问题，这里仅用 timing 作为 key，会导致不知道是不是 gen 阶段的吧？
                        timing_raw.update(combined_gen_output.meta_info["timing"]) # J：更新 timing_raw 字典，包含 llm Server 生成序列的时间
                        combined_gen_output.meta_info.pop("timing", None) # J：从 combined_gen_output 中移除 timing 字段（应该是避免其他不必要影响）

                    # J：这里理论上只有 REMAX 优势估计器中 combined_gen_output 会多出来 baseline ，其他优势估计器中 combined_gen_output 就是 gen_batch_output
                    gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts) # J：slice(start, end)，提取 policy rollout 的输出
                    if "__do_sample__" in gen_batch_output.non_tensor_batch:
                        gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"]) # J：后续阶段不需要 __do_sample__（仅用于 REMAX 优势估计器生成区分 policy rollout 和 baseline） 这个字段了

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None) # J：slice(start, end)，提取 baseline rollout 的输出
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"]) # J：后续阶段不需要 __do_sample__（仅用于 REMAX 优势估计器生成区分 policy rollout 和 baseline） 这个字段了

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys(): # J：如果使用 REMAX 优势估计器，且使用 Reward Model，且 baseline rollout 的输出中没有 rm_scores 张量
                            # J：给 baseline rollout 计算 reward score
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output) # J：计算 reward score 并返回包含 rm_scores 张量和 reward_extra_info 字段的 DataProto 对象，注：仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0
                            gen_baseline_output = gen_baseline_output.union(baseline_reward) # J：将 baseline_reward 合并到 gen_baseline_output 中(不是按照行合并，是按照 key 合并)，合并后就有了 “rm_scores” 张量

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1) # J：对 baseline rollout 的 rm_scores 张量进行求和，得到每个 prompt 的 reward score，注：一般来说，RLHF 中，仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0，实际上就是取了最后一个 Response token 的 reward score
                        batch.batch["reward_baselines"] = reward_baseline_tensor # J：将 reward score 赋值给 batch 中的 reward_baselines 字段

                        del gen_baseline_output # J：删除 baseline rollout 的输出，避免占用显存
                    del combined_gen_batch, combined_gen_output # J：删除 combined_gen_batch 和 combined_gen_output，避免占用显存
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True) # J：将 batch 中的 tensor 重复 rollout_n 次，每个 tensor 都会逐元素重复，用于跟 policy rollout 数量（gen_batch_output）对齐
                    batch = batch.union(gen_batch_output) # J：将 gen_batch_output 按照列合并到 batch 中，与 batch 中的 tensor 一一对应（注意：是 repeated batch 数量才对得上）

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch) # J：计算 Response 部分的注意力掩码，读取 batch.batch["attention_mask"]
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).

                    if self.config.trainer.balance_batch: # J：当前一般是打开的
                        self._balance_batch(batch, metrics=metrics) # J：平衡每个 DP rank 上的 token 数量, 并更新 metrics 中 sequence length 不平衡相关的指标

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist() # J：计算每个样本的有效 token 数量，维度为 [batch_size*rollout_n]
                    # get images_seqlens
                    images_seqlens_all = []
                    # non_tensor_batch["multi_modal_inputs"] 是一个 list, 每个样本一个的 dict
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]: # J：遍历每个样本的 multi_modal_inputs 字段，for 多模态输入
                        # J：image_grid_thw[:, 0] → 每张图的帧数 T
                        # J：image_grid_thw[:, 1] * image_grid_thw[:, 2] → 每张图的 H*W （一帧的 patch 数）
                        if "image_grid_thw" not in multi_modal_input.keys(): # J：跳过纯文本样本
                            continue
                        # J：images_seqlens 是 派生量 ，每帧的视觉 token 数（= H×W），可用于后续统计视觉 token 数量，计算视觉 FLOPs 需要
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist()) # J：将每个样本的 images_seqlens 列表添加到 images_seqlens_all 中
                    batch.meta_info["images_seqlens"] = images_seqlens_all # J：将 images_seqlens_all 赋值给 batch.meta_info["images_seqlens"] 字段
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch) # J：计算 reward score 并返回包含 rm_scores 张量和 reward_extra_info 字段的 DataProto 对象，注：仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0
                            # J: 特别说明：自定义的 奖励函数中，可以返回任意
                            batch = batch.union(batch_reward) # J：将 batch_reward 合并到 batch 中(不是按照行合并，是按照 key 合并)，合并后就有了 “rm_scores” 张量

                        # J：问题，score 指标（原始模型打分）似乎没有被上报到 metrics 中？
                        # J：回答：上报了，在后面的 compute_data_metrics 函数中收集到 critic/score/xxx 指标中，然后汇总以后上报的
                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch) # J：从 batch 中提取 reward_tensor（"rm_scores"） 和 reward_extra_infos_dict（meta_info["reward_extra_keys"] 对应的 non_tensor_batch 中的数据） 字段

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ) # J：没有 π_old
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode( # J：应用 bypass 模式，此时是直接设置 old_log_probs = rollout_log_probs，回填到 batch["old_log_probs"] 字段
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs； J：重新计算 old_log_probs，作为 proximal anchor
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch) # J：计算 old_log_probs，包含 ”entropy" 字段和 “old_log_probs” 字段 等
                            entropys = old_log_prob.batch["entropys"] # J：从 old_log_prob 中提取 entropy 字段，这里的熵是 Token 粒度的，shape = [batch_size*rollout_n, seq_len]
                            response_masks = batch.batch["response_mask"] # J：shape 与 entropys 相同
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss( # J：聚合 entropy（根据 loss_agg_mode 聚合，类似 loss 聚合一样）
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode, # J：聚合 entropy 的方式，复用 actor loss 的 loss_agg_mode 配置
                                loss_scale_factor=actor_config.loss_scale_factor, # J：聚合 entropy 的缩放因子，复用 actor loss 的 loss_scale_factor 配置
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(), # J：上报熵
                                "perf/mfu/actor_infer": old_log_prob_mfu, # J：上报 actor 推理时间
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                # J：不能同时使用 R2 模式和 R3 模式，R2: "routed_experts" in batch.batch; R3: "routed_experts" in old_log_prob.batch
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch)) # J：计算 rollout vs actor logprobs 相关指标，用于调试（这个 diff 不能太大）

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}' # J：计算完成后需要确保 old_log_probs 存在

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"): # J：计算 "ref"
                            ref_log_prob = self._compute_ref_log_prob(batch) # J：计算 "ref" 的 log_probs，回填到 batch["ref_log_prob"] 字段
                            batch = batch.union(ref_log_prob) # J：将 "ref" 的 log_probs 合并到 batch 中，添加 ["ref_log_prob"] 字段

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch) # J：执行一次推理，获取更新前的 Critic 预估值（values），返回 batch["values"] 字段的 DataProto 类型
                            batch = batch.union(values) # J：将 values 合并到 batch 中，添加 ["values"] 字段

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor # J：reward_tensor 是前面计算得到的 reward 信息，这里赋值为 token_level_scores，注意，从此不再是 rm_scores 字段

                        if reward_extra_infos_dict: # J：如果有 reward_extra_infos_dict，说明有额外的奖励信息，需要合并到 batch 中
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty( # J：应用 KL 惩罚项，根据 config 中的配置
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"] # J：包含 KL 惩罚的信息，这里赋值为 rewards

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode # J：bapass 模式下不会计算 megatron/FSDP 的策略 logprobs，此时使用的训推 logprobs 完全相同，训推策略比值永远为 1
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config) # J：计算 IS 和 RS 的效果，相关字段和指标都存到 batch 中
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True # J：默认对 GRPO 的 Advantage 进行归一化
                        )  # GRPO adv normalization factor

                        batch = compute_advantage( # J：计算 Advantage（核心函数），计算后的很多指标会直接添加到返回的 batch 里面
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch) # J：更新 Critic 网络
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics) # J：将 Critic 网络的指标合并到 metrics 中

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps: # J：如果 Critic 网络的预热步数大于当前步数，继续预热
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        # J：为什么需要同步 rollout replicas 的权重？这里还在 Critic 网络，不需要同步 Rollout Replicas 的权重吧
                        # J：理解，训练 Critic 也需要用到 Rollout 策略，所以是需要同步的，但是实际上只需要同步一次即可，这里是为了确保没问题（每个 step 都同步了一次，除了浪费时间，没有别的问题）
                        self.checkpoint_manager.update_weights(self.global_steps) # J：将 Actor 网络的权重同步到 Rollout Replicas 中
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch) # J：更新 Actor 网络

                        # J: 如果购买的是云厂商的 ESI 实例，需要检查是否接近过期时间，如果接近，强制保存检查点(真是贴心的设计)
                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi( # J：检查是否接近过期时间，暂时先不用管
                            max_steps_duration=self.max_steps_duration, # J: 这里传入历史单步训练最长时间，用于预估单步训练需要的最长时间
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ): # J：如果满足保存检查点的条件，保存检查点（1.最后一步；2.当前步数是 save_freq 的倍数；3. ESI 实例接近过期时间）
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"): # J：记录保存检查点的时间消耗
                                self._save_checkpoint() # J：保存 ckpt

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps) # J：更新 Actor 网络的权重到 rollout replicas

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics) # J：将 Actor 更新相关的指标合并到 metrics 中

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir) # J：异步 dump 生成样本到 JSONL 文件

                # validate
                if self.config.trainer.test_freq > 0 and ( # J：仅在 test_freq > 0 时，启动模型评估
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0 # J：在最后一步或当前步数是 test_freq 的倍数时，评估模型
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate() # J：评估当前模型
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # J：注：到这里本 Step 训练已经完成了
                with marked_timer("stop_profile", timing_raw): # J：记录停止 profile 所需要 的时间消耗，注：停止 profiling 本身也需要时间
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"] # J：获取当前 Step 训练的时间消耗
                self.max_steps_duration = max(self.max_steps_duration, steps_duration) # J：更新历史单步训练最长时间

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic)) # J：收集数据指标（奖励、损失、梯度范数等）用于上报
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None) # J: 配置示例 '["accuracy_reward", "format_reward"]'
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys: # J：针对 GDPO，分维度分别上报信息（这里应该是每个 维度的奖励信息）
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw)) # J：上报各阶段时间相关指标，包括 gen, ref, values, adv, update_critic, update_actor 等
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)) # J：上报吞吐量指标，包括 total_num_tokens, time_per_step, throughput 等
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None) # J：获取梯度范数指标
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm)) # J：上报梯度方差监控诊断指标（refer to OTB paper）
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics( # J：计算 speculative decoding 指标，如 accept_rate, accept_length, etc
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                # J：核心，这一步将所有指标记录到后端，可以是文件、数据库、可视化工具等
                logger.log(data=metrics, step=self.global_steps, backend=["file"]) # J：将指标记录到后端

                progress_bar.update(1) # J：更新进度条
                self.global_steps += 1 # J：更新全局步数

                if is_last_step: # J：如果是最后一个 Step，阻塞等待所有异步保存真正完成，管理 dump executor 行器的关闭
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True) # J：训练结束前 阻塞 等待所有异步保存真正完成，避免进程退出时还有未落盘的 ckpt
                    self._shutdown_dump_executor() # J：关闭 dump executor 行器，确保所有异步 dump 操作完成（正确执行完异步任务后退出）
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"): # J：只有当 self.train_dataset 实现了 on_batch_end 方法时才调用
                    # The dataset may be changed after each training batch
                    # J：把当前训练完的 batch 传回 dataset，让 dataset 有机会 根据上一轮训练结果动态更新自身
                    # J：可能场景如：基于上一轮训练反馈调整下一批数据（比如课程学习中根据当前分数调整下一批数据难度？）
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
