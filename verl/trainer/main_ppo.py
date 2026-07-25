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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other mpain.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import deprecated


@deprecated(
    "main_ppo.py is deprecated, and wil be replaced by main_ppo_sync.py in v0.8.0, please use main_ppo_sync.py instead."
)
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        # J：dict[int, Worker], role_worker_mapping Key 为 Role 的枚举值，Value 为 Role 类的 Ray 远程对象
        # J: dict[int, str], 键是 Role 的枚举值，值是资源池 ID（如 "global_pool" 或 "teacher_pool" 等）
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config): # J：添加 actor_rollout Worker
        """Add actor rollout worker using the unified model engine implementation."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import ActorRolloutRefWorker

        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        # Ref policy is fused into ActorRolloutRefWorker unless LoRA is used with a dedicated ref model.
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        # J：将 ActorRolloutRefWorker 类初始化添加到 role_worker_mapping 中（Key 为 Role 枚举值，Value 为 ActorRolloutRefWorker 类的 Ray 远程对象）
        # J：并将其对应的 Role 枚举值映射到 global_pool 资源池（Key 为 Role 枚举值，Value 为 "global_pool"）
        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls # J：返回 ActorRolloutRefWorker 类和 RayWorkerGroup 类, ray_worker_group_cls 用于传入训练循环中创建 RayWorkerGroup 实例

    def add_critic_worker(self, config):
        """Add critic worker to role mapping using the unified model engine implementation."""
        from verl.trainer.ppo.ray_trainer import Role
        from verl.workers.engine_workers import TrainingWorker

        # The model-engine TrainingWorker handles all critic backends (fsdp/fsdp2/megatron/...)
        # internally based on ``config.critic.strategy``.
        self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config): # J：初始化资源池管理器，构造 ResourcePoolManager 对象
        """Initialize resource pool manager."""

        global_pool_id = "global_pool"
        resource_pool_spec = { # J: dict[str, list[int]], 键是池名，值是一个列表，长度等于节点数，每个元素是该节点上的 GPU 数
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes, # J: 表示 global_pool 跨 nnodes 个节点、每节点 n_gpus_per_node 张卡，共 n_gpus_per_node*nnodes 张卡
        }

        # J：判断 reward_model 是否需要独立资源池来处理
        if config.reward.reward_model.enable_resource_pool:
            # J：若 reward_model 启用独立资源池，则需要指定 reward_pool 跨的节点数和每节点上的 GPU 数
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool # J: 添加一个名字为 "reward_pool" 的 Key，用于 reward_pool 独立使用资源池
        else:
            # J: 否则共用资源池(不用添加 ”reward_pool“ 这个 Key)，同时将 reward 的参数设置回去
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        # J：判断 distillation 是否需要独立资源池来处理， OPD 时会创建独立的 Teacher
        distillation_config = config.get("distillation")
        if is_distillation_enabled(distillation_config):
            if distillation_config.n_gpus_per_node <= 0:
                raise ValueError("config.distillation.n_gpus_per_node must be greater than 0")
            if distillation_config.nnodes <= 0:
                raise ValueError("config.distillation.nnodes must be greater than 0")

            teacher_pool = [distillation_config.n_gpus_per_node] * distillation_config.nnodes # J: 使用 distillation 配置下对应的 n_gpus_per_node 和 nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool # J: 添加一个名字为 "teacher_pool" 的 Key，用于 teacher_pool 独立使用资源池

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        # J: 构造 ResourcePoolManager 数据结构并返回
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_resource_pool(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            # we do not use reward model workers, so we only register reward model in resource pool
            # without continue to register reward model worker in role mapping
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_teacher_model_resource_pool(self, config):
        """Add teacher model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if is_distillation_enabled(config.get("distillation")):
            # we do not use teacher model workers, so we only register teacher model in resource pool
            # without registering a teacher model worker in role-worker mapping
            self.mapping[Role.TeacherModel] = "teacher_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Ref policy is fused into ActorRolloutRefWorker in the unified model engine.

        Kept for backward compatibility with subclasses that still invoke it; the method
        is now a no-op because the reference policy lives on the same worker group as
        the actor/rollout.
        """
        return

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # J: Worker 决策映射 & 注册，注：在统一模型引擎中，ref policy 已被融合到 ActorRolloutRefWorker 中，因此不再需要单独的 ref policy worker
        # J: 返回得到的是：
        #        # actor_rollout_cls = ActorRolloutRefWorker
        #        # ray_worker_group_cls = RayWorkerGroup
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config) # J: 主要修改 self.role_worker_mapping 和 self.mapping
        self.add_critic_worker(config) # J: 主要修改 self.role_worker_mapping 和 self.mapping

        # J：可能新增 reward_pool 资源池, 需要 Reward 且 config.reward.reward_model.enable_resource_pool 为 True 时，会新增 reward_pool 资源池
        self.add_reward_model_resource_pool(config) # J: 主要修改 self.mapping

        # J：可能新增 teacher_pool 资源池, 用于 distillation 时独立使用资源池
        self.add_teacher_model_resource_pool(config) # J: 主要修改 self.mapping

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls) # J: 已经废弃了，当前实现是空函数

        # J: 验证配置是否符合预期，部分配置之间有一定的依赖关系, 例如：训练 Batch 大小必须是 DP 大小的整数倍
        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # J：将文件或目录从远程存储（如 HDFS）复制到本地缓存 ，并可选地加载到共享内存（shm）中加速访问
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # J：实例化 tokenizer 和 processor
        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # J：初始化资源池管理器，构造 ResourcePoolManager 对象
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # J：数据处理
        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset( # J：创建训练数据集，这里可能会经过一次 shuffle 操作（与实现类有关）
            config.data.train_files,
            config.data, # J：传递数据配置，用于确定数据集类等
            tokenizer,
            processor,
            is_train=True, # J：指定为训练数据集
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset( # J：创建验证数据集
            config.data.val_files,
            config.data, # J：传递数据配置，用于确定数据集类等
            tokenizer,
            processor,
            is_train=False, # J：指定为验证数据集
            max_samples=config.data.get("val_max_samples", -1),
        )
        # J：创建训练数据集的采样器
        # # 如果数据配置中启用了 shuffle 选项, 则创建随机采样器（配置参数为 config.data.shuffle）
        # # 如果未启用 shuffle 选项, 则创建顺序采样器
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer( # J：初始化 PPO 训练器，核心函数
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping, # J: Key 为 Role 的枚举值，Value 为 Role 类的 Ray 远程对象
            resource_pool_manager=resource_pool_manager, # J: ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
            ray_worker_group_cls=ray_worker_group_cls, # J: ray_worker_group_cls == RayWorkerGroup
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn, # J：自定义的 collate_fn 函数，用于将 batch 个的样本字典转换为同 key 名的一个整体 batch 的 Tensor 和 numpy 数组
            train_sampler=train_sampler, # J：训练数据集的采样器
        )
        # Initialize the workers of the trainer.
        trainer.init_workers() # J：初始化训练器的 worker，包括 ActorRolloutRefWorker、CriticWorker 等

        # Start the training process.
        trainer.fit() # J：开始训练过程，会执行训练循环，包括数据加载、模型更新、损失计算、优化器更新等


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    dataset_cls = get_dataset_class(data_config) # J：根据数据配置获取数据集类，默认使用 RLHFDataset 类

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls( # J：实例化数据集类，dataset_cls 为默认类（RLHFDataset）时，会初始化数据集并执行超长过滤等操作
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset): # J：创建数据集的采样器，跟数据集的 shuffle 选项选择随机采样器或顺序采样器
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle: # J：如果数据配置中启用了 shuffle 选项, 则创建随机采样器
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        # J：创建随机采样器，用于在训练时随机采样数据集中的样本
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # J：如果数据配置中未启用 shuffle 选项, 则创建顺序采样器
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
