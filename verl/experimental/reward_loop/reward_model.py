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

import asyncio
import logging
import os

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager: # J：RewardModelManager 是 reward model 的管理类，负责初始化 reward model 的 replica 和路由
    """Reward model manager."""

    def __init__(
        self,
        config: RewardModelConfig, # J：config 是 reward model 的配置
        resource_pool: RayResourcePool = None,
    ):
        """
        Initialize the reward model manager.

        Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """
        self.config = config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_router()
        assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_config = self.config.rollout
        rollout_world_size = ( # J：rollout_world_size 是 rollout 角色的 world_size，即 rollout 角色的 GPU 数量的乘积
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        world_size = ( # J：world_size 是所有 worker 的数量或所有节点的 GPU 数量的乘积（standalone mode 下为 n_gpus_per_node * nnodes）
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size # J：num_replicas 是 rollout_world_size 的整数倍
        assert num_replicas > 0, ( # J：num_replicas 必须大于 0，否则无法运行 reward model，小于等于 0 时，表示 world_size 小于 rollout_world_size
            f"Not enough GPUs to run the reward model. "
            f"world_size ({world_size}) < rollout_world_size ({rollout_world_size}). "
            f"Check your resource pool or standalone config (n_gpus_per_node, nnodes)."
        )

        rollout_replica_class = get_rollout_replica_class(rollout_config.name)
        model_config = HFModelConfig(path=self.config.model_path) # J：model_config 是 reward model 的配置，包含模型路径、分层配置等
        self.tokenizer = model_config.get_processor()
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            self._run_all(
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_router(self):
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        # TODO (dyy): sglang router is not ready yet.
        # if self.config.rollout.name == "sglang":
        #     from .router.inner_sglang_router import launch_router_process
        # else:
        #     from .router.naive_router import launch_router_process

        from .router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self):
        return self.router_address

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
