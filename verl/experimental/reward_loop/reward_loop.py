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

import aiohttp
import numpy as np
import ray
from omegaconf import DictConfig, open_dict
from ray.actor import ActorHandle
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import load_reward_manager, resolve_reward_manager_cls
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.ray_utils import get_event_loop

from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def migrate_legacy_reward_impl(config):
    """
    Migrate the legacy reward model implementation to the new one.
    """
    # 1. reward workers migration
    # config.reward_model.num_workers -> config.reward.num_workers
    if config.reward_model.num_workers is not None:
        config.reward.num_workers = config.reward_model.num_workers

    # 2. reward manager migration
    # config.reward_model.reward_manager -> config.reward.reward_manager
    if config.reward_model.reward_manager is not None:
        config.reward.reward_manager.name = config.reward_model.reward_manager
    if config.reward_model.reward_loop_source is not None:
        config.reward.reward_manager.source = config.reward_model.reward_loop_source
        config.reward.reward_manager.module.path = config.reward_model.reward_loop_module_path
        config.reward.reward_manager.module.name = config.reward_model.reward_loop_class_name

    # 3. custom reward function migration
    # config.custom_reward_function -> config.reward.custom_reward_function
    if not all(v is None for v in config.custom_reward_function.values()):
        config.reward.custom_reward_function = config.custom_reward_function

    # 4. reward model migration
    # config.reward_model -> config.reward.reward_model
    for key in ["enable", "enable_resource_pool", "n_gpus_per_node", "nnodes"]:
        if config.reward_model.get(key) is not None:
            config.reward.reward_model[key] = config.reward_model[key]
    if config.reward_model.model.path is not None:
        config.reward.reward_model.model_path = config.reward_model.model.path
    # config.reward_model.reward_kwargs -> config.reward.reward_kwargs (for dapo algo)
    if config.reward_model.get("reward_kwargs") is not None:
        with open_dict(config.reward):
            config.reward["reward_kwargs"] = config.reward_model["reward_kwargs"]
    # config.reward_model.rollout -> config.reward.reward_model.rollout
    legacy_rollout = config.reward_model.rollout
    for key in legacy_rollout.keys():
        if legacy_rollout[key] is not None:
            config.reward.reward_model.rollout[key] = legacy_rollout[key]

    # 5. sandbox_fusion migration
    # config.sandbox_fusion -> reward.sandbox_fusion
    if not all(v is None for v in config.sandbox_fusion.values()):
        config.reward.sandbox_fusion = config.sandbox_fusion

    # 6. delete legacy config from configs
    with open_dict(config):
        del config.reward_model
        del config.custom_reward_function
        del config.sandbox_fusion

    return config


class RewardLoopWorker: # J: Reward 处理的 worker，负责计算一个 batch 数据的奖励分数
    # J：关系：RewardLoopManager 持有多个 RewardLoopWorker（Ray Worker），RewardLoopWorker 包含一个 RewardManager, RewardManager 负责真实的 reward 计算过程
    """
    RewardLoopWork can tackle reward computation:
    (1) rule-based reward computation
    (2) reward model-based reward computation (both disrm and genrm)
    (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

    Reward Computation Logic:
    - if user-customized reward function is provided:
        -> directly use user-customized reward function
    - if user-customized reward function is not provided:
        -> rm is not enabled: use default rule-based reward function
        -> rm is disrm: compute reward score using disrm
        -> rm is genrm: raise error (user-costomized reward func must be provided) # J：GenRM 必须配置 custom_reward_function （因为要解析生成的判断文本）
    """

    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn() # J: 初始化 RewardManager，回填到 self.reward_manager 中
        self.loop = get_event_loop()

    def _init_reward_fn(self): # J：初始化 RewardManager 对象
        input_tokenizer_path = self.config.actor_rollout_ref.model.tokenizer_path
        if input_tokenizer_path is None:
            input_tokenizer_path = self.config.actor_rollout_ref.model.path
        input_tokenizer_local_path = copy_to_local(input_tokenizer_path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward.reward_model.enable:
            # J：加载 reward model 的 tokenizer
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward.reward_model.model_path) # J：复制文件到本地路径，并返回本地路径
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)

        self.reward_manager = load_reward_manager( # J：加载 RewardManager 类对象，reward_manager 是真实执行奖励计算的对象
            # J：关系：RewardLoopManager 持有多个 RewardLoopWorker（Ray Worker），RewardLoopWorker 包含一个 RewardManager, RewardManager 负责真实的 reward 计算过程
            self.config,
            self.input_tokenizer,
            reward_router_address=self.reward_router_address,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]: # J：计算一个 batch 的奖励分数
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, data: DataProto) -> dict: # J：计算奖励分数，判别式 RM 是特殊函数调用，其他情况（包括 RLVR 和 GenRM 场景都是 reward_manager 负责）
        if self.config.reward.custom_reward_function.path is not None: # J：自定义配置的奖励打分函数
            # directly use user-customized reward function
            return await self.reward_manager.run_single(data) # J：执行一次奖励打分
        else:
            if self.config.reward.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                # J: 这里 disrm 是 Discriminative Reward Model 的含义，说明是调用判别式奖励模型计算奖励分数，与这个对应的是可以考虑使用 GenRM
                # J: 类文档 reward_loop.py 也写明： GenRM 必须配置 custom_reward_function （因为要解析生成的判断文本），所以代码里注释 # we assume the rm is disrm，只要走到 compute_score_disrm 这条分支，就假定它是 DisRM
                return await self.compute_score_disrm(data[-1:])
            else:
                return await self.reward_manager.run_single(data)

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.reward_router_address}/{endpoint}"
        last_exception = None
        for attempt in range(max_retries):
            try:
                # It's safer to have a timeout instead of None, which can hang indefinitely.
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientResponseError as e:
                # Do not retry on 4xx client errors, but retry on 5xx server errors.
                if 400 <= e.status < 500:
                    logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                    raise
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:
                # Using exponential backoff is generally better than a fixed sleep.
                backoff_seconds = 2**attempt
                await asyncio.sleep(min(backoff_seconds, 30))

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
        if last_exception:
            raise last_exception

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        rollout_response = self.input_tokenizer.decode(valid_response_ids)
        rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

        chat.append({"role": "assistant", "content": rollout_response})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]

        return rm_prompt

    async def compute_score_disrm(self, data: DataProto) -> dict: # J：判别式 RM 的奖励分数计算
        disrm_prompt = await self._preprocess_reward_inputs(data)
        engine_name = self.config.reward.reward_model.rollout.name
        model_name = self.config.reward.reward_model.model_path
        # J：区分不同推理引擎得到结果
        if engine_name == "vllm": # J：vllm 引擎
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "use_activation": False,
            }
            output = await self._post_request(payloads, "classify") # J：真实调用函数
            rm_score = output["data"][-1]["probs"][-1]
        elif engine_name == "sglang": # J: sglang 引擎
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        elif engine_name == "trtllm":
            # TODO: remove this once TRT-LLM switches to TorchSampler
            raise ValueError("TensorRT-LLM backend does not support reward models currently.")

            payloads = {
                "model": model_name,
                "prompt": disrm_prompt,
                "return_context_logits": True,
            }
            output = await self._post_request(payloads, "v1/completions")
            rm_score = output["choices"][0]["context_logits"]
            assert isinstance(rm_score, list) and len(rm_score) > 0, (
                "TensorRT-LLM OpenAI server response for reward score is not in the expected format."
            )

            rm_score = float(rm_score[0][0])
            logger.debug(f"rm score: {rm_score}")
        else:
            raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")

        return {"reward_score": rm_score}


class RewardLoopManager: # J：RewardLoopManager 类定义，RewardLoopManager 与主流程交互，包含了 RewardManager 类
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        if self.config.reward.reward_model.enable: # J：如果打开 reward_model 则，需要创建 RewardModelManager 对象
            # J：目前都采用将 reward model 放到一个节点上，然后打开 ip 服务，其他节点通过 ip 调用 的方式实现（基于 vllm 或者 sglang 引擎）
            self.reward_model_manager = RewardModelManager(config.reward.reward_model, rm_resource_pool) # J：reward_model_manager 是 reward model 的管理类，负责初始化 reward model 的 replica 和路由
            self.reward_router_address = self.reward_model_manager.get_router_address() # J：reward_router_address 是 reward model 的路由地址，用于其他节点调用
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self.reward_loop_workers_class = ray.remote(RewardLoopWorker) # J：注册 RewardLoopWorker 类为远程
        self.reward_manager_cls = resolve_reward_manager_cls(config) # J：根据配置文件解析 RewardManager 类，这个类在 RewardLoopManager 中不会初始化，仅用于调用一些静态类函数
        self._init_reward_loop_workers() # J：调用 reward_loop_workers_class 类创建对象并存储到  self.reward_loop_workers 中

    @property
    def reward_loop_worker_handles(self) -> list[ActorHandle]:
        """Return worker handles for agent loop worker to compute reward score.

        Only return worker handles when reward computation can be parallelized with rollout:
        (1) rule-based reward without reward model
        (2) reward model with extra resource pool
        """
        if not self.config.reward.reward_model.enable or self.config.reward.reward_model.enable_resource_pool:
            return self.reward_loop_workers
        return None

    def _init_reward_loop_workers(self): # J：初始化 reward loop worker
        self.reward_loop_workers = [] # J：可以有多个 reward loop worker
        num_workers = self.config.reward.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers): # J：每个 reward loop worker 都分配到一个节点上
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]

            self.reward_loop_workers.append( # J：reward_loop_workers 是 reward loop worker 的列表
                self.reward_loop_workers_class.options( # J：reward_loop_workers_class 是 RewardLoopWorker 类的 Ray 远程对象
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address) # J：创建 reward loop worker 并分配到节点上
            )

    def compute_rm_score(self, data: DataProto) -> DataProto: # J：计算 reward score 并返回包含 rm_scores 张量和 reward_extra_info 字段的 DataProto 对象，注：仅每个样本的最后一个 Response token 被赋值，其余 Token 都是 0
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        chunks = data.chunk(len(self.reward_loop_workers)) # J：将数据分 num_workers 块
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk) # J：每个 reward loop worker 负责计算一个 chunk 的 reward score，并发完成计算
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist] # J：将所有 reward loop worker 的输出展平为一个列表

        # compute rm score
        scores = [item["reward_score"] for item in outputs_flat] # J：抽取 “reward_score” 字段
        rm_scores = self.reward_manager_cls.assemble_rm_scores(data, scores) # J：将每个样本的 reward score 赋值给 rm_scores 张量的最后一个 Response token
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data)) # J：len(data) 和 rm_scores.size(0) 相同

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat] # J：抽取 reward_extra_info 的信息
        reward_extra_keys = list(reward_extra_infos[0].keys())
        non_tensor_batch = {}
        for key in reward_extra_keys:
            # J：将每个 reward的 reward_extra_info 字段转换为 numpy 数组，并存储在 non_tensor_batch 中
            # J：比如在 gdpo 中，reward_extra_info 包含 accuracy_reward 和 format_reward 两个字段，这里需要把对应的奖励记录下来
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()


        return DataProto( # J：返回包含 rm_scores 张量和 reward_extra_info 字段的 DataProto 对象
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
