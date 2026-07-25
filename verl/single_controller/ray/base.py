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
import inspect
import logging
import os
import socket
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import ray
from ray.experimental.state.api import get_actor
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.protocol import DataProto, _padding_size_key
from verl.single_controller.base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.utils.device import get_device_name, is_torch_npu_available
from verl.utils.py_functional import temp_env_var

__all__ = ["Worker"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_random_string(length: int) -> str:
    import random
    import string

    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


# J：根据方法名、分发函数、收集函数、执行函数和是否阻塞，动态生成一个函数对象
# J：返回一个函数对象（已经初始化好的 Functor 子类对象），可像调用普通函数一样调用，会完成 参数分发（dispatch_fn）和 方法调用（execute_fn）和结果的收集（collect_fn）等操作
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs) # J: 调用 dispatch_fn 函数分发参数
            padding_count = kwargs.pop(_padding_size_key, 0) # J: 从 kwargs 中移除 _padding_size_key 字段值赋值 padding_count，返回其值，默认值为 0
            output = execute_fn(method_name, *args, **kwargs) # J: 调用 execute_fn 函数执行方法
            if blocking: # J: 如果是阻塞调用，等待 Ray Actor 执行完成
                output = ray.get(output) # J: 阻塞等待 Ray Actor 执行完成
            output = collect_fn(self, output) # J: 根据 collect_fn 函数收集输出
            if padding_count > 0: # J: 如果 padding_count 大于 0，说明已经填充过，需要去掉填充部分
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count] # J：生成索引列表，去掉填充部分的索引
                    output = output.select_idxs(indices) # J: 从 output 中选择索引为 indices 的元素，去掉填充部分
                elif isinstance(output, list):
                    output = output[:-padding_count] # J: 列表切片，去掉填充部分
            return output

    # use class type to pass the method_name to get a better observability
    # J：type(name, bases, dict) 是 Python 的元类构造函数，用于动态创建新的类
    # J：name 是类的名称，bases 是基类的元组，dict 是类的属性字典，用于定义类的属性和方法
    # J：这里使用 Functor 类作为基类，因为 Functor 类是一个函数对象，可以像调用普通函数一样调用，而不会触发类的初始化
    # J：外层的 () 是实例化操作符，用于创建 Functor 类的实例对象
    return type(method_name, (Functor,), {})()


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]: # J：根据节点 IP 排序 placement group
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.
    # J: 排序的目的是为了保证在分布式训练（如 FSDP）中，当从检查点恢复时，各节点的 RANK（全局进程排名）保持一致。因为 FSDPCheckpointManager 会在本地存储分片模型和优化器状态，如果节点顺序发生变化，恢复时可能出现不匹配
    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.
    # J: 通过排序，如果只有一个资源池且节点拓扑未变，那么即使整个 Ray 集群重启，多次 Ray 作业之间的 RANK 分配也会保持一致（注：如果节点拓扑发生变化，RANK 会重新分配）
    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    # J：ray.nodes() 获取当前 Ray 集群中所有节点的信息列表，每个节点是一个字典
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {} # J：placement group id 映射到节点 ip 地址
    for pg in pgs:
        # J：调用 Ray 内部 API ray._private.state.state.placement_group_table(pg.id)，获取该 Placement Group 的详细状态表（specs）
        # J：这个表是一个字典，包含了该 PG 的所有信息，例如资源束到节点的映射
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        node_id = specs["bundles_to_node_id"][0] # J：获取该 PG 第一个 bundle 对应的节点 ID（因为同一个 placement group 的所有 bundle 都在同一个节点上）
        pg_ip[pg.id] = node_ip[node_id] # J：将该 PG 的 ID 映射到该节点的 IP 地址
    return sorted(pgs, key=lambda pg: pg_ip[pg.id]) # J：根据节点 IP 升序排序 placement group


@ray.remote
def get_master_addr_port(master_port_range: Optional[list[int]] = None) -> tuple[str, str]:
    addr = ray.util.get_node_ip_address().strip("[]")

    if master_port_range is None:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
    else:
        port = master_port_range[0]
        while port < master_port_range[1]:
            try:
                with socket.socket() as s:
                    s.bind(("", port))
                    break
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
        else:
            raise RuntimeError(f"Could not find a free port in range {master_port_range}")
    return addr, str(port)


class RayResourcePool(ResourcePool): # J：Ray 资源池，用于管理 Ray 节点上的资源，包括进程数和 GPU 分配情况, Rank 信息等
    def __init__(
        self,
        process_on_nodes: Optional[list[int]] = None, # J: 列表的长度代表将使用多少个 Ray 节点，而每个元素的值代表该节点上将启动的进程数
                                                      # J：例如，[2, 1] 表示在第一个节点上运行 2 个进程，在第二个节点上运行 1 个进程
        use_gpu: bool = True,
        name_prefix: str = None,  # J: 资源池名称前缀，例如 "global_pool" 或 "teacher_pool" 等
        max_colocate_count: int = 10,
        detached=False, # J：取默认值 False
        accelerator_type: Optional[str] = None,
    ) -> None:
        super().__init__(process_on_nodes, max_colocate_count) # J: 初始化资源池基类
        self.use_gpu = use_gpu
        # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix # J：如果没有指定名称前缀，就随机生成一个6位前缀
        self.pgs = None # J: 资源池的 placement group
        self.detached = detached # J: 创建的 Placement Group 是否应该是“detached”（分离的）。如果为 True，Placement Group 的生命周期将独立于创建它的 Python 脚本，即使脚本退出，Placement Group 也会在集群中持续存在
        self.accelerator_type = accelerator_type # J: 加速器类型，暂未传入，默认 None

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        if self.pgs is not None:
            return self.pgs

        # J：生成 placement group 名称前缀，若没有传入 name，就使用默认的名称前缀和节点数生成一个
        pg_name_prefix = ( # J：举例：pg_name_prefix = "global_pool_verl_group_8_8:" 等
            name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        )
        # print(f"pg_name_prefix = {pg_name_prefix}")
        if device_name == "npu":
            device_name = "NPU"
        elif device_name == "cuda":
            device_name = "GPU"

        # J：每个 bundle 表示一个进程的资源需求，包含 CPU 核心、GPU、NPU 等资源
        bundle = {"CPU": self.max_colocate_count} # J: self.max_colocate_count 个 CPU 核心，对应每个 colocate 都有一个处理器
        if self.use_gpu:
            bundle[device_name] = 1 # J: 每个进程一个 GPU（为每个 GPU 分配一个进程）
            if self.accelerator_type is not None:
                # J: 如果指定了 accelerator_type，则还会添加一个对该类型加速器的微小需求（1e-4）。这是一种常用的技巧，用于“标记”或“提示”调度器倾向于选择具有特定加速器类型的节点，而不实际消耗大量该类资源
                bundle[self.accelerator_type] = 1e-4
        # J：为每个节点生成 placement group 方案，每个 placement group 对应一个节点，包含 process_count 个 bundle
        # J：例如，[2, 1] 表示在第一个节点上运行 2 个进程，在第二个节点上运行 1 个进程
        # J: 得到的 pg_scheme 是一个两层嵌套列表，每个元素是一个列表，列表的长度代表该节点上将启动的进程数
        # J：比如 [[{"CPU": 10, "GPU": 1}, {"CPU": 10, "GPU": 1}], [{"CPU": 10, "GPU": 1}]] 表示两个节点，第一个节点上运行 2 个进程，第二个节点上运行 1 个进程
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        lifetime = "detached" if self.detached else None # J：一般取 detached 为 False

        pgs = [
            # J：placement_group 函数用于在集群中原子性地预留一组资源（即“全部成功或全部失败”的 gang scheduling）
            # J: lifetime="detached" 表示 placement group 会在 Ray 节点重启后继续存在
            # J：为每个节点分配一个 placement group（以节点为单位申请原子资源），每个 placement group 包含 process_count 个 bundle，每个 bundle 包含 self.max_colocate_count 个 CPU 核心和 1 个 GPU
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        # pg.ready() 返回一个 Ray ObjectRef，当 Placement Group 成功创建（即所需资源在集群中成功预留）时，这个 ObjectRef 会变得 ready 状态
        ray.get([pg.ready() for pg in pgs]) # J：等待所有 placement group 都准备就绪

        self.pgs = sort_placement_group_by_node_ip(pgs) # J：根据节点 IP 升序排序 placement group
        return pgs


class SubRayResourcePool(RayResourcePool):
    def __init__(
        self,
        placement_groups: list[PlacementGroup],
        start_bundle_index: int,
        subgroup_world_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pgs = placement_groups
        self.start_bundle_index = start_bundle_index
        self.subgroup_world_size = subgroup_world_size

    @property
    def world_size(self):
        return self.subgroup_world_size


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """
    # J: dict[str, list[int]], 键是池名，值是一个列表，长度等于节点数，每个元素是该节点上的 GPU 数
    resource_pool_spec: dict[str, list[int]] # J: Key: pool_name, Value: [n_gpus_per_node, ...] e.g. {"global_pool": [3, 4, ...]}
    # J: dict[int, str], 键是 Role 的枚举值，值是资源池 ID（如 "global_pool" 或 "teacher_pool" 等）
    mapping: dict[int, str] # J: Key: role, Value: pool_name e.g. {Role.Actor: "global_pool", Role.Critic: "global_pool"}
    max_colocate_count: int = 3
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self): # J：创建 Ray 资源池
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        # J：创建资源池并申请资源，每个资源池对应一个 RayResourcePool 对象
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True, # J: 是否使用 GPU
                max_colocate_count=self.max_colocate_count,
                name_prefix=resource_pool_name, # J: 资源池名称前缀，例如 "global_pool" 或 "teacher_pool" 等
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool # J：将资源池对象存储到字典中，键是池名，值是资源池对象

        self._check_resource_available() # J：检查资源池是否足够，确保所有节点上的 GPU 数足够满足所有资源池的 GPU 数需求

    def get_resource_pool(self, role) -> RayResourcePool: # J：根据 Role 获取对应的资源池对象
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self): # J：检查资源池是否足够，确保所有节点上的 GPU 数足够满足所有资源池的 GPU 数需求
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node() # J：获取所有节点上的可用资源
        node_available_gpus = { # J：计算每个节点上的可用 GPU/NPU 数
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values()) # J：计算所有节点上的总 GPU 数
        total_required_gpus = sum( # J：计算所有资源池的总 GPU 数
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError( # J：如果总可用 GPU 数小于总所需 GPU 数，抛出异常，否则通过检查
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def extract_pg_from_exist(
    resource_pools: dict[str, RayResourcePool], src_role_names: list[str], resource_pool: RayResourcePool
) -> list:
    src_pgs = [
        pg
        for role_name, resource_pool in resource_pools.items()
        for pg in resource_pool.get_placement_groups()
        if role_name in src_role_names
    ]

    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: list[tuple[int, PlacementGroup]] = []
    searching_idx = 0
    for request_process, original_idx in sorted_process_on_nodes:
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, (
            f"requesting {request_process} processes, bundle count cannot satisfy"
        )
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    return [pg for _, pg in sorted(unsorted_pgs)]


# split a RayResourcePool or SubRayResourcePool into multiple SubRayResourcePool
def split_resource_pool(
    resource_pool: RayResourcePool | SubRayResourcePool, split_size: int | list[int]
) -> list[SubRayResourcePool]:
    """
    Split a RayResourcePool into multiple SubRayResourcePool.
    resouce_pool can also be a SubRayResourcePool (have been splited) for multiple-time spliting.

    Args:
        resource_pool (RayResourcePool | SubRayResourcePool): The resource pool to split.
        split_size (int | list[int]): The size of each split. If int, all splits will have the same size.
            If list[int], each element in the list represents the size of a split.

    Returns:
        list[SubRayResourcePool]: A list of SubRayResourcePool after splitting.
    """
    # convert split_size to list[int]
    if isinstance(split_size, int):
        assert resource_pool.world_size % split_size == 0, "split_size must be a divisor of world_size"
        num_replica = resource_pool.world_size // split_size
        split_size_list = [split_size] * num_replica
    else:
        split_size_list = split_size

    assert sum(split_size_list) == resource_pool.world_size, "split_size must sum up to world_size"

    # judge if this resource pool has been splited
    if isinstance(resource_pool, SubRayResourcePool):
        start_bundle_idx_list = np.cumsum([resource_pool.start_bundle_index] + split_size_list[:-1])
    else:
        start_bundle_idx_list = np.cumsum([0] + split_size_list[:-1])

    # ensure resource_pool.pgs has been initialized
    device = "npu" if is_torch_npu_available(check_device=False) else "cuda"
    placement_groups = resource_pool.get_placement_groups(device_name=device)
    split_resource_pools = [
        SubRayResourcePool(
            process_on_nodes=resource_pool.store,
            use_gpu=resource_pool.use_gpu,
            name_prefix=f"{resource_pool.name_prefix}_split_{split_idx}",
            max_colocate_count=resource_pool.max_colocate_count,
            placement_groups=placement_groups,
            start_bundle_index=start_bundle_idx_list[split_idx],
            subgroup_world_size=split_size_list[split_idx],
        )
        for split_idx in range(len(split_size_list))
    ]
    return split_resource_pools


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    assert rp1.use_gpu == rp2.use_gpu, "Both RayResourcePool must either use_gpu or not"
    assert rp1.max_colocate_count == rp2.max_colocate_count, "Both RayResourcePool must has the same max_colocate_count"
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, "Both RayResourcePool must has the same n_gpus_per_node"
    assert rp1.detached == rp2.detached, "Detached ResourcePool cannot be merged with non-detached ResourcePool"

    new_store = rp1.store + rp2.store

    merged = type(rp1)(
        new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}", rp1.max_colocate_count, rp1.detached
    )
    merged.pgs = rp1.get_placement_groups(device_name=get_device_name()) + rp2.get_placement_groups(
        device_name=get_device_name()
    )

    return merged


class RayClassWithInitArgs(ClassWithInitArgs): # J: 类包装器，用于延迟实例化类，将类的构造函数参数存储起来
    """A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        """Set additional resource requirements for the actor.

        Args:
            additional_resource: Dictionary specifying additional resource requirements
        """
        self._additional_resource = additional_resource

    def update_options(self, options: dict):
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__( # J: 用于创建 Ray Actor 的方法
        self,
        placement_group,
        placement_group_bundle_idx,
        use_gpu: bool = True,
        num_gpus=1,
        sharing_with=None,
        device_name="cuda",
    ) -> Any:
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            use_gpu: Whether to use GPU resources
            num_gpus: Number of GPUs to allocate
            sharing_with: Actor to share resources with
            device_name: Device for training

        Returns:
            A Ray actor handle with the configured options
        """
        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args, cuda_visible_devices=visible_devices, **self.kwargs)

        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx
            )
        }
        options.update(self._options)

        if use_gpu and device_name == "cuda":
            options["num_gpus"] = num_gpus
        if use_gpu and device_name == "npu":
            options["resources"] = {"NPU": num_gpus}

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


class RayWorkerGroup(WorkerGroup): # J: Ray 工作进程组类，用于管理 Ray 工作进程组，每个资源池创建一个 RayWorkerGroup 对象
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.
    """

    def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached=False,
        worker_names=None,
        worker_handles: list[ray.actor.ActorHandle] = None,
        ray_wait_register_center_timeout: int = 300,
        **kwargs,
    ) -> None:
        """Initialize a RayWorkerGroup.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            name_prefix: Prefix for worker names
            detached: Whether workers should be detached
            worker_names: Names of existing workers to attach to
            ray_wait_register_center_timeout: Timeout for waiting on register center
            **kwargs: Additional keyword arguments
        """
        self._master_addr = kwargs.pop("master_addr", None)
        self._master_port = kwargs.pop("master_port", None)
        self.use_gpu = kwargs.pop("use_gpu", resource_pool.use_gpu if resource_pool is not None else True)
        self._ray_master_port_range = kwargs.pop("master_port_range", None)
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init # J: RayClassWithInitArgs 类对象，用于初始化 Ray Actor 对象
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix # J: 工作进程组名称前缀，默认用于唯一标识工作进程组
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout # J: 等待注册中心超时时间，默认 300 秒
        # Whether the WorkerGroup is a Colocate WorkerGroup created by FusedWorker.
        # J: 如果 ray_cls_with_init 为 None，则默认认为是不使用 FusedWorker 类，否则根据 ray_cls_with_init.fused_worker_used 判断是否使用了 FusedWorker 类
        self.fused_worker_used = False if ray_cls_with_init is None else ray_cls_with_init.fused_worker_used
        # if a WorkerGroup is spawned from Colocate WorkerGroup, this indicates which sub-class is binded to
        # this WorkerGroup.
        self.sub_cls_name = ""
        self.device_name = kwargs.get("device_name", "cuda")
        self.profile_steps = kwargs.get("profile_steps", None)
        self.worker_nsight_options = kwargs.get("worker_nsight_options", None)
        self.customized_worker_env = kwargs.get("worker_env", {})
        if self.worker_nsight_options is not None and self.worker_nsight_options["capture-range-end"] is None:
            self.worker_nsight_options["capture-range-end"] = f"repeat-shutdown:{6 * len(self.profile_steps)}"

        if worker_names is not None and (not self.fused_worker_used):
            assert self._is_init_with_detached_workers # J: 如果 worker_names 不为 None，且不使用 FusedWorker 类，则必须是 ResourcePool=None 的情况
            self._worker_names = worker_names # J: 工作进程组名称列表，用于唯一标识工作进程组

        if self._is_init_with_detached_workers: # J：_is_init_with_detached_workers=True 表示 resource_pool 为 None
            # J: ResourcePool=None 时，需要通过 worker_names 或 worker_handles（二选一） 来获取工作进程组
            # J：ResourcePool=None 时，workers 已经初始化完成了，这里只是获取已经存在的 workers
            self._init_with_detached_workers(worker_names=worker_names, worker_handles=worker_handles)
        elif isinstance(resource_pool, SubRayResourcePool):
            self._init_with_subresource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached, # J: 是否创建离散的工作进程，默认 False
                worker_env=self.customized_worker_env,
            )
        else:
            self._init_with_resource_pool(
                resource_pool=resource_pool, # J: 指向资源池的引用，包含 placement group 实例，和 world_size 等信息
                ray_cls_with_init=ray_cls_with_init, # J: RayClassWithInitArgs 类对象，用于初始化 Ray Actor 对象
                bin_pack=bin_pack, # J: 是否使用严格 bin packing 资源分配，默认 True
                detached=detached, # J: 是否创建离散的工作进程，默认 False
                worker_env=self.customized_worker_env, # J: 自定义的工作进程环境变量（worker_env 字段，默认空字典
            )

        if ray_cls_with_init is not None:
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator) # J: 绑定 Ray Actor 类的方法到 self（RayWorkerGroup 对象） 上，同时返回绑定的方法名称列表

        self.wg_dict = None
        self.method_names = []

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        """Check if a worker actor is still alive.

        Args:
            worker: Ray actor handle to check

        Returns:
            bool: True if the worker is alive, False otherwise
        """
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_detached_workers(self, worker_names, worker_handles):
        # ray.get_actor holds a weak reference to the actor, which causes actors garbage collected unexpectedly
        # if we only hold spawn RayWorkerGroup. By passing actor handle explicitly, spawn RayWorkerGroup have
        # strong reference to these actors.
        # https://github.com/ray-project/ray/pull/45699
        # J: 如果 worker_handles 为 None，则借用 worker_names 获取工作进程句柄，此时要求注册时的有名字的 Actor
        # J: 如果 worker_handles 不为 None，则直接使用 worker_handles 中的工作进程句柄
        workers = worker_handles if worker_handles else [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(workers) # J: 工作进程组大小，即工作进程数量

    def _get_master_addr_port(self, pg, bundle_index=0, master_port_range=None):
        """Get master addr and port for this worker group"""
        if self._master_addr is None and self._master_port is None:
            self._master_addr, self._master_port = ray.get(
                get_master_addr_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=bundle_index
                    ),
                ).remote(master_port_range=master_port_range)
            )
        elif self._master_addr is not None and self._master_port is not None:
            logger.debug(f"{self._master_addr=} {self._master_port=}")
        else:
            raise ValueError(
                "Both 'master_addr' and 'master_port' must be provided if you intend to manually specify them, "
                "or neither should be provided to use Ray's default assignment."
            )

    def _init_with_resource_pool( # J: 初始化工作进程组，从资源池创建新工作进程，每个 rank 对应一个工作进程，都一一完成 Worker 初始化
        self,
        resource_pool, # J: 指向资源池的引用，包含 placement group 实例，和 world_size 等信息
        ray_cls_with_init, # J: RayClassWithInitArgs 类对象，用于初始化 Ray Actor 对象
        bin_pack, # J: 是否使用严格 bin packing 资源分配，默认 True
        detached, # J: 是否创建离散的工作进程，默认 False
        worker_env=None, # J: 自定义的工作进程环境变量（worker_env 字段，默认空字典）
    ):
        """Initialize the worker group by creating new workers from a resource pool.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        self.resource_pool = resource_pool
        strategy = "PACK"
        if bin_pack: # J: 如果使用严格 bin packing，则使用 STRICT_PACK 策略
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "
            if pg_idx == 0:
                self._get_master_addr_port(pg, bundle_index=0, master_port_range=self._ray_master_port_range)

            for local_rank in range(local_world_size): # J: 遍历本地工作进程组中的每个工作进程
                rank += 1
                self._create_worker( # J: 根据配置创建一个工作进程，初始化 Worker 实例
                    rank=rank, # J: 每个工作进程都有一个唯一的 rank，用于标识和管理
                    pg_idx=pg_idx, # J: 每个工作进程所属的 placement group 索引，用于确定工作进程在资源池中的位置
                    pg=pg, # J: 每个工作进程所属的 placement group 实例，用于指定工作进程在资源池中的位置
                    local_rank=local_rank, # J: 每个工作进程在本地工作进程组中的 rank，用于确定工作进程在本地资源池中的位置
                    resource_pool=resource_pool, # J: 指向资源池的引用，用于获取资源池的 world_size 等信息
                    ray_cls_with_init=ray_cls_with_init, # J: RayClassWithInitArgs 类实例，包含初始化参数和待初始化的 Ray Actor 类
                    worker_env=worker_env, # J: 自定义的工作进程环境变量（worker_env 字段，默认空字典）
                    detached=detached, # J: 是否创建离散的工作进程，默认 False
                )

    def _init_with_subresource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached, worker_env=None):
        """Initialize the worker group by creating new workers from a resource pool or sub resource pool.
        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size # J: 工作进程组大小，即工作进程数量

        rank = -1
        local_world_size = resource_pool.store[0]
        self._get_master_addr_port(
            pgs[resource_pool.start_bundle_index // local_world_size],
            bundle_index=resource_pool.start_bundle_index % local_world_size,
            master_port_range=self._ray_master_port_range,
        )
        for curr_rank in range(resource_pool.start_bundle_index, resource_pool.start_bundle_index + world_size):
            pg_idx = curr_rank // local_world_size
            pg = pgs[pg_idx]
            local_rank = curr_rank % local_world_size
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "

            rank += 1
            self._create_worker(
                rank=rank,
                pg_idx=pg_idx,
                pg=pg,
                local_rank=local_rank,
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                worker_env=worker_env,
                detached=detached,
            )

    # J: 创建一个工作进程
    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached): # J: 创建一个工作进程
        world_size = resource_pool.world_size
        use_gpu = resource_pool.use_gpu
        if self.use_gpu and not use_gpu:
            raise ValueError("use_gpu is True but resource_pool.use_gpu is False")
        local_world_size = resource_pool.store[0]
        num_gpus = 1 / resource_pool.max_colocate_count

        # we pass in environment variable at option so that Worker can use environment variable to set
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "WG_PREFIX": self.name_prefix,
            "WG_BACKEND": "ray",
            "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
            "MASTER_ADDR": self._master_addr,
            "MASTER_PORT": self._master_port,
        }
        if worker_env is not None:
            logging.debug(f"Appending ray class env, origin: {env_vars}, customized env: {worker_env}")
            conflict_env_vars = set(env_vars.keys()) & set(worker_env.keys())
            if len(conflict_env_vars) > 0:
                logging.error(
                    f"User customized env vars conflict with system env: {conflict_env_vars} "
                    f"Overriding may cause unexpected behavior."
                )
                raise ValueError(f"Cannot override protected system env: {conflict_env_vars}")
            env_vars.update(worker_env)
        import re

        # J：cia 是 class with init args 的简称
        cia_name = type(ray_cls_with_init.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
        cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
        name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

        if self.profile_steps and self.device_name == "cuda":
            ray_cls_with_init.update_options(
                {
                    "runtime_env": {
                        "env_vars": env_vars,
                        "nsight": self.worker_nsight_options,
                    },
                    "name": name,
                }
            )
        else:
            ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})

        if detached:
            ray_cls_with_init.update_options({"lifetime": "detached"})

        # create a worker
        worker = ray_cls_with_init( # J: 创建一个工作进程，初始化 Ray Actor 类实例（真正的 Worker 实例创建）
            placement_group=pg, # J: 指定工作进程所属的 placement group
            placement_group_bundle_idx=local_rank, # J: 使用 local_rank 来指定工作进程在 placement group 中的 bundle 索引
            use_gpu=self.use_gpu,
            num_gpus=num_gpus,
            device_name=self.device_name,
        )
        # J: _workers 和 _worker_names 是对应关系，每个工作进程都有一个名称
        self._workers.append(worker) # J: 将新创建的工作进程添加到 self._workers 列表中
        self._worker_names.append(name) # J: 将新创建的工作进程的名称添加到 self._worker_names 列表中

    @property
    def worker_names(self):
        return self._worker_names

    @classmethod
    def from_detached(
        cls,
        name_prefix=None,
        worker_names=None,
        worker_handles=None,
        ray_cls_with_init=None,
        **kwargs,
    ):
        """Create a worker group from existing detached workers.

        Args:
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            ray_cls_with_init: Class with initialization arguments for workers

        Returns:
            A new RayWorkerGroup instance
        """
        worker_group = cls(
            resource_pool=None,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix=name_prefix,
            worker_names=worker_names,
            worker_handles=worker_handles,
            **kwargs,
        )
        return worker_group

    def spawn(self, prefix_set):
        """Spawn to a dictionary of worker groups, each with a subset of method with prefix.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        if self.fused_worker_used:
            return self.spawn_fused(prefix_set)

        def _rebind_actor_methods(worker_group, actor_name):
            prefix: str = actor_name + "_"
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_worker_group = self.from_detached(
                name_prefix=self.name_prefix,
                worker_names=self._worker_names,
                worker_handles=self._workers,
                ray_cls_with_init=self.ray_cls_with_init,
                profile_steps=self.profile_steps,
                worker_nsight_options=self.worker_nsight_options,
            )

            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def spawn_fused(self, prefix_set):
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        wg_dict = dict()
        for key in prefix_set:
            new_wg = deepcopy(self)
            new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
            new_wg.sub_cls_name = key
            wg_dict[key] = new_wg
        return wg_dict

    def fuse(self, prefix_set):
        """Fuse multiple worker groups into the current worker group.

        Args:
            prefix_set: Set of prefixes to fuse into the worker group
        """
        if self.wg_dict is None:
            self.wg_dict = self.spawn(prefix_set)
        for role_name, role_wg in self.wg_dict.items():
            setattr(self, role_name, role_wg)
        self.method_names = self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        if self.fused_worker_used and method_name not in self.method_names:
            remote_call = getattr(worker, self.fused_worker_execute_fn_name)
            return remote_call.remote(f"{self.sub_cls_name}_fwmn_{method_name}", *args, **kwargs)
        # fused worker not used
        remote_call = getattr(worker, method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method execution
        """
        return ray.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self._execute_remote_single_worker(self._workers[0], method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """Alias for execute_rank_zero_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        """Alias for execute_all_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all workers
        """
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(
                        self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs)
                    )
                return result

        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def workers(self):
        return self._workers

    @property
    def world_size(self):
        return self._world_size


"""
Utilities that enables creating workers inside the same ray.Actor,
with code written in separate ray.Actors.
"""


# deprecated, switching to FusedWorker
# J：绑定 user_defined_cls 类方法到 cls 类
def _bind_workers_method_to_parent(cls, key, user_defined_cls): # J：很好的做法，可以防止大面积重新定义方法
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """

    for method_name in dir(user_defined_cls): # J：dir(user_defined_cls) 返回 user_defined_cls 类的所有属性名（包括方法）
        try:
            method = getattr(user_defined_cls, method_name) # J：获取方法对象
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable" # J：检查方法是否可调用
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR): # J：检查方法是否有 MAGIC_ATTR 属性（魔法属性由装饰器 register 添加），避免访问 __init__ 等内建方法

            def generate_function(name, key=key): # J：key 是角色名
                # J：生成包装函数，用于将调用分发到实际的 Worker
                # J：封装后，从外部调用者的“语法”和“参数传递”角度看，完全一样；但从“调用目标对象”的本质上讲，不一样
                # J：封装前：直接调用 worker.some_method()
                # J：封装后：调用 proxy_instance.some_method()，实际执行的是 proxy_instance.worker_dict[key].some_method()
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs) # J：将调用分发到实际的 Worker，返回得到的是函数的调用结果

                async def async_func(self, *args, **kwargs): # J：协程函数，用于异步调用实际的 Worker 方法
                    # dispatch to the actual worker
                    return await getattr(self.worker_dict[key], name)(*args, **kwargs)

                # J：根据方法是否异步(协程，async def)，选择不同的包装函数，保证与原始方法的同步/异步调用签名一致
                # J：注，noqa: B023 是一个代码静态检查（Linter）的忽略指令，专门用于压制特定类型的代码风格或潜在错误警告
                wrapper = async_func if inspect.iscoroutinefunction(method) else func  # noqa: B023

                return wrapper

            func = generate_function(method_name) # J：生成包装函数
            # pass MAGIC_ATTR for outer worker group
            attrs = getattr(method, MAGIC_ATTR) # J：获取方法的 MAGIC_ATTR 属性（魔法属性由装饰器 register 添加）
            setattr(func, MAGIC_ATTR, attrs) # J：将 MAGIC_ATTR 属性传递给包装函数
            try:
                # bind direct rollout method to class without prefix
                # J：注册 Worker 时会添加类似 @register(dispatch_mode=Dispatch.ONE_TO_ALL) 的装饰器注解，用于指定方法的分发模式
                # J：如果方法的分发模式为 Dispatch.DIRECT_ROLLOUT_METHOD，且方法名中包含 "rollout"，则直接将方法绑定到 cls 类，方法名不包含角色名
                # J：理解：这里相当于默认让 workerGroup 调用方法名为 Rollout 角色的方法名
                # J: 目前似乎没有分发模式为 DIRECT_ROLLOUT_METHOD 的方法
                if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
                    assert not hasattr(cls, method_name), ( # J：检查 cls 类是否有该方法名，避免冲突，理论上此时不该有该方法
                        f"conflict direct rollout method {method_name} with role {key}"
                    )
                    setattr(cls, method_name, func) # J：将包装函数绑定到 cls 类，方法名不包含角色名
                    print(f"bind role {key} method {method_name} to class {cls}") # J：打印绑定信息
                else:
                    method_name_with_prefix = key + "_" + method_name # J：方法名前加上 角色名 为前缀
                    setattr(cls, method_name_with_prefix, func) # J：将包装函数绑定到 cls 类，方法名前缀为角色名
                    # J：忘记打印绑定信息了
            except Exception as e:
                raise ValueError(f"Fail to set method_name {method_name}") from e


def _unwrap_ray_remote(cls): # J：解除 @ray.remote 装饰
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__ # J：cls 是 @ray.remote 装饰后类，cls.__ray_actor_class__ 是没有经过 @ray.remote 装饰的类
    return cls


# J：返回角色的 Ray Actor 类包装器对象的基类（MegatronWorker 或 Worker（FSDP 工作进程））
def _determine_fsdp_megatron_base_class(mros: list): # J：根据 MRO 中的基类名称确定角色的 Ray Actor 类包装器对象的基类是否为 MegatronWorker 或 Worker（FSDP）
    """
    - megatron: base class should be MegatronWorker
    - fsdp: base class should be Worker
    """
    for cls in mros[0]:
        if cls.__name__ == "MegatronWorker": # J：Megatron 工作进程，MegatronWorker 是 Worker 的子类
            return cls
        if cls.__name__ == "Worker": # J：FSDP 工作进程
            return cls
    raise ValueError(f"Cannot determine base class for {mros}")


# deprecated, switching to FusedWorker
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]): # J：返回封装了 WorkerDict 类的 RayClassWithInitArgs 类对象
    # J：class_dict 是一个 dict 对象（键 是[str(角色)]，值是 RayClassWithInitArgs 对象）
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _determine_fsdp_megatron_base_class( # J：返回角色的 Ray Actor 类对象的基类（MegatronWorker 或 Worker（FSDP 工作进程））
        # J：cls.cls 将 RayClassWithInitArgs 提出为待实例化的具体 Actor 类，详情见 RayClassWithInitArgs 类的 父类定义
        [cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()] # J：获取每个角色的 Ray Actor 类对象的 MRO（父类到子类的整个继承关系）
    )
    # J：确保确定的基类是 Worker，MegatronWorker 是 Worker 的子类
    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"
    print(f"colocated worker base class {worker_cls}") # J：打印确定的基类

    for key, cls in class_dict.items(): # J：遍历每个角色的 RayClassWithInitArgs 类对象
        cls_dict[key] = cls.cls # J：cls.cls 将 RayClassWithInitArgs 提出为待实例化的具体 Actor 类，key 是角色名
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys() # J：确保 cls_dict 和 init_args_dict 中的键名一致

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls): # J：创建一个类，继承自 worker_cls（MegatronWorker 或 Worker）
        def __init__(self):
            super().__init__()
            self.worker_dict = {} # J：创建一个空字典，用于存储每个角色的 Ray Actor 类实例
            for key, user_defined_cls in cls_dict.items(): # J：遍历每个角色的 Ray Actor 类对象
                user_defined_cls = _unwrap_ray_remote(user_defined_cls) # J：解除 @ray.remote 装饰，不再是 Ray Actor 类，是普通类
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker>
                # when DISABLE_WORKER_INIT == 1 it will return immediately
                with temp_env_var("DISABLE_WORKER_INIT", "1"): # J：跳过这个等待 Head 服务就绪的初始化，立即返回普通类的实例化
                    self.worker_dict[key] = user_defined_cls( # J：普通类的实例化，注意 DISABLE_WORKER_INIT=1 使得不用等待 Head 服务就绪
                        *init_args_dict[key].get("args", ()), **init_args_dict[key].get("kwargs", {})
                    )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items(): # J：cls_dict 是一个 dict 对象（键 是[str(角色)]，值是 Ray Actor 类 类）
        user_defined_cls = _unwrap_ray_remote(user_defined_cls) # J：解除 @ray.remote 装饰，不再是 Ray Actor 类，是普通类
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls) # J：绑定 user_defined_cls 类方法到 cls 类，key 是角色名

    remote_cls = ray.remote(WorkerDict) # J：将 WorkerDict 类注册为 Ray Actor 类
    remote_cls = RayClassWithInitArgs(cls=remote_cls) # J：将 WorkerDict 类注册为 RayClassWithInitArgs 类，后续延迟实例化
    return remote_cls # J：返回 封装了 WorkerDict 类的 RayClassWithInitArgs 类对象


FusedWorkerCLSName = "FusedWorker"


def create_colocated_worker_raw_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function returns a FusedWorker class.

    `FusedWorker.{class_name}` -> FusedClass
        Use `class_name` as a param to directly access the underlying class.

    `FusedWorker._fuw_execute("{class_name}_fwmn_{method_name}", *args, **kwargs)`
        First param must be "{class_name}_fwmn_{method_name}" in order to access `method_name`
        of underlying class `{class_name}`.

    `FusedWorker.fused_worker_dict` -> {"class_name": FusedClass}
        Stores all underlying classes.

    `FusedClass.fused_worker_dict` -> {"class_name": FusedClass}
        The same as `FusedWorker.fused_worker_dict`, enables underlying class to access other
        underlying classes.
    """
    raw_cls_dict = {cls_name: _unwrap_ray_remote(cia.cls) for cls_name, cia in class_dict.items()}
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    cls_names = list(class_dict.keys())

    # FusedWorker_Actor_Critic
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    class FusedWorker(Worker): # J: FusedWorker 类，继承自 Worker 类
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cls_names = cls_names
            self.raw_cls_dict = raw_cls_dict
            self.init_args_dict = init_args_dict
            self.init_kwargs_dict = init_kwargs_dict

            for cls_name, udc, ud_args, ud_kwargs in zip(
                self.cls_names,
                self.raw_cls_dict.values(),
                self.init_args_dict.values(),
                self.init_kwargs_dict.values(),
                strict=True,
            ):
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    udc._get_ray_actor_cls_name = lambda x, name_renamed=class_name_renamed: name_renamed
                    udc._get_ray_method_prefix = lambda x, name_prefixed=cls_name: f"{name_prefixed}_"
                    # cls_name = "actor", "critic", udc = ActorWorker, CriticWorker
                    self.fused_worker_dict[cls_name] = udc(*ud_args, **ud_kwargs)
                    setattr(self, cls_name, self.fused_worker_dict[cls_name])

            # injecting fused_worker to each sub worker so they can be aware of existence of each other
            for _, worker in self.fused_worker_dict.items():
                setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

        def _fuw_execute(self, method_name: str, *args, **kwargs):
            # for fused_worker, method_name is in a form of "{cls_name}_fwmn_{method_name}"
            # where fwmn stands "fused worker method name"
            names = method_name.split("_fwmn_")
            cls_name = names[0]
            method_name = names[1]

            assert cls_name in self.fused_worker_dict, (
                f"calling {cls_name}'s {method_name}, but {cls_name} not in fused_worker_dict"
            )
            udc_method = getattr(self.fused_worker_dict[cls_name], method_name)
            return udc_method(*args, **kwargs)

    renamed_fused_worker_cls = type(class_name_renamed, (FusedWorker,), {})
    renamed_fused_worker_cls.is_fused_worker = True
    renamed_fused_worker_cls.raw_cls_dict = raw_cls_dict

    return renamed_fused_worker_cls


def create_colocated_worker_cls_fused(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function returns a RayClassWithInitArgs instance of FusedWorker, which is an replacement
    of `create_colocated_worker_cls`. WorkerGroup constructed using this class will be a colocated
    WorkerGroup, which will be referenced as `ColocateWorkerGroup` below.

    `ColocateWorkerGroup.spawn(prefix_set)`
        returns a dict of WorkerGroup {"class_name": WorkerGroup}, WorkerGroup in this dict will
        have methods of underlying class `class_name` attached.

    `ColocateWorkerGroup.fuse(prefix_set)`
        After executing this function, `ColocateWorkerGroup.{class_name}` will return WorkerGroup
        with methods of underlying class `class_name` attached.
    """
    raw_colocated_worker_cls = create_colocated_worker_raw_cls(class_dict) # J: 这里面会创建 FusedWorker 类

    remote_cls = ray.remote(raw_colocated_worker_cls)
    cia = RayClassWithInitArgs(cls=remote_cls)
    cia.fused_worker_used = True # J: 标记为使用了 FusedWorker 类

    return cia
