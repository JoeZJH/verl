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
the class of WorkerGroup
"""

import logging
import signal
import threading
import time
from typing import Any, Callable

from .decorator import MAGIC_ATTR, Dispatch, get_predefined_dispatch_fn, get_predefined_execute_fn


class ResourcePool: # J: 资源池基类，用于管理多个节点上的资源，包括进程数和 GPU 分配情况, Rank 信息等
    """
    Manages a pool of resources across multiple nodes, tracking process counts and GPU allocations.
    The class provides methods to calculate world size, local world sizes, and local ranks
    across all nodes in the pool.
    """

    def __init__(self, process_on_nodes=None, max_colocate_count: int = 10, n_gpus_per_node=8) -> None:
        """Initialize the ResourcePool with node processes and GPU configuration.

        Args:
            process_on_nodes (List[int], optional): List of process counts per node. Defaults to empty list.
            max_colocate_count (int, optional): Maximum number of processes that can be colocated. Defaults to 10.
            n_gpus_per_node (int, optional): Number of GPUs available per node. Defaults to 8.
        """
        if process_on_nodes is None:
            process_on_nodes = []
        self._store = process_on_nodes # J：列表的长度代表将使用多少个 Ray 节点，而每个元素的值代表该节点上将启动的进程数
        self.max_colocate_count = max_colocate_count # J: 最大可共置的进程数，默认 10
        self.n_gpus_per_node = n_gpus_per_node  # this is left for future huawei GPU that contains 16 GPUs per node

    def add_node(self, process_count):
        self._store.append(process_count) # J: 添加一个节点的进程数，只需要在末尾添加该节点的进程数即可

    @property
    def world_size(self): # J: 所有节点上进程数的总和即为 world size
        """Total number of processes across all nodes in the pool."""
        return sum(self._store) # J: 返回所有节点上进程数的总和即为 world size

    def __call__(self) -> Any: # J: 返回所有节点上进程数的列表
        return self._store

    @property
    def store(self): # J: 返回所有节点上进程数的列表
        return self._store

    def local_world_size_list(self) -> list[int]: # J: 展平并返回所有节点上进程数的列表（列表长度为 world size），每个元素为该节点上进程数
        """Returns a flat list where each process has its local world size."""
        nested_local_world_size_list = [ 
            # J：两层列表，每个节点一个列表（根据每个节点上进程数，生成一个列表，每个元素都为该节点上进程总数）
            # J：举例：如果 self._store = [2, 3, 4]，则 nested_local_world_size_list = [[2, 2], [3, 3, 3], [4, 4, 4, 4]]
            [local_world_size for _ in range(local_world_size)] for local_world_size in self._store
        ]
        # J：将两层列表展开为一维列表
        # J：举例：如果 nested_local_world_size_list = [[2, 2], [3, 3, 3], [4, 4, 4, 4]]，则返回 [2, 2, 3, 3, 3, 4, 4, 4, 4]
        return [item for row in nested_local_world_size_list for item in row]

    def local_rank_list(self) -> list[int]: # J: 展平并返回所有节点上进程数的列表（列表长度为 world size），每个元素为该节点上进程的本地 rank
        """Returns a flat list of local ranks for all processes across all nodes."""
        # J：举例：如果 self._store = [2, 3, 4]，则 nested_local_rank_list = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
        nested_local_rank_list = [[i for i in range(local_world_size)] for local_world_size in self._store]
        # J：将两层列表展开为一维列表
        # J：举例：如果 nested_local_rank_list = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]，则返回 [0, 1, 0, 1, 2, 0, 1, 2, 3]
        return [item for row in nested_local_rank_list for item in row]


class ClassWithInitArgs: # J: 类包装器，用于延迟实例化类，将类的构造函数参数存储起来
    """
    Wrapper class that stores constructor arguments for deferred instantiation.
    This class is particularly useful for remote class instantiation where
    the actual construction needs to happen at a different time or location.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        """Initialize the ClassWithInitArgs instance.

        Args:
            cls: The class to be instantiated later
            *args: Positional arguments for the class constructor
            **kwargs: Keyword arguments for the class constructor
        """
        self.cls = cls # J：存储后续要实例化的类
        self.args = args # J：存储要实例化的类的构造函数的位置参数
        self.kwargs = kwargs # J：存储要实例化的类的构造函数的关键字参数  

        self.fused_worker_used = False # J: 是否使用了 FusedWorker 类，默认 False

    def __call__(self) -> Any:
        """Instantiate the stored class with the stored arguments."""
        return self.cls(*self.args, **self.kwargs)


def check_workers_alive(workers: list, is_alive: Callable, gap_time: float = 1) -> None:
    """Continuously monitors worker processes and raises SIGABRT if any worker dies.

    Args:
        workers (List):
            List of worker objects to monitor
        is_alive (Callable):
            Function to check if a worker is alive
        gap_time (float):
            Time interval between checks
    """
    import time

    while True:
        for worker in workers:
            if not is_alive(worker):
                logging.warning(f"worker {worker} is not alive sending signal to main thread")
                signal.raise_signal(signal.SIGABRT)
        time.sleep(gap_time)


class WorkerGroup: # J: 工作进程组类，用于管理工作进程组，每个资源池创建一个 WorkerGroup 对象
    """
    Base class for managing a group of workers in a distributed system.
    The class provides methods for worker management, aliveness checking, and method binding.
    """

    fused_worker_execute_fn_name = "_fuw_execute"

    def __init__(self, resource_pool: ResourcePool, **kwargs) -> None:
        # J：当 resource_pool 为 None 时， WorkerGroup 没有资源池可用，无法自己创建 workers；从后文看，这种情况下 workers 已经初始化完成了，只需要获取已经存在的 workers
        # J：ResourcePool 描述了要在多少个节点上启动多少个进程；当 resource_pool 不为 None 时， WorkerGroup 可以用这些资源来 创建并绑定新的 workers（每个 worker 都分配一个资源）
        self._is_init_with_detached_workers = resource_pool is None # J：如果 resource_pool 为 None，则认为是初始化时没有绑定 workers，否则认为是初始化时绑定了 workers？

        self.fused_worker_used = False

        if resource_pool is not None:
            # handle the case when WorkGroup is attached to an existing one
            self._process_dispatch_config = resource_pool()
        else:
            self._process_dispatch_config = None

        self._workers = []
        self._worker_names = []

        self._dispatch_info = {}
        self._collect_info = {}

        self._master_addr = None
        self._master_port = None

        self._checker_thread: threading.Thread = None

    def _is_worker_alive(self, worker):
        """Check if a worker is alive. Must be implemented by derived classes."""
        raise NotImplementedError("WorkerGroup._is_worker_alive called, should be implemented in derived class.")

    def _block_until_all_workers_alive(self) -> None:
        """Blocks until all workers in the group are alive."""
        while True:
            all_state = [self._is_worker_alive(worker) for worker in self._workers]
            if False in all_state:
                time.sleep(1)
            else:
                break

    def start_worker_aliveness_check(self, every_n_seconds=1) -> None:
        """Starts a background thread to monitor worker aliveness.

        Args:
            every_n_seconds (int): Interval between aliveness checks
        """
        # before starting checking worker aliveness, make sure all workers are already alive
        self._block_until_all_workers_alive()

        self._checker_thread = threading.Thread(
            target=check_workers_alive, args=(self._workers, self._is_worker_alive, every_n_seconds)
        )
        self._checker_thread.start()

    @property
    def world_size(self):
        """Number of workers in the group."""
        return len(self._workers)

    def _bind_worker_method(self, user_defined_cls, func_generator):
        """Binds worker methods to the WorkerGroup based on registered attributes.

        Args:
            user_defined_cls (type): The class containing methods to bind
            func_generator (Callable): Function that generates the bound method

        Returns:
            List[str]: List of method names that were successfully bound
        """
        method_names = []
        for method_name in dir(user_defined_cls): # J：遍历用户定义类的所有方法名
            try:
                method = getattr(user_defined_cls, method_name)
                assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                # if it is a property, it will fail because Class doesn't have instance property
                continue

            if hasattr(method, MAGIC_ATTR): # J：仅处理被注册的方法，即有 MAGIC_ATTR 属性的方法，忽略 __init__ 等自建方法
                # this method is decorated by register
                attribute = getattr(method, MAGIC_ATTR) # J：获取方法的注册属性字典
                assert isinstance(attribute, dict), f"attribute must be a dictionary. Got {type(attribute)}"
                assert "dispatch_mode" in attribute, "attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"] # J：获取分发模式
                execute_mode = attribute["execute_mode"] # J：获取执行模式
                blocking = attribute["blocking"] # J：获取是否阻塞

                # get dispatch fn
                if isinstance(dispatch_mode, Dispatch):
                    # get default dispatch fn
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode) # J：根据预定义的分发模式获取分发函数字典
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else: # J：如果分发模式不是 Dispatch 类型，即自定义分发模式
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode # J：自定义分发模式必须包含 dispatch_fn 键
                    assert "collect_fn" in dispatch_mode # J：自定义分发模式必须包含 collect_fn 键
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                # get execute_fn_name
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]

                # get execute_fn from string
                try:
                    execute_fn = getattr(self, wg_execute_fn_name) # J：根据字符串获取执行函数对象
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception:
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    raise

                # bind a new method to the RayWorkerGroup
                func = func_generator( # J：根据方法名、分发函数、收集函数、执行函数和是否阻塞，动态生成一个函数对象
                                       # J：这个函数对象可像调用普通函数一样调用，会完成 参数分发（dispatch_fn）和 方法调用（execute_fn）和结果的收集（collect_fn）等操作
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )

                try:
                    setattr(self, method_name, func) # J：将动态生成的函数对象绑定到当前实例的属性上
                                                     # J：这样就可以在 RayWorkerGroup 实例上直接调用这个方法了
                    method_names.append(method_name) # J：将方法名添加到方法名列表中
                except Exception as e:
                    raise ValueError(f"Fail to set method_name {method_name}") from e

        return method_names # J：返回成功绑定的方法名列表
