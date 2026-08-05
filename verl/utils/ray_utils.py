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
Contains commonly used utilities for ray
"""

import asyncio
import concurrent.futures
import functools
import inspect
import os
from typing import Any, Optional

import ray


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def parallel_put(data_list: list[Any], max_workers: Optional[int] = None):
    """
    Puts a list of data into the Ray object store in parallel using a thread pool.

    Args:
        data_list (List[Any]): A list of Python objects to be put into the Ray object store.
        max_workers (int, optional): The maximum number of worker threads to use.
                                     Defaults to min(len(data_list), 16).

    Returns:
        List[ray.ObjectRef]: A list of Ray object references corresponding to the input data_list,
                             maintaining the original order.
    """
    assert len(data_list) > 0, "data_list must not be empty"

    def put_data(index, data):
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())

        # reorder based on index
        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref

    return output


def get_event_loop(): # J：获取当前线程的 event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop

# J：auto_await 装饰器，处理下面三种方式
# # 0. Case 0：直接返回结果（func 是普通函数）
# # 1. Case 1：asyncio.run(coro)，直到协程完成，返回结果（func 是异步函数（协程），调用方是同步代码（脚本顶层 / 普通 def）无 event loop）
# # 2. Case 2：返回 coroutine()，调用方会 await（func 是异步函数（协程），调用方是 async 函数，有 event loop，会 await）
# # 3. Case 3：开线程跑 asyncio.run(coro) 并等待结果返回（func 是异步函数（协程），调用方是同步函数，但被某个 async 栈间接调到，有 event loop 但不能用 await）
def auto_await(func): # J：verl 自定义的装饰器，用于自动处理异步函数的调用方式
    """Auto await a coroutine function.

    Handles three cases:
    1. When the decorated function is called with await: returns the coroutine
       so the caller can await it.
    2. When called directly and there is no running event loop: runs the
       coroutine with asyncio.run() and returns the result.
    3. When called directly and the event loop is already running: runs the
       coroutine (e.g. in a thread pool to avoid deadlock) and returns the result.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs) # J：若为异步函数，返回 coroutine，若为同步函数，直接返回结果

        # J: case 0, func 是普通函数，直接返回结果
        if not inspect.iscoroutine(coro): # J：如果不是 coroutine，直接返回结果
            return coro

        try:
            # # J：理解：
            # # # J：当前线程无运行中的 event loop 的场景：1) 脚本顶层 或 2) 普通 def foo() 内等 都是
            # # # J：当前线程有运行中的 event loop 的场景：1)async def foo() 内 + 写 await foo() 或 2）同步函数被某个 async 函数间接调用
            loop = asyncio.get_running_loop() # J：获取当前运行中的 event loop，如果没有则抛出 RuntimeError
        except RuntimeError:
            loop = None

        # Case 1: No running loop -> run with asyncio.run()
        # J: case 1, func 是异步函数（协程），调用方是同步代码（脚本顶层 / 普通 def），无 event loop
        if loop is None: # J：如果没有运行中的 event loop（即不是使用类似 await func(...) 的方式调用的），则用 asyncio.run() 同步执行并返回结果
            return asyncio.run(coro) # J：同步执行并返回结果，注：如果协程内部有 await 操作，会阻塞当前线程，直到 await 的结果返回

        # Case 2: Running loop -> return coro if caller will await
        # J: Case 2, func 是异步函数（协程），调用方是 async 函数，有 event loop，会 await
        caller_frame = inspect.currentframe() # J：返回 当前这一行所在的栈帧 ，也就是 wrapper 函数自身的帧（C Python 中是 PyFrameObject 的引用），注：栈帧保存了函数调用现场：局部变量、返回地址、对应的 code object 等
        if caller_frame is not None: # J：如果有调用帧，说明有其他调用方使用 await func(...) 的方式本函数，考虑直接返回给调用方即可
            caller_frame = caller_frame.f_back # J：frame.f_back 是栈帧的 回溯指针 ，指向调用当前函数的那一帧
        
        # J：若调用方式是 await func(...) 调用当前函数且调用方是 async 函数，则可以直接返回 coroutine，交给调用方的 event loop 自行处理
        caller_is_async = caller_frame is not None and (caller_frame.f_code.co_flags & inspect.CO_COROUTINE) != 0 # J：进一步判断调用方是不是 async 函数
        if caller_is_async: # J：如果是 async 函数，直接返回 coroutine
            return coro # J：返回 coroutine，交给调用方的 event loop 处理，注："等待"这件事由调用方的 await 来做，wrapper 不需要、也不能代劳

        # J：如果没有 Case 2，所有"有运行中 loop"的情况都会进 Case 3——开新线程 + asyncio.run 。这能工作但有代价：
        # 1. 线程切换开销 ：每次调用都要起一个线程
        # 2. 失去并发语义 ：调用方明明是 async，本可以 asyncio.gather 多个 mgr.generate_sequences(...) 并发，结果被强行塞到独立线程里串行等待
        # 3. event loop 隔离 ：内部协程跑在另一个线程的新 loop 上，与原 loop 的任务（比如 client session、connection pool）无法共享
        # J：所以 Case 2 的存在意义是： 识别出"调用方会 await"这种最自然的异步使用方式，把控制权完整交还给调用方的 event loop ，让 @auto_await 既兼容同步调用，又不损害异步调用的并发能力

        # Case 3: Running loop -> run coro in thread pool
        # (cannot block the loop thread without deadlock)
        # J：Case 3, func 是异步函数（协程），调用方是同步函数，但被某个 async 栈间接调到，有 event loop 但不能用 await
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool: # J：创建一个线程池，最多有一个线程
            future = pool.submit(asyncio.run, coro) # J：提交一个任务到线程池，任务是 asyncio.run(coro)
            return future.result() # J：等待线程池中的任务完成，返回结果

    return wrapper
