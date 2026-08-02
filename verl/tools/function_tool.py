# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Lightweight function-based tool registration.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from transformers.utils import get_json_schema

from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# global registry
FUNCTION_TOOL_REGISTRY: dict[str, FunctionTool] = {}

_LOADED_FUNCTION_TOOL_PATHS: dict[str, list[FunctionTool]] = {}


@dataclass
class FunctionTool: # J：函数工具对象，用于存储函数工具的元数据和调用逻辑，同时定义了 call 方法
    """Carrier object stored in :data:`FUNCTION_TOOL_REGISTRY`.

    Exposes the minimal interface that the agent loop relies on:

    - ``name``: tool name (matches ``tool_schema.function.name``)
    - ``tool_schema``: ``OpenAIFunctionToolSchema`` for prompt assembly
    - ``fn``: the underlying callable
    """

    name: str
    fn: Callable[..., Any] # J：底层函数工具对象，可调用的函数
    tool_schema: OpenAIFunctionToolSchema
    is_async: bool = False # J：是否为协程

    async def call(self, parameters: dict[str, Any]) -> Any: # J：调用函数工具，根据 is_async 参数选择不同的调用方式
        """Invoke the underlying function with the LLM-supplied parameters."""
        if self.is_async: # J：如果是 is_async=True（即 fn 是协程函数），使用协程方式调用函数
            return await self.fn(**parameters) # J：调用协程函数
        return await asyncio.to_thread(self.fn, **parameters) # J：否则 fn 为普通函数，此时使用多线程调用普通函数

# J：注册函数工具的装饰器函数
# J：在 Python 中，装饰器有两种常见写法：
# J:    不带括号：@function_tool （此时 function_tool 直接接收被装饰的函数作为第一个参数）
# J:    带括号：@function_tool("custom_name") （此时 function_tool 先执行，返回一个真正的装饰器，然后再接收被装饰的函数）
# J: 
def function_tool( 
    name: Optional[str | Callable] = None, # J：工具名称，或函数对象本身
    *,
    schema: Optional[OpenAIFunctionToolSchema | dict] = None,
):
    """Register a Python function as a verl tool.

    The OpenAI tool schema is inferred from the function via
    :func:`transformers.utils.get_json_schema`, so the function **must** carry:

    - a Google-style docstring summarising the tool;
    - a ``Args:`` block describing every parameter;
    - a type hint on every parameter.

    If any of those are missing, ``transformers`` raises
    ``DocstringParsingException`` / ``TypeHintParsingException`` at
    registration time.

    Supports both decorator forms::

        @function_tool                          # bare, name = fn.__name__
        def web_search(...): ...

        @function_tool("web_search")            # rename the tool
        def search(...): ...

    Args:
        name: Tool name exposed to the LLM. Defaults to the function name.
            When used as a bare ``@function_tool`` (no parentheses), this
            position receives the function being decorated.
        schema: Skip schema inference entirely and use the supplied
            ``OpenAIFunctionToolSchema`` (or a dict matching that shape) as-is.
            Use this only if your function's signature can't be expressed in
            JSON Schema -- the normal path is to fix the function.
    """

    def _make_decorator(tool_name_override: Optional[str]):
        def decorator(fn: Callable): # J：装饰器函数，用于注册函数工具
            tool_name = tool_name_override or fn.__name__ # J：根据 tool_name_override 或 fn.__name__ 确定工具名称

            if isinstance(schema, OpenAIFunctionToolSchema):
                built_schema = schema
            elif isinstance(schema, dict):
                built_schema = OpenAIFunctionToolSchema.model_validate(schema)
            else:
                built_schema = _build_schema_from_fn(fn, tool_name)

            entry = FunctionTool( # J：创建函数工具对象
                name=tool_name, # J：工具名称
                fn=fn, # J：底层函数工具对象，可调用的函数
                tool_schema=built_schema, # J：函数工具的 OpenAI 的工具模式（Pydantic 模型）
                is_async=inspect.iscoroutinefunction(fn), # J：fn 是否为协程函数的标志位
            )

            existing = FUNCTION_TOOL_REGISTRY.get(tool_name)
            if existing is not None and existing.fn is not fn: # J：如果已注册过该函数工具，且不是当前函数，抛出异常
                raise ValueError(
                    f"Function tool '{tool_name}' is already registered to "
                    f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
                    f"refusing to overwrite with {fn.__module__}.{fn.__qualname__}."
                )
            FUNCTION_TOOL_REGISTRY[tool_name] = entry # J：注册函数工具
            logger.info("Registered function tool '%s' from %s.%s", tool_name, fn.__module__, fn.__qualname__)
            return fn # J：返回底层函数工具对象本身

        return decorator
    # if callable(name) and schema is None: 用来处理装饰器的两种写法（不带括号和带括号）的差异
    if callable(name) and schema is None: # J： name 是函数对象，且未指定 schema
        fn = name # J：将 name 赋值给 fn，作为函数工具对象的底层函数
        return _make_decorator(None)(fn) # J：_make_decorator(None)(fn) 返回底层函数工具对象本身，会默认使用 fn.__name__ 作为工具名称，其中_make_decorator(None)返回 decorator 函数

    return _make_decorator(name) # J：返回装饰器，用于注册函数工具


def get_function_tool(name: str) -> FunctionTool: # J：根据名称获取已注册的函数工具，如果未注册则抛出异常
    """Look up a registered function tool by name. Raises ``KeyError`` if absent."""
    if name not in FUNCTION_TOOL_REGISTRY:
        raise KeyError(
            f"Function tool '{name}' not found in registry. Make sure its defining "
            f"file is referenced via the rollout `function_tool_path` config."
        )
    return FUNCTION_TOOL_REGISTRY[name]


def load_function_tools_from_path(path: str) -> list[FunctionTool]: # J：从 python 文件加载函数工具，通过 @function_tool 装饰器注册的函数工具
    """Execute a Python file at ``path`` and return its registered function tools."""
    abs_path = os.path.abspath(path) # J：获取绝对路径
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"function_tool_path does not exist: {path}")

    if abs_path in _LOADED_FUNCTION_TOOL_PATHS: # J：如果已加载过该路径的函数工具，直接返回已加载的函数工具
        return _LOADED_FUNCTION_TOOL_PATHS[abs_path]

    before = set(FUNCTION_TOOL_REGISTRY) # J：获取当前已注册的函数工具名称

    # Use a path-derived synthetic module name so the imported file can
    # ``from X import Y`` its siblings via ``sys.modules``.
    # J：从路径派生的合成模块名称，用于导入文件可以 ``from X import Y`` 其兄弟模块
    module_name = "_verl_function_tools_" + abs_path.replace(os.sep, "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for function_tool_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module) # J：执行函数工具模块，注册函数工具

    new_names = sorted(set(FUNCTION_TOOL_REGISTRY) - before) # J：获取新注册的函数工具名称
    if not new_names:
        logger.warning(
            "function_tool_path '%s' loaded but no @function_tool decorators found; "
            "did you forget to apply the decorator?",
            path,
        )
    else:
        logger.info("Loaded %d function tool(s) from %s: %s", len(new_names), path, new_names)

    tools = [FUNCTION_TOOL_REGISTRY[name] for name in new_names] # J：获取新注册的函数工具
    _LOADED_FUNCTION_TOOL_PATHS[abs_path] = tools # J：缓存已加载的函数工具
    return tools # J：返回新注册的函数工具（注意：仅返回新注册的函数工具，之前已注册的函数工具不会被返回）


def _build_schema_from_fn(fn: Callable, tool_name: str) -> OpenAIFunctionToolSchema:
    """Infer the OpenAI tool schema for ``fn`` via transformers.

    The heavy lifting (signature inspection + Google-style docstring parsing
    + JSON-Schema type mapping) is delegated to
    :func:`transformers.utils.get_json_schema`.
    """
    sig = inspect.signature(fn)
    variadic = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if variadic:
        raise ValueError(
            f"@function_tool '{tool_name}' ({fn.__module__}.{fn.__qualname__}) "
            f"declares variadic parameter(s) {variadic}, which can't be "
            f"expressed in an OpenAI tool schema. Replace them with explicit "
            f"named parameters."
        )

    raw = get_json_schema(fn)
    raw["function"]["name"] = tool_name
    return OpenAIFunctionToolSchema.model_validate(raw)


def normalize_function_tool_return(ret: Any) -> tuple[ToolResponse, float, dict]: # J：将结果统一为 ToolResponse, reward, metrics 元组
    """Coerce a function's return value into the ``(ToolResponse, reward, metrics)`` triple.

    Accepted shapes:

    - ``ToolResponse``  -> as-is, reward 0.0, metrics {}
    - ``str``           -> ``ToolResponse(text=ret)``
    - ``dict``          -> ``ToolResponse(text=json.dumps(ret))``
    - ``(response,)`` / ``(response, reward)`` / ``(response, reward, metrics)``
      -- ``reward`` may be ``None`` (treated as ``0.0``); ``metrics`` may be
      ``None`` (treated as ``{}``).
    - anything else     -> ``ToolResponse(text=str(ret))``

    Tuples of length 0 or >= 4 raise ``TypeError`` rather than being silently
    stringified, since they almost always signal a tool authoring bug.
    ``None`` is detected via ``is None`` rather than truthiness, so a
    legitimate ``0`` / ``0.0`` / ``False`` reward is preserved.
    """
    if isinstance(ret, ToolResponse):
        return ret, 0.0, {}
    if isinstance(ret, str):
        return ToolResponse(text=ret), 0.0, {}
    if isinstance(ret, dict):
        return ToolResponse(text=json.dumps(ret, ensure_ascii=False)), 0.0, {}
    if isinstance(ret, tuple):
        if not 1 <= len(ret) <= 3:
            raise TypeError(
                f"@function_tool return tuple must have length 1, 2, or 3 "
                f"(got length {len(ret)}: {ret!r}). Use (response,), "
                f"(response, reward), or (response, reward, metrics)."
            )
        response = _coerce_response(ret[0])
        reward = 0.0 if len(ret) < 2 or ret[1] is None else float(ret[1])
        metrics = {} if len(ret) < 3 or ret[2] is None else dict(ret[2])
        return response, reward, metrics
    return ToolResponse(text=str(ret)), 0.0, {}


def _coerce_response(value: Any) -> ToolResponse:
    if isinstance(value, ToolResponse):
        return value
    if isinstance(value, str):
        return ToolResponse(text=value)
    if isinstance(value, dict):
        return ToolResponse(text=json.dumps(value, ensure_ascii=False))
    return ToolResponse(text=str(value))
