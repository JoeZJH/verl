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

from __future__ import annotations

import importlib
import logging
import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, Optional

from omegaconf import OmegaConf

from verl.tools.function_tool import FunctionTool, load_function_tools_from_path
from verl.tools.schemas import OpenAIFunctionToolSchema

if TYPE_CHECKING:
    from verl.tools.base_tool import BaseTool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolType(Enum):
    # MCP tool is removed for now.
    NATIVE = "native"


def get_tool_class(cls_name): # J：根据类名获取工具类
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    tool_cls = getattr(module, class_name) # J：从模块中获取类名对应的类
    return tool_cls


def initialize_tools_from_config(tools_config_file) -> list: # J：从 yaml 配置文件加载 Native 工具
    """Instantiate ``BaseTool`` subclasses declared in a yaml config."""
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    # J：tools_config 中会包含一个叫做 tools 的字段，里面列出了所有工具的定义
    for tool_config in tools_config.tools: # J：遍历配置文件中的所有工具配置
        cls_name = tool_config.class_name # J：工具类名
        tool_type = ToolType(tool_config.config.type) # J：工具类型，当前仅支持 NATIVE
        tool_cls = get_tool_class(cls_name) # J：根据类名动态获取工具类，包括加载模块和类

        match tool_type:
            case ToolType.NATIVE:
                if tool_config.get("tool_schema", None) is None:
                    tool_schema = None
                else:
                    # J：将配置文件中的工具模式转换为 Pydantic 模型
                    tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True), # J：将配置文件中的工具配置转换为 Python 字典
                    tool_schema=tool_schema, # J：工具模式（Pydantic 模型）
                )
                tool_list.append(tool)
            case _:
                raise NotImplementedError(f"Unsupported tool type: {tool_type}")

    return tool_list


def load_all_tools( # J：加载所有工具，包括 NativeTool 和 FunctionTool
    tool_config_path: Optional[str],
    function_tool_path: Optional[str],
) -> list[BaseTool | FunctionTool]:
    """Load native + function tools, check for name collisions, return merged list."""
    # J：从配置文件加载本地工具，包括 NativeTool 和 FunctionTool
    # # J：从 yaml 配置文件（通过 tool_config_path 指定）加载 Native 工具，YAML 配置文件 + BaseTool 的子类
    # # J：返回的 BaseTool 包含 execute 携程方法用于执行工具
    # # J：NativeTool 一般都都继承自：verl.tools.base_tool.BaseTool（具体类名 yaml 配置中需要写出）
    # # J:   NativeTool 有状态，包含 config, tool_shema 以及其他可自定的参数等，且包含 get_openai_tool_schema，create, execute, release 和 calc_reward 等函数
    # # J：   NativeTool 用于执行自定义的沙盒执行， 搜索引擎和 爬虫等有状态的工具
    native_tools: list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
    # # J：从 python 文件（通过 function_tool_path 指定）加载 Function 工具
    # # J：返回的 FunctionTool 包含 call 协程方法用于执行工具
    # # J：   FunctionTool 是无状态的，是一个 FunctionTool 类对象，包含 name, fn, 和 tool_shema 等，核心是 call 函数
    # # J：   FunctionTool 用于天气查询，计算器等无状态函数
    function_tools: list[FunctionTool] = load_function_tools_from_path(function_tool_path) if function_tool_path else []

    if function_tools and native_tools:
        existing = {t.name for t in native_tools} # J：获取所有 Native 工具的名称
        collisions = sorted(t.name for t in function_tools if t.name in existing) # J：获取所有 Function 工具的名称，与 Native 工具名称冲突
        if collisions: # J：如果有冲突则抛出异常，native_tools 中的名称不能与 Function 工具名称冲突
            raise ValueError(
                f"Function tool name(s) {collisions} collide with tools already declared in "
                f"'{tool_config_path}'. Each tool name must be unique across `tool_config_path` "
                f"and `function_tool_path`; rename one of them."
            )

    return native_tools + function_tools # J：合并 Native 工具和 Function 工具
