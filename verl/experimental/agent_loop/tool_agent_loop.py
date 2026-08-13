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
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)


class AgentState(Enum): # J：定义 Agent 状态枚举
    PENDING = "pending" # J：待处理状态，需要处理输入 messages 为 prompt_ids 并切换到 GENERATING 状态
    GENERATING = "generating" # J：生成状态，需要生成 response_ids 并切换到 PROCESSING_TOOLS 状态或 TERMINATED 状态
    PROCESSING_TOOLS = "processing_tools" # J：处理工具状态，需要执行工具调用并准备工具响应，返回 GENERATING 状态 或 TERMINATED 状态
    TERMINATED = "terminated" # J：终止状态，需要返回最终 response_ids


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase): # J：定义工具 Agent 循环
    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop. # J：核心参数，指定要使用的工具列表
        """
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns # J：最大用户轮数
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns # J：最大助手轮数
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls # J：最大并行调用数
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length # J：最大工具响应长度
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side # J：工具响应截断侧，默认从中间截断

        tool_list = tools.tools if tools else [] # J：获取工具列表，默认为空列表
        self.tools = {tool.name: tool for tool in tool_list} # J：将工具列表转换为字典，键为工具名称，值为工具对象
        # J：注，这里的工具是 Pydantic 模型，需要使用 model_dump 方法转换为 dict 格式
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list] # J：将工具列表转换为 dict 列表(按照工具定义的 schema 转换)
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer) # J：根据工具 Parser 名称获取工具 Parser，并输入 tokenizer 为参数初始化 ToolParser 实例
        self.tool_parser_name = self.rollout_config.multi_turn.format # J：获取工具 Parser 名称(就使用 format)

        self.prompt_length = self.rollout_config.prompt_length # J：最大 prompt 长度
        self.response_length = self.rollout_config.response_length # J：最大响应长度

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput: # J：AgentLoop 的核心函数
        messages = list(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages) # J：从消息中提取多模态输入
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios) # J：获取多模态处理器的参数

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData( # J：初始化 Agent 数据
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection") # J：从 extra_info 中获取工具选择列表，默认全选
        if tool_selection and self.tools:
            # J：selected 包含同时在 extra_info.tool_selection 中和 self.tools 中的工具名称
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools} # J：根据 extra_info.tool_selection 选择工具，selected 是一个字典，键是工具名称，值是工具对象
            agent_data._active_tools = selected # J：将 active_tools 设置为 selected，即只使用 extra_info.tool_selection 中的工具
            agent_data._active_tool_schemas = [ # J：将 active_tools 中的工具 schema 转换为 dict 列表，后续传入 apply_chat_template 方法
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else: # J：如果没有 tool_selection，默认全选所有工具
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING # J：初始化状态为 PENDING，只会进入依次 PENDING 状态
        while state != AgentState.TERMINATED: # J：状态机循环，直到状态为 TERMINATED
            if state == AgentState.PENDING: # J：处理 PENDING 状态
                state = await self._handle_pending_state(agent_data, sampling_params) # J：处理 PENDING 状态，准备 prompt_ids 并返回 GENERATING 状态
            elif state == AgentState.GENERATING: # J：处理 GENERATING 状态
                state = await self._handle_generating_state(agent_data, sampling_params) # J：处理 GENERATING 状态，生成 response_ids 并返回 PROCESSING_TOOLS 状态 或 TERMINATED 状态
            elif state == AgentState.PROCESSING_TOOLS: # J：处理 PROCESSING_TOOLS 状态
                state = await self._handle_processing_tools_state(agent_data) # J：处理 PROCESSING_TOOLS 状态，执行工具调用并准备工具响应，返回 GENERATING 状态 或 TERMINATED 状态
            else: # J：处理其他错误状态
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :] # J：注意，response_mask 不包含最前面的 prompt_ids 的位置内容，也就是说 response_mask 的长度仅和新生成的 token_ids 数量相关（包括工具返回和模型生成结果）
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)] # J：取出原始 prompt_ids（不包含新生成的 工具调用和 模型输出 token_ids）
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput( # J：生成 AgentLoopOutput 对象
            prompt_ids=prompt_ids, # J：原始 prompt_ids（不包含新生成的 工具调用和 模型输出 token_ids）
            response_ids=response_ids[: self.response_length], # J：截取 response_ids 中前 self.response_length（最大响应长度） 个 token_ids
            response_mask=agent_data.response_mask[: self.response_length], # J：截取 response_mask 中前 self.response_length（最大响应长度） 个 token_ids，与 response_ids 对应
            multi_modal_data=multi_modal_data, # J：多模态输入数据
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards}) # J：将 turn_scores 和 tool_rewards 添加到 extra_fields 中
        return output # J：返回 AgentLoopOutput 对象

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState: # J：处理 PENDING 状态，准备 prompt_ids 并返回 GENERATING 状态
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas) # J：获取 active_tools 中的 tool_schemas，默认使用 self.tool_schemas
        prompt_ids = await self.apply_chat_template( # J：应用聊天模板，生成 prompt_ids
            agent_data.messages, # J：针对 agent_data 中当前的所有 messages 进行 apply_chat_template
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
        )
        agent_data.prompt_ids = prompt_ids # J: 设置 agent_data.prompt_ids 的内容为 prompt_ids（当前应该只会进入一次 pending 状态）
        # J：注意：这里仅仅对 prompt_ids 进行赋值，没有考虑 response_mask
        return AgentState.GENERATING # J：返回 GENERATING 状态

    async def _handle_generating_state( # J：处理 GENERATING 状态，生成 response_ids 并返回 PROCESSING_TOOLS 状态或 TERMINATED 状态
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # Inject tool parser stop tokens so generation halts after each tool call
        if self.tool_parser.stop_token_ids: # J：如果有工具解析器的 stop_token_ids，则将其添加到 sampling_params 中
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        with simple_timer("generate_sequences", agent_data.metrics): # J：记录生成 response_ids 的时间，还会包含 num_preempted 等指标
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        agent_data.assistant_turns += 1 # J：增加助手轮数
        agent_data.response_ids = output.token_ids # J：将模型生成的 token_ids 赋值给 response_ids（注意：仅包含本轮生成的 token_ids）
        agent_data.prompt_ids += agent_data.response_ids # J：将 response_ids 添加到 prompt_ids 中(prompt_ids 包含所有轮的 token_ids，是记录轨迹的核心字段)
        agent_data.response_mask += [1] * len(agent_data.response_ids) # J：所有响应都是模型输出，所以 response_mask 中所有元素都是 1
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs # J：将 response_ids 的 log_probs 添加到 response_logprobs 中

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts # J：将路由专家添加到路由专家列表中

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length: # J：如果 response_ids 长度超过最大长度长度，则终止
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns: # J：如果助手轮数超过最大轮数，则终止
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns: # J：如果用户轮数超过最大轮数，则终止
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()] # J：获取当前激活的工具列表
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools) # J：从 response_ids 中提取工具调用

        if agent_data.tool_calls: # J：如果有工具调用，则继续跳转到工具调用处理状态
            return AgentState.PROCESSING_TOOLS # J：返回 PROCESSING_TOOLS 状态
        else:
            return AgentState.TERMINATED # J：返回 TERMINATED 状态

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState: # J：处理 PROCESSING_TOOLS 状态，执行工具调用并准备工具响应，返回 GENERATING 状态 或 TERMINATED 状态
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]: # J：遍历工具调用列表，最多执行 max_parallel_calls 个工具调用
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data)) # J：启动工具调用任务，将任务添加到任务列表中，实现并行调用
            tool_call_names.append(tool_call.name) # J：将工具调用名称添加到 tool_call_names 中

        with simple_timer("tool_calls", agent_data.metrics): # J：记录 工具调用时间
            responses = await asyncio.gather(*tasks) # J：获取工具调用结果

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses: # J：依次处理多个工具调用的结果
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message) # J：将工具响应消息添加到 add_messages 中

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages) # J：将包含所有工具响应的 message 添加到 agent_data.messages 中

        if self.tool_parser_name == "gpt-oss": # J: 针对 gpt-oss 的工具调用格式特殊梳理
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "gemma4": # J: 针对 gemma4 的工具调用格式特殊梳理
            # Gemma4's chat template drops tool responses when passed without the preceding
            # assistant tool_call message. Manually format the response tokens.
            # Format: <|tool_response>response:func_name{value:<|"|>content<|"|>}<tool_response|>
            parts = []
            for msg, name in zip(add_messages, tool_call_names, strict=True):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
            tool_response_text = "".join(parts)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else: # J: 处理通用的 工具格式
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template( # J：针对 add_messages 进行 apply_chat_template
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length: # J：如果 response_ids 超过最大长度，返回 TERMINATED 状态
            return AgentState.TERMINATED # J：返回 TERMINATED 状态
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids # J: 将当前轮次的 token_ids 添加添加到 agent_data.prompt_ids 中（prompt_ids 包含所有轮的 token_ids，是记录轨迹的核心字段）
        agent_data.response_mask += [0] * len(response_ids) # J：工具调用的结果都不是模型生成的
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids) # J：不需要记录这里的 logprobs，这部分不是模型输出
        agent_data.user_turns += 1 # J：user_turns 轮次 +1
        return AgentState.GENERATING # J：返回 GENERATING 状态

    async def _call_tool( # J：单个调用工具，返回工具响应结果
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle. # J：注意，FunctionCall 不是 FunctionTool，这里的 tool_call 是 FunctionCall 类型，真实工具调用可以是 FunctionTool 类型，也可以是 BaseTool 类型的
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        # Validate tool name
        tool_name = tool_call.name
        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Validate tool arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Execute tool
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name] # J：获取工具实例

            if isinstance(tool, FunctionTool): # J：如果是 FunctionTool 类型
                # Function-based tools have no lifecycle; call directly.
                # Note: tools_kwargs (create_kwargs / release_kwargs) is intentionally
                # ignored here. Function tools are stateless and per-trajectory state
                # injection is not supported by design; use a BaseTool subclass instead.
                raw = await tool.call(tool_args) # J：直接调用工具，获取原始结果
                tool_execution_response, tool_reward, res = normalize_function_tool_return(raw) # J：将结果统一为 ToolResponse, reward, metrics 元组
            else: # J：否则，如果是 BaseTool 类型
                # BaseTool subclass
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {})) # J：创建工具实例
                tool_execution_response, tool_reward, res = await tool.execute( # J：执行工具调用返回结果
                    instance_id, tool_args, agent_data=agent_data
                )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0, {}
        finally:
            # Only BaseTool instances need release (function tools never set instance_id).
            if tool and instance_id and not isinstance(tool, FunctionTool):
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length: # J：如果 tool_response_text 超过最大长度
            if self.tool_response_truncate_side == "left": # J：如果配置为左侧截断
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            elif self.tool_response_truncate_side == "right": # J：如果配置为右侧截断
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            else: # J：否则，配置为默认截断中间
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:] # J：截断中间部分，两边各保留 length 个字符

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text} # J：创建工具响应参数字典，包含文本内容

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name): # J：如果 tool_execution_response 有该属性（image 或 video）
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value # J：将属性值添加到 tool_response_kwargs 中

        return ToolResponse(**tool_response_kwargs), tool_reward, res # J：返回工具响应、奖励和额外信息
