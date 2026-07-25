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
import json
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse

# J：沙盒实现可参考 examples.tutorial.agent_loop_get_started.sandbox.SandboxTool
class BaseTool: # J：基础工具类，所有工具的基类
    """Base class for tools.

    A tool should support the following methods:

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema() # J：工具 schema
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name # J：工具名称
        print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2)) # J：打印工具 schema

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema: # J：获取工具的 OpenAI 格式 schema
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]: # J：创建工具实例
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            return str(uuid4()), ToolResponse()
        else:
            return instance_id, ToolResponse()

    # J：执行工具，详情请参考 examples.tutorial.agent_loop_get_started.sandbox.SandboxTool.execute
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]: # J：执行工具，返回工具响应、奖励分数和指标
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing text, image, and/or video content.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        return ToolResponse(text="Updated the tool state."), 0.0, {} # J：返回工具响应、奖励分数和指标

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        pass
