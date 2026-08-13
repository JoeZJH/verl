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
import re

import aiohttp
from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse

class SandboxTool(BaseTool): # J：沙盒工具类
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Different model may use different code pattern, e.g. python, py, etc.
        self.code_pattern = re.compile(r"```py(.*?)```", re.DOTALL)

    async def code_interpreter(self, code: str) -> str: # J：通过沙盒（HTTP API 请求）执行代码
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            str: The output of the code execution.
        """
        async with aiohttp.ClientSession() as session: # J：创建异步会话
            async with session.post( 
                self.config.get("sandbox_fusion_url"), # J：沙盒融合 URL
                json={"code": code}, # J：请求体 JSON 格式
            ) as resp: # J：异步发送 POST 请求并接收响应
                resp.raise_for_status()
                result = await resp.json()
                stdout, stderr = result["run_result"]["stdout"], result["run_result"]["stderr"]
                return stdout + stderr # J：返回标准输出和标准错误输出的拼接

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.code_interpreter) # J：获取 JSON schema, 通过 code_interpreter 函数的 docstring (参数和返回值类型)
        return OpenAIFunctionToolSchema(**schema) # J：返回 OpenAI 格式的工具 schema

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[str, float, dict]: # J：执行工具实现
        code = parameters["code"] # J：从参数中获取代码
        matches = self.code_pattern.findall(code) # J：从代码中提取 Python 代码块
        if matches:
            code = matches[0].strip() # J：提取第一个匹配项并移除首尾空格

        # NOTE: Some script may not explicitly print result, we need to add a print statement to the end of the script.
        # More better way is to SFT the model to make it print result by default, we skip SFT stage in this tutorial.
        lines = code.split("\n") # J：将代码按行分割
        # J：从后往前遍历代码行，找到第一个不是 print 语句的行，然后在该行前添加 print 语句
        for i, line in reversed(list(enumerate(lines))):
            if line == "": # J：跳过空行
                continue
            if not lines[i].startswith("print"): # J：如果当前行不是 print 语句，则为第一个不是 print 语句的行
                lines[i] = f"print({line})" # J：在当前行前添加 print 语句（第一个不是 print 语句的行，仅添加一次）
            break # J：找到第一个不是 print 语句的行后，跳出循环
        code = "\n".join(lines) # J：将代码重新组合成字符串

        result = await self.code_interpreter(code) # J：真正执行，通过沙盒（HTTP API 请求）执行代码
        return ToolResponse(text=result), 0.0, {} # J：返回执行结果和额外信息
