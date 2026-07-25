# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""``@function_tool`` examples loaded by the tests.

Includes both sync and async functions to exercise the dispatcher's
``asyncio.iscoroutinefunction`` branch.
"""

import asyncio

from verl.tools.function_tool import function_tool

@function_tool # J：注册函数工具到工具列表 verl.tools.function_tool.FUNCTION_TOOL_REGISTRY
def get_weather(city: str) -> dict: # J：获取天气函数工具，测试用
    """Get the current weather for a city.

    Args:
        city: The city to look up, e.g. "Tokyo" or "San Francisco".
    """
    # Stubbed lookup table; in production this would hit a weather API. The
    # values are deliberately unusual so a test can distinguish a real tool
    # response from a number the model guessed.
    table = {
        "tokyo": {"temperature_c": 17.3, "condition": "drizzle"},
        "san francisco": {"temperature_c": 14.8, "condition": "fog"},
        "new york": {"temperature_c": 21.6, "condition": "sunny"},
    }
    return table.get(city.lower(), {"temperature_c": -273.15, "condition": "unknown"}) # J：返回天气信息, 如果城市不存在则返回未知天气


@function_tool # J：注册函数工具到工具列表 verl.tools.function_tool.FUNCTION_TOOL_REGISTRY
def calculator(expression: str) -> str: # J：计算函数工具
    """Evaluate an arithmetic expression and return the result.

    Supports +, -, *, /, **, parentheses, and unary minus. Use this for any
    numerical computation instead of doing mental arithmetic.

    Args:
        expression: A Python-style arithmetic expression, e.g. "(3+4)*5".
    """
    import ast
    import operator as op

    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):  # noqa: UP038 # J：常量值
            return node.value # J：返回常量值
        if isinstance(node, ast.BinOp): # J：二元运算符
            return ops[type(node.op)](_eval(node.left), _eval(node.right)) # J：递归计算二元运算符的结果
        if isinstance(node, ast.UnaryOp): # J：一元运算符
            return ops[type(node.op)](_eval(node.operand)) # J：递归计算一元运算符的结果
        raise ValueError(f"unsupported node: {ast.dump(node)}")

    try:
        return str(_eval(ast.parse(expression, mode="eval").body)) # J：计算表达式的结果
    except Exception as e:
        return f"ERROR: {e}"


@function_tool # J：注册函数工具到工具列表 verl.tools.function_tool.FUNCTION_TOOL_REGISTRY
async def fetch_url(url: str) -> str:
    """Fetch the contents of a URL (async).

    This is a stubbed example that demonstrates ``async def`` tools; the
    real implementation would use ``aiohttp`` or ``httpx``.

    Args:
        url: The URL to fetch.
    """
    # Yield once so the tool actually behaves like an awaitable, then return
    # a deterministic stub payload that tests can assert on.
    await asyncio.sleep(0)
    return f"<stub body for {url}>" # J：返回模拟的URL内容, 这里只能用于测试
