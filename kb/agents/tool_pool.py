"""工具池（ToolPool）：Agent 可用工具的统一定义、注册与调用。

把散落的工具函数 + JSON Schema 升级为可扩展的池：
- 注册：register(name, description, parameters, func, category) —— 新增工具只加一行
- 枚举：schemas() → OpenAI Function Calling 需要的 JSON Schema 列表
- 调用：call(name, **args) —— 统一执行 + 调用计数（统计供前端/评估）
- 查询：get(name) / list_all() / stats()

相比 MCP（跨应用标准协议），这是本地版的"工具池"：工具在进程内注册表里，
LLM 通过 Function Calling 按 schema 调度。未来接 MCP 时，只需把池里每个工具
包一层 @mcp.tool()，执行器从"本地函数"换成"MCP 客户端"即可，架构不变。
"""
from __future__ import annotations

import inspect
import threading
from typing import Callable


class ToolSpec:
    """一个工具的完整定义：给 LLM 看的 schema + 实际执行函数。"""

    def __init__(self, name: str, description: str, parameters: dict,
                 func: Callable, category: str = "内置", required: list[str] | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func
        self.category = category
        self.required = required or []
        self.calls = 0

    def schema(self) -> dict:
        """OpenAI Function Calling 格式的 JSON Schema。"""
        params = dict(self.parameters)
        if self.required:
            params["required"] = self.required
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def __call__(self, **kwargs) -> str:
        self.calls += 1
        return self.func(**kwargs)


class ToolPool:
    """工具池：注册表 + 执行器 + 调用统计，单例使用。"""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._lock = threading.Lock()

    # ---------- 注册 / 查询 ----------

    def register(self, name: str, description: str, parameters: dict,
                 func: Callable, category: str = "内置",
                 required: list[str] | None = None) -> "ToolPool":
        with self._lock:
            self._tools[name] = ToolSpec(name, description, parameters, func,
                                         category=category, required=required)
        return self

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    # ---------- 给 LLM 的说明书 ----------

    def schemas(self) -> list[dict]:
        """OpenAI Function Calling 完整 tools 参数。"""
        return [t.schema() for t in self._tools.values()]

    # ---------- 执行 ----------

    def call(self, name: str, **args) -> str:
        """按名字调用工具并计数；未注册/执行异常都转成可回喂 LLM 的文本。"""
        spec = self._tools.get(name)
        if spec is None:
            return f"工具不存在: {name}"
        try:
            return spec(**args)
        except Exception as e:
            return f"工具执行失败: {type(e).__name__}: {e}"

    # ---------- 统计（前端 / 评估用） ----------

    def stats(self) -> list[dict]:
        return [{"name": t.name, "category": t.category, "description": t.description,
                 "calls": t.calls} for t in self._tools.values()]


# ============ 全局单例：全项目共享同一个工具池 ============
_pool = ToolPool()


def get_pool() -> ToolPool:
    return _pool
