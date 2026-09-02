"""工具注册表：Agent 可调用的工具（Function Calling）。
每个工具 = 一个函数 + 一份 JSON Schema 描述（告诉 LLM 这个工具怎么用）。
迁移自旧项目 app/tools.py，检索改用 kb.kb_rag.query。"""
import json

from .llm import LLMClient
from .kb_rag import query
from .web_search import web_search as _ws
from .prompts import TOOL_ASSISTANT_PROMPT


def search_knowledge(query_text: str, topic: str = "") -> str:
    """检索知识库，返回最相关的片段文本（供 LLM 看）。"""
    hits = query(query_text, topic=topic, top_k=3)
    if not hits:
        return "知识库中没有相关内容"
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)


def execute_python(code: str) -> str:
    """执行一段 Python 代码，返回运行输出（5 秒超时 + 半隔离）。"""
    import io, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__}, {})
        return output.getvalue() or "(无输出，代码运行成功)"
    except Exception as e:
        return f"运行报错: {type(e).__name__}: {e}"


def web_search_tool(query: str) -> str:
    """联网搜索，返回结果标题列表。"""
    results = _ws(query)
    return "\n".join(results) if results else "无返回内容"


# ---------- 工具注册表：给 LLM 看的"工具说明书" ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在个人知识库中检索与问题相关的笔记片段。参数 topic 可限定领域/分类（如 notes_draft）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词/问题"},
                    "topic": {"type": "string", "description": "领域名（可选）"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "运行一段 Python 代码并返回输出。用于验证代码、算数、处理数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "联网搜索当前问题，返回相关网页标题。用于知识库没有的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            },
        },
    },
]

# 工具名 → 执行函数 的映射（执行器）
TOOL_FUNCS = {
    "search_knowledge": search_knowledge,
    "execute_python": execute_python,
    "web_search_tool": web_search_tool,
}


# ---------- Agent 工具主循环（Function Calling 核心） ----------
def run_agent_with_tools(question: str, max_rounds: int = 5) -> str:
    """让 LLM 自主决定：要不要调工具、调哪个、调几次，直到给出最终回答。"""
    messages = [
        {"role": "system", "content": TOOL_ASSISTANT_PROMPT},
        {"role": "user", "content": question},
    ]
    llm = LLMClient()
    for _ in range(max_rounds):
        msg = llm.chat_with_tools(messages, TOOLS)
        if msg is None or not getattr(msg, "tool_calls", None):
            return msg.content if msg is not None else "（mock 模式：配置真实模型后才能演示工具调用）"
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            print(f"  [工具] 调用: {name} {args}")
            try:
                result = TOOL_FUNCS[name](**args)
            except Exception as e:
                result = f"工具执行失败: {type(e).__name__}: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    return "（达到工具轮次上限，未能完成回答）"


if __name__ == "__main__":
    print("\033[92m kb/tools.py 文件\033[0m")
    print(run_agent_with_tools("用 python 算 2 的 10 次方"))
