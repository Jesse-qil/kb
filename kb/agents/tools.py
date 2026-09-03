"""工具注册表：Agent 可调用的工具（Function Calling）。
每个工具 = 一个函数 + 一份 JSON Schema 描述（告诉 LLM 这个工具怎么用）。
迁移自旧项目 app/tools.py，检索改用 kb.kb_rag.query。"""
import json

from ..llm import LLMClient
from ..storage.vector_store import query
from ..web_search import web_search as _ws
from ..prompts import TOOL_ASSISTANT_PROMPT


# 报错详情截断长度（防止子进程刷屏污染回答）
_MAX_DETAIL = 300


def search_knowledge(query_text: str, topic: str = "") -> str:
    """检索知识库，返回最相关的片段文本（供 LLM 看）。"""
    hits = query(query_text, topic=topic, top_k=3)
    if not hits:
        return "知识库中没有相关内容"
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)


def execute_python(code: str, timeout: float = 5.0) -> str:
    """用【子进程】执行代码：超时直接杀进程，不会拖垮主程序。
    比线程安全：进程内存/锁全隔离，超时 terminate 真能终止。
    退出码约定：0 = 子进程包装层正常结束（含代码运行期异常已转文字）
               1 = 代码抛了运行期异常（wrapper 内 sys.exit(1)）
               ≠0 = 语法错误/崩溃（stderr 有 traceback）"""
    import subprocess, sys

    #   关键：wrapper 是"子进程的源码"，必须用普通字符串拼接（非 f-string），
    #   里面的 type(e).__name__ 留给子进程执行时才求值——父进程拼死会拿不到子进程异常
    wrapper = (
        "import sys\n"
        "code = sys.argv[1]\n"
        "try:\n"
        "    exec(code)\n"
        "except Exception as e:\n"
        "    print('运行报错: ' + type(e).__name__ + ': ' + str(e))\n"
        "    sys.exit(1)\n"     # 代码异常也标退出码 1，调用方好区分
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper, code],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        # 非 0 退出码：代码异常(exit 1)详情在 stdout；语法错/崩溃在 stderr
        if proc.returncode != 0:
            detail = out or err or f"exit code {proc.returncode}"
            return f"运行失败: {detail[:_MAX_DETAIL]}"

        if out:
            return out
        if err:
            return f"(无输出，但有 stderr) {err[:_MAX_DETAIL]}"
        return "(无输出，代码运行成功)"
    except subprocess.TimeoutExpired:
        return f"运行超时：超过 {timeout}s（已终止）"



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
