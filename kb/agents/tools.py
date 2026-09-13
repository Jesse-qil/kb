"""工具池：Agent 可调用的工具（Function Calling）。
工具统一注册进 ToolPool（kb/agents/tool_pool.py）：name/description/schema/执行函数/调用统计。
迁移自旧项目 app/tools.py，检索改用 kb.kb_rag.query。"""
import json

from ..llm import LLMClient
from ..storage.vector_store import query
from ..web_search import web_search as _ws
from ..prompts import TOOL_ASSISTANT_PROMPT
from ..config import agent_cfg
from .tool_pool import get_pool


# 报错详情截断长度（防止子进程刷屏污染回答）
_MAX_DETAIL = 300


def search_knowledge(query_text: str, topic: str = "") -> str:
    """检索知识库，返回最相关的片段文本（供 LLM 看）。"""
    hits = query(query_text, topic=topic, top_k=agent_cfg()["query_top_k"])
    if not hits:
        return "知识库中没有相关内容"
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)


def execute_python(code: str, timeout: float = 5.0) -> str:
    """用【子进程】执行代码：超时直接杀进程，不会拖垮主程序。
    比线程安全：进程内存/锁全隔离，超时 terminate 真能终止。
    退出码约定：0 = 子进程包装层正常结束（含代码运行期异常已转文字）
               1 = 代码抛了运行期异常（wrapper 内 sys.exit(1)）
               ≠0 = 语法错误/崩溃（stderr 有 traceback）"""
    import subprocess, sys, os

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
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, "-c", wrapper, code],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", env=env,
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


def search_graph(query_text: str) -> str:
    """在知识图谱中查找与问题关联的文档与实体关系。
    知识库检索的是"内容片段"，图谱检索的是"笔记间的关系"：
    问题命中实体/标签（如 LangGraph、装饰器）时，返回它关联的文档清单，
    帮 LLM 找出跨文档的关联线索（比如"LangGraph 相关笔记有哪些"）。"""
    from ..kg.retrieve import expand_sources
    from ..kg import store
    if not query_text.strip():
        return "查询内容为空"
    g = store.load()
    if not g["nodes"]:
        return "知识图谱尚未构建（运行 scripts/build_graph.py 或在前端点「重建图谱」）"
    sources = expand_sources(query_text)
    if not sources:
        return "图谱中没有与问题直接关联的文档（可改用知识库检索）"
    terms = list(dict.fromkeys(
        nd["name"] for nd in g["nodes"]
        if nd.get("type") in ("entity", "tag")
        and len(str(nd.get("name", ""))) >= 2
        and nd["name"] in query_text
    ))
    lines = []
    if terms:
        lines.append("问题命中的图谱实体/标签: " + "、".join(terms[:10]))
    lines.append("关联文档:")
    lines += [f" - {s}" for s in sources]
    return "\n".join(lines)


# ---------- 工具池注册：新增工具在这里加一行 ----------
# ToolPool 统一注册（schema 给 LLM 看 + 函数实际执行 + 调用统计），
# 工具名/描述/参数/执行函数/分类 五要素齐全，前端可枚举展示。
_pool = get_pool()

_pool.register(
    "search_knowledge",
    "在个人知识库中检索与问题相关的笔记片段。参数 topic 可限定领域/分类。",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词/问题"},
            "topic": {"type": "string", "description": "领域名（可选）"},
        },
    },
    search_knowledge,
    category="检索", required=["query"],
)
_pool.register(
    "execute_python",
    "运行一段 Python 代码并返回输出。用于验证代码、算数、处理数据。",
    {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
    },
    execute_python,
    category="执行", required=["code"],
)
_pool.register(
    "web_search_tool",
    "联网搜索当前问题，返回相关网页标题。用于知识库没有的内容。",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "搜索关键词"}},
    },
    web_search_tool,
    category="检索", required=["query"],
)
_pool.register(
    "search_graph",
    "在知识图谱中查找与问题关联的文档与实体关系。用于找笔记之间的关联。",
    {
        "type": "object",
        "properties": {"query_text": {"type": "string", "description": "要查关联的问题/实体"}},
    },
    search_graph,
    category="检索", required=["query_text"],
)

# ---------- 兼容旧引用：TOOLS / TOOL_FUNCS 由池派生 ----------
# （graph.py 等仍按原接口 import；新代码建议直接用 get_pool()）
TOOLS = _pool.schemas()
TOOL_FUNCS = {t.name: t.func for t in _pool.list_all()}


# ---------- Agent 工具主循环（Function Calling 核心） ----------
def run_agent_with_tools(question: str, max_rounds: int = 0) -> str:
    """让 LLM 自主决定：要不要调工具、调哪个、调几次，直到给出最终回答。"""
    max_rounds = max_rounds or agent_cfg()["tool_max_rounds"]
    messages = [
        {"role": "system", "content": TOOL_ASSISTANT_PROMPT},
        {"role": "user", "content": question},
    ]
    llm = LLMClient()
    pool = get_pool()
    for _ in range(max_rounds):
        msg = llm.chat_with_tools(messages, pool.schemas())
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
            result = pool.call(name, **args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    return "（达到工具轮次上限，未能完成回答）"


if __name__ == "__main__":
    print("\033[92m kb/tools.py 文件\033[0m")
    print(run_agent_with_tools("用 python 算 2 的 10 次方"))
