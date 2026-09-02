"""多 Agent 知识库问答（LangGraph 编排）
Planner(规划) → Retriever(检索) → Answerer(回答) → Reviewer(审查)
审查不通过打回 Answerer 重写，最多 3 轮。迁移自旧项目 app/graph.py。
改动：检索走 kb.kb_rag.query（含阈值过滤由调用方做），LLM 走 kb.llm.LLMClient。"""
import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .kb_rag import query, list_contents
from .llm import LLMClient
from .web_search import web_search
from .config import recall_cfg
from .prompts import (PLANNER_SYSTEM, ANSWERER_PROMPT,
                    ANSWERER_NO_DATA_PROMPT, REVIEWER_SYSTEM)


# ---------- 1. 状态（Agent 之间的"快递单"） ----------
class KBState(TypedDict):
    question: str            # 用户的问题
    topic: str               # Planner 判断出的领域（如 "notes_draft"）
    queries: list[str]       # Planner 拆出的检索查询
    hits: list[dict]         # 检索结果
    answer: str
    review_feedback: str
    rounds: int
    from_web: bool           # 是否走了联网兜底
    is_overview: bool        # 是否"知识库总览"类问题


def _chat(system: str, user: str) -> str:
    return LLMClient().chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])


# ---------- 候选领域：从台账/向量库拿（无 chroma 时可从 doc_index） ----------
def _get_topics() -> list[str]:
    """返回库里已有的领域列表（Planner 判断用）。优先看台账，避免每次起 chroma。"""
    try:
        from .doc_index import load
        topics = []
        for info in load().values():
            t = info.get("topic", "默认")
            if t not in topics:
                topics.append(t)
        return topics
    except Exception:
        return []


# ---------- 2. Agent 1：Planner ----------
def planner(state: KBState) -> dict:
    topics = _get_topics()
    cand = "、".join(topics) if topics else "默认"
    system = PLANNER_SYSTEM.format(cand=cand)
    user = f"""对问题做三件事，只输出 JSON（不要多余文字）：
    {{"overview": true/false, "topic": "领域名（必须是候选之一，都不像填 默认）", "queries": ["查询1", "查询2"]}}
    overview=true 当且仅当问题是在问"知识库里有什么/包含哪些/目录"这类总览问题。
    问题：{state["question"]}"""
    try:
        data = json.loads(_chat(system, user))
        topic = data.get("topic", "默认")
        queries = data.get("queries") or [state["question"]]
        is_overview = bool(data.get("overview", False))
    except Exception:
        topic, queries, is_overview = "默认", [state["question"]], False
    print(f"[Planner] 领域={topic} 查询={queries} 总览={is_overview}")
    return {"topic": topic, "queries": queries, "is_overview": is_overview}


# ---------- 2.5 总览节点 ----------
def overview(state: KBState) -> dict:
    """回答"库里有什么"：列领域 → 文件。"""
    try:
        contents = list_contents()
    except Exception:
        contents = {}
    if not contents:
        # 向量库空 → 用台账兜底
        from .doc_index import load
        contents = {}
        for rel, info in load().items():
            t = info.get("topic", "默认")
            contents.setdefault(t, [])
            if rel.split("/")[-1] not in contents[t]:
                contents[t].append(rel.split("/")[-1])
    lines = []
    for topic, files in contents.items():
        lines.append(f"[{topic}]：{'、'.join(files)}")
    answer = "我的知识库目前包含这些内容：\n" + "\n".join(lines) + \
             "\n\n问具体知识点，我可以详细讲解。"
    print("[总览] 列出知识库目录")
    return {"answer": answer}


# ---------- 3. Agent 2：Retriever ----------
def retriever(state: KBState) -> dict:
    """按领域多查询检索；弱相关（最高分<阈值）转联网。"""
    threshold = recall_cfg()["score_threshold"]
    hits, seen = [], set()
    for q in state["queries"]:
        for h in query(q, topic=state["topic"] if state["topic"] != "默认" else "", top_k=3):
            key = (h["source"], h["text"][:30])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    if not hits or hits[0]["score"] < threshold:
        print("[检索] 知识库弱相关，转联网…")
        web = web_search(state["question"])
        if web:
            return {"hits": [{"source": "网络", "text": t, "heading": "", "score": 0.0, "topic": ""} for t in web],
                    "from_web": True}
        return {"hits": [], "from_web": False}
    print(f"[检索] 命中 {len(hits[:6])} 条")
    return {"hits": hits[:6], "from_web": False}


# ---------- 4. Agent 3：Answerer ----------
def answerer(state: KBState) -> dict:
    if not state["hits"]:
        system = ANSWERER_NO_DATA_PROMPT
        user = (f"问题：{state['question']}\n\n"
                "（知识库和网络都没找到相关资料，请如实说明，可以凭常识简单回答，但要明确说这不是笔记内容）")
        answer = _chat(system, user)
        print(f"[回答] 第 {state['rounds'] + 1} 轮回答完成（无资料）")
        return {"answer": answer, "rounds": state["rounds"] + 1}
    context = "\n\n".join(f"[来自 {h['source']}]\n{h['text']}" for h in state["hits"])
    system = ANSWERER_PROMPT
    user = f"问题：{state['question']}\n\n参考资料：\n{context}"
    if state["review_feedback"]:
        user += f"\n\n上一轮审查意见（必须按此修改）：{state['review_feedback']}"
    answer = _chat(system, user)
    print(f"[回答] 第 {state['rounds'] + 1} 轮回答完成")
    return {"answer": answer, "rounds": state["rounds"] + 1}


# ---------- 5. Agent 4：Reviewer ----------
def reviewer(state: KBState) -> dict:
    system = REVIEWER_SYSTEM
    user = f"问题：{state['question']}\n\n参考资料：\n" + \
           "\n".join(f"[{h['source']}] {h['text'][:100]}" for h in state["hits"]) + \
           f"\n\n回答：\n{state['answer']}"
    fb = _chat(system, user).strip()
    print(f"[审查] {fb}")
    return {"review_feedback": fb}


# ---------- 6. 条件边 ----------
def should_retry(state: KBState) -> str:
    if state["review_feedback"] == "pass" or state["rounds"] >= 3:
        return "accept"
    return "retry"


def route_after_planner(state: KBState) -> str:
    return "overview" if state["is_overview"] else "retriever"


# ---------- 7. 组装图 ----------
def build_graph():
    g = StateGraph(KBState)
    g.add_node("planner", planner)
    g.add_node("overview", overview)
    g.add_node("retriever", retriever)
    g.add_node("answerer", answerer)
    g.add_node("reviewer", reviewer)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_after_planner,
                            {"overview": "overview", "retriever": "retriever"})
    g.add_edge("overview", END)
    g.add_edge("retriever", "answerer")
    g.add_edge("answerer", "reviewer")
    g.add_conditional_edges("reviewer", should_retry,
                            {"accept": END, "retry": "answerer"})
    return g.compile()


# ---------- 8. 对外接口 ----------
def ask(question: str) -> dict:
    result = build_graph().invoke({
        "question": question, "topic": "", "queries": [],
        "hits": [], "answer": "", "review_feedback": "", "rounds": 0,
        "from_web": False, "is_overview": False,
    })
    threshold = recall_cfg()["score_threshold"]
    if result.get("is_overview"):
        return {"answer": result["answer"], "sources": [], "topic": result["topic"]}
    if result["from_web"]:
        sources = [{"source": "网络搜索", "heading": "", "score": 0}]
    elif result["hits"] and result["hits"][0]["score"] >= threshold:
        seen, sources = set(), []
        for h in result["hits"]:
            key = (h["source"], h["text"][:20])
            if key not in seen:
                seen.add(key)
                sources.append({"source": h["source"], "heading": h.get("heading", ""), "score": h["score"]})
    else:
        sources = []
    return {"answer": result["answer"], "sources": sources, "topic": result["topic"]}


if __name__ == "__main__":
    print("\033[92m kb/graph.py 文件\033[0m")
    g = build_graph()
    print("图节点:", list(g.get_graph().nodes))
    r = ask("知识库里有什么内容？")
    print("回答:", r["answer"])
    print("来源:", r["sources"])
