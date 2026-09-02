"""问答：检索知识库 → 拼上下文 → 交给 LLM 回答（带来源）。
复用旧项目 rag.py 的思路，但只依赖 kb 包内部函数，配置全走 kb_config.yaml。"""
from ..storage.vector_store import query
from ..llm import LLMClient
from ..config import recall_cfg
from ..prompts import KB_ASSISTANT_PROMPT as SYSTEM_PROMPT


def ask(question: str, topic: str = "") -> dict:
    """问一句 → 返回 {"answer":..., "sources":[...]}。"""
    hits = query(question, topic=topic)
    threshold = recall_cfg()["score_threshold"]

    # 弱相关（最高分低于阈值）或没查到 → 不给弱资料
    used_kb = bool(hits) and hits[0]["score"] >= threshold
    if used_kb:
        context = "\n\n".join(f"[来自 {h['source']}]\n{h['text']}" for h in hits)
        user = f"问题：{question}\n\n参考资料：\n{context}"
    else:
        user = (f"问题：{question}\n\n"
                "（知识库里没有找到相关资料，请如实说明，可以凭常识简单回答，但要说这不是笔记内容）")

    answer = LLMClient().chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])

    if used_kb:
        seen, sources = set(), []
        for h in hits:
            key = (h["source"], h["text"][:20])
            if key not in seen:
                seen.add(key)
                sources.append({"source": h["source"], "heading": h["heading"], "score": h["score"]})
    else:
        sources = []
    return {"answer": answer, "sources": sources}
