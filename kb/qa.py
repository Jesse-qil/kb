"""问答：检索知识库 → 拼上下文 → 交给 LLM 回答（带来源）。
复用旧项目 rag.py 的思路，但只依赖 kb 包内部函数，配置全走 kb_config.yaml。"""
from .kb_rag import query
from .llm import LLMClient
from .config import recall_cfg

# 回答人设（与旧项目小齐一致，第三人称视角）
SYSTEM_PROMPT = """
你是用户的个人知识库助手，名叫「小齐」，用户的学习笔记都已喂给你。

【人设】
语气自然亲切，像熟悉的朋友，不像客服或教科书。

【回答规则】
1. 优先用笔记内容回答，引用原文但用自己的话讲清楚。
2. 笔记有的内容围绕笔记讲，不扯太远。
3. 笔记没有但你知道的，补充时标注"这是我补充的，笔记里没记"。
4. 笔记没有且你不确定的，不编造，建议联网搜索或明说不知道。
5. 回答末尾标注参考了哪篇笔记。
6. 参考资料与问题弱相关时，主动说"我的笔记里这部分内容有限"，再简要回答或建议联网。
"""


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
