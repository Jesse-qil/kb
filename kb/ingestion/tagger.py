# -*- coding: utf-8 -*-
"""文档标签器：根据文档内容给出 3-6 个稳定标签。

三条路（结果合并，取并集）：
- LLM 零样本打标（按 tag schema 候选）
- embedding 语义打标：标签文本向量化 vs 文档向量算余弦（bge 可用时才启用）
- 本地启发式规则兜底，保证永远不空
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import tag_schema_file
from ..llm import LLMClient

_MAX_TAGS = 6
_DEFAULT_SCHEMA = {
    "version": 1,
    "allowed_tags": [
        "Python",
        "Python基础",
        "语法",
        "函数",
        "装饰器",
        "面向对象",
        "集合",
        "文件处理",
        "正则",
        "Web",
        "FastAPI",
        "CLI",
        "前端",
        "数据结构",
        "算法",
        "面试",
        "项目",
        "笔记",
        "总结",
        "资料",
        "LLM",
        "RAG",
        "Agent",
        "LangGraph",
        "向量检索",
        "工具调用",
        "会话记忆",
        "文档处理",
        "Markdown",
        "PDF",
    ],
    "fallback_tags": ["笔记", "资料", "项目"],
    "topic_hints": {
        "notes_draft": ["笔记", "总结"],
        "reference": ["资料", "教程"],
        "project_material": ["项目", "实现", "调试"],
        "AI-Agent": ["LLM", "Agent", "RAG"],
    },
}

_TAG_RE = re.compile(r'[^\w\u4e00-\u9fff+-]+')


def _load_schema() -> dict:
    p = tag_schema_file()
    if not p.exists():
        return _DEFAULT_SCHEMA
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _DEFAULT_SCHEMA
        data.setdefault("allowed_tags", _DEFAULT_SCHEMA["allowed_tags"])
        data.setdefault("fallback_tags", _DEFAULT_SCHEMA["fallback_tags"])
        data.setdefault("topic_hints", _DEFAULT_SCHEMA["topic_hints"])
        return data
    except Exception:
        return _DEFAULT_SCHEMA


def _clean_tag(tag: str) -> str:
    tag = _TAG_RE.sub("", str(tag or "").strip())
    return tag[:24]


def _dedupe(tags: list[str]) -> list[str]:
    out = []
    seen = set()
    for tag in tags:
        clean = _clean_tag(tag)
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out[:_MAX_TAGS]


def _heuristic_tags(text: str, title: str = "", topic: str = "") -> list[str]:
    schema = _load_schema()
    blob = "\n".join(filter(None, [title, topic, text[:2000]])).lower()
    tags: list[str] = []

    def add(tag: str) -> None:
        clean = _clean_tag(tag)
        if clean and clean not in tags:
            tags.append(clean)

    if topic:
        add(topic)
    for hint in schema.get("topic_hints", {}).get(topic, []):
        add(hint)

    for tag in schema.get("allowed_tags", []):
        if not tag:
            continue
        low = str(tag).lower()
        if low in blob or str(tag) in text or str(tag) in title or str(tag) == topic:
            add(tag)

    if len(tags) < 3:
        for tag in schema.get("fallback_tags", []):
            add(tag)

    if len(tags) < 3:
        for tag in schema.get("allowed_tags", []):
            add(tag)
            if len(tags) >= 3:
                break

    return _dedupe(tags)


def _extract_json_blob(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


# ---------- embedding 语义打标通道 ----------
# 标签向量缓存：{tags元组: 向量列表}，标签表不变就只算一次
_TAG_VEC_CACHE: dict[tuple, list[list[float]]] = {}


def embedding_tags(text: str, schema: dict, top_n: int = 5,
                   threshold: float = 0.40) -> list[str]:
    """语义打标：把候选标签（可带描述）和文档分别向量化，余弦超阈值即入选。
    bge 不可用（hash 降级）时返回空列表——hash 向量没有语义，打了也是噪声。"""
    from .. import embedding as E
    if E._use_fallback:
        return []
    tags = [str(t) for t in schema.get("allowed_tags", []) if t]
    if not tags:
        return []
    try:
        key = tuple(tags)
        if key not in _TAG_VEC_CACHE:
            descs = schema.get("tag_descriptions", {}) or {}
            texts = [f"{t}：{descs[t]}" if descs.get(t) else t for t in tags]
            _TAG_VEC_CACHE[key] = E.embed_texts(texts)
        tvecs = _TAG_VEC_CACHE[key]
        dvec = E.embed_texts([text[:1000]])[0]
        scored = sorted(
            ((t, sum(a * b for a, b in zip(dvec, v))) for t, v in zip(tags, tvecs)),
            key=lambda x: -x[1])
        return [t for t, s in scored if s >= threshold][:top_n]
    except Exception:
        return []


def suggest_tags(text: str, title: str = "", topic: str = "") -> list[str]:
    """根据文档内容给出标签列表：LLM + embedding 语义 + 启发式三路合并。
    LLM/embedding 都失效时启发式兜底，保证不空。"""
    schema = _load_schema()
    fallback = _heuristic_tags(text, title=title, topic=topic)

    # 1) LLM 零样本（失败返回空，不抛）
    llm_tags: list[str] = []
    try:
        llm = LLMClient()
        if llm.provider != "mock":
            prompt = (
                "你是知识库打标器。根据文档内容输出 JSON，只能是 "
                '{"tags":["标签1","标签2"]}。'
                f"尽量从候选标签里选 3-5 个；如果候选都不合适，可以补充少量新标签。"
                f"\n候选标签：{', '.join(schema.get('allowed_tags', []))}"
                f"\n当前主题：{topic or '默认'}"
                f"\n文档标题：{title or '（无）'}"
                f"\n文档开头：\n{text[:1500]}"
            )
            out = llm.chat([
                {"role": "system", "content": "你只输出 JSON。"},
                {"role": "user", "content": prompt},
            ])
            raw_tags = _extract_json_blob(out).get("tags", [])
            if isinstance(raw_tags, list):
                llm_tags = _dedupe([str(t) for t in raw_tags])
    except Exception:
        llm_tags = []

    # 2) embedding 语义通道（与 LLM 结果互补，LLM 漏掉的补上）
    emb_tags = embedding_tags(text, schema)

    # 3) 合并：LLM 优先，embedding 补充，都弱时启发式兜底
    merged = _dedupe(llm_tags + emb_tags)
    if len(merged) < 3:
        merged = _dedupe(merged + fallback)
    return merged or fallback
