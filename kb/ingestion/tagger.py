# -*- coding: utf-8 -*-
"""文档标签器：根据文档内容给出 3-6 个稳定标签。

主通道（v2，标签自治）：
- 标签向量库检索：文档向量 → tag_store 检索 Top5
  - 最高相似度 ≥ 0.60 → 直接复用（不调 LLM）
  - 0.45 ~ 0.60 → LLM 从候选标签里确认/补充（低成本）
  - < 0.45 → LLM 生成新标签，写入标签库（后续内容即可检索到）
- 本地启发式规则兜底，保证永远不空
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import tag_schema_file
from ..llm import LLMClient
from . import tag_store

_MAX_TAGS = 6
# 分层阈值（由 scripts/tag_sim_probe.py 实测校准：
# 101 篇文档与 35 个标签的 Top1 相似度 p50=0.597、max=0.741，0.60 可覆盖 ~47%）
_REUSE_MIN = 0.60   # 最高相似度 ≥ 此值 → 直接复用标签，不调 LLM
_GRAY_MIN = 0.45    # 最高相似度 < 此值 → 全新领域，LLM 生成新标签并写回
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


# ---------- embedding 语义打标通道（v1，已由 tag_store 标签库检索取代） ----------
# 保留函数定义仅为兼容历史 import；新流程不再调用。
_TAG_VEC_CACHE: dict[tuple, list[list[float]]] = {}


def embedding_tags(text: str, schema: dict, top_n: int = 5,
                   threshold: float = 0.40) -> list[str]:
    """v1 语义打标：对固定候选表做相似度。已被 tag_store.retrieve 取代——
    标签库检索覆盖候选表 + 动态新增标签，功能等价且不依赖写死的候选表。"""
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


def _llm_suggest(text: str, title: str, topic: str,
                 candidates: list[tuple[str, float]], prefer_existing: bool = True) -> list[str]:
    """LLM 打标：候选标签来自标签库检索结果。
    prefer_existing=True（灰色地带）：优先从候选里选，不够才补新标签；
    prefer_existing=False（全新领域）：允许放开生成，但语义相同必须复用候选。
    返回空列表表示失败（调用方走启发式兜底）。
    """
    schema = _load_schema()
    try:
        llm = LLMClient()
        if llm.provider == "mock":
            return []
        cand_txt = ", ".join(f"{t}(相似度{s:.2f})" for t, s in candidates) if candidates else "（无）"
        if prefer_existing:
            instruction = "优先从候选标签里选 3-5 个；候选不够合适可补充少量新标签，避免近义重复。"
        else:
            instruction = ("这是全新领域。按内容生成 3-5 个精准标签；"
                           "若与候选标签语义相同，必须复用候选标签，不要造近义词。")
        prompt = (
            "你是知识库打标器。根据文档内容输出 JSON，只能是 "
            '{"tags":["标签1","标签2"]}。'
            f"\n{instruction}"
            f"\n候选标签（来自标签库检索，越靠前越相似）：{cand_txt}"
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
            return _dedupe([str(t) for t in raw_tags])
    except Exception:
        pass
    return []


def suggest_tags(text: str, title: str = "", topic: str = "") -> list[str]:
    """根据文档内容给出标签列表（v2 标签自治流程）：

    1. 文档向量 → 标签向量库检索 Top5
    2. 最高相似度 ≥ 0.60 → 直接复用达标标签（免 LLM，复用不足 3 个由启发式补）
    3. 0.45 ~ 0.60 → LLM 从候选标签确认/补充（低成本）
    4. < 0.45 → LLM 生成新标签（写回标签库，后续内容可检索到）
    LLM/标签库都失效时启发式兜底，保证不空。
    """
    fallback = _heuristic_tags(text, title=title, topic=topic)

    # 1) 标签库检索（主通道：覆盖种子候选 + 动态新增标签）
    hits = tag_store.retrieve(text, top_k=5)
    llm_tags: list[str] = []

    if hits and hits[0][1] >= _REUSE_MIN:
        # 2) 高分复用：不调 LLM。复用全部达标标签，不足 3 个由启发式补足
        reuse = [t for t, s in hits if s >= _REUSE_MIN]
        merged = _dedupe(reuse + fallback)
        return merged or fallback

    # 3/4) 灰色地带或全新领域：LLM 参与
    if hits:
        prefer_existing = hits[0][1] >= _GRAY_MIN
        llm_tags = _llm_suggest(text, title, topic, hits, prefer_existing=prefer_existing)
    else:
        llm_tags = _llm_suggest(text, title, topic, [], prefer_existing=False)

    # 新标签写回标签库（自治：下次任何文档都能检索到）
    if llm_tags:
        known = {t for t, _ in hits}
        new_tags = [t for t in llm_tags if t not in known]
        if new_tags:
            tag_store.add_tags(new_tags, source=f"{topic}/{title}" if title else topic)

    merged = _dedupe(llm_tags)
    if len(merged) < 3:
        merged = _dedupe(merged + fallback)
    return merged or fallback
