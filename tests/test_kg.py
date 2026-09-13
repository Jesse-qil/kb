# -*- coding: utf-8 -*-
"""知识图谱测试：元数据层构建 + 存储读写 + 检索增强（不依赖真实知识库与 LLM）。"""
from __future__ import annotations

import json

from kb.kg import build, retrieve, store


def test_metadata_graph():
    """元数据层：doc/topic/tag 节点与属于/含标签边，零 LLM。"""
    rows = {
        "AI-Agent/01.md": {"topic": "AI-Agent", "tags": ["LangGraph", "Agent"]},
        "AI-Agent/02.md": {"topic": "AI-Agent", "tags": ["LangGraph"]},
    }
    nodes, edges = build._metadata_graph(rows)
    types = {nd["type"] for nd in nodes}
    assert types == {"doc", "topic", "tag"}
    # 2 条"属于" + 3 条"含标签"
    assert len(edges) == 5
    assert {"src": "doc:AI-Agent/01.md", "dst": "topic:AI-Agent",
            "rel": "属于"} in edges
    assert {"src": "doc:AI-Agent/02.md", "dst": "tag:LangGraph",
            "rel": "含标签"} in edges


def test_store_roundtrip(tmp_path, monkeypatch):
    """图谱存储：save → load 一致；损坏文件返回空结构（不抛异常）。"""
    monkeypatch.setattr(store, "_GRAPH_FILE", tmp_path / "kg.json")
    g = {"meta": {"docs": 1}, "nodes": [{"id": "a", "name": "A", "type": "doc"}],
         "edges": [], "doc_meta": {}}
    store.save(g)
    out = store.load()
    assert out["nodes"][0]["id"] == "a"
    assert "updated" in out["meta"]

    (tmp_path / "kg.json").write_text("{bad json", encoding="utf-8")
    assert store.load()["nodes"] == []


def _fake_graph() -> dict:
    """构造一个小型图谱：两篇文档，标签+实体各一。"""
    return {
        "meta": {"docs": 2},
        "nodes": [
            {"id": "doc:AI-Agent/01.md", "name": "01.md", "type": "doc",
             "topic": "AI-Agent"},
            {"id": "doc:AI-Agent/02.md", "name": "02.md", "type": "doc",
             "topic": "AI-Agent"},
            {"id": "tag:LangGraph", "name": "LangGraph", "type": "tag"},
            {"id": "ent:Agent", "name": "Agent", "type": "entity"},
        ],
        "edges": [
            {"src": "doc:AI-Agent/01.md", "dst": "tag:LangGraph", "rel": "含标签"},
            {"src": "doc:AI-Agent/02.md", "dst": "ent:Agent", "rel": "提及"},
        ],
        "doc_meta": {},
    }


def test_expand_sources(tmp_path, monkeypatch):
    """检索增强：问题命中标签/实体 → 拉出关联文档。"""
    f = tmp_path / "kg.json"
    f.write_text(json.dumps(_fake_graph(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(store, "_GRAPH_FILE", f)

    out = retrieve.expand_sources("LangGraph 和 Agent 怎么用", limit=5)
    assert "AI-Agent/01.md" in out
    assert "AI-Agent/02.md" in out


def test_expand_sources_no_hit(tmp_path, monkeypatch):
    """无实体命中 → 空列表（图谱通道不干扰主链路）。"""
    f = tmp_path / "kg.json"
    f.write_text(json.dumps(_fake_graph(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(store, "_GRAPH_FILE", f)
    assert retrieve.expand_sources("今天天气怎么样", limit=5) == []


def test_expand_sources_missing_graph(tmp_path, monkeypatch):
    """图谱文件不存在 → 空列表。"""
    monkeypatch.setattr(store, "_GRAPH_FILE", tmp_path / "none.json")
    assert retrieve.expand_sources("LangGraph", limit=5) == []
