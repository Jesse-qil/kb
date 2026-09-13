"""知识图谱存储层：knowledge/graph/kg.json 的读写。

图结构：
{
  "meta": {"version": 1, "updated": 123, "docs": 131, "triples": 320},
  "nodes": [{"id": "doc:相对路径", "name": "...", "type": "doc|topic|tag|entity", "topic": "..."}],
  "edges": [{"src": "node_id", "dst": "node_id", "rel": "属于|含标签|提及|... "}],
  "doc_meta": {"相对路径": {"hash": "...", "triples": 5}},   # 实体抽取增量用
}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import KNOWLEDGE_DIR
_GRAPH_DIR = KNOWLEDGE_DIR / "graph"
_GRAPH_FILE = _GRAPH_DIR / "kg.json"


def _empty() -> dict:
    return {"meta": {"version": 1, "updated": 0, "docs": 0, "triples": 0},
            "nodes": [], "edges": [], "doc_meta": {}}


def load() -> dict:
    """读图谱；文件不存在/损坏返回空结构（绝不抛异常）。"""
    if not _GRAPH_FILE.exists():
        return _empty()
    try:
        data = json.loads(_GRAPH_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _empty()
        for k in ("meta", "nodes", "edges", "doc_meta"):
            data.setdefault(k, _empty()[k])
        return data
    except Exception:
        return _empty()


def save(g: dict) -> None:
    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    g.setdefault("meta", {})
    g["meta"]["updated"] = time.time()
    _GRAPH_FILE.write_text(
        json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")


def graph_file() -> Path:
    return _GRAPH_FILE
