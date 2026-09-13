"""知识图谱构建。

两层：
1. 元数据层（确定性，零 LLM）：doc - topic - tag 节点与边，来自台账 doc_index
2. 实体层（LLM 增量抽取）：每篇文档抽 (head, relation, tail) 三元组，
   只处理新增/变化文档（doc_meta 里 hash 对比），单篇失败静默跳过不中断。
"""
from __future__ import annotations

import json
import re
import time

from ..config import kg_cfg, raw_dir
from ..llm import LLMClient
from ..prompts import KG_EXTRACT_PROMPT
from ..storage import doc_index
from . import store

_MAX_EXTRACT_CHARS = 2000   # 每篇文档送入抽取的正文上限（标题在前）
_MAX_TRIPLES = 6            # 每篇最多抽几条


# ---------- 元数据层 ----------

def _metadata_graph(rows: dict) -> tuple[list, list]:
    """从台账 rows={rel: {topic, tags,...}} 构建 doc/topic/tag 节点与边。"""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_n: set = set()
    seen_e: set = set()

    def add_node(nid: str, name: str, typ: str, topic: str = "") -> None:
        if nid not in seen_n:
            seen_n.add(nid)
            nodes.append({"id": nid, "name": name, "type": typ, "topic": topic})

    def add_edge(src: str, dst: str, rel: str) -> None:
        key = (src, dst, rel)
        if key not in seen_e:
            seen_e.add(key)
            edges.append({"src": src, "dst": dst, "rel": rel})

    for rel, info in rows.items():
        topic = info.get("topic", "默认")
        doc_id = f"doc:{rel}"
        add_node(doc_id, rel.rsplit("/", 1)[-1], "doc", topic=topic)
        add_node(f"topic:{topic}", topic, "topic")
        add_edge(doc_id, f"topic:{topic}", "属于")
        for t in (info.get("tags") or []):
            t = str(t).strip()
            if not t:
                continue
            add_node(f"tag:{t}", t, "tag")
            add_edge(doc_id, f"tag:{t}", "含标签")
    return nodes, edges


# ---------- 实体层（LLM 抽取） ----------

def _read_head(f) -> str:
    """读文件前 _MAX_EXTRACT_CHARS 字符（Markdown 去代码块，省 token）。"""
    try:
        text = f.read_text(encoding="utf-8")
    except Exception:
        return ""
    # 去掉 ``` 代码块（占 token 且实体稀疏）
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    return text[:_MAX_EXTRACT_CHARS]


def _parse_triples(content: str) -> list[tuple]:
    """解析 LLM 输出 JSON，容错：剥 code block、正则取第一个 {...}。"""
    if not content:
        return []
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        return []
    out: list[tuple] = []
    for tr in (data.get("triples") or [])[:_MAX_TRIPLES]:
        h = str(tr.get("head", "")).strip()
        r = str(tr.get("rel", "")).strip()
        t = str(tr.get("tail", "")).strip()
        if h and r and t and len(h) <= 20 and len(t) <= 20:
            out.append((h, r, t))
    return out


def _extract_one(llm: LLMClient, rel: str) -> list[tuple]:
    """对单篇文档抽三元组；任何异常返回 []（静默降级）。"""
    f = raw_dir() / rel
    if not f.exists():
        return []
    text = _read_head(f)
    if len(text) < 20:
        return []
    content = llm.chat([
        {"role": "system", "content": KG_EXTRACT_PROMPT},
        {"role": "user", "content": text},
    ])
    return _parse_triples(content)


def _entity_layer(g: dict, rows: dict, verbose: bool) -> tuple[int, list, list, dict]:
    """增量抽取实体三元组，返回 (新增三元组数, 节点, 边, 新 doc_meta)。"""
    old_meta = g.get("doc_meta", {}) or {}
    new_meta = dict(old_meta)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_n: set = {nd["id"] for nd in g.get("nodes", [])}
    seen_e: set = {(e["src"], e["dst"], e["rel"]) for e in g.get("edges", [])}
    todo = [rel for rel, info in rows.items()
            if old_meta.get(rel, {}).get("hash") != info.get("hash")]
    total = 0

    if not todo:
        if verbose:
            print("[kg] 实体层无新增/变化文档，跳过抽取")
        return 0, [], [], new_meta

    llm = LLMClient()
    for i, rel in enumerate(todo):
        if verbose:
            print(f"[kg] 抽取实体 {i + 1}/{len(todo)} {rel}")
        triples = _extract_one(llm, rel)
        new_meta[rel] = {"hash": rows[rel].get("hash", ""), "triples": len(triples)}
        total += len(triples)
        for h, r, t in triples:
            hid, tid = f"ent:{h}", f"ent:{t}"
            if hid not in seen_n:
                seen_n.add(hid)
                nodes.append({"id": hid, "name": h, "type": "entity", "topic": ""})
            if tid not in seen_n:
                seen_n.add(tid)
                nodes.append({"id": tid, "name": t, "type": "entity", "topic": ""})
            if (f"doc:{rel}", hid, "提及") not in seen_e:
                seen_e.add((f"doc:{rel}", hid, "提及"))
                edges.append({"src": f"doc:{rel}", "dst": hid, "rel": "提及"})
            if (hid, tid, r) not in seen_e:
                seen_e.add((hid, tid, r))
                edges.append({"src": hid, "dst": tid, "rel": r})
    return total, nodes, edges, new_meta


# ---------- 总入口 ----------

def build(extract_entities: bool | None = None, verbose: bool = True) -> dict:
    """构建/增量更新知识图谱。extract_entities=None 时按配置 kg.extract_entities。"""
    rows = doc_index.load()
    if not rows:
        print("[kg] 台账为空（还没入库？），图谱构建中止")
        return {"nodes": 0, "edges": 0, "triples": 0, "docs": 0}

    nodes, edges = _metadata_graph(rows)
    doc_meta: dict = {}
    triples_total = 0

    extract = kg_cfg().get("extract_entities", False)
    if extract_entities is not None:
        extract = extract_entities
    if extract:
        n, ent_nodes, ent_edges, doc_meta = _entity_layer(store.load(), rows, verbose)
        nodes += ent_nodes
        edges += ent_edges
        triples_total = n

    g = {
        "meta": {"version": 1, "updated": time.time(),
                 "docs": len(rows), "triples": triples_total},
        "nodes": nodes, "edges": edges, "doc_meta": doc_meta,
    }
    store.save(g)
    if verbose:
        print(f"[kg] 图谱构建完成：{len(nodes)} 节点 / {len(edges)} 边 / "
              f"{triples_total} 新三元组 / {len(rows)} 文档")
    return {"nodes": len(nodes), "edges": len(edges),
            "triples": triples_total, "docs": len(rows)}
