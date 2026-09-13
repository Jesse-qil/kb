"""图谱检索增强：问题 → 实体/标签命中 → 关联文档 source 列表。

原理：把"问题和图谱的匹配"当第三路召回通道（向量 + BM25 之外）。
问题里出现的实体名/标签名（如"LangGraph"、"装饰器"）在图中命中节点，
沿 实体/标签 --提及/含标签--> 文档 的边拉出关联文档，片段加权进候选池。
治"标签对、但向量分排不进前排"的漏检。
"""
from __future__ import annotations

from ..config import kg_cfg
from . import store


def _hit_terms(question: str, g: dict) -> list[tuple[str, str]]:
    """问题中包含的 entity/tag 节点名（>=2 字），返回 [(node_id, name)]。"""
    terms: list[tuple[str, str]] = []
    for nd in g.get("nodes", []):
        if nd.get("type") not in ("entity", "tag"):
            continue
        name = str(nd.get("name", "")).strip()
        if len(name) >= 2 and name in question:
            terms.append((nd["id"], name))
    return terms


def expand_sources(question: str, topic: str = "", limit: int = 0) -> list[str]:
    """返回与问题关联的文档 source 列表（按关联文档数排序）。
    图谱不存在/无命中 → 空列表（调用方当没有图谱通道）。
    """
    if not question or not store.graph_file().exists():
        return []
    g = store.load()
    nodes = g.get("nodes") or []
    edges = g.get("edges") or []
    if not nodes or not edges:
        return []
    limit = limit or int(kg_cfg().get("expand_top_n", 5))

    terms = _hit_terms(question, g)
    if not terms:
        return []
    term_ids = {t[0] for t in terms}

    # 一跳：实体/标签节点 → 相邻文档节点（边方向 doc->node 或 node->doc 都算）
    doc_hits: dict[str, int] = {}   # doc_id -> 命中实体数
    for e in edges:
        src, dst = e.get("src", ""), e.get("dst", "")
        if src.startswith("doc:"):
            if dst in term_ids:
                doc_hits[src] = doc_hits.get(src, 0) + 1
        elif dst.startswith("doc:"):
            if src in term_ids:
                doc_hits[dst] = doc_hits.get(dst, 0) + 1

    # 二跳：实体-实体关系（head ->rel-> tail），从命中实体经关系边到相邻实体 → 其提及文档
    if doc_hits:
        neighbor_ents: set[str] = set()
        for e in edges:
            if e.get("src") in term_ids and not e.get("dst", "").startswith("doc:"):
                neighbor_ents.add(e["dst"])
        for e in edges:
            if e.get("src") in neighbor_ents and e.get("dst", "").startswith("doc:"):
                doc_hits[e["dst"]] = doc_hits.get(e["dst"], 0) + 1

    # 按命中数排序取前 limit，映射回 source 路径
    ordered = sorted(doc_hits.items(), key=lambda kv: -kv[1])
    sources: list[str] = []
    for doc_id, _ in ordered:
        src = doc_id[len("doc:"):]
        if topic and not src.startswith(topic + "/"):
            continue
        if src not in sources:
            sources.append(src)
        if len(sources) >= limit:
            break
    return sources
