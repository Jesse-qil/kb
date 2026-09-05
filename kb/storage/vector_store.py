"""向量库封装：只管"片段怎么存、怎么按向量找"。
纯存储职责：不知道文档从哪来、怎么分块（那是 ingestion 的事）。

优先使用 Chroma；如果环境里没有 chromadb，则自动退到本地 JSON 存储。
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import chromadb
except Exception:
    chromadb = None

from ..config import KNOWLEDGE_DIR, chroma_dir, recall_cfg
from ..embedding import embed_query

_COLLECTION = "notes"
_FALLBACK_FILE = KNOWLEDGE_DIR / "vector_store" / "fallback_store.json"


def _client():
    if chromadb is None:
        return None
    d = chroma_dir()
    d.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(d))


def _collection():
    client = _client()
    if client is None:
        return None
    return client.get_or_create_collection(
        _COLLECTION, metadata={"hnsw:space": "cosine"})


def _load_store() -> dict:
    f = _FALLBACK_FILE
    if not f.exists():
        return {"chunks": {}}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"chunks": {}}
        data.setdefault("chunks", {})
        return data
    except Exception:
        return {"chunks": {}}


def _save_store(data: dict) -> None:
    f = _FALLBACK_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_matches(stored_source: str, requested_source: str, topic: str = "") -> bool:
    if stored_source == requested_source:
        return True
    if topic and stored_source == f"{topic}/{requested_source}":
        return True
    if stored_source.endswith("/" + requested_source):
        if not topic:
            return True
        return stored_source.startswith(topic + "/")
    return False


def delete_by_source(source: str, topic: str = "") -> None:
    """删除某来源文件的片段。
    source 应传相对路径（topic/.../文件名，全局唯一）。
    topic 可选：用于兼容旧格式数据。
    """
    col = _collection()
    if col is not None:
        if topic:
            where = {"$and": [{"source": source}, {"topic": topic}]}
        else:
            where = {"source": source}
        col.delete(where=where)
        return

    store = _load_store()
    kept = {}
    for cid, row in store["chunks"].items():
        md = row.get("metadata", {})
        stored_source = md.get("source", "")
        stored_topic = md.get("topic", "")
        if _source_matches(stored_source, source, topic=topic) and (not topic or stored_topic == topic):
            continue
        kept[cid] = row
    store["chunks"] = kept
    _save_store(store)


def _tags_str(tags) -> str:
    """标签列表 → 逗号拼接字符串（chroma metadata 只收标量）。"""
    if not tags:
        return ""
    return ",".join(str(t).strip() for t in tags if str(t).strip())


def _tags_list(md: dict) -> list[str]:
    """metadata 里的 tags 字符串 → 列表。"""
    raw = md.get("tags", "") or ""
    return [t for t in raw.split(",") if t] if raw else []


def _tag_bool_md(tags) -> dict:
    """每个标签一个布尔键（t:标签名），供 chroma where 精确过滤。"""
    out = {}
    for t in (tags or []):
        t = str(t).strip()
        if t:
            out[f"t:{t}"] = True
    return out


def _metadata_for(c: dict) -> dict:
    md = {"source": c["source"], "heading": c["heading"], "topic": c["topic"],
          "tags": _tags_str(c.get("tags"))}
    md.update(_tag_bool_md(c.get("tags")))
    return md


def upsert(chunks: list[dict]) -> None:
    """写入/更新片段。chunks 元素需含 id/text/source/heading/topic/embedding，
    可选 tags（标签列表，进 metadata 供检索加权）。"""
    col = _collection()
    if col is not None:
        col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[_metadata_for(c) for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
        )
        return

    store = _load_store()
    for c in chunks:
        store["chunks"][c["id"]] = {
            "document": c["text"],
            "metadata": _metadata_for(c),
            "embedding": c["embedding"],
        }
    _save_store(store)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


_TAG_BOOST = 0.04        # 每命中一个标签的加分
_TAG_BOOST_CAP = 0.12    # 加分上限（3 个标签封顶，防止标签刷分压过语义）


def _rerank_by_tags(hits: list[dict], tags: list[str]) -> None:
    """标签软加权：命中标签越多的片段加分越多，原地重排。
    加分直接并入 score（融合分）：排序、下游阈值判断用同一把尺子。"""
    tset = {t for t in tags if t}
    if not tset:
        return
    for h in hits:
        overlap = len(tset & set(h.get("tags") or []))
        h["score"] = round(h["score"] + min(_TAG_BOOST * overlap, _TAG_BOOST_CAP), 4)
    hits.sort(key=lambda h: h["score"], reverse=True)


def _merge_hits(hits: list[dict], pool: dict) -> None:
    """按 (source, 文本前30字) 去重合并进 pool。"""
    for h in hits:
        key = (h["source"], h["text"][:30])
        if key not in pool:
            pool[key] = h


def _search_by_tags(question: str, tags: list[str], k: int,
                    topic: str = "") -> list[dict]:
    """标签召回通道：只在这几个标签的片段里做向量搜索。
    保证"标签对、但向量分排不进前排"的片段（如单 chunk 小文档）
    一定进候选池，再由 _rerank_by_tags 统一排序。"""
    clean = [t for t in tags if t]
    if not clean:
        return []
    qv = embed_query(question)
    col = _collection()
    if col is not None:
        total = col.count()
        if not total:
            return []
        conds = [{"t:" + t: True} for t in clean]
        where = conds[0] if len(conds) == 1 else {"$or": conds}
        if topic:
            where = {"$and": [where, {"topic": topic}]}
        n = max(1, min(k, total))
        try:
            res = col.query(query_embeddings=[qv], n_results=n,
                            include=["documents", "metadatas", "distances"], where=where)
        except Exception:
            return []
        hits = []
        for i in range(len(res["ids"][0])):
            md = res["metadatas"][0][i]
            hits.append({
                "text": res["documents"][0][i],
                "source": md["source"],
                "topic": md.get("topic", ""),
                "heading": md.get("heading", ""),
                "tags": _tags_list(md),
                "score": round(1 - res["distances"][0][i], 4),
            })
        return hits

    # JSON 降级存储：客户端过滤
    store = _load_store()
    hits = []
    tset = set(clean)
    for row in store["chunks"].values():
        md = row.get("metadata", {})
        if topic and md.get("topic", "") != topic:
            continue
        if not (tset & set(_tags_list(md))):
            continue
        emb = row.get("embedding")
        if not emb:
            continue
        hits.append({
            "text": row.get("document", ""),
            "source": md.get("source", "未知"),
            "topic": md.get("topic", ""),
            "heading": md.get("heading", ""),
            "tags": _tags_list(md),
            "score": round(_dot(qv, emb), 4),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def query(question: str, topic: str = "", top_k: int = 0,
          tags: list[str] | None = None) -> list[dict]:
    """检索：问题 → 向量 → 相似度搜索。
    topic 非空则按领域过滤；tags 非空则双通道召回：
    1) 常规向量搜索（候选池扩到 k*5）
    2) 标签过滤搜索（带标签的片段保底进池）
    合并后按标签重叠加分重排（软加权：归错目录/标签不全也不会丢内容）。"""
    col = _collection()
    k = top_k or recall_cfg()["top_k"]
    fetch = k * 5 if tags else k
    pool: dict = {}

    if col is not None:
        kwargs = {}
        if topic:
            kwargs["where"] = {"topic": topic}
        total = col.count()
        n = max(1, min(fetch, total)) if total else 1
        res = col.query(query_embeddings=[embed_query(question)],
                        n_results=n, include=["documents", "metadatas", "distances"], **kwargs)
        for i in range(len(res["ids"][0])):
            md = res["metadatas"][0][i]
            _merge_hits([{
                "text": res["documents"][0][i],
                "source": md["source"],
                "topic": md.get("topic", ""),
                "heading": md.get("heading", ""),
                "tags": _tags_list(md),
                "score": round(1 - res["distances"][0][i], 4),
            }], pool)
    else:
        qv = embed_query(question)
        for row in _load_store()["chunks"].values():
            md = row.get("metadata", {})
            if topic and md.get("topic", "") != topic:
                continue
            emb = row.get("embedding")
            if not emb:
                continue
            _merge_hits([{
                "text": row.get("document", ""),
                "source": md.get("source", "未知"),
                "topic": md.get("topic", ""),
                "heading": md.get("heading", ""),
                "tags": _tags_list(md),
                "score": round(_dot(qv, emb), 4),
            }], pool)

    if tags:
        _merge_hits(_search_by_tags(question, tags, k, topic=topic), pool)
        hits = list(pool.values())
        _rerank_by_tags(hits, tags)
        return hits[:k]

    hits = sorted(pool.values(), key=lambda h: -h["score"])
    return hits[:k]


def count() -> int:
    """库里片段总数。"""
    col = _collection()
    if col is not None:
        return col.count()
    return len(_load_store()["chunks"])


def list_contents() -> dict:
    """返回 {主题: [文件名...]}，用于"库里有什么"。"""
    col = _collection()
    result = {}
    if col is not None:
        data = col.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        for md in metadatas:
            t, s = md.get("topic", "默认"), md.get("source", "未知")
            if "/" in s and s.split("/", 1)[0] == t:
                s = s.split("/", 1)[1]
            result.setdefault(t, [])
            if s not in result[t]:
                result[t].append(s)
        return result

    store = _load_store()
    for row in store["chunks"].values():
        md = row.get("metadata", {})
        t, s = md.get("topic", "默认"), md.get("source", "未知")
        if "/" in s and s.split("/", 1)[0] == t:
            s = s.split("/", 1)[1]
        result.setdefault(t, [])
        if s not in result[t]:
            result[t].append(s)
    return result
