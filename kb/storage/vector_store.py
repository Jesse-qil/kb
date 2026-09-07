"""向量库封装：只管"片段怎么存、怎么按向量找"。
纯存储职责：不知道文档从哪来、怎么分块（那是 ingestion 的事）。

优先使用 Chroma；如果环境里没有 chromadb，则自动退到本地 JSON 存储。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import chromadb
except Exception:
    chromadb = None

try:
    import jieba
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    jieba = None
    BM25Okapi = None
    _HAS_BM25 = False

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


# ---------- 混合检索：BM25 关键词通道 ----------
# 与向量通道双路召回：专有名词/术语（LangChain、pydantic…）靠关键词精确命中保底进池，
# 再由 _rerank_by_bm25 归一化融合。无 jieba/rank_bm25 时自动降级为纯向量（_HAS_BM25=False）。
_bm25_cache: dict = {"count": -1, "bm25": None, "texts": [], "metas": [], "lookup": {}}

# 查询分词停用词：虚词 + 高频疑问词。BM25 对这类词无区分力，只会抬噪音片段。
_BM25_STOPWORDS = {
    "的", "了", "和", "与", "或", "及", "是", "在", "吗", "呢", "啊", "吧",
    "个", "中", "上", "下", "里", "这", "那", "有", "用", "把", "被", "对",
    "如何", "怎么", "怎样", "什么", "为什么", "哪些", "哪个", "区别", "之间",
    "比较", "对比", "介绍", "讲解", "说说", "请问", "一下",
}


_jieba_loaded = False


def _load_jieba_userdict() -> None:
    """把知识库标签词注入 jieba 用户词典（'装饰器' 不被误切成 '装饰'+'器'）。"""
    global _jieba_loaded
    if _jieba_loaded or jieba is None:
        return
    _jieba_loaded = True
    try:
        import json
        from ..config import tag_schema_file
        data = json.loads(tag_schema_file().read_text(encoding="utf-8"))
        for t in data.get("allowed_tags", []):
            t = str(t).strip()
            if len(t) >= 2:
                jieba.add_word(t)
    except Exception:
        pass


def _tokenize(text: str, drop_stop: bool = False) -> list[str]:
    """中文分词：有 jieba 用 jieba；无 jieba 退回 英文词 + 中文 bigram 兜底。
    drop_stop=True 时过滤停用词 + 中文单字（BM25 查询用；建索引不过滤，保 recall）。"""
    if not text:
        return []
    if jieba is not None:
        _load_jieba_userdict()
        toks = [t for t in jieba.lcut(text) if t.strip()]
    else:
        import re
        toks = re.findall(r"[a-z0-9_+\-]+", text.lower())
        for seg in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            toks.append(seg)
            toks.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    if drop_stop:
        out = []
        for t in toks:
            if t in _BM25_STOPWORDS:
                continue
            if len(t) == 1 and "\u4e00" <= t <= "\u9fff":
                continue  # 中文单字噪音（如"器"来自"装饰器"误切）
            out.append(t)
        return out
    return toks


def _bm25_index(col):
    """构建/复用 BM25 索引。col.count() 变化（入库/删除）时自动重建。
    每次 query 都新建 chroma client，count 是最新的，可作失效信号。"""
    global _bm25_cache
    try:
        total = col.count()
    except Exception:
        return None
    if _bm25_cache["count"] == total and _bm25_cache["bm25"] is not None:
        return _bm25_cache["bm25"]
    data = col.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    texts = [str(d) for d in docs]
    corpus = [_tokenize(t) for t in texts]
    if not corpus or not any(corpus):
        _bm25_cache = {"count": total, "bm25": None, "texts": [], "metas": [], "lookup": {}}
        return None
    bm25 = BM25Okapi(corpus)
    lookup = {(md.get("source", ""), t[:30]): i
              for i, (md, t) in enumerate(zip(metas, texts))}
    _bm25_cache = {"count": total, "bm25": bm25, "texts": texts,
                   "metas": metas, "lookup": lookup}
    return bm25


def _search_by_bm25(question: str, k: int, topic: str = "") -> list[dict]:
    """BM25 关键词召回通道：专有名词/术语精确命中保底进候选池。
    分数先留 0.0，统一由 _rerank_by_bm25 融合重排。"""
    if not _HAS_BM25:
        return []
    col = _collection()
    if col is None:
        return []
    bm25 = _bm25_index(col)
    if bm25 is None:
        return []
    toks = _tokenize(question, drop_stop=True) or _tokenize(question)
    if not toks:
        return []
    try:
        scores = bm25.get_scores(toks)
    except Exception:
        return []
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        if i >= len(_bm25_cache["metas"]):
            continue
        md = _bm25_cache["metas"][i]
        if topic and md.get("topic", "") != topic:
            continue
        out.append({
            "text": _bm25_cache["texts"][i],
            "source": md.get("source", "未知"),
            "topic": md.get("topic", ""),
            "heading": md.get("heading", ""),
            "tags": _tags_list(md),
            "score": 0.0,
        })
        if len(out) >= k:
            break
    return out


def _rerank_by_bm25(hits: list[dict], bm25, tokens: list[str]) -> None:
    """BM25 词命中加权：按『命中不同查询词数 / 查询词总数』加分并重排。
    加分 = boost × ratio。对词频/文档长度不敏感：两篇同词文档加分相同，排序仍由
    向量语义分主导；多词查询时命中词越多加越多，专有名词组合（chroma+milvus、
    LangChain+RAG）保底进前排。加分并入 score，排序与下游阈值用同一把尺子。"""
    if not hits or bm25 is None or not tokens:
        return
    boost = float(recall_cfg().get("bm25_boost", 0.10))
    if boost <= 0:
        return
    try:
        per_word = [bm25.get_scores([t]) for t in tokens]
    except Exception:
        return
    n = len(tokens)
    for h in hits:
        i = _bm25_cache["lookup"].get((h["source"], h["text"][:30]))
        if i is None:
            continue
        hit_words = sum(1 for s in per_word if float(s[i]) > 0)
        h["score"] = round(h["score"] + boost * (hit_words / n), 4)
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

    # ---- 混合检索：BM25 关键词通道保底进池 + 融合重排 ----
    # 向量是内容相似度主通道；BM25 保证专有名词/术语的精确命中不被语义近邻挤掉。
    # 融合方式与标签加权同构：归一化 BM25 分按 boost 并入 score，不改变分数尺度。
    bm25, bm25_tokens = None, []
    if col is not None and _HAS_BM25:
        bm25 = _bm25_index(col)
        if bm25 is not None:
            toks = _tokenize(question, drop_stop=True) or _tokenize(question)
            if toks:
                bm25_tokens = toks
                bk = int(recall_cfg().get("bm25_top_k", 10))
                _merge_hits(_search_by_bm25(question, bk, topic=topic), pool)

    hits = list(pool.values())
    if tags:
        _rerank_by_tags(hits, tags)
    if bm25 is not None and bm25_tokens:
        _rerank_by_bm25(hits, bm25, bm25_tokens)
    hits.sort(key=lambda h: -h["score"])
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
