"""向量库封装：只管"片段怎么存、怎么按向量找"。
纯存储职责：不知道文档从哪来、怎么分块（那是 ingestion 的事）。"""
import chromadb

from ..config import chroma_dir, recall_cfg
from ..embedding import embed_query

_COLLECTION = "notes"


def _client():
    d = chroma_dir()
    d.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(d))


def _collection():
    return _client().get_or_create_collection(
        _COLLECTION, metadata={"hnsw:space": "cosine"})


def delete_by_source(source: str) -> None:
    """删除某来源文件的所有旧片段（重导前清理，避免脏数据累积）。"""
    _collection().delete(where={"source": source})


def upsert(chunks: list[dict]) -> None:
    """写入/更新片段。chunks 元素需含 id/text/source/heading/topic/embedding。"""
    col = _collection()
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "heading": c["heading"],
                    "topic": c["topic"]} for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
    )


def query(question: str, topic: str = "", top_k: int = 0) -> list[dict]:
    """检索：问题 → 向量 → 相似度搜索。topic 非空则按领域过滤。"""
    col = _collection()
    k = top_k or recall_cfg()["top_k"]
    kwargs = {}
    if topic:
        kwargs["where"] = {"topic": topic}
    res = col.query(query_embeddings=[embed_query(question)],
                    n_results=k, include=["documents", "metadatas", "distances"], **kwargs)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "topic": res["metadatas"][0][i].get("topic", ""),
            "heading": res["metadatas"][0][i].get("heading", ""),
            "score": round(1 - res["distances"][0][i], 4),
        })
    return hits


def count() -> int:
    """库里片段总数。"""
    return _collection().count()


def list_contents() -> dict:
    """返回 {主题: [文件名...]}，用于"库里有什么"。"""
    data = _collection().get(include=["metadatas"])
    result = {}
    for md in data.get("metadatas") or []:
        t, s = md.get("topic", "默认"), md.get("source", "未知")
        result.setdefault(t, [])
        if s not in result[t]:
            result[t].append(s)
    return result
