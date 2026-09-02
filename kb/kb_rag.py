"""知识库核心：入库（扫描→分块→向量化→写Chroma→更新台账）与检索。
阶段1：全量入库（每篇都重embed），增量跳过逻辑在 ingest() 里已预留（unchanged 跳过）。
"""
import hashlib
from pathlib import Path

import chromadb

from .config import raw_dir, chroma_dir, split_cfg, recall_cfg, KNOWLEDGE_DIR
from . import file_scanner, doc_index
from .embedder import embed_texts, embed_query

_COLLECTION = "notes"


def _client():
    d = chroma_dir()
    d.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(d))


def _chunk_md(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按段落切分，再合并成不超过 chunk_size 字、相邻重叠 overlap 字的片段。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
                cur = cur[-overlap:] + "\n\n" + p
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks


def _read_file(path: Path) -> str:
    if path.suffix == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return path.read_text(encoding="utf-8")


def ingest(verbose: bool = True) -> dict:
    """扫描 raw → 台账对比 → 未变跳过，变了/新增的重新分块入库 → 更新台账。"""
    scan_rows = file_scanner.scan()
    index_rows = doc_index.load()
    cls = file_scanner.classify(scan_rows, index_rows)

    root = raw_dir()
    todo = cls["new"] + cls["changed"]       # 只处理新增 + 修改
    all_chunks = []
    for rel in todo:
        f = root / rel
        text = _read_file(f)
        cfg = split_cfg()
        for i, chunk in enumerate(_chunk_md(text, cfg["chunk_size"], cfg["chunk_overlap"])):
            info = scan_rows[rel]
            all_chunks.append({
                "id": hashlib.md5(f"{rel}:{i}".encode()).hexdigest(),
                "text": chunk,
                "source": f.name,
                "topic": info["topic"],
                "heading": chunk.splitlines()[0][:50] if chunk.splitlines() else "",
                "doc_id": index_rows.get(rel, {}).get("id", ""),   # inode 编号（更新后回填）
            })

    stats = {"new": len(cls["new"]), "changed": len(cls["changed"]),
             "unchanged": len(cls["unchanged"]), "removed": len(cls["removed"])}

    if all_chunks:
        # 已存在的旧片段先删（同 doc 重导，避免脏数据累积）
        client = _client()
        col = client.get_or_create_collection(_COLLECTION,
                                              metadata={"hnsw:space": "cosine"})
        for c in all_chunks:
            col.delete(where={"source": c["source"]})
        embs = embed_texts([c["text"] for c in all_chunks])
        col.upsert(
            ids=[c["id"] for c in all_chunks],
            documents=[c["text"] for c in all_chunks],
            metadatas=[{"source": c["source"], "heading": c["heading"],
                        "topic": c["topic"]} for c in all_chunks],
            embeddings=embs,
        )
    # 更新台账（含移除已删除的文档）
    doc_index.update(scan_rows)
    if verbose:
        print(f"[入库] 新增{stats['new']} 修改{stats['changed']} 未变{stats['unchanged']} 删除{stats['removed']}，片段 {len(all_chunks)} 条")
    return {"chunks": len(all_chunks), **stats}


def query(question: str, topic: str = "", top_k: int = 0) -> list[dict]:
    """检索：问题→向量→Chroma 相似度搜索。topic 非空则按分类过滤。"""
    col = _client().get_or_create_collection(_COLLECTION)
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
    """返回库里片段总数（Chroma collection.count）。"""
    col = _client().get_or_create_collection(_COLLECTION)
    return col.count()

def list_contents() -> dict:
    """返回 {主题: [文件名...]}，用于"库里有什么"。"""
    col = _client().get_or_create_collection(_COLLECTION)
    data = col.get(include=["metadatas"])
    result = {}
    for md in data.get("metadatas") or []:
        t, s = md.get("topic", "默认"), md.get("source", "未知")
        result.setdefault(t, [])
        if s not in result[t]:
            result[t].append(s)
    return result
