"""入库编排：扫描 → 对比台账 → 分块 → 向量化 → 写库 → 更新台账。
编排者只串流程，具体步骤委托给各模块。"""
import hashlib

from ..config import raw_dir, split_cfg
from ..embedding import embed_texts
from ..storage import vector_store
from ..storage import doc_index
from . import scanner, splitter


def ingest(verbose: bool = True) -> dict:
    """增量入库：只处理新增 + 修改，未变跳过；台账里已删除的剔除。"""
    scan_rows = scanner.scan()
    index_rows = doc_index.load()
    cls = scanner.classify(scan_rows, index_rows)

    root = raw_dir()
    todo = cls["new"] + cls["changed"]
    all_chunks = []
    for rel in todo:
        f = root / rel
        text = splitter.read_file(f)
        cfg = split_cfg()
        info = scan_rows[rel]
        for i, chunk in enumerate(splitter.chunk_md(text, cfg["chunk_size"], cfg["chunk_overlap"])):
            all_chunks.append({
                "id": hashlib.md5(f"{rel}:{i}".encode()).hexdigest(),
                "text": chunk,
                "source": f.name,
                "topic": info["topic"],
                "heading": chunk.splitlines()[0][:50] if chunk.splitlines() else "",
            })

    stats = {"new": len(cls["new"]), "changed": len(cls["changed"]),
             "unchanged": len(cls["unchanged"]), "removed": len(cls["removed"])}

    if all_chunks:
        # 重导前清掉同文件的旧片段（防脏数据），再向量化写入
        for c in all_chunks:
            vector_store.delete_by_source(c["source"])
        embs = embed_texts([c["text"] for c in all_chunks])
        for c, emb in zip(all_chunks, embs):
            c["embedding"] = emb
        vector_store.upsert(all_chunks)

    # 更新台账（含剔除已删除文档）
    doc_index.update(scan_rows)
    if verbose:
        print(f"[入库] 新增{stats['new']} 修改{stats['changed']} 未变{stats['unchanged']} 删除{stats['removed']}，片段 {len(all_chunks)} 条")
    return {"chunks": len(all_chunks), **stats}
