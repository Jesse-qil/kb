"""入库编排：扫描 → 对比台账 → 分块 → 向量化 → 写库 → 更新台账。
编排者只串流程，具体步骤委托给各模块。
支持 progress 回调：(阶段名, 百分比0-100)，前端进度条用。"""
import hashlib

from ..config import raw_dir, split_cfg
from ..embedding import embed_texts
from ..storage import vector_store
from ..storage import doc_index
from . import scanner, splitter

# embed 分批大小：太大前端进度半天不动，太小模型调用频繁
_EMBED_BATCH = 16


def _report(progress, stage: str, pct: float):
    if progress:
        progress(stage, int(pct))


def ingest(verbose: bool = True, progress=None) -> dict:
    """增量入库：只处理新增 + 修改，未变跳过；台账里已删除的剔除。
    progress: 可选回调 progress(stage: str, percent: int)"""
    _report(progress, "扫描文档", 2)
    scan_rows = scanner.scan()
    index_rows = doc_index.load()
    _report(progress, "对比台账", 8)
    cls = scanner.classify(scan_rows, index_rows)

    root = raw_dir()
    todo = cls["new"] + cls["changed"]
    total_files = max(len(todo), 1)

    all_chunks = []
    for fi, rel in enumerate(todo):
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
        # 分块阶段 8% → 30%（按文件数推进）
        _report(progress, f"分块({f.name})", 8 + 22 * (fi + 1) / total_files)

    stats = {"new": len(cls["new"]), "changed": len(cls["changed"]),
             "unchanged": len(cls["unchanged"]), "removed": len(cls["removed"])}

    # 已删除的文件：把它的旧片段也从向量库清掉（防残留）
    if cls["removed"]:
        # 台账 key 形如 "topic/文件名" → 拆开，按 topic+source 精确定位
        # （只按文件名删会误删其他 topic 下的同名文件）
        for rel in cls["removed"]:
            topic, _, fname = rel.rpartition("/")
            vector_store.delete_by_source(fname, topic=topic)
        _report(progress, f"清理已删除({len(cls['removed'])})", 90)

    if all_chunks:
        # 重导前清掉该文件旧片段（防脏数据）。
        # ★ 注意：文件可能换过 topic（如 notes_draft → AI-Agent），旧片段在旧 topic 下，
        #   所以重导按 source 全删（不限定 topic），但按 source 去重避免重复删几十次。
        to_clean = {c["source"] for c in all_chunks}
        for source in to_clean:
            vector_store.delete_by_source(source)
        # 分批向量化：30% → 85%，每批推进（embed 是主要耗时）
        total_chunks = len(all_chunks)
        done = 0
        for start in range(0, total_chunks, _EMBED_BATCH):
            batch = all_chunks[start:start + _EMBED_BATCH]
            embs = embed_texts([c["text"] for c in batch])
            for c, emb in zip(batch, embs):
                c["embedding"] = emb
            done += len(batch)
            _report(progress, f"向量化({done}/{total_chunks})", 30 + 55 * done / total_chunks)
        _report(progress, "写入向量库", 88)
        vector_store.upsert(all_chunks)

    # 更新台账（含剔除已删除文档）
    _report(progress, "更新台账", 95)
    doc_index.update(scan_rows)
    _report(progress, "完成", 100)
    if verbose:
        print(f"[入库] 新增{stats['new']} 修改{stats['changed']} 未变{stats['unchanged']} 删除{stats['removed']}，片段 {len(all_chunks)} 条")
    return {"chunks": len(all_chunks), **stats}
