"""入库编排：扫描 → 对比台账 → 分块 → 向量化 → 写库 → 更新台账。
编排者只串流程，具体步骤委托给各模块。
支持 progress 回调：
  progress(stage, percent, *, current="", done=0, total=0)
阶段示例：
  ("扫描文档", 2, current="raw/...")
  ("对比台账", 8, total=12)
  ("分块/打标", 18, current="example.md", done=3, total=12)
  ("向量化", 65, current="64/180", done=64, total=180)
前端按 600ms 轮询；回调越密，前端越清楚「现在到底在干啥」。"""
import hashlib
import time

from ..config import raw_dir, split_cfg, embedding_cfg
from ..embedding import embed_texts
from ..storage import vector_store
from ..storage import doc_index
from . import scanner, splitter, chunk_cache, tagger

# embed 分批大小：太大前端进度半天不动，太小模型调用频繁
_EMBED_BATCH = 16


def _report(progress, stage: str, pct: float, **kw) -> None:
    """统一进度回调入口；progress 为 None 时静默。
    kw 透传给前端：current（当前文件名）、done/total（计数器）。"""
    if progress:
        progress(stage, int(pct), **kw)


def _first_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:80]
    return fallback


def _embed_chunks(chunks: list[dict], progress=None, offset=0.0, span=1.0) -> None:
    """分批向量化。每个 batch 完成都回调一次（让进度条动起来）。"""
    total = len(chunks)
    if total <= 0:
        return
    done = 0
    for start in range(0, total, _EMBED_BATCH):
        batch = chunks[start:start + _EMBED_BATCH]
        embs = embed_texts([c["text"] for c in batch])
        for c, emb in zip(batch, embs):
            c["embedding"] = emb
        done += len(batch)
        _report(progress, "向量化",
                offset + span * done / total,
                current=f"{done}/{total}", done=done, total=total)


def ingest(verbose: bool = True, progress=None) -> dict:
    """增量入库：只处理新增 + 修改，未变跳过；台账里已删除的剔除。
    progress: 可选回调 progress(stage, percent, *, current="", done=0, total=0)"""
    started = time.time()
    _report(progress, "扫描文档", 2, current="扫描 raw/ 目录…")
    scan_rows = scanner.scan()
    index_rows = doc_index.load()
    _report(progress, "对比台账", 8, current="和台账对比…")
    cls = scanner.classify(scan_rows, index_rows)

    root = raw_dir()
    todo = cls["new"] + cls["changed"]
    total_files = max(len(todo), 1)
    cfg = split_cfg()
    emb_cfg = embedding_cfg()
    model_name = emb_cfg["model_name"]

    all_chunks = []
    cache_write_rows = []
    cache_hits = 0
    cache_misses = 0
    for fi, rel in enumerate(todo):
        f = root / rel
        info = scan_rows[rel]
        cached = chunk_cache.get(info["hash"], cfg["chunk_size"], cfg["chunk_overlap"], model_name)
        file_chunks: list[dict]
        file_tags: list[str]
        use_cache = False
        if cached and cached.get("chunks") is not None and cached.get("tags") is not None:
            file_tags = list(cached.get("tags") or [])
            file_chunks = []
            for i, chunk in enumerate(cached.get("chunks") or []):
                file_chunks.append({
                    "id": hashlib.md5(f"{rel}:{i}".encode()).hexdigest(),
                    "text": chunk.get("text", ""),
                    # source 用相对路径（topic/.../文件名）：全局唯一，
                    # 防止不同目录同名文件互相删片段/覆盖来源
                    "source": rel,
                    "topic": info["topic"],
                    "heading": chunk.get("heading", ""),
                    "embedding": chunk.get("embedding"),
                })
            if all(c.get("embedding") is not None for c in file_chunks):
                cache_hits += 1
                use_cache = True
                _report(progress, f"缓存命中({f.name})", 8 + 22 * (fi + 1) / total_files,
                        current=f"{fi + 1}/{len(todo)} {f.name}",
                        done=fi + 1, total=len(todo))
        if not use_cache:
            cache_misses += 1
            _report(progress, f"分块/打标({f.name})", 8 + 22 * (fi + 1) / total_files,
                    current=f"{fi + 1}/{len(todo)} {f.name}",
                    done=fi + 1, total=len(todo))
            text = splitter.read_file(f)
            _report(progress, f"LLM 打标({f.name})", 8 + 22 * (fi + 1) / total_files,
                    current=f"LLM 打标 {f.name}",
                    done=fi + 1, total=len(todo))
            file_tags = tagger.suggest_tags(text, title=_first_title(text, f.name), topic=info["topic"])
            _report(progress, f"分块({f.name})", 8 + 22 * (fi + 1) / total_files,
                    current=f"分块 {f.name}",
                    done=fi + 1, total=len(todo))
            pieces = splitter.chunk_md(text, cfg["chunk_size"], cfg["chunk_overlap"])
            file_chunks = []
            for i, chunk in enumerate(pieces):
                file_chunks.append({
                    "id": hashlib.md5(f"{rel}:{i}".encode()).hexdigest(),
                    "text": chunk,
                    # source 用相对路径（topic/.../文件名）：全局唯一，
                    # 防止不同目录同名文件互相删片段/覆盖来源
                    "source": rel,
                    "topic": info["topic"],
                    "heading": chunk.splitlines()[0][:50] if chunk.splitlines() else "",
                    "tags": list(file_tags),
                })
            cache_write_rows.append({
                "content_hash": info["hash"],
                "chunk_size": cfg["chunk_size"],
                "chunk_overlap": cfg["chunk_overlap"],
                "model_name": model_name,
                "tags": file_tags,
                "chunks": file_chunks,
            })
        scan_rows[rel]["tags"] = file_tags
        all_chunks.extend(file_chunks)

    stats = {"new": len(cls["new"]), "changed": len(cls["changed"]),
             "unchanged": len(cls["unchanged"]), "removed": len(cls["removed"]),
             "cache_hits": cache_hits, "cache_misses": cache_misses}

    missing_chunks = [c for c in all_chunks if c.get("embedding") is None]
    if missing_chunks:
        _report(progress, f"向量化 0/{len(missing_chunks)}", 30,
                current=f"准备向量化 {len(missing_chunks)} 片段",
                done=0, total=len(missing_chunks))
        _embed_chunks(missing_chunks, progress=progress, offset=30, span=55)
    else:
        _report(progress, "缓存命中，跳过向量化", 85,
                current=f"0/{len(all_chunks)} 全部命中缓存")

    # 批量写缓存：一次加载+一次落盘，避免逐文件读写导致 Windows 短句柄冲突
    if cache_write_rows:
        _report(progress, f"写缓存({len(cache_write_rows)})", 86,
                current=f"写 chunk_cache.jsonl · {len(cache_write_rows)} 文件")
        chunk_cache.put_many(
            [
                {
                    "content_hash": row["content_hash"],
                    "chunk_size": row["chunk_size"],
                    "chunk_overlap": row["chunk_overlap"],
                    "model_name": row["model_name"],
                    "tags": row["tags"],
                    "chunks": [
                        {"text": c["text"], "heading": c["heading"], "embedding": c["embedding"]}
                        for c in row["chunks"]
                    ],
                }
                for row in cache_write_rows
            ]
        )

    # 已删除的文件：把它的旧片段也从向量库清掉（防残留）
    if cls["removed"]:
        _report(progress, f"清理已删除({len(cls['removed'])})", 90,
                current=f"清理 {len(cls['removed'])} 个已删文件的旧片段")
        for rel in cls["removed"]:
            # 新格式：source=相对路径，直接精确删
            vector_store.delete_by_source(rel)
            # 旧格式数据 source 存的是纯文件名 → 按 (文件名, topic) 兼容清理
            fname = rel.rsplit("/", 1)[-1]
            topic = rel.split("/", 1)[0]
            vector_store.delete_by_source(fname, topic=topic)

    if all_chunks:
        # 重导前清掉该文件旧片段（防脏数据）。
        # 按文件去重：新格式 source=rel 天然唯一；
        # 旧格式数据 source=文件名 → 再按 (文件名, topic) 清一遍。
        # （文件换 topic 时旧 topic 的残片可能残留，属旧格式固有问题，不会误删他人）
        _report(progress, "清旧片段", 92,
                current=f"清理 {len({(c['source'], c['topic']) for c in all_chunks})} 个文件的旧片段")
        cleaned = set()
        for c in all_chunks:
            key = (c["source"], c["topic"])
            if key in cleaned:
                continue
            cleaned.add(key)
            vector_store.delete_by_source(c["source"])
            fname = c["source"].rsplit("/", 1)[-1]
            vector_store.delete_by_source(fname, topic=c["topic"])
        # 对未命中缓存的文件，前面已经完成分块/向量化。
        # 这里只剩整体写库阶段。
        _report(progress, "写入向量库", 94,
                current=f"chroma upsert {len(all_chunks)} 片段")
        vector_store.upsert(all_chunks)

    # 更新台账（含剔除已删除文档）
    _report(progress, "更新台账", 98,
            current=f"doc_index.update · {len(scan_rows)} 条")
    doc_index.update(scan_rows)
    _report(progress, "完成", 100,
            current=f"用时 {int(time.time() - started)} 秒")
    if verbose:
        print(f"[入库] 新增{stats['new']} 修改{stats['changed']} 未变{stats['unchanged']} 删除{stats['removed']}，"
              f"片段 {len(all_chunks)} 条，缓存命中{stats['cache_hits']} 文件，用时 {int(time.time() - started)} 秒")
    return {"chunks": len(all_chunks), **stats, "elapsed": int(time.time() - started)}
