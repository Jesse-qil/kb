"""命令行入口：python -m kb.cli <命令> [参数]
命令：
  scan            扫描 raw，对比台账，打印 增/改/删/未变
  ingest          入库（新增+修改的文档重新向量化）
  query <问题>     检索并打印结果
  list            列出知识库内容（库里有什么）
  reset           清空向量库和台账（重建用）
  retag           重新给所有文档打标签（改了 tag_schema.json 或
                  换了 LLM 模型后用；更新台账并回填向量库 metadata）
  import-folder <路径> [--review] [--topic X]
                  批量导入本地文件夹：自动识别 .md/.docx/.pdf，
                  LLM 自动分类主题后入库；--review 只进待审查区；
                  --topic 强制指定主题
"""
import sys


def cmd_scan():
    from .ingestion import scanner
    from .storage import doc_index
    scan_rows = scanner.scan()
    cls = scanner.classify(scan_rows, doc_index.load())
    print(f"扫描到 {len(scan_rows)} 篇：新增{len(cls['new'])} 修改{len(cls['changed'])} 未变{len(cls['unchanged'])} 删除{len(cls['removed'])}")
    for k in ("new", "changed"):
        for rel in cls[k]:
            print(f"  [{k}] {rel}")


def cmd_ingest():
    from .ingestion.pipeline import ingest
    ingest()


def cmd_query(q: str):
    from .storage.vector_store import query
    hits = query(q)
    if not hits:
        print("没检索到内容")
        return
    for h in hits:
        print(f"[{h['topic']}] {h['source']} (相似度 {h['score']})")
        print(f"  {h['text'][:120]}")
        print()


def cmd_list():
    from .storage.vector_store import list_contents
    for topic, files in list_contents().items():
        print(f"[目录] {topic}: {', '.join(files)}")




def cmd_ask(q: str):
    from .agents.single import ask
    res = ask(q)
    print(res["answer"])
    for s in res["sources"]:
        print(f"  [来源] {s['source']} (相似度 {s['score']})")


def cmd_reset():
    import shutil
    from .config import chroma_dir, KNOWLEDGE_DIR, index_file, chunks_file
    shutil.rmtree(chroma_dir(), ignore_errors=True)
    if index_file().exists():
        index_file().unlink()
    if chunks_file().exists():
        chunks_file().unlink()
    fallback_store = KNOWLEDGE_DIR / "vector_store" / "fallback_store.json"
    if fallback_store.exists():
        fallback_store.unlink()
    print("已清空向量库 + 台账")


def cmd_import_folder():
    from .ingestion.folder_import import import_folder
    args = sys.argv[2:]
    review = "--review" in args
    topic = ""
    if "--topic" in args:
        i = args.index("--topic")
        if i + 1 < len(args):
            topic = args[i + 1]
    paths = [a for a in args if not a.startswith("--") and a != topic]
    if not paths:
        print("用法: python -m kb.cli import-folder <文件夹路径> [--review] [--topic 主题名]")
        return
    try:
        s = import_folder(paths[0], auto=not review, topic=topic)
    except ValueError as e:
        print(f"导入失败: {e}")
        return
    for d in s["details"]:
        if d["status"] == "done":
            print(f"  [入库] {d['file']} → [{d['topic']}] {d['path']}")
        elif d["status"] == "pending":
            print(f"  [待审] {d['file']} 建议[{d['suggest_topic'] or '待定'}]")
        elif d["status"] == "duplicate":
            print(f"  [跳过] {d['file']}（{d['note']}）")
        elif d["status"] == "error":
            print(f"  [错误] {d['file']}：{d['note']}")
    if s["imported"]:
        print("触发入库…")
        from .ingestion.pipeline import ingest
        ingest()


def cmd_retag():
    """全量重打标签：标签表/模型变了以后，不用重导文档。
    流程：重打标 → 更新台账 → 借 chunk_cache 回填向量库 metadata（不重算 embedding）。"""
    import hashlib
    from .config import raw_dir, split_cfg, embedding_cfg
    from .storage import doc_index, vector_store
    from .ingestion import chunk_cache
    from .ingestion.tagger import suggest_tags
    from .ingestion.splitter import read_file

    cfg = split_cfg()
    model_name = embedding_cfg()["model_name"]
    rows = doc_index.load()
    root = raw_dir()
    if not rows:
        print("台账为空，先 ingest")
        return
    for rel, info in rows.items():
        f = root / rel
        if not f.exists():
            print(f"  [跳过] {rel}（文件不存在）")
            continue
        text = read_file(f)
        title = text.splitlines()[0].lstrip("#").strip()[:80] if text.splitlines() else f.name
        tags = suggest_tags(text, title=title, topic=info.get("topic", ""))
        info["tags"] = tags
        cached = chunk_cache.get(info["hash"], cfg["chunk_size"], cfg["chunk_overlap"], model_name)
        if not cached or cached.get("chunks") is None:
            print(f"  [已打标] {rel}: {tags}（向量库回填将随下次 ingest 完成）")
            continue
        chunks = []
        for i, ch in enumerate(cached["chunks"]):
            if ch.get("embedding") is None:
                break
            chunks.append({
                "id": hashlib.md5(f"{rel}:{i}".encode()).hexdigest(),
                "text": ch.get("text", ""),
                "source": rel,
                "topic": info.get("topic", ""),
                "heading": ch.get("heading", ""),
                "tags": list(tags),
                "embedding": ch["embedding"],
            })
        if chunks and len(chunks) == len(cached["chunks"]):
            vector_store.delete_by_source(rel)
            vector_store.upsert(chunks)
            print(f"  [已打标] {rel}: {tags}（{len(chunks)} chunks 回填完成）")
        else:
            print(f"  [已打标] {rel}: {tags}（缓存缺向量，回填将随下次 ingest 完成）")
    doc_index.update(rows)
    print("retag 完成")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "scan":
        cmd_scan()
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "query" and len(sys.argv) >= 3:
        cmd_query(" ".join(sys.argv[2:]))
    elif cmd == "ask" and len(sys.argv) >= 3:
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "list":
        cmd_list()
    elif cmd == "reset":
        cmd_reset()
    elif cmd == "retag":
        cmd_retag()
    elif cmd == "import-folder":
        cmd_import_folder()
    else:
        print(__doc__)


if __name__ == "__main__":
    print("\033[92m cli.py 文件\033[0m")
    main()
