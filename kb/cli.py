"""命令行入口：python -m kb.cli <命令> [参数]
命令：
  scan            扫描 raw，对比台账，打印 增/改/删/未变
  ingest          入库（新增+修改的文档重新向量化）
  query <问题>     检索并打印结果
  list            列出知识库内容（库里有什么）
  reset           清空向量库和台账（重建用）
"""
import sys


def cmd_scan():
    from . import file_scanner, doc_index
    scan_rows = file_scanner.scan()
    cls = file_scanner.classify(scan_rows, doc_index.load())
    print(f"扫描到 {len(scan_rows)} 篇：新增{len(cls['new'])} 修改{len(cls['changed'])} 未变{len(cls['unchanged'])} 删除{len(cls['removed'])}")
    for k in ("new", "changed"):
        for rel in cls[k]:
            print(f"  [{k}] {rel}")


def cmd_ingest():
    from .kb_rag import ingest
    ingest()


def cmd_query(q: str):
    from .kb_rag import query
    hits = query(q)
    if not hits:
        print("没检索到内容")
        return
    for h in hits:
        print(f"[{h['topic']}] {h['source']} (相似度 {h['score']})")
        print(f"  {h['text'][:120]}")
        print()


def cmd_list():
    from .kb_rag import list_contents
    for topic, files in list_contents().items():
        print(f"[目录] {topic}: {', '.join(files)}")




def cmd_ask(q: str):
    from .qa import ask
    res = ask(q)
    print(res["answer"])
    for s in res["sources"]:
        print(f"  [来源] {s['source']} (相似度 {s['score']})")


def cmd_reset():
    import shutil
    from .config import chroma_dir, KNOWLEDGE_DIR, index_file
    shutil.rmtree(chroma_dir(), ignore_errors=True)
    if index_file().exists():
        index_file().unlink()
    print("已清空向量库 + 台账")


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
    else:
        print(__doc__)


if __name__ == "__main__":
    print("\033[92m cli.py 文件\033[0m")
    main()
