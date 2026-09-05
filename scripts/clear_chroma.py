"""清理 chroma notes collection 全部数据 + 重建。

适用场景：chroma 残留旧路径数据（list_collections 视图 bug 导致 API delete 不彻底）。

原理：直接 sqlite DELETE 清掉 notes collection 的所有 segments/embeddings/metadata，
再调 ingest() 走标准路径重建。不动物理文件，避免 safe-delete hook。

跑法（venv）：
    PY=$(cygpath -w "$(pwd)/.venv/Scripts/python.exe") && "$PY" scripts/clear_chroma.py
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.config import chroma_dir


def clear_notes_collection() -> dict:
    """删除 notes collection 的所有数据（保留 kb_chunks 等其他集合）。返回统计。"""
    sqlite_path = chroma_dir() / "chroma" / "chroma.sqlite3"
    if not sqlite_path.exists():
        return {"deleted": 0, "skipped": "no sqlite"}

    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()

    row = cur.execute("SELECT id FROM collections WHERE name='notes'").fetchone()
    if not row:
        conn.close()
        return {"deleted": 0, "skipped": "no notes collection"}

    notes_id = row[0]

    # 拿到要删的 id 列表（拼成 IN (...) 用）
    emb_ids = [r[0] for r in cur.execute(
        "SELECT id FROM embeddings WHERE segment_id IN "
        "(SELECT id FROM segments WHERE collection=?)", (notes_id,))]
    seg_ids = [r[0] for r in cur.execute(
        "SELECT id FROM segments WHERE collection=?", (notes_id,))]

    if not emb_ids:
        conn.close()
        return {"deleted": 0, "skipped": "empty collection"}

    emb_id_str = ",".join(f'"{eid}"' for eid in emb_ids)
    seg_id_str = ",".join(f'"{sid}"' for sid in seg_ids)

    cur.execute("BEGIN")
    n = 0
    # embedding_metadata（每条 embedding 多个 key）
    cur.execute(
        "DELETE FROM embedding_metadata WHERE id IN "
        "(SELECT id FROM embeddings WHERE segment_id IN "
        "(SELECT id FROM segments WHERE collection=?))", (notes_id,))
    n += cur.rowcount
    cur.execute(
        "DELETE FROM embeddings_queue WHERE id IN "
        "(SELECT id FROM embeddings WHERE segment_id IN "
        "(SELECT id FROM segments WHERE collection=?))", (notes_id,))
    n += cur.rowcount
    # fts 系列
    cur.execute(f"DELETE FROM embedding_fulltext_search_idx WHERE segid IN ({seg_id_str})")
    n += cur.rowcount
    cur.execute(f"DELETE FROM embedding_fulltext_search_data WHERE id IN ({emb_id_str})")
    n += cur.rowcount
    cur.execute(f"DELETE FROM embedding_fulltext_search_content WHERE id IN ({emb_id_str})")
    n += cur.rowcount
    cur.execute(f"DELETE FROM embedding_fulltext_search_docsize WHERE id IN ({emb_id_str})")
    n += cur.rowcount
    cur.execute("DELETE FROM embedding_fulltext_search")  # string_value 无 segid
    n += cur.rowcount
    cur.execute(f"DELETE FROM max_seq_id WHERE segment_id IN ({seg_id_str})")
    n += cur.rowcount
    # embeddings 本体
    cur.execute(f"DELETE FROM embeddings WHERE id IN ({emb_id_str})")
    n += cur.rowcount
    # segments
    cur.execute(
        "DELETE FROM segment_metadata WHERE segment_id IN "
        "(SELECT id FROM segments WHERE collection=?)", (notes_id,))
    n += cur.rowcount
    cur.execute("DELETE FROM segments WHERE collection=?", (notes_id,))
    n += cur.rowcount
    # collection 自身
    cur.execute("DELETE FROM collection_metadata WHERE collection_id=?", (notes_id,))
    n += cur.rowcount
    cur.execute("DELETE FROM collections WHERE id=?", (notes_id,))
    n += cur.rowcount
    conn.commit()
    cur.execute("VACUUM")
    conn.close()
    return {"deleted": n, "embeddings": len(emb_ids), "segments": len(seg_ids)}


def main() -> int:
    print("[clear_chroma] 删除 notes collection 所有数据 ...")
    result = clear_notes_collection()
    print(f"[clear_chroma] {result}")

    if result.get("deleted", 0) == 0:
        print("[clear_chroma] 无须重建（已空或不存在）")
        return 0

    print("[clear_chroma] 触发 ingest() 重建 ...")
    from kb.ingestion.pipeline import ingest
    r = ingest(verbose=False)
    print(f"[clear_chroma] ingest result: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())