"""文档台账 doc_index.json：仿 inode 的登记表。
每篇文档记录：id（稳定编号，尽量不变）/ path / hash（内容指纹）/ mtime / tags
用途：增量更新（对比 hash 判断改没改过）、删除检测、前端列文档。"""
import json
from pathlib import Path
from ..config import index_file


def load() -> dict:
    """读台账；文件不存在返回空字典。"""
    f = index_file()
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def save(rows: dict) -> None:
    f = index_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def find_by_hash(content_hash: str) -> str | None:
    """在已入库台账里找相同内容 hash 的文档，返回其相对路径；没有返回 None。"""
    if not content_hash:
        return None
    for rel, info in load().items():
        if info.get("hash") == content_hash:
            return rel
    return None


def next_id(existing: dict) -> str:
    """分配下一个未占用的编号 kb-0001 递增。"""
    used = {v.get("id", "") for v in existing.values()}
    n = 1
    while f"kb-{n:04d}" in used:
        n += 1
    return f"kb-{n:04d}"


def update(scan_rows: dict) -> dict:
    """把扫描结果合并进台账：
    - 同 path 且 hash 相同 → 未变，保留原 id
    - 同 path 但 hash 变了 → 已修改，保留原 id（编号=inode 不轻易变）
    - 新 path → 分配新 id
    - 台账里有但这次扫描没有的 path → 文档已删除，剔除
    返回新台账。"""
    old = load()
    new_rows = {}
    for rel, info in scan_rows.items():
        prev = old.get(rel)
        doc_id = prev["id"] if prev else next_id({**old, **new_rows})
        new_rows[rel] = {
            "id": doc_id,
            "topic": info["topic"],
            "hash": info["hash"],
            "mtime": info["mtime"],
            "tags": prev.get("tags", []) if prev else [],
        }
    save(new_rows)
    return new_rows
