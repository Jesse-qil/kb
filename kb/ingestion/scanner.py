"""文件扫描器：遍历 raw/ 找 .md/.docx，算 hash/mtime/topic。
topic = 相对 raw 的父目录名（如 raw/notes_draft/xx.md → notes_draft）
顶层目录 = 大领域/分类；一篇笔记一个 topic，将来可用 tags 补多标签。"""
import hashlib
from pathlib import Path
from ..config import raw_dir, CONFIG


def _hash_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _topic_of(path: Path, root: Path) -> str:
    rel = path.parent.relative_to(root)
    if rel.parts:
        return rel.parts[0]      # 顶层目录 = 领域/分类
    return "默认"


def scan() -> dict:
    """扫描 raw/，返回 {相对路径: {topic, hash, mtime}}。"""
    root = raw_dir()
    if not root.exists():
        return {}
    suffix = CONFIG["scan"]["file_suffix"]
    result = {}
    for suf in suffix:
        for f in sorted(root.rglob(f"*{suf}")):
            rel = f.relative_to(root).as_posix()
            result[rel] = {
                "topic": _topic_of(f, root),
                "hash": _hash_of(f),
                "mtime": int(f.stat().st_mtime),
            }
    return result


def classify(scan_rows: dict, index_rows: dict) -> dict:
    """对比台账，把文件分成 新增/修改/未变/删除 四类，供入库只处理变动的。"""
    new, changed, unchanged, removed = [], [], [], []
    for rel, info in scan_rows.items():
        prev = index_rows.get(rel)
        if prev is None:
            new.append(rel)
        elif prev.get("hash") != info["hash"]:
            changed.append(rel)
        else:
            unchanged.append(rel)
    for rel in index_rows:
        if rel not in scan_rows:
            removed.append(rel)
    return {"new": new, "changed": changed, "unchanged": unchanged, "removed": removed}
