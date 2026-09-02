"""待审查区管理：上传文件先进 pending/，人工确认后才移入 raw/。
数据：文件实体在 knowledge/pending/ 下；索引在 pending.json。"""
import json, shutil, time
from pathlib import Path

from ..config import pending_dir, raw_dir

_INDEX = "pending.json"


def _pending_file() -> Path:
    return pending_dir() / _INDEX


def _load() -> dict:
    f = _pending_file()
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8"))


def _save(rows: dict) -> None:
    pending_dir().mkdir(parents=True, exist_ok=True)
    _pending_file().write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def add(filename: str, suggest_topic: str, size: int, content_hash: str = "") -> dict:
    """登记一条待审查（文件本身已由 API 写入 pending/）。
    content_hash: 内容 md5，用于待审查区内部去重。"""
    rows = _load()
    # 同名已存在 → 覆盖旧文件与记录
    rows[filename] = {
        "suggest_topic": suggest_topic,
        "size": size,
        "hash": content_hash,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save(rows)
    return rows[filename]


def find_by_hash(content_hash: str) -> str | None:
    """在待审查区里找相同内容 hash 的文件，返回文件名；没有返回 None。"""
    if not content_hash:
        return None
    for name, info in _load().items():
        if info.get("hash") == content_hash:
            return name
    return None


def list_all() -> list[dict]:
    """返回待审查列表（含文件是否真实存在）。"""
    rows = _load()
    out = []
    for name, info in rows.items():
        p = pending_dir() / name
        out.append({**info, "filename": name, "exists": p.exists()})
    return out


def approve(filename: str, topic: str) -> dict:
    """审查通过：文件从 pending/ 移到 raw/<topic>/，返回落盘路径；失败抛异常。"""
    src = pending_dir() / filename
    if not src.exists():
        raise FileNotFoundError(f"待审查文件不存在: {filename}")
    topic = (topic or "").strip().strip("/\\")
    if not topic:
        raise ValueError("主题不能为空")
    dest_dir = raw_dir() / topic
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.move(str(src), str(dest))

    rows = _load()
    rows.pop(filename, None)
    _save(rows)
    return {"path": f"{topic}/{filename}", "dest": str(dest)}


def reject(filename: str) -> bool:
    """丢弃待审查文件（删除实体 + 记录）。"""
    src = pending_dir() / filename
    if src.exists():
        src.unlink()
    rows = _load()
    removed = rows.pop(filename, None) is not None
    _save(rows)
    return removed


def reset() -> None:
    """清空整个待审查区（文件 + 记录）。"""
    d = pending_dir()
    if d.exists():
        shutil.rmtree(d)
