# -*- coding: utf-8 -*-
"""切片缓存：按文档内容 hash 缓存分块结果（文本/标签/embedding）。

v2 拆分存储（解决 43 MB 单文件线性膨胀问题）：
- chunk_cache.meta.jsonl：JSON Lines，存元数据 + 文本 + 标签，**不含 embedding**
- chunk_cache.emb.npy：1D numpy 数组，shape=(N_total_chunks * 512,), dtype=float32
  所有 chunks 的 embedding 按写入顺序顺序拼接；每行记录里 chunks[i] 用 emb_off 索引进 emb.npy
- 旧格式 chunk_cache.jsonl（行内带 embedding 数组）首次加载时**自动迁移**到新格式

性能对比：
- 大小：43 MB → 5-6 MB（去掉 JSON 冗余、embedding 走紧凑 numpy）
- 读：43 MB JSON 全量 parse → mmap 5 MB meta + numpy memmap 取切片
- 写：43 MB 全量重写 → 5 MB meta 重写 + 7 MB emb append

并发安全：
- 进程内：threading.Lock 串行所有写
- 跨进程：Windows msvcrt.locking / Linux fcntl.flock 锁 .lock 文件
- 写入策略：写 tmp → 加锁 → os.replace（带重试）→ 释放
- 重试：偶发 WinError 5（目标被另一进程短暂持有）退避 50/150/300/600ms
- 事务锁：独立的 .transaction.lock 保护「读-改-写」整段，避免多进程互相覆盖
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import chunks_file


EMB_DIM = 512            # bge-small-zh-v1.5 默认维度
EMB_DTYPE = np.float32   # 必须与 embed_texts 返回类型一致


# ---------- 跨进程文件锁 ----------
@contextmanager
def _file_lock(target: Path):
    """对 target 旁的 .lock 文件加跨进程排他锁；不支持锁时降级为无锁（仍靠 OS 原子 rename）。"""
    lock_path = target.with_name(target.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    locked = False
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        if sys.platform == "win32":
            try:
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                locked = True
            except (ImportError, OSError):
                locked = False
        else:
            try:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                locked = True
            except (ImportError, OSError):
                locked = False
        yield
    finally:
        if fh is not None:
            if locked:
                try:
                    if sys.platform == "win32":
                        import msvcrt

                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            try:
                fh.close()
            except Exception:
                pass


# ---------- 进程内写锁 ----------
_WRITE_LOCK = threading.Lock()

# 跨进程事务锁：保护「读 → 改 → 写」整段，避免多个进程读同一份 meta 后
# 互相覆盖写入。用一个独立的 .transaction.lock 文件，避免和 meta/emb 的写入锁互斥。
_TX_LOCK_PATH = chunks_file().with_name(".chunk_cache.transaction.lock")

# 模块级 rows 缓存：避免每次 get 都全文 parse 3.4MB meta.jsonl
# 失效触发点：put_many、_migrate_legacy、clear
_ROWS_CACHE: dict[str, dict] | None = None
_EMB_CACHE: np.ndarray | None = None


# ---------- 文件路径 ----------
def _meta_file() -> Path:
    """元数据 + 文本 + 标签 文件。"""
    return chunks_file().with_name("chunk_cache.meta.jsonl")


def _emb_file() -> Path:
    """embedding 二进制文件。"""
    return chunks_file().with_name("chunk_cache.emb.npy")


def _legacy_file() -> Path:
    """旧版单文件 JSONL（行内带 embedding 数组），仅迁移期读一次。"""
    return chunks_file()  # chunk_cache.jsonl


# ---------- key ----------
def _key(content_hash: str, chunk_size: int, chunk_overlap: int, model_name: str) -> str:
    return f"{content_hash}|{chunk_size}|{chunk_overlap}|{model_name}"


# ---------- 原子写 ----------
def _atomic_write_no_lock(f: Path, data: bytes) -> None:
    """无文件锁版（调用方已持有事务锁），仅做 tmp + os.replace。"""
    tmp = f.with_name(f"{f.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, f)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


def _atomic_write(f: Path, data: bytes) -> None:
    """写 tmp + 跨进程加锁 + os.replace（带重试）。"""
    last_err: Exception | None = None
    for delay in (0.05, 0.15, 0.3, 0.6):
        tmp: Path | None = None
        try:
            with _file_lock(f):
                tmp = f.with_name(f"{f.name}.{os.getpid()}.{time.time_ns()}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, f)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            try:
                if tmp is not None and tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(delay)
    if last_err is None:
        raise OSError("chunk_cache write failed: unknown error")
    raise last_err


def _atomic_write_lines(f: Path, lines: list[str]) -> None:
    """文本写：JSON Lines 全量重写。"""
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _atomic_write(f, payload)


# ---------- emb.npy IO ----------
def _read_emb() -> np.ndarray:
    """读取 embedding 数组；返回**普通内存数组**（立刻释放文件句柄）。
    重要：如果返回 mmap，整个 put_many 期间 mmap 句柄会持有 emb.npy，
    导致其他进程的 os.replace 报 WinError 5。
    """
    global _EMB_CACHE
    f = _emb_file()
    if not f.exists():
        return np.zeros((0, EMB_DIM), dtype=EMB_DTYPE)
    if _EMB_CACHE is not None:
        return _EMB_CACHE
    arr = np.load(f, mmap_mode="r")
    out = np.array(arr, dtype=EMB_DTYPE) if arr.size else np.zeros((0, EMB_DIM), dtype=EMB_DTYPE)
    _EMB_CACHE = out
    return out


def _invalidate_cache() -> None:
    """失效 rows + emb 缓存（put_many / migrate / clear 后调用）。"""
    global _ROWS_CACHE, _EMB_CACHE
    _ROWS_CACHE = None
    _EMB_CACHE = None


# ---------- 旧版迁移 ----------
_MIGRATED_FLAG = ".jsonl.migrated.flag"


def _is_legacy_format() -> bool:
    """检测是否还有旧格式未迁移。"""
    legacy = _legacy_file()
    if not legacy.exists():
        return False
    # 迁移完成标记存在（即使 rename 失败）→ 跳过
    flag = legacy.with_name(_MIGRATED_FLAG)
    if flag.exists():
        return False
    meta = _meta_file()
    if not meta.exists():
        return True
    # 检查 meta 内容：v2 格式的 chunk 没有 embedding 字段，有 emb_off 字段
    try:
        first = next(iter(meta.read_text(encoding="utf-8").splitlines()), "")
        if not first.strip():
            return legacy.exists() and legacy.stat().st_size > 0
        row = json.loads(first)
        sample = (row.get("chunks") or [{}])[0]
        if "embedding" in sample:
            return True
        return False
    except Exception:
        return False


def _migrate_legacy() -> None:
    """把旧版 chunk_cache.jsonl（行内带 embedding）迁移到 meta+emb 拆分格式。
    使用事务锁保护：多进程同时启动时只有一个执行迁移，其他等待。"""
    legacy = _legacy_file()
    if not legacy.exists():
        return
    meta_path = _meta_file()
    emb_path = _emb_file()
    flag = legacy.with_name(_MIGRATED_FLAG)

    # 收集所有 chunks 的 embedding（保证顺序拼接）
    all_vecs: list[np.ndarray] = []
    meta_rows: list[dict] = []
    text = legacy.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("key"):
            continue
        new_chunks = []
        for ch in row.get("chunks") or []:
            emb = ch.get("embedding")
            if emb is None:
                new_chunks.append({"text": ch.get("text", ""), "heading": ch.get("heading", ""),
                                   "emb_off": -1, "emb_len": 0})
                continue
            vec = np.asarray(emb, dtype=EMB_DTYPE)
            if vec.size == 0:
                new_chunks.append({"text": ch.get("text", ""), "heading": ch.get("heading", ""),
                                   "emb_off": -1, "emb_len": 0})
                continue
            emb_off = sum(v.shape[0] for v in all_vecs)
            new_chunks.append({
                "text": ch.get("text", ""),
                "heading": ch.get("heading", ""),
                "emb_off": emb_off,
                "emb_len": vec.shape[0],
            })
            all_vecs.append(vec)
        row["chunks"] = new_chunks
        meta_rows.append(row)

    # 拼成大数组
    if all_vecs:
        big = np.concatenate([np.ascontiguousarray(v) for v in all_vecs], axis=0)
    else:
        big = np.zeros((0, EMB_DIM), dtype=EMB_DTYPE)

    # 写 meta（jsonl）和 emb（npy），并打标已迁移（无论 rename 成功与否）
    lines = [json.dumps(r, ensure_ascii=False) for r in meta_rows]
    payload_emb = BytesIO()
    np.save(payload_emb, big, allow_pickle=False)
    with _WRITE_LOCK:
        with _file_lock(_TX_LOCK_PATH):
            _atomic_write_no_lock(meta_path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
            _atomic_write_no_lock(emb_path, payload_emb.getvalue())
            _invalidate_cache()
            # 创建迁移标记（防止下次再触发迁移；即使 rename 失败也不影响）
            try:
                flag.write_text("migrated\n", encoding="utf-8")
            except OSError:
                pass
            # 删旧文件（rename 或 unlink；rename 失败就放过，反正不影响新格式读写）
            bak = legacy.with_suffix(".jsonl.migrated")
            try:
                os.replace(legacy, bak)
            except OSError:
                try:
                    legacy.unlink()
                except OSError:
                    pass


# ---------- 加载 ----------
def _load_meta_rows() -> dict[str, dict]:
    """从 meta.jsonl 加载元数据索引 dict[key -> row]（带模块级缓存）。"""
    global _ROWS_CACHE
    if _ROWS_CACHE is not None:
        return _ROWS_CACHE
    f = _meta_file()
    if not f.exists():
        _ROWS_CACHE = {}
        return _ROWS_CACHE
    rows: dict[str, dict] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        key = row.get("key")
        if key:
            rows[key] = row
    _ROWS_CACHE = rows
    return rows


def _load_rows() -> dict[str, dict]:
    """对外暴露：触发迁移 + 返回 rows 索引。"""
    if _is_legacy_format():
        _migrate_legacy()
    return _load_meta_rows()


# ---------- 写 meta ----------
def _save_meta_rows(rows: dict[str, dict]) -> None:
    f = _meta_file()
    lines = [json.dumps(r, ensure_ascii=False) for r in rows.values()]
    _atomic_write_lines(f, lines)


def _save_rows(rows: dict[str, dict]) -> None:
    """对外暴露（兼容旧调用）：只重写 meta，不动 emb.npy。"""
    with _WRITE_LOCK:
        _save_meta_rows(rows)


# ---------- 公开 API ----------
def get(content_hash: str, chunk_size: int, chunk_overlap: int, model_name: str) -> dict | None:
    """读取缓存命中，返回整条记录（chunks 含 embedding 还原为 list）。
    没有命中返回 None。"""
    if not content_hash:
        return None
    key = _key(content_hash, chunk_size, chunk_overlap, model_name)
    with _WRITE_LOCK:
        if _is_legacy_format():
            _migrate_legacy()
        rows = _load_meta_rows()
        row = rows.get(key)
        if row is None:
            return None
        # 还原：emb_off 从 emb.npy 取数组转 list
        emb_arr = _read_emb()
        chunks = []
        for ch in row.get("chunks") or []:
            off = ch.get("emb_off", -1)
            ln = ch.get("emb_len", 0)
            if off >= 0 and ln > 0 and off + ln <= emb_arr.shape[0]:
                emb_list = emb_arr[off:off + ln].tolist()
            else:
                emb_list = None
            chunks.append({
                "text": ch.get("text", ""),
                "heading": ch.get("heading", ""),
                "embedding": emb_list,
            })
        return {
            "key": row["key"],
            "content_hash": row.get("content_hash", ""),
            "chunk_size": row.get("chunk_size", chunk_size),
            "chunk_overlap": row.get("chunk_overlap", chunk_overlap),
            "model_name": row.get("model_name", model_name),
            "tags": row.get("tags") or [],
            "chunks": chunks,
            "updated": row.get("updated", 0.0),
        }


def put(
    content_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    *,
    tags: list[str],
    chunks: list[dict],
) -> dict | None:
    """写入单条缓存记录（兼容旧调用）。"""
    put_many([{
        "content_hash": content_hash,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "model_name": model_name,
        "tags": tags,
        "chunks": chunks,
    }])
    return get(content_hash, chunk_size, chunk_overlap, model_name)


def put_many(rows_in: Iterable[dict]) -> int:
    """批量写入：每个元素需含 content_hash/chunk_size/chunk_overlap/model_name/tags/chunks。
    使用事务锁（独立 .transaction.lock）保护读-改-写整段，避免多进程互相覆盖写入。
    """
    rows_in = list(rows_in)
    if not rows_in:
        return 0
    now = time.time()
    new_meta: dict[str, dict] = {}
    new_vecs: list[np.ndarray] = []
    for r in rows_in:
        key = _key(r["content_hash"], r["chunk_size"], r["chunk_overlap"], r["model_name"])
        new_chunks = []
        for ch in r.get("chunks") or []:
            emb = ch.get("embedding")
            if emb is None:
                new_chunks.append({"text": ch.get("text", ""), "heading": ch.get("heading", ""),
                                   "emb_off": -1, "emb_len": 0})
                continue
            vec = np.asarray(emb, dtype=EMB_DTYPE)
            emb_off = sum(v.shape[0] for v in new_vecs)
            new_chunks.append({
                "text": ch.get("text", ""),
                "heading": ch.get("heading", ""),
                "emb_off": emb_off,
                "emb_len": vec.shape[0],
            })
            new_vecs.append(vec)
        new_meta[key] = {
            "key": key,
            "content_hash": r["content_hash"],
            "chunk_size": r["chunk_size"],
            "chunk_overlap": r["chunk_overlap"],
            "model_name": r["model_name"],
            "tags": r.get("tags") or [],
            "chunks": new_chunks,
            "updated": now,
        }

    with _WRITE_LOCK:
        with _file_lock(_TX_LOCK_PATH):
            if _is_legacy_format():
                _migrate_legacy_inner_unlocked()
            rows = _load_meta_rows()
            existing_emb = _read_emb()
            old_total = int(existing_emb.shape[0])
            # 把 new_meta 里的 emb_off 加上 old_total 偏移
            for meta_row in new_meta.values():
                for ch in meta_row["chunks"]:
                    if ch.get("emb_off", -1) >= 0:
                        ch["emb_off"] = ch["emb_off"] + old_total
            rows.update(new_meta)
            # 合并 emb
            if new_vecs:
                new_big = np.concatenate([np.ascontiguousarray(v) for v in new_vecs], axis=0)
                combined = (np.concatenate([existing_emb, new_big], axis=0)
                            if existing_emb.shape[0] > 0 else new_big)
            else:
                combined = existing_emb
            # 双写（事务锁已持，不再加锁）
            lines = [json.dumps(r, ensure_ascii=False) for r in rows.values()]
            _atomic_write_no_lock(
                _meta_file(),
                ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"),
            )
            if new_vecs or not _emb_file().exists():
                buf = BytesIO()
                np.save(buf, np.ascontiguousarray(combined), allow_pickle=False)
                _atomic_write_no_lock(_emb_file(), buf.getvalue())
            _invalidate_cache()
    return len(new_meta)


def _migrate_legacy_inner_unlocked() -> None:
    """迁移版（要求调用方已持有事务锁）：避免 put_many 嵌套锁。"""
    legacy = _legacy_file()
    if not legacy.exists():
        return
    meta_path = _meta_file()
    emb_path = _emb_file()
    flag = legacy.with_name(_MIGRATED_FLAG)

    all_vecs: list[np.ndarray] = []
    meta_rows: list[dict] = []
    text = legacy.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("key"):
            continue
        new_chunks = []
        for ch in row.get("chunks") or []:
            emb = ch.get("embedding")
            if emb is None:
                new_chunks.append({"text": ch.get("text", ""), "heading": ch.get("heading", ""),
                                   "emb_off": -1, "emb_len": 0})
                continue
            vec = np.asarray(emb, dtype=EMB_DTYPE)
            if vec.size == 0:
                new_chunks.append({"text": ch.get("text", ""), "heading": ch.get("heading", ""),
                                   "emb_off": -1, "emb_len": 0})
                continue
            emb_off = sum(v.shape[0] for v in all_vecs)
            new_chunks.append({
                "text": ch.get("text", ""),
                "heading": ch.get("heading", ""),
                "emb_off": emb_off,
                "emb_len": vec.shape[0],
            })
            all_vecs.append(vec)
        row["chunks"] = new_chunks
        meta_rows.append(row)

    if all_vecs:
        big = np.concatenate([np.ascontiguousarray(v) for v in all_vecs], axis=0)
    else:
        big = np.zeros((0, EMB_DIM), dtype=EMB_DTYPE)

    lines = [json.dumps(r, ensure_ascii=False) for r in meta_rows]
    _atomic_write_no_lock(meta_path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
    buf = BytesIO()
    np.save(buf, big, allow_pickle=False)
    _atomic_write_no_lock(emb_path, buf.getvalue())
    _invalidate_cache()
    try:
        flag.write_text("migrated\n", encoding="utf-8")
    except OSError:
        pass
    bak = legacy.with_suffix(".jsonl.migrated")
    try:
        os.replace(legacy, bak)
    except OSError:
        try:
            legacy.unlink()
        except OSError:
            pass


def clear() -> None:
    """清空 meta + emb + 旧文件。"""
    with _WRITE_LOCK:
        with _file_lock(_TX_LOCK_PATH):
            for p in (_meta_file(), _emb_file(), _legacy_file()):
                if p.exists():
                    p.unlink()
            _invalidate_cache()