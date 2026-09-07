# -*- coding: utf-8 -*-
"""标签向量库：标签自治的核心模块。

把"固定候选表"升级为"动态增长的标签池"：
- 种子：tag_schema.json 的候选标签（不存在则用默认表）首次使用时灌入 chroma tags 集合
- 每条标签一个向量（embed_texts，与检索口径一致）
- 新笔记打标时：文档向量 → 标签库检索 TopN → 高分复用 / 低分由 LLM 生成新标签并写回
- 标签随使用增长，后续新内容即可检索到刚加的标签，全程不用改代码

降级：chromadb 不可用时退回 内存+JSON 名单（无向量检索，retrieve 返回空，
打标流程自动走 LLM 兜底），保证功能不中断。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

try:
    import chromadb
except Exception:
    chromadb = None

from ..config import KNOWLEDGE_DIR, chroma_dir, tag_schema_file
from ..embedding import embed_texts

_COLLECTION = "tags"
# 降级时保存标签名单（无向量，仅供 list_tags/add_tags 维持标签不丢）
_FALLBACK_FILE = KNOWLEDGE_DIR / "tags" / "tag_names.json"

_lock = threading.Lock()
_seeded = False


def _clean_tag(tag: str) -> str:
    """延迟导入 tagger 的清洗函数（避免模块级循环依赖）。"""
    from .tagger import _clean_tag as _ct
    return _ct(tag)


def _client():
    if chromadb is None:
        return None
    d = chroma_dir()
    d.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(d))


def _collection():
    client = _client()
    if client is None:
        return None
    return client.get_or_create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})


def _seed_tags() -> list[str]:
    """种子标签：tag_schema.json 优先，否则默认候选表 + 兜底 + 主题提示。"""
    from .tagger import _DEFAULT_SCHEMA
    tags: list[str] = []
    schema = {}
    p = tag_schema_file()
    if p.exists():
        try:
            schema = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            schema = {}
    for key in ("allowed_tags", "fallback_tags"):
        src = schema.get(key) if key in schema else _DEFAULT_SCHEMA.get(key)
        tags.extend(str(t) for t in (src or []) if t)
    hints = schema.get("topic_hints") if "topic_hints" in schema else _DEFAULT_SCHEMA["topic_hints"]
    for vals in (hints or {}).values():
        tags.extend(str(v) for v in vals if v)
    out, seen = [], set()
    for t in tags:
        t = _clean_tag(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def ensure_seeded() -> None:
    """惰性初始化：标签库为空时把种子标签灌进去（幂等，线程安全）。"""
    global _seeded
    if _seeded:
        return
    with _lock:
        if _seeded:
            return
        tags = _seed_tags()
        col = _collection()
        if col is not None:
            try:
                if col.count() == 0:
                    vecs = embed_texts(tags)
                    col.upsert(ids=tags, documents=tags, embeddings=vecs,
                               metadatas=[{"source": "seed"} for _ in tags])
            except Exception:
                pass
        else:
            try:
                f = _FALLBACK_FILE
                f.parent.mkdir(parents=True, exist_ok=True)
                cur = set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()
                cur.update(tags)
                f.write_text(json.dumps(sorted(cur), ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        _seeded = True


def list_tags() -> list[str]:
    """标签库全部标签（按名排序）。"""
    ensure_seeded()
    col = _collection()
    if col is not None:
        try:
            return sorted(col.get(include=["documents"])["documents"] or [])
        except Exception:
            return []
    try:
        if _FALLBACK_FILE.exists():
            return sorted(json.loads(_FALLBACK_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return []


def retrieve(text: str, top_k: int = 3) -> list[tuple[str, float]]:
    """文档内容 → 标签库相似度检索，返回 [(标签, 相似度)] 按分降序。

    口径与实测校准一致：embed_texts（无指令前缀）、文档取前 1000 字符、
    chroma cosine 空间 distance→相似度 = 1 - distance。
    """
    ensure_seeded()
    if not text or not text.strip():
        return []
    col = _collection()
    if col is None:
        return []
    try:
        if col.count() == 0:
            return []
        qv = embed_texts([text[:1000]])[0]
        n = min(max(1, top_k), col.count())
        res = col.query(query_embeddings=[qv], n_results=n,
                        include=["documents", "distances"])
        out = []
        for i in range(len(res["ids"][0])):
            out.append((res["documents"][0][i], round(1 - res["distances"][0][i], 4)))
        return out
    except Exception:
        return []


def add_tags(tags: list[str], source: str = "") -> int:
    """把新标签写入标签库（幂等：已存在的跳过），返回本次新增数量。

    source：哪个文档贡献的标签（留痕，便于追溯标签来源）。
    """
    ensure_seeded()
    clean: list[str] = []
    seen = set()
    for t in tags or []:
        t = _clean_tag(t)
        if t and t not in seen:
            seen.add(t)
            clean.append(t)
    if not clean:
        return 0
    added = 0
    col = _collection()
    with _lock:
        if col is not None:
            try:
                existing = set(col.get(ids=clean, include=["documents"])["ids"])
                missing = [t for t in clean if t not in existing]
                if missing:
                    vecs = embed_texts(missing)
                    col.upsert(ids=missing, documents=missing, embeddings=vecs,
                               metadatas=[{"source": str(source)[:200], "created_by": "llm"}
                                          for _ in missing])
                    added = len(missing)
            except Exception:
                return 0
        else:
            try:
                f = _FALLBACK_FILE
                f.parent.mkdir(parents=True, exist_ok=True)
                cur = set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()
                missing = [t for t in clean if t not in cur]
                if missing:
                    cur.update(missing)
                    f.write_text(json.dumps(sorted(cur), ensure_ascii=False, indent=2),
                                 encoding="utf-8")
                    added = len(missing)
            except Exception:
                return 0
    return added



def delete_tag(tag: str) -> bool:
    """从标签库删除一个标签。返回是否成功删除。"""
    tag = _clean_tag(tag)
    if not tag:
        return False
    ensure_seeded()
    col = _collection()
    with _lock:
        if col is not None:
            try:
                existing = col.get(ids=[tag], include=["documents"])["ids"]
                if not existing:
                    return False
                col.delete(ids=[tag])
                return True
            except Exception:
                return False
        else:
            try:
                f = _FALLBACK_FILE
                if f.exists():
                    cur = set(json.loads(f.read_text(encoding="utf-8")))
                    if tag in cur:
                        cur.discard(tag)
                        f.write_text(json.dumps(sorted(cur), ensure_ascii=False, indent=2),
                                     encoding="utf-8")
                        return True
            except Exception:
                pass
    return False


def merge_tag(from_tag: str, to_tag: str) -> dict:
    """合并标签：把 from_tag 合并到 to_tag，删除 from_tag。
    返回 {ok, from_tag, to_tag, deleted_from, added_to}。
    注意：此操作只改标签库，文档片段上的旧标签需要另行更新。"""
    from_tag = _clean_tag(from_tag)
    to_tag = _clean_tag(to_tag)
    if not from_tag or not to_tag or from_tag == to_tag:
        return {"ok": False, "error": "标签名无效或相同"}
    ensure_seeded()
    col = _collection()
    with _lock:
        if col is not None:
            try:
                # 确保目标标签存在
                existing = col.get(ids=[to_tag], include=["documents"])["ids"]
                added_to = False
                if not existing:
                    vecs = embed_texts([to_tag])
                    col.upsert(ids=[to_tag], documents=[to_tag], embeddings=vecs,
                               metadatas=[{"source": "merge_from:" + from_tag, "created_by": "merge"}])
                    added_to = True
                # 删除源标签
                from_existing = col.get(ids=[from_tag], include=["documents"])["ids"]
                deleted_from = False
                if from_existing:
                    col.delete(ids=[from_tag])
                    deleted_from = True
                return {"ok": True, "from_tag": from_tag, "to_tag": to_tag,
                        "deleted_from": deleted_from, "added_to": added_to}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        else:
            try:
                f = _FALLBACK_FILE
                cur = set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()
                added_to = to_tag not in cur
                cur.add(to_tag)
                deleted_from = from_tag in cur
                cur.discard(from_tag)
                f.write_text(json.dumps(sorted(cur), ensure_ascii=False, indent=2),
                             encoding="utf-8")
                return {"ok": True, "from_tag": from_tag, "to_tag": to_tag,
                        "deleted_from": deleted_from, "added_to": added_to}
            except Exception as e:
                return {"ok": False, "error": str(e)}


def reset() -> None:
    """清空标签库（测试/重建用）。"""
    global _seeded
    with _lock:
        col = _collection()
        if col is not None:
            try:
                col.delete(where={"source": {"$ne": ""}})
            except Exception:
                try:
                    ids = col.get(include=["documents"])["ids"]
                    if ids:
                        col.delete(ids=ids)
                except Exception:
                    pass
        _seeded = False
