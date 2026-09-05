# -*- coding: utf-8 -*-
"""会话管理：多会话 + 文件持久化。
- 每个会话存 knowledge/sessions/<id>.json：完整历史 + 标题 + 创建时间

- 服务重启历史不丢
- get_text() 仍只取最近 N 轮喂 prompt（窗口），但文件里保留完整历史
状态层：只管存取，不决定怎么用。"""
import json
import threading
import time
import uuid
from pathlib import Path

from .config import KNOWLEDGE_DIR

_SESSIONS_DIR = KNOWLEDGE_DIR / "sessions"
_LOCK = threading.Lock()

MAX_TURNS = 6          # 喂给 LLM 的窗口轮数（不是存储上限！）


def _sanitize(session_id: str) -> str:
    # 防目录穿越：只接受我们生成的 id（s_xxxxxx）
    return session_id.replace("/", "_").replace("\\", "_")


def _dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy()
    return _SESSIONS_DIR


_MIGRATED = False


def _migrate_legacy() -> None:
    """旧数据兼容（幂等，进程内只跑一次）：
    历史上 _path() 多拼过一层 s_，产生过 s_s_xxx.json 这类文件——
    列表能列出（读的是文件内 id），但按 sid 找不到文件、历史读不到。
    现在按【文件内记录的 id】重命名为 <id>.json，保证列表可见 ⇔ 历史可读。"""
    global _MIGRATED
    if _MIGRATED:
        return
    _MIGRATED = True
    for p in _SESSIONS_DIR.glob("*.json"):
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
            sid = s.get("id", "")
            if not sid or not isinstance(sid, str):
                continue
            target = p.parent / f"{_sanitize(sid)}.json"
            if target != p and not target.exists():
                p.rename(target)
        except Exception:
            continue


def _path(session_id: str) -> Path:
    return _dir() / f"{_sanitize(session_id)}.json"


def _load(session_id: str) -> dict:
    p = _path(session_id)
    if not p.exists():
        return {"id": session_id, "messages": [], "title": "新会话",
                "created": time.time()}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"id": session_id, "messages": [], "title": "新会话",
                "created": time.time()}


def _save(session: dict) -> None:
    _path(session["id"]).write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 会话级操作 ----------
def create(title: str = "新会话") -> str:
    """新建会话，返回 session_id（形如 s_xxxxxxxx）。"""
    sid = "s_" + uuid.uuid4().hex[:10]
    _save({"id": sid, "messages": [], "title": title, "created": time.time()})
    return sid


def list_sessions() -> list[dict]:
    """返回会话列表（按创建时间倒序）：[{id, title, created, msg_count}]"""
    out = []
    with _LOCK:
        for p in _dir().glob("*.json"):
            try:
                s = json.loads(p.read_text(encoding="utf-8"))
                out.append({
                    "id": s["id"],
                    "title": s.get("title", "新会话"),
                    "created": s.get("created", 0),
                    "msg_count": len(s.get("messages", [])),
                })
            except Exception:
                continue
    out.sort(key=lambda x: -x["created"])
    return out


def delete(session_id: str) -> bool:
    """删除会话文件。"""
    p = _path(session_id)
    if p.exists():
        with _LOCK:
            p.unlink()
        return True
    return False


# ---------- 消息级操作 ----------
def append(session_id: str, role: str, content: str) -> None:
    """记一条消息（保留完整历史，不再裁剪丢弃）。"""
    if not session_id or not content:
        return
    with _LOCK:
        s = _load(session_id)
        # 第一条用户消息自动当标题（截断）
        if not s["messages"] and role == "user":
            s["title"] = content.replace("\n", " ")[:20]
        s["messages"].append({"role": role, "content": content,
                              "time": time.time()})
        _save(s)


def get_history(session_id: str) -> list[dict]:
    """返回完整历史消息列表（前端回看用）。"""
    return _load(session_id).get("messages", [])


def get_text(session_id: str, max_turns: int = MAX_TURNS) -> str:
    """把【最近 max_turns 轮】拼成给 LLM 的文本（窗口裁剪只在这发生）。"""
    msgs = get_history(session_id)
    if len(msgs) > max_turns * 2:
        msgs = msgs[-max_turns * 2:]
    if not msgs:
        return ""
    lines = []
    for m in msgs:
        who = "我" if m["role"] == "user" else "小齐"
        text = m["content"].replace("\n", " ")[:200]
        lines.append(f"{who}：{text}")
    return "\n".join(lines)


def clear(session_id: str) -> None:
    """清空某会话消息（保留会话文件）。"""
    with _LOCK:
        s = _load(session_id)
        s["messages"] = []
        _save(s)
