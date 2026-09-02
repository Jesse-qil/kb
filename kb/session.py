# -*- coding: utf-8 -*-
"""会话记忆（短期上下文）：按 session_id 存最近 N 轮对话。
状态层：只管"存历史/取历史/裁剪"，不决定怎么用历史（那是 agent 层的事）。
内存版：服务重启后历史清空。"""
import threading
import time

# session_id -> [{role, content, time}, ...]（只存 user/assistant）
_HISTORY: dict[str, list[dict]] = {}
_LOCK = threading.Lock()

MAX_TURNS = 6          # 保留最近几轮（1 轮 = 一问一答）
_MAX_MSGS = MAX_TURNS * 2


def append(session_id: str, role: str, content: str) -> None:
    """记一条消息；超过轮数上限裁掉最旧的。"""
    if not session_id or not content:
        return
    with _LOCK:
        msgs = _HISTORY.setdefault(session_id, [])
        msgs.append({"role": role, "content": content, "time": time.time()})
        # 只留最近 N 轮，避免 prompt 无限膨胀
        if len(msgs) > _MAX_MSGS:
            del msgs[: len(msgs) - _MAX_MSGS]


def get_text(session_id: str, max_turns: int = MAX_TURNS) -> str:
    """把历史拼成给 LLM 看的一段文本（按时间从旧到新）。
    例：我：xxx\n小齐：xxx\n...（截断为最近 max_turns 轮）"""
    with _LOCK:
        msgs = _HISTORY.get(session_id, [])
    if len(msgs) > max_turns * 2:
        msgs = msgs[-max_turns * 2:]
    if not msgs:
        return ""
    lines = []
    for m in msgs:
        who = "我" if m["role"] == "user" else "小齐"
        text = m["content"].replace("\n", " ")[:200]   # 单条截断，防爆
        lines.append(f"{who}：{text}")
    return "\n".join(lines)


def clear(session_id: str) -> None:
    with _LOCK:
        _HISTORY.pop(session_id, None)
