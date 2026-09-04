# -*- coding: utf-8 -*-
"""长期记忆（profile）数据层：跨会话"用户画像"的读写。
只管存/取/合并结构，不决定"提炼什么"（那是 agent 层的事，见 kb/agents/）。
文件：knowledge/memory/profile.json（已 gitignore）"""
import json
import time
from pathlib import Path

from .config import KNOWLEDGE_DIR

_MEMORY_DIR = KNOWLEDGE_DIR / "memory"
_PROFILE_FILE = _MEMORY_DIR / "profile.json"

# profile 结构：facts(事实) / prefs(偏好) / goals(目标)，每类最多条数（防无限膨胀）
_MAX_PER_CATEGORY = 20


def _dir() -> Path:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return _MEMORY_DIR


def load() -> dict:
    """读 profile；没有则返回空结构。"""
    if not _PROFILE_FILE.exists():
        return {"facts": [], "prefs": [], "goals": [], "last_updated": 0}
    try:
        return json.loads(_PROFILE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"facts": [], "prefs": [], "goals": [], "last_updated": 0}


def save(profile: dict) -> None:
    """写 profile（自动补缺失字段 + 截断防膨胀 + 更新时间戳）。"""
    _dir()
    p = profile if profile else {}
    p.setdefault("facts", [])
    p.setdefault("prefs", [])
    p.setdefault("goals", [])
    for key in ("facts", "prefs", "goals"):
        # 去重 + 截断（保序去重）
        seen, cleaned = set(), []
        for item in p[key]:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                cleaned.append(text)
        p[key] = cleaned[-_MAX_PER_CATEGORY:]
    p["last_updated"] = time.time()
    _PROFILE_FILE.write_text(
        json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")


def to_text() -> str:
    """把 profile 拼成给 LLM 的一段文本（提问前回灌用）。
    空 → 返回 ""（前端/agent 判断无需回灌）。"""
    p = load()
    parts = []
    if p.get("facts"):
        parts.append("用户已知信息：" + "；".join(p["facts"]))
    if p.get("prefs"):
        parts.append("用户偏好：" + "；".join(p["prefs"]))
    if p.get("goals"):
        parts.append("用户目标：" + "；".join(p["goals"]))
    return "\n".join(parts)


def reset() -> None:
    """清空记忆（测试/用户主动清）。"""
    if _PROFILE_FILE.exists():
        _PROFILE_FILE.unlink()


if __name__ == "__main__":
    print("\033[92m memory.py 文件\033[0m")
    save({"facts": ["用户叫小明"], "prefs": ["喜欢比喻"], "goals": ["学会 Agent"]})
    print("to_text:")
    print(to_text())
    save({"facts": ["用户叫小明", "用户叫小明", "在学 Python"]})  # 测试去重
    print("合并后 facts:", load()["facts"])
    reset()
    print("reset 后:", load())
