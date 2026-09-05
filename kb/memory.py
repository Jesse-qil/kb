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
    # 用 _PROFILE_FILE.parent 而非 _MEMORY_DIR 常量，方便测试 patch
    _PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    return _PROFILE_FILE.parent


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


def _parse_extract_json(raw: str) -> dict:
    """解析 LLM 提炼出的 JSON：容错处理 ```json 包裹 / 前置说明文字。
    无效 JSON → 返回空结构（绝不抛异常）。"""
    text = (raw or "").strip()
    if not text:
        return {"facts": [], "prefs": [], "goals": []}
    # 去掉 markdown 包裹
    if text.startswith("```"):
        # 找到第一个换行后的内容，到最后一个 ```
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if "```" in text:
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    # 兜底：找第一个 { 和最后一个 }
    if "{" in text and "}" in text:
        text = text[text.index("{"): text.rindex("}") + 1]
    try:
        data = json.loads(text)
    except Exception:
        return {"facts": [], "prefs": [], "goals": []}
    # 容错：确保三类都是 list of str（过滤 None/int/空字符串）
    out = {"facts": [], "prefs": [], "goals": []}
    if not isinstance(data, dict):
        return out
    for cat in ("facts", "prefs", "goals"):
        v = data.get(cat, [])
        if isinstance(v, list):
            for x in v:
                # 只接受非空字符串（None / int / 空串都丢）
                if isinstance(x, str) and x.strip():
                    out[cat].append(x.strip())
    return out


def update_from_conversation(question: str, answer: str, llm) -> bool:
    """从一轮对话中提炼用户画像并合并写回 profile。
    失败静默不影响主对话（异常会被调用方 try/except 兜住）。
    返回 True=成功写入 / False=失败。"""
    try:
        from .prompts import MEMORY_EXTRACTOR_PROMPT
        user = f"用户问：{question}\n\n助手答：{answer}"
        raw = llm.chat([
            {"role": "system", "content": MEMORY_EXTRACTOR_PROMPT},
            {"role": "user", "content": user},
        ])
        extracted = _parse_extract_json(raw)
        # 没新信息就不写盘
        if not any(extracted.values()):
            return False
        existing = load()
        for cat in ("facts", "prefs", "goals"):
            existing[cat] = list(existing.get(cat, [])) + extracted[cat]
        save(existing)
        return True
    except Exception as e:
        # 不抛 — 让主对话流不受影响
        print(f"[memory] update_from_conversation 失败（已忽略）: {e}")
        return False


if __name__ == "__main__":
    print("\033[92m memory.py 文件\033[0m")
    save({"facts": ["用户叫小明"], "prefs": ["喜欢比喻"], "goals": ["学会 Agent"]})
    print("to_text:")
    print(to_text())
    save({"facts": ["用户叫小明", "用户叫小明", "在学 Python"]})  # 测试去重
    print("合并后 facts:", load()["facts"])
    # 测试提炼 JSON 容错解析
    print("解析 markdown 包裹:", _parse_extract_json("```json\n{\"facts\":[\"x\"],\"prefs\":[],\"goals\":[]}\n```"))
    print("解析带前置文字:", _parse_extract_json("好的，下面是结果：{\"facts\":[\"y\"],\"prefs\":[],\"goals\":[]}"))
    reset()
    print("reset 后:", load())
