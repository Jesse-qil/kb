# -*- coding: utf-8 -*-
"""kb/memory.py 长期记忆接线测试。
覆盖：to_text 回灌、update_from_conversation 提炼、_parse_extract_json 容错、reset。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb import memory


class _FakeLLM:
    """模拟 LLM：调用 chat 时返回预置 JSON。"""

    def __init__(self, response: str):
        self._resp = response
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self._resp


# ===================== 基础读写 =====================
class TestLoadSave:
    def test_load_empty_returns_default(self, tmp_path, monkeypatch):
        """profile.json 不存在时 load 返回空结构。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        p = memory.load()
        assert p == {"facts": [], "prefs": [], "goals": [], "last_updated": 0}

    def test_save_dedup_and_truncate(self, tmp_path, monkeypatch):
        """save 应去重 + 截断到 _MAX_PER_CATEGORY。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        # 25 条 facts + 5 条重复 → 去重后 25，截断到 20（保尾段）
        facts = [f"事实{i}" for i in range(20)] + ["重复A", "重复A", "重复B", "重复B", "重复C"]
        memory.save({"facts": facts, "prefs": ["p"], "goals": ["g"]})
        loaded = memory.load()
        assert len(loaded["facts"]) == memory._MAX_PER_CATEGORY
        # 末尾应是 "重复C"，不是被截掉
        assert loaded["facts"][-1] == "重复C"

    def test_save_sets_last_updated(self, tmp_path, monkeypatch):
        """save 应刷新 last_updated。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        before = memory.load()["last_updated"]
        memory.save({"facts": ["x"]})
        after = memory.load()["last_updated"]
        assert after > before


# ===================== to_text 回灌格式 =====================
class TestToText:
    def test_empty_profile_returns_empty_string(self, tmp_path, monkeypatch):
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        assert memory.to_text() == ""

    def test_to_text_includes_all_categories(self, tmp_path, monkeypatch):
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        memory.save({
            "facts": ["学 Python 半年"],
            "prefs": ["喜欢比喻"],
            "goals": ["准备面试"],
        })
        text = memory.to_text()
        assert "用户已知信息" in text
        assert "学 Python 半年" in text
        assert "用户偏好" in text
        assert "喜欢比喻" in text
        assert "用户目标" in text
        assert "准备面试" in text


# ===================== _parse_extract_json 容错 =====================
class TestParseExtractJson:
    def test_plain_json(self):
        data = memory._parse_extract_json('{"facts":["a"],"prefs":[],"goals":[]}')
        assert data == {"facts": ["a"], "prefs": [], "goals": []}

    def test_markdown_wrapped(self):
        """LLM 偶发 ```json\n...\n``` 包裹 → 必须解包。"""
        raw = "```json\n{\"facts\":[\"a\"],\"prefs\":[\"b\"],\"goals\":[]}\n```"
        data = memory._parse_extract_json(raw)
        assert data["facts"] == ["a"]
        assert data["prefs"] == ["b"]

    def test_with_preamble_text(self):
        """LLM 偶发先写「好的，下面是结果：」再 JSON → 必须从 { 起截到 } 止。"""
        raw = "好的，下面是结果：\n{\"facts\":[\"x\"],\"prefs\":[],\"goals\":[]}"
        data = memory._parse_extract_json(raw)
        assert data["facts"] == ["x"]

    def test_empty_response(self):
        """LLM 返回空 → 不抛异常，返回空结构。"""
        data = memory._parse_extract_json("")
        assert data == {"facts": [], "prefs": [], "goals": []}

    def test_invalid_json_returns_empty(self):
        """完全无效 JSON → 返回空结构（容错）。"""
        data = memory._parse_extract_json("这不是 JSON")
        assert data == {"facts": [], "prefs": [], "goals": []}

    def test_filters_non_string_items(self):
        """非字符串条目（如 None / int）应被过滤。"""
        raw = '{"facts":[null, 1, "x", ""],"prefs":[],"goals":[]}'
        data = memory._parse_extract_json(raw)
        # 空字符串和 None/数字都被过滤，只留 "x"
        assert data["facts"] == ["x"]


# ===================== update_from_conversation =====================
class TestUpdateFromConversation:
    def test_writes_extracted_facts_into_profile(self, tmp_path, monkeypatch):
        """LLM 返回正常 JSON → 写入 profile 并去重。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        memory.save({"facts": ["已有事实"], "prefs": [], "goals": []})

        llm = _FakeLLM('{"facts":["新事实","已有事实"], "prefs":["新偏好"], "goals":[]}')
        ok = memory.update_from_conversation("在学 Python", "好的", llm)
        assert ok is True

        loaded = memory.load()
        # 新事实被加进去；已有的"已有事实"去重保留一份
        assert "新事实" in loaded["facts"]
        assert "已有事实" in loaded["facts"]
        assert loaded["facts"].count("已有事实") == 1
        assert "新偏好" in loaded["prefs"]

    def test_no_new_info_returns_false(self, tmp_path, monkeypatch):
        """LLM 返回空 JSON → 不写盘，返回 False。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        memory.save({"facts": ["x"], "prefs": [], "goals": []})

        llm = _FakeLLM('{"facts":[], "prefs":[], "goals":[]}')
        ok = memory.update_from_conversation("...", "...", llm)
        assert ok is False

    def test_markdown_wrapped_response_works(self, tmp_path, monkeypatch):
        """LLM 用 ```json 包裹 → 仍能解析成功。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")

        llm = _FakeLLM('```json\n{"facts":["x"], "prefs":[], "goals":[]}\n```')
        ok = memory.update_from_conversation("q", "a", llm)
        assert ok is True
        assert memory.load()["facts"] == ["x"]

    def test_invalid_json_does_not_raise(self, tmp_path, monkeypatch):
        """LLM 返回无效 JSON → 静默返回 False，不抛异常。"""
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")

        llm = _FakeLLM("随便说什么")
        ok = memory.update_from_conversation("q", "a", llm)
        assert ok is False
        # profile 不被破坏
        assert memory.load()["facts"] == []

    def test_llm_raises_does_not_propagate(self, tmp_path, monkeypatch):
        """LLM 调用抛异常 → update_from_conversation 捕获并返回 False，不影响调用方。"""

        class BoomLLM:
            def chat(self, messages):
                raise RuntimeError("网络炸了")

        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")

        ok = memory.update_from_conversation("q", "a", BoomLLM())
        assert ok is False


# ===================== reset =====================
class TestReset:
    def test_reset_removes_file(self, tmp_path, monkeypatch):
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        memory.save({"facts": ["x"]})
        assert memory._PROFILE_FILE.exists()
        memory.reset()
        assert not memory._PROFILE_FILE.exists()
        assert memory.load()["facts"] == []

    def test_reset_when_no_file_is_noop(self, tmp_path, monkeypatch):
        fake = tmp_path / "knowledge"
        fake.mkdir()
        monkeypatch.setattr(memory, "_PROFILE_FILE", fake / "memory" / "profile.json")
        # 没文件也不抛异常
        memory.reset()
        assert memory.load()["facts"] == []