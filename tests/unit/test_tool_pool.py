"""ToolPool 工具池测试：注册 / 枚举 / 调用 / 统计 / 异常兜底。"""
import pytest

from kb.agents.tool_pool import ToolPool


def make_pool() -> ToolPool:
    pool = ToolPool()
    pool.register(
        "add", "两个数相加",
        {"type": "object", "properties": {
            "a": {"type": "number"}, "b": {"type": "number"}}},
        lambda a, b: str(a + b),
        category="执行", required=["a", "b"],
    )
    pool.register(
        "boom", "必然抛异常的工具",
        {"type": "object", "properties": {"x": {"type": "string"}}},
        lambda x: (_ for _ in ()).throw(RuntimeError("爆炸")),
    )
    return pool


def test_register_and_enumerate():
    pool = make_pool()
    assert pool.names() == ["add", "boom"]
    specs = pool.list_all()
    assert [s.name for s in specs] == ["add", "boom"]
    assert specs[0].category == "执行"


def test_schemas_openai_format():
    pool = make_pool()
    schemas = pool.schemas()
    assert len(schemas) == 2
    assert schemas[0]["type"] == "function"
    fn = schemas[0]["function"]
    assert fn["name"] == "add"
    assert fn["parameters"]["required"] == ["a", "b"]


def test_call_and_stats():
    pool = make_pool()
    assert pool.call("add", a=2, b=3) == "5"
    assert pool.call("add", a=10, b=1) == "11"
    stats = {s["name"]: s["calls"] for s in pool.stats()}
    assert stats["add"] == 2
    assert stats["boom"] == 0


def test_call_unknown_tool():
    pool = make_pool()
    result = pool.call("no_such_tool", x=1)
    assert "工具不存在" in result
    # 未注册工具不增加任何统计
    assert all(s["calls"] == 0 for s in pool.stats())


def test_call_exception_returns_text():
    pool = make_pool()
    result = pool.call("boom", x="y")
    assert "工具执行失败" in result
    assert "RuntimeError" in result
    # 异常也会计数（确实尝试调用了）
    stats = {s["name"]: s["calls"] for s in pool.stats()}
    assert stats["boom"] == 1


def test_get_pool_singleton():
    from kb.agents.tool_pool import get_pool
    assert get_pool() is get_pool()
