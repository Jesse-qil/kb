"""对 retrieval_cases.json 跑检索，验证 hit@1 不退化。

不做全量 ingest、不跑 benchmark 大报告，只验证：
1. 每个 case 能召回 ≥1 条
2. filtered 路径下 hit@1 ≥ 期望阈值
3. 跨主题混淆用例（react_confusion / agent_confusion / vector_db）能命中正确 topic

依赖：chroma + chunk_cache 已有数据（跑 `python tests/benchmark_retrieval.py --no-ingest` 即可）。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_FILE = Path(__file__).with_name("retrieval_cases.json")


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def query_fn():
    from kb.storage.vector_store import query
    return query


# 各 group 的最低 hit@1 阈值（按用户预期设定）
GROUP_THRESHOLDS = {
    "decorator": 1.0,         # 装饰器用例，应 100% 命中
    "async": 0.6,             # 异步用例
    "agent": 0.6,             # AI-Agent 用例
    "react_confusion": 0.3,   # 至少 1/3 命中 AI-Agent
    "agent_confusion": 0.0,   # LangChain 难命中，容许 0
    "vector_db": 1.0,         # 向量数据库 应 100% 命中
    "pydantic": 0.0,          # Pydantic 容许 0（容错）
}


class TestRetrievalQuality:
    def test_no_case_dropped(self, cases):
        """所有 case 必须有 expected_source + topic + group 字段。"""
        for c in cases:
            assert "expected_source" in c, f"case {c['id']} 缺 expected_source"
            assert "topic" in c, f"case {c['id']} 缺 topic"
            assert "group" in c, f"case {c['id']} 缺 group"

    def test_all_groups_have_thresholds(self, cases):
        """每个 group 必须有预设阈值（防止新增用例忘记设阈值）。"""
        groups = {c["group"] for c in cases}
        missing = groups - GROUP_THRESHOLDS.keys()
        assert not missing, f"group {missing} 未设阈值，先在 GROUP_THRESHOLDS 加"

    def test_filtered_recall_per_group(self, cases, query_fn):
        """各 group 在 filtered 路径下的 hit@1 不低于阈值。"""
        # debug: 确认真实 chroma 状态
        from kb.config import KNOWLEDGE_DIR, chroma_dir
        from kb.storage.vector_store import _collection
        c = _collection()
        print(f'\n[DEBUG] KNOWLEDGE_DIR={KNOWLEDGE_DIR}, chroma_dir={chroma_dir()}, count={c.count()}', flush=True)
        by_group = defaultdict(list)
        debug_log = []
        for c_ in cases:
            hits = query_fn(c_["question"], topic=c_["topic"], top_k=3)
            top1 = hits[0]["source"] if hits else None
            by_group[c_["group"]].append(top1 == c_["expected_source"])
            if not (top1 == c_["expected_source"]):
                debug_log.append(f'  {c_["id"]} expected={c_["expected_source"]!r}, got={top1!r}')

        for group, results in by_group.items():
            threshold = GROUP_THRESHOLDS[group]
            hit1 = sum(results) / len(results)
            assert hit1 >= threshold, (
                f"group={group} hit@1={hit1:.3f} < 阈值 {threshold} "
                f"({sum(results)}/{len(results)})\n未命中明细:\n"
                + "\n".join(debug_log)
            )

    def test_global_recall_minimum(self, cases, query_fn):
        """global 路径 hit@1 ≥ 0.2（topic 过滤去掉后还能搜到一部分）。"""
        # global：不过滤 topic，全库搜
        results = []
        for c in cases:
            hits = query_fn(c["question"], topic="", top_k=5)
            top1 = hits[0]["source"] if hits else None
            results.append(top1 == c["expected_source"])
        hit1 = sum(results) / len(results)
        assert hit1 >= 0.2, f"global hit@1={hit1:.3f} < 0.2（{sum(results)}/{len(results)}）"

    def test_react_not_confused_with_react_js(self, query_fn):
        """ReAct 不应首位命中 React.js 前端。"""
        hits = query_fn("ReAct 是什么", topic="AI-Agent", top_k=3)
        if hits:
            top1 = hits[0]["source"]
            assert "React总纲" not in top1 and "React入门" not in top1, (
                f"ReAct 误命中 React.js: {top1}"
            )