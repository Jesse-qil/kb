# -*- coding: utf-8 -*-
"""reranker 测试：核心是"模型不可用静默降级、不破坏主链路" + 打分重排正确性。"""
from __future__ import annotations

import kb.rerank as rr


def test_no_model_returns_original(monkeypatch):
    """无模型（加载失败/未装依赖）→ 原样返回，不抛异常、不加分。"""
    monkeypatch.setattr(rr, "_load_model", lambda: None)
    hits = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
    out = rr.rerank("测试", hits)
    assert out == hits
    assert "rerank_score" not in out[0]


def test_empty_and_single_short_circuit(monkeypatch):
    """空/单元素候选直接返回，不做无谓打分。"""
    assert rr.rerank("q", []) == []
    one = [{"text": "x", "score": 0.5}]
    assert rr.rerank("q", one) == one


def test_rerank_sorts_by_score(monkeypatch):
    """加分微调：rerank 分差×权重并入 score，高分靠前、低分靠后，长度不变。"""

    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [float(len(pairs) - i) for i in range(len(pairs))]

    monkeypatch.setattr(rr, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(rr, "rerank_cfg",
                        lambda: {"top_n": 30, "weight": 0.15})
    hits = [{"text": f"c{i}", "score": 0.1, "source": f"s{i}"} for i in range(5)]
    out = rr.rerank("q", hits)
    assert len(out) == 5
    assert out[0]["text"] == "c0"          # rerank 5.0 最高（base=1，加 0.6）
    assert out[-1]["text"] == "c4"         # rerank 1.0 最低（加 0）
    assert all("rerank_score" in h for h in out)
    assert out[0]["rerank_score"] == 5.0
    # 加分并入 score：0.1 + 0.15*(5-1) = 0.7
    assert abs(out[0]["score"] - 0.7) < 1e-9


def test_rerank_predict_failure_degrades(monkeypatch):
    """打分抛异常 → 原样返回（静默降级）。"""

    class BadModel:
        def predict(self, pairs, show_progress_bar=False):
            raise RuntimeError("boom")

    monkeypatch.setattr(rr, "_load_model", lambda: BadModel())
    hits = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
    out = rr.rerank("q", hits)
    assert out == hits
