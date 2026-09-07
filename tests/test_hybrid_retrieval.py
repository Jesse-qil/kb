"""混合检索（向量 + BM25 关键词通道）单元测试。

覆盖：
1. BM25 通道召回：专有名词查询能保底召回关键词命中片段（pydantic / LangGraph 等）
2. 融合排序：命中更多查询词的片段排在同池单命中片段之前
3. 分词：停用词 + 中文单字过滤、标签词注入 jieba 用户词典
4. 降级：_HAS_BM25=False 时行为与纯向量一致、不报错
5. 返回结构完整：text/source/topic/heading/tags/score

依赖：chroma + chunk_cache 已有数据（跑 `python tests/benchmark_retrieval.py --no-ingest` 即可）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.storage import vector_store as vs
from kb.storage.vector_store import _search_by_bm25, _tokenize, query


@pytest.fixture(autouse=True)
def reset_bm25_cache():
    """每个用例前重置 BM25 索引缓存，避免跨用例状态污染。"""
    vs._bm25_cache = {"count": -1, "bm25": None, "texts": [], "metas": [], "lookup": {}}
    yield


class TestBM25Channel:
    def test_search_by_bm25_returns_exact_term_hits(self):
        """专有名词查询：BM25 通道应召回含该词的片段。"""
        hits = _search_by_bm25("pydantic BaseModel", k=5)
        assert hits, "BM25 通道应召回 pydantic 片段"
        texts = " ".join(h["text"] for h in hits).lower()
        assert "pydantic" in texts, "BM25 top 片段应包含 pydantic 关键词"

    def test_bm25_topic_filter(self):
        """topic 过滤：BM25 通道只返回指定主题。"""
        hits = _search_by_bm25("LangGraph", k=10, topic="AI-Agent")
        assert hits, "AI-Agent 主题下 LangGraph 应有关键词命中"
        assert all(h["topic"] == "AI-Agent" for h in hits)

    def test_bm25_empty_question(self):
        assert _search_by_bm25("", k=5) == []


class TestHybridFusion:
    def test_multi_term_hit_outranks_single_term(self):
        """chroma+milvus 双词命中文档应排在单命中片段之前（向量数据库文档进 top1）。"""
        hits = query("chroma 和 milvus 区别", topic="数据库", top_k=3)
        assert hits, "混合检索应返回结果"
        top1 = hits[0]["source"]
        assert "向量数据库" in top1, f"期望向量数据库文档排前，实际 top1={top1}"

    def test_hybrid_score_range(self):
        """融合分并入 score：向量 ≤1 + BM25 boost 0.10 + 标签加分上限 0.12。"""
        hits = query("LangGraph 多智能体怎么做", topic="AI-Agent", top_k=5)
        assert hits
        for h in hits:
            assert 0 <= h["score"] <= 1.3, f"score 越界: {h['score']}"

    def test_query_result_structure(self):
        """返回结构完整：text/source/topic/heading/tags/score。"""
        hits = query("装饰器怎么写", topic="Python基础", top_k=3)
        assert hits, "装饰器查询应返回结果"
        for h in hits:
            for key in ("text", "source", "topic", "heading", "tags", "score"):
                assert key in h, f"结果缺 {key} 字段"


class TestTokenize:
    def test_stopword_and_single_char_filter(self):
        """停用词 + 中文单字过滤：'装饰器怎么写' → ['装饰器']（标签词注入 jieba）。"""
        assert _tokenize("装饰器怎么写", drop_stop=True) == ["装饰器"]

    def test_chroma_milvus_tokens(self):
        """停用词过滤：'和'/'区别' 不应进 BM25 查询词。"""
        toks = _tokenize("chroma 和 milvus 区别", drop_stop=True)
        assert "和" not in toks and "区别" not in toks
        assert "chroma" in toks and "milvus" in toks

    def test_index_tokenize_keeps_all(self):
        """建索引分词不过滤停用词（保 recall）。"""
        toks = _tokenize("的 和 是", drop_stop=False)
        assert len(toks) >= 3


class TestDegradation:
    def test_no_bm25_falls_back_to_vector(self, monkeypatch):
        """_HAS_BM25=False（缺 jieba/rank_bm25）时行为与纯向量一致，不报错。"""
        monkeypatch.setattr(vs, "_HAS_BM25", False)
        hits = query("装饰器怎么写", topic="Python基础", top_k=3)
        assert hits, "降级路径应正常返回"
        assert all(0 <= h["score"] <= 1.0 for h in hits)
