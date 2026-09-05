# -*- coding: utf-8 -*-
"""chunk_cache 核心 API 测试：put / get / put_many / clear / 缓存一致性。

所有测试使用 isolated_cache fixture 重定向到临时目录，不污染真实知识库。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from kb.ingestion import chunk_cache


# ===================== put + get =====================
class TestPutGet:
    """put 与 get 的往返行为。"""

    def test_put_get_roundtrip(self, isolated_cache, sample_row):
        """put 后 get 能完整取回所有字段。"""
        chunk_cache.put(
            sample_row["content_hash"],
            sample_row["chunk_size"],
            sample_row["chunk_overlap"],
            sample_row["model_name"],
            tags=sample_row["tags"],
            chunks=sample_row["chunks"],
        )
        hit = chunk_cache.get(
            sample_row["content_hash"],
            sample_row["chunk_size"],
            sample_row["chunk_overlap"],
            sample_row["model_name"],
        )
        assert hit is not None
        assert hit["content_hash"] == sample_row["content_hash"]
        assert hit["tags"] == ["Python基础", "装饰器"]
        assert len(hit["chunks"]) == 2
        # embedding 必须保持 512 维
        assert len(hit["chunks"][0]["embedding"]) == 512
        assert len(hit["chunks"][1]["embedding"]) == 512
        # embedding 值精度（float32 → 7 位有效数字）
        np.testing.assert_allclose(
            hit["chunks"][0]["embedding"],
            sample_row["chunks"][0]["embedding"],
            atol=1e-5,
        )
        # 文本/标题完整保留
        assert hit["chunks"][0]["text"] == "这是第一个 chunk 的文本"
        assert hit["chunks"][0]["heading"] == "简介"

    def test_get_miss_returns_none(self, isolated_cache):
        """get 不存在的 key 返回 None，不抛异常。"""
        assert chunk_cache.get("nonexistent_hash", 400, 120, "any_model") is None

    def test_get_distinguishes_chunk_size(self, isolated_cache, sample_row):
        """相同 hash 不同 chunk_size 视为不同缓存条目。"""
        chunk_cache.put(
            sample_row["content_hash"], 400, 120, sample_row["model_name"],
            tags=sample_row["tags"], chunks=sample_row["chunks"],
        )
        # 换 chunk_size：miss
        assert chunk_cache.get(
            sample_row["content_hash"], 600, 120, sample_row["model_name"],
        ) is None
        # 原 chunk_size：hit
        assert chunk_cache.get(
            sample_row["content_hash"], 400, 120, sample_row["model_name"],
        ) is not None

    def test_get_distinguishes_model_name(self, isolated_cache, sample_row):
        """相同 hash 不同 model_name 视为不同缓存条目。"""
        chunk_cache.put(
            sample_row["content_hash"], 400, 120, "model-A",
            tags=sample_row["tags"], chunks=sample_row["chunks"],
        )
        assert chunk_cache.get(
            sample_row["content_hash"], 400, 120, "model-B",
        ) is None
        assert chunk_cache.get(
            sample_row["content_hash"], 400, 120, "model-A",
        ) is not None

    def test_get_empty_hash_returns_none(self, isolated_cache):
        """content_hash 为空字符串直接返回 None（不写索引污染）。"""
        assert chunk_cache.get("", 400, 120, "any_model") is None

    def test_put_overwrites_same_key(self, isolated_cache, sample_row):
        """同一 key 二次 put → 后者覆盖前者，tags/chunks 都更新。"""
        chunk_cache.put(
            sample_row["content_hash"], 400, 120, sample_row["model_name"],
            tags=["old_tag"], chunks=[{"text": "old", "heading": "h", "embedding": [0.1] * 512}],
        )
        chunk_cache.put(
            sample_row["content_hash"], 400, 120, sample_row["model_name"],
            tags=["new_tag"], chunks=[{"text": "new", "heading": "h", "embedding": [0.2] * 512}],
        )
        hit = chunk_cache.get(
            sample_row["content_hash"], 400, 120, sample_row["model_name"],
        )
        assert hit["tags"] == ["new_tag"]
        assert len(hit["chunks"]) == 1
        assert hit["chunks"][0]["text"] == "new"


# ===================== put_many =====================
class TestPutMany:
    """put_many 的批量写入语义。"""

    def test_put_many_returns_count(self, isolated_cache):
        """put_many 返回实际写入条数。"""
        rows = [
            {
                "content_hash": f"hm_{i}",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": [f"t{i}"],
                "chunks": [{"text": f"c{i}", "heading": "h", "embedding": [float(i)] * 512}],
            }
            for i in range(5)
        ]
        n = chunk_cache.put_many(rows)
        assert n == 5

    def test_put_many_empty_input(self, isolated_cache):
        """空输入返回 0，不写文件不报错。"""
        assert chunk_cache.put_many([]) == 0

    def test_put_many_appends_to_emb_array(self, isolated_cache):
        """多次 put_many 后 emb.npy 应拼接所有 chunks（行数 = 总 chunks 数）。"""
        for batch in range(3):
            chunk_cache.put_many([
                {
                    "content_hash": f"app_b{batch}_h{h}",
                    "chunk_size": 400,
                    "chunk_overlap": 120,
                    "model_name": "M",
                    "tags": [],
                    "chunks": [
                        {"text": f"t{batch}_{h}_{k}", "heading": "h", "embedding": [float(k)] * 512}
                        for k in range(2)  # 每行 2 chunks
                    ],
                }
                for h in range(2)  # 每批 2 行
            ])
        emb = np.load(isolated_cache / "chunks" / "chunk_cache.emb.npy")
        # emb.npy 是 1D：3 批 × 2 行 × 2 chunks = 12 chunks × 512 dim = 6144
        assert emb.shape == (12 * 512,)
        assert emb.dtype == np.float32

    def test_put_many_overwrite_same_key(self, isolated_cache):
        """put_many 写入同 key 多次 → 后写覆盖前写，但 emb 数组会膨胀（旧 emb 残留）。

        这是已知实现取舍：覆盖语义下 emb.npy 不做 GC，简单靠 meta 指向新 offset。
        验证：get 拿到的总以最新为准，且最新 chunks 数与新 rows 一致。"""
        chunk_cache.put_many([
            {
                "content_hash": "dup", "chunk_size": 400, "chunk_overlap": 120,
                "model_name": "M", "tags": ["v1"],
                "chunks": [
                    {"text": "v1_c1", "heading": "h", "embedding": [1.0] * 512},
                    {"text": "v1_c2", "heading": "h", "embedding": [1.0] * 512},
                ],
            }
        ])
        chunk_cache.put_many([
            {
                "content_hash": "dup", "chunk_size": 400, "chunk_overlap": 120,
                "model_name": "M", "tags": ["v2"],
                "chunks": [
                    {"text": "v2_c1", "heading": "h", "embedding": [2.0] * 512},
                ],
            }
        ])
        hit = chunk_cache.get("dup", 400, 120, "M")
        assert hit["tags"] == ["v2"]
        assert len(hit["chunks"]) == 1
        assert hit["chunks"][0]["text"] == "v2_c1"

    def test_put_many_with_chunks_without_embedding(self, isolated_cache):
        """chunk 缺 embedding 时 → 写入 meta 但 emb_off=-1，get 返回 embedding=None。"""
        chunk_cache.put_many([
            {
                "content_hash": "no_emb", "chunk_size": 400, "chunk_overlap": 120,
                "model_name": "M", "tags": [],
                "chunks": [{"text": "no emb", "heading": "h"}],  # 没有 embedding 字段
            }
        ])
        hit = chunk_cache.get("no_emb", 400, 120, "M")
        assert hit is not None
        assert hit["chunks"][0]["text"] == "no emb"
        assert hit["chunks"][0]["embedding"] is None


# ===================== 磁盘文件布局 =====================
class TestFileLayout:
    """新格式磁盘文件正确生成。"""

    def test_meta_and_emb_files_created(self, isolated_cache, sample_row):
        """put_many 后应生成 .meta.jsonl + .emb.npy，不再用旧的 .jsonl。"""
        chunk_cache.put_many([sample_row])
        kb = isolated_cache / "chunks"
        assert (kb / "chunk_cache.meta.jsonl").exists()
        assert (kb / "chunk_cache.emb.npy").exists()

    def test_meta_jsonl_format(self, isolated_cache, sample_row):
        """meta.jsonl 行结构：含 key/content_hash/tags/chunks，但 chunks 内只存文本+offset，无 embedding。"""
        chunk_cache.put_many([sample_row])
        meta_file = isolated_cache / "chunks" / "chunk_cache.meta.jsonl"
        lines = [ln for ln in meta_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["content_hash"] == sample_row["content_hash"]
        assert row["tags"] == ["Python基础", "装饰器"]
        assert "embedding" not in row["chunks"][0]  # v2 关键：meta 不再含 embedding 数组
        assert "emb_off" in row["chunks"][0]
        assert "emb_len" in row["chunks"][0]

    def test_emb_npy_is_1d_float32(self, isolated_cache, sample_row):
        """emb.npy 是 1D numpy 数组，shape=(N_total_chunks * 512,)，dtype=float32。
        每条 chunk 的 embedding 长度固定 512，按写入顺序顺序拼接；通过 meta.emb_off 切片读取。
        """
        chunk_cache.put_many([sample_row])
        emb_file = isolated_cache / "chunks" / "chunk_cache.emb.npy"
        arr = np.load(emb_file)
        assert arr.ndim == 1
        assert arr.dtype == np.float32
        # sample_row 含 2 chunks × 512 dim = 1024
        assert arr.shape == (1024,)


# ===================== clear =====================
class TestClear:
    def test_clear_removes_all_files(self, isolated_cache, sample_row):
        chunk_cache.put_many([sample_row])
        chunk_cache.clear()
        kb = isolated_cache / "chunks"
        # clear 应删掉 meta + emb + 旧 jsonl
        assert not (kb / "chunk_cache.meta.jsonl").exists()
        assert not (kb / "chunk_cache.emb.npy").exists()
        # get 应 miss
        assert chunk_cache.get(
            sample_row["content_hash"], 400, 120, sample_row["model_name"],
        ) is None


# ===================== 模块级缓存一致性 =====================
class TestCacheInvalidation:
    """put_many / clear 后模块级 _ROWS_CACHE 和 _EMB_CACHE 必须失效。"""

    def test_put_many_invalidates_module_cache(self, isolated_cache, sample_row):
        """第一次 get 后模块级 rows 缓存住；put_many 新数据后 get 必须能读到。"""
        # 先 get miss（让 _load_meta_rows 跑一次，建立空缓存）
        assert chunk_cache.get("first", 400, 120, "M") is None
        # 再 put_many
        chunk_cache.put_many([sample_row])
        # 必须能读到（不读到 = 缓存没失效 = bug）
        hit = chunk_cache.get(
            sample_row["content_hash"],
            sample_row["chunk_size"],
            sample_row["chunk_overlap"],
            sample_row["model_name"],
        )
        assert hit is not None
        assert hit["tags"] == sample_row["tags"]

    def test_clear_invalidates_module_cache(self, isolated_cache, sample_row):
        chunk_cache.put_many([sample_row])
        chunk_cache._load_meta_rows()  # 触发缓存填充
        chunk_cache._read_emb()        # 触发 emb 缓存填充
        chunk_cache.clear()
        # 缓存应已清空
        assert chunk_cache._ROWS_CACHE is None
        assert chunk_cache._EMB_CACHE is None