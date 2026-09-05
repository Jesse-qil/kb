# -*- coding: utf-8 -*-
"""chunk_cache 自动迁移测试：旧版 chunk_cache.jsonl（行内带 embedding）
首次加载时应自动迁移到 v2 meta+emb 拆分格式。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from kb.ingestion import chunk_cache


def _seed_legacy_format(isolated_cache: Path, rows: list[dict]) -> Path:
    """手工写一个旧版 chunk_cache.jsonl，返回文件路径。"""
    legacy = isolated_cache / "chunks" / "chunk_cache.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return legacy


class TestMigration:
    def test_migration_creates_meta_and_emb(self, isolated_cache):
        """旧 .jsonl 首次 _load_rows() 后应生成 meta.jsonl + emb.npy。"""
        rows = [
            {
                "key": f"k{i}|400|120|M",
                "content_hash": f"k{i}",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": [f"tag{i}"],
                "chunks": [
                    {"text": f"text {i}", "heading": "h", "embedding": [float(i) / 256] * 512}
                ],
                "updated": 0.0,
            }
            for i in range(3)
        ]
        _seed_legacy_format(isolated_cache, rows)
        # 触发迁移
        loaded = chunk_cache._load_rows()
        assert len(loaded) == 3
        # 新格式文件应已生成
        kb = isolated_cache / "chunks"
        assert (kb / "chunk_cache.meta.jsonl").exists()
        assert (kb / "chunk_cache.emb.npy").exists()

    def test_migration_preserves_embedding_precision(self, isolated_cache):
        """迁移后 get 拿到的 embedding 与原 JSON 行内值**完全一致**（< 1e-6）。"""
        rows = [
            {
                "key": f"precise|400|120|M",
                "content_hash": "precise",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": ["精度"],
                "chunks": [
                    {"text": "精度测试", "heading": "h1", "embedding": [0.123456] * 512},
                    {"text": "第二条", "heading": "h2", "embedding": [-0.987654] * 512},
                ],
                "updated": 0.0,
            }
        ]
        _seed_legacy_format(isolated_cache, rows)
        # 触发迁移
        chunk_cache._load_rows()
        # 读回验证精度
        hit = chunk_cache.get("precise", 400, 120, "M")
        assert hit is not None
        assert len(hit["chunks"]) == 2
        np.testing.assert_allclose(
            hit["chunks"][0]["embedding"],
            [0.123456] * 512,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            hit["chunks"][1]["embedding"],
            [-0.987654] * 512,
            atol=1e-6,
        )

    def test_migration_flag_prevents_repeat(self, isolated_cache):
        """迁移标记存在时，即使旧 .jsonl 残留也不再重复迁移。"""
        rows = [
            {
                "key": "flag|400|120|M",
                "content_hash": "flag",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": [],
                "chunks": [{"text": "x", "heading": "h", "embedding": [0.0] * 512}],
                "updated": 0.0,
            }
        ]
        _seed_legacy_format(isolated_cache, rows)
        chunk_cache._load_rows()  # 第一次迁移
        kb = isolated_cache / "chunks"
        legacy = kb / "chunk_cache.jsonl"
        flag = kb / ".jsonl.migrated.flag"
        # 迁移标记必须存在
        assert flag.exists(), "迁移标记应已创建"
        # 旧 .jsonl 可能被成功 rename 删除（属合法状态）
        # 模拟「rename 失败 → 旧文件还在」：仅在不存在时回填占位
        if not legacy.exists():
            legacy.write_text("placeholder\n", encoding="utf-8")
        chunk_cache._invalidate_cache()
        loaded = chunk_cache._load_rows()
        assert len(loaded) == 1
        # 二次 _load_rows() 不应再触发迁移（meta mtime 不变）
        meta_mtime = (kb / "chunk_cache.meta.jsonl").stat().st_mtime
        time.sleep(0.05)
        chunk_cache._load_rows()
        assert (kb / "chunk_cache.meta.jsonl").stat().st_mtime == meta_mtime

    def test_migration_handles_missing_embedding_field(self, isolated_cache):
        """旧数据里某些 chunk 缺 embedding 字段 → 迁移后 emb_off=-1，get 返回 embedding=None。"""
        rows = [
            {
                "key": "missing|400|120|M",
                "content_hash": "missing",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": [],
                "chunks": [
                    {"text": "有 emb", "heading": "h", "embedding": [0.5] * 512},
                    {"text": "缺 emb", "heading": "h"},  # 没 embedding 字段
                ],
                "updated": 0.0,
            }
        ]
        _seed_legacy_format(isolated_cache, rows)
        chunk_cache._load_rows()
        hit = chunk_cache.get("missing", 400, 120, "M")
        assert hit is not None
        assert hit["chunks"][0]["embedding"] is not None
        assert hit["chunks"][1]["embedding"] is None

    def test_migration_idempotent_via_flag(self, isolated_cache):
        """预先放好 .jsonl.migrated.flag + 旧 .jsonl → _load_rows() 不报错，跳过迁移。"""
        kb = isolated_cache / "chunks"
        kb.mkdir(parents=True, exist_ok=True)
        # 旧 .jsonl 写脏数据（验证不会被迁移读取）
        legacy = kb / "chunk_cache.jsonl"
        legacy.write_text("garbage\n", encoding="utf-8")
        # 迁移标记
        flag = kb / ".jsonl.migrated.flag"
        flag.write_text("migrated\n", encoding="utf-8")
        # 不应抛异常，rows 应为空（meta 不存在）
        rows = chunk_cache._load_rows()
        assert rows == {}

    def test_migration_emb_offset_correct(self, isolated_cache):
        """多文件迁移后 emb_off 应当是连续的（顺序拼接）。"""
        rows = []
        for i in range(5):
            n_chunks = i + 1  # 文件 0: 1 chunk, 文件 1: 2 chunks, ...
            rows.append({
                "key": f"off_{i}|400|120|M",
                "content_hash": f"off_{i}",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": [],
                "chunks": [
                    {"text": f"f{i}_c{j}", "heading": "h", "embedding": [float(j)] * 512}
                    for j in range(n_chunks)
                ],
                "updated": 0.0,
            })
        _seed_legacy_format(isolated_cache, rows)
        loaded = chunk_cache._load_rows()
        # emb.npy 是 1D：总 chunk 数 = 1+2+3+4+5 = 15，乘以 512
        emb = np.load(isolated_cache / "chunks" / "chunk_cache.emb.npy")
        assert emb.shape == (15 * 512,)
        # meta 里 emb_off 应该按 512 维连续递增（emb_off 是 dim 单位，不是 chunk 单位）
        cur = 0
        for i in range(5):
            row = loaded[f"off_{i}|400|120|M"]
            for j in range(i + 1):
                assert row["chunks"][j]["emb_off"] == cur, (
                    f"file={i} chunk={j} 期望 emb_off={cur}, 实际 {row['chunks'][j]['emb_off']}"
                )
                cur += 512  # 每条 chunk 占用 512 dim


class TestLegacyDetection:
    def test_no_legacy_no_migration(self, isolated_cache):
        """meta.jsonl 不存在且无旧 .jsonl → 不触发迁移，rows 为空。"""
        rows = chunk_cache._load_rows()
        assert rows == {}

    def test_meta_present_skips_legacy(self, isolated_cache):
        """已有 meta + 旧 .jsonl 残留 + flag → 跳过迁移，rows 从 meta 加载。"""
        kb = isolated_cache / "chunks"
        kb.mkdir(parents=True, exist_ok=True)
        # meta 写一条新格式数据
        meta = kb / "chunk_cache.meta.jsonl"
        meta.write_text(
            json.dumps({
                "key": "new|400|120|M",
                "content_hash": "new",
                "chunk_size": 400,
                "chunk_overlap": 120,
                "model_name": "M",
                "tags": ["新格式"],
                "chunks": [{"text": "x", "heading": "h", "emb_off": 0, "emb_len": 512}],
                "updated": 0.0,
            }, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # emb 也写一个 1×512 的数组
        np.save(kb / "chunk_cache.emb.npy", np.zeros((1, 512), dtype=np.float32))
        # 旧 .jsonl 写垃圾数据（验证不会被读取）
        (kb / "chunk_cache.jsonl").write_text("garbage\n", encoding="utf-8")
        # 迁移标记也放
        (kb / ".jsonl.migrated.flag").write_text("migrated\n", encoding="utf-8")
        # 加载
        rows = chunk_cache._load_rows()
        assert "new|400|120|M" in rows
        # 不应抛 JSON parse 异常