# -*- coding: utf-8 -*-
"""pytest 全局 fixture：把 chunk_cache 重定向到临时目录，避免污染真实知识库。

策略：monkeypatch `kb.config.KNOWLEDGE_DIR` → 重新加载 `kb.ingestion.chunk_cache`
（因为 `_TX_LOCK_PATH` 是模块加载时计算的常量）→ 清空模块级缓存。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# 确保项目根在 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """把 chunk_cache 所有文件操作重定向到 tmp_path/knowledge/。

    返回值：临时知识库目录路径。
    用法：def test_xxx(isolated_cache):
        kb_dir = isolated_cache
        ...
    """
    from kb import config
    from kb.ingestion import chunk_cache

    fake_root = tmp_path / "knowledge"
    fake_root.mkdir(parents=True, exist_ok=True)
    # 1) 改 KNOWLEDGE_DIR（chunks_file() 每次重新拼 → 自动跟随）
    monkeypatch.setattr(config, "KNOWLEDGE_DIR", fake_root)
    # 2) 重新加载 chunk_cache（让 _TX_LOCK_PATH 跟着重算）
    importlib.reload(chunk_cache)
    # 3) 清模块级缓存
    chunk_cache._invalidate_cache()
    yield fake_root
    # 4) teardown：再 reload 一次回原状态
    importlib.reload(chunk_cache)
    chunk_cache._invalidate_cache()


@pytest.fixture
def sample_row():
    """构造一条标准的 put_many 行（512 维 float32 embedding）。"""
    import numpy as np

    emb = [float(i) / 512 for i in range(512)]
    return {
        "content_hash": "hash_sample_001",
        "chunk_size": 400,
        "chunk_overlap": 120,
        "model_name": "BAAI/bge-small-zh-v1.5",
        "tags": ["Python基础", "装饰器"],
        "chunks": [
            {
                "text": "这是第一个 chunk 的文本",
                "heading": "简介",
                "embedding": emb,
            },
            {
                "text": "第二个 chunk",
                "heading": "用法",
                "embedding": [0.5] * 512,
            },
        ],
    }


@pytest.fixture
def legacy_jsonl(tmp_path):
    """手工生成旧版 chunk_cache.jsonl（行内带 embedding 数组）。"""
    import json
    legacy = tmp_path / "chunk_cache.jsonl"
    rows = []
    for i in range(3):
        emb = [float(j) / 1024 for j in range(512)]
        rows.append({
            "key": f"legacy_hash_{i}|400|120|BAAI/bge-small-zh-v1.5",
            "content_hash": f"legacy_hash_{i}",
            "chunk_size": 400,
            "chunk_overlap": 120,
            "model_name": "BAAI/bge-small-zh-v1.5",
            "tags": ["旧格式", f"标签{i}"],
            "chunks": [
                {
                    "text": f"旧格式 chunk {i}",
                    "heading": f"旧标题 {i}",
                    "embedding": emb,
                }
            ],
            "updated": 0.0,
        })
    legacy.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return legacy