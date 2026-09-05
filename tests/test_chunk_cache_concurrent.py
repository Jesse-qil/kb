# -*- coding: utf-8 -*-
"""chunk_cache 多进程并发安全测试。

核心场景：4 个进程同时 put_many（写同一 key，触发读-改-写竞态）。
验证：
1. 全部成功，无 WinError 5 / 进程崩溃
3. emb.npy 不丢数据
4. meta.jsonl 不丢数据

注：multiprocessing 用 spawn 上下文（Windows 默认），子进程独立内存。
通过环境变量把临时目录传给子进程，子进程在入口 monkeypatch KNOWLEDGE_DIR。
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _worker(pid_seed: int, n_rows: int, kb_dir: str, result_queue):
    """子进程入口：monkeypatch KNOWLEDGE_DIR → 写 n_rows 条同名 key 触发竞态。"""
    # 1) 改 KB 路径（不动真实知识库）
    sys.path.insert(0, str(ROOT))
    from kb import config
    config.KNOWLEDGE_DIR = Path(kb_dir)
    from kb.ingestion import chunk_cache
    importlib.reload(chunk_cache)

    # 2) 构造测试数据：用同 hash 不同 chunk 文本（每个进程写自己的版本）
    rows = []
    for i in range(n_rows):
        rows.append({
            "content_hash": "concurrent_same_key",  # 所有进程写同一 key
            "chunk_size": 400,
            "chunk_overlap": 120,
            "model_name": "BAAI/bge-small-zh-v1.5",
            "tags": [f"pid_{pid_seed}_batch_{i}"],
            "chunks": [
                {
                    "text": f"text-from-pid-{pid_seed}-chunk-{i}",
                    "heading": "h",
                    "embedding": [float(pid_seed) / 4 + i / 100] * 512,
                }
            ],
        })
    try:
        n = chunk_cache.put_many(rows)
        result_queue.put(("ok", pid_seed, n))
    except Exception as e:
        result_queue.put(("err", pid_seed, type(e).__name__, str(e)))


@pytest.mark.timeout(120)
class TestConcurrent:
    def test_four_workers_no_data_loss(self, tmp_path):
        """4 进程并发 put_many 同一 key：最终只保留一个 winner，
        但事务锁保证 4 个进程都不报错，meta+emb 不丢/不破坏。"""
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir(parents=True, exist_ok=True)

        n_workers = 4
        n_rows_each = 3
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = []
        for seed in range(n_workers):
            p = ctx.Process(
                target=_worker,
                args=(seed, n_rows_each, str(kb_dir), q),
            )
            p.start()
            procs.append(p)

        # 收集结果
        results = []
        for _ in procs:
            results.append(q.get(timeout=60))
        for p in procs:
            p.join(timeout=15)
            assert not p.is_alive(), f"子进程 {p.pid} 未正常退出"

        # 1) 4 个进程都成功（不允许 WinError 5 / OSError）
        for r in results:
            assert r[0] == "ok", f"子进程失败: {r}"

        # 2) 数据一致性：从主进程独立加载一次（用 try/finally 保证回滚 KNOWLEDGE_DIR）
        sys.path.insert(0, str(ROOT))
        from kb import config
        from kb.ingestion import chunk_cache
        original_kb_dir = config.KNOWLEDGE_DIR
        try:
            config.KNOWLEDGE_DIR = kb_dir
            importlib.reload(chunk_cache)

            rows = chunk_cache._load_rows()
            # 同 key 多次 put_many，最后写赢；总条数 = 1（同一 key）
            assert len(rows) == 1
            winner = next(iter(rows.values()))
            assert winner["content_hash"] == "concurrent_same_key"
            # winner 应有 1 个 chunk
            assert len(winner["chunks"]) == 1
            # winner 的 text 必须来自某个 pid（4 个可能之一）
            assert "text-from-pid-" in winner["chunks"][0]["text"]

            # 3) emb.npy 维度必须正确（不会因为竞态被破坏）
            emb = np.load(kb_dir / "chunks" / "chunk_cache.emb.npy")
            assert emb.ndim == 1
            assert emb.dtype == np.float32
            assert emb.shape[0] >= 512  # 至少能放 winner 的一条 chunk
            # winner 的 chunk 必须在 emb.npy 里能取到 512 维
            off = winner["chunks"][0]["emb_off"]
            ln = winner["chunks"][0]["emb_len"]
            assert ln == 512
            assert off + ln <= emb.shape[0]
            vec = emb[off:off + ln]
            assert vec.shape == (512,)
        finally:
            # 关键：恢复原 KNOWLEDGE_DIR，否则后续 test 会读到空 kb
            config.KNOWLEDGE_DIR = original_kb_dir
            importlib.reload(chunk_cache)

    @pytest.mark.timeout(120)
    def test_concurrent_different_keys_independent(self, tmp_path):
        """4 进程各写不同 key：应都能落盘，最终 4 条共存。"""
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir(parents=True, exist_ok=True)

        n_workers = 4
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = []
        for seed in range(n_workers):
            p = ctx.Process(
                target=_worker_different_keys,
                args=(seed, str(kb_dir), q),
            )
            p.start()
            procs.append(p)

        results = []
        for _ in procs:
            results.append(q.get(timeout=60))
        for p in procs:
            p.join(timeout=15)

        for r in results:
            assert r[0] == "ok", f"子进程失败: {r}"

        # 验证 4 条独立数据都落盘（用 try/finally 保证回滚 KNOWLEDGE_DIR）
        sys.path.insert(0, str(ROOT))
        from kb import config
        from kb.ingestion import chunk_cache
        original_kb_dir = config.KNOWLEDGE_DIR
        try:
            config.KNOWLEDGE_DIR = kb_dir
            importlib.reload(chunk_cache)
            rows = chunk_cache._load_rows()
            assert len(rows) == n_workers
            for seed in range(n_workers):
                key = f"unique_{seed}|400|120|BAAI/bge-small-zh-v1.5"
                assert key in rows, f"缺失 pid={seed} 的数据"
        finally:
            config.KNOWLEDGE_DIR = original_kb_dir
            importlib.reload(chunk_cache)


def _worker_different_keys(pid_seed: int, kb_dir: str, result_queue):
    """子进程：写 1 条独有 key。"""
    sys.path.insert(0, str(ROOT))
    from kb import config
    config.KNOWLEDGE_DIR = Path(kb_dir)
    from kb.ingestion import chunk_cache
    importlib.reload(chunk_cache)

    rows = [{
        "content_hash": f"unique_{pid_seed}",
        "chunk_size": 400,
        "chunk_overlap": 120,
        "model_name": "BAAI/bge-small-zh-v1.5",
        "tags": [f"pid_{pid_seed}"],
        "chunks": [
            {
                "text": f"unique text from pid {pid_seed}",
                "heading": "h",
                "embedding": [float(pid_seed)] * 512,
            }
        ],
    }]
    try:
        n = chunk_cache.put_many(rows)
        result_queue.put(("ok", pid_seed, n))
    except Exception as e:
        result_queue.put(("err", pid_seed, type(e).__name__, str(e)))