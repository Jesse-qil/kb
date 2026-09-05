from __future__ import annotations

import json
import statistics
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.ingestion.pipeline import ingest
from kb.config import KNOWLEDGE_DIR, chroma_dir, chunks_file, index_file
from kb.storage.vector_store import query

CASES_FILE = Path(__file__).with_name("retrieval_cases.json")
RESULTS_DIR = Path(__file__).with_name("results")
LATEST_JSON = RESULTS_DIR / "latest.json"
LATEST_MD = RESULTS_DIR / "latest.md"


def load_cases() -> list[dict]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


def ensure_indexed() -> None:
    """重置向量库 + 台账 + chunk_cache（meta+emb），重跑入库。
    chunk_cache 保留与否取决于你想测什么：
    - 保留：跳过 LLM 打标 + embedding，30 秒完成（关心检索代码路径）
    - 删除：从原始 markdown 重建，~8 分钟（关心打标/embedding 链路）
    benchmark 默认清理冷启动测全链路。"""
    shutil.rmtree(chroma_dir(), ignore_errors=True)
    if index_file().exists():
        index_file().unlink()
    # 新格式：meta + emb + 旧 jsonl + migration flag 全删
    for name in ("chunk_cache.meta.jsonl", "chunk_cache.emb.npy",
                 "chunk_cache.jsonl", ".jsonl.migrated.flag"):
        p = chunks_file().with_name(name)
        if p.exists():
            p.unlink()
    fallback_store = KNOWLEDGE_DIR / "vector_store" / "fallback_store.json"
    if fallback_store.exists():
        fallback_store.unlink()
    ingest(verbose=False)


def evaluate_case(case: dict, top_k: int = 3, topic_filter: str = "") -> dict:
    hits = query(case["question"], topic=topic_filter, top_k=top_k)
    ranked = [h["source"] for h in hits]
    expected = case["expected_source"]
    rank = ranked.index(expected) + 1 if expected in ranked else None
    return {
        **case,
        "ranked": ranked,
        "top1": ranked[0] if ranked else None,
        "hit1": bool(ranked) and ranked[0] == expected,
        "hit3": expected in ranked[:3],
        "rank": rank,
        "rr": 1 / rank if rank else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    hit1 = sum(1 for r in rows if r["hit1"])
    hit3 = sum(1 for r in rows if r["hit3"])
    mrr = statistics.mean(r["rr"] for r in rows) if rows else 0.0
    by_group = defaultdict(list)
    for row in rows:
        by_group[row.get("group", "default")].append(row)
    grouped = {}
    for group, items in by_group.items():
        grouped[group] = {
            "count": len(items),
            "hit1": sum(1 for r in items if r["hit1"]) / len(items),
            "hit3": sum(1 for r in items if r["hit3"]) / len(items),
            "mrr": statistics.mean(r["rr"] for r in items),
        }
    return {
        "total": total,
        "hit1": hit1 / total if total else 0.0,
        "hit3": hit3 / total if total else 0.0,
        "mrr": mrr,
        "by_group": grouped,
    }


def write_report(summary: dict, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("# Retrieval Benchmark")
    lines.append("")
    lines.append(f"- total: {summary['total']}")
    lines.append(f"- filtered hit@1: {summary['filtered']['hit1']:.3f}")
    lines.append(f"- filtered hit@3: {summary['filtered']['hit3']:.3f}")
    lines.append(f"- filtered MRR: {summary['filtered']['mrr']:.3f}")
    lines.append(f"- global hit@1: {summary['global']['hit1']:.3f}")
    lines.append(f"- global hit@3: {summary['global']['hit3']:.3f}")
    lines.append(f"- global MRR: {summary['global']['mrr']:.3f}")
    lines.append("")
    lines.append("| id | q | expected | filtered top1 | global top1 | f hit1 | g hit1 |")
    lines.append("|---|---|---|---|---|---:|---:|")
    for row in rows:
        q = row["question"].replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {q} | {row['expected_source']} | "
            f"{row['filtered_top1'] or ''} | {row['global_top1'] or ''} | "
            f"{int(row['filtered_hit1'])} | {int(row['global_hit1'])} |"
        )
    lines.append("")
    lines.append("## By Group")
    for mode in ("filtered", "global"):
        lines.append(f"### {mode}")
        for group, info in summary[mode]["by_group"].items():
            lines.append(f"- {group}: hit@1={info['hit1']:.3f}, hit@3={info['hit3']:.3f}, MRR={info['mrr']:.3f}")
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_indexed()
    cases = load_cases()
    filtered_rows = [evaluate_case(case, topic_filter=case.get("topic", "")) for case in cases]
    global_rows = [evaluate_case(case, topic_filter="") for case in cases]
    report_rows = []
    for frow, grow in zip(filtered_rows, global_rows):
        report_rows.append({
            "id": frow["id"],
            "question": frow["question"],
            "expected_source": frow["expected_source"],
            "filtered_top1": frow["top1"],
            "global_top1": grow["top1"],
            "filtered_hit1": frow["hit1"],
            "global_hit1": grow["hit1"],
        })
    summary = {
        "total": len(cases),
        "filtered": summarize(filtered_rows),
        "global": summarize(global_rows),
    }
    write_report(summary, report_rows)

    print(f"total={summary['total']}")
    print(f"filtered hit@1={summary['filtered']['hit1']:.3f}")
    print(f"filtered hit@3={summary['filtered']['hit3']:.3f}")
    print(f"filtered mrr={summary['filtered']['mrr']:.3f}")
    print(f"global hit@1={summary['global']['hit1']:.3f}")
    print(f"global hit@3={summary['global']['hit3']:.3f}")
    print(f"global mrr={summary['global']['mrr']:.3f}")
    for mode in ("filtered", "global"):
        for group, info in summary[mode]["by_group"].items():
            print(f"{mode}:{group}: hit@1={info['hit1']:.3f}, hit@3={info['hit3']:.3f}, mrr={info['mrr']:.3f}")
    print(f"report={LATEST_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
