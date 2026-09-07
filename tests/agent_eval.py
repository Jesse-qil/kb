from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb.agents.graph import run as run_graph
from kb.agents.single import ask as ask_single
from kb.llm import LLMClient
from kb.config import recall_cfg
CASES_FILE = Path(__file__).with_name("agent_eval_cases.json")
RESULTS_DIR = Path(__file__).with_name("results")
HISTORY_DIR = RESULTS_DIR / "history"
LATEST_JSON = RESULTS_DIR / "agent_eval_latest.json"
LATEST_MD = RESULTS_DIR / "agent_eval_latest.md"


def load_cases() -> list[dict]:
    return json.loads(CASES_FILE.read_text(encoding="utf-8"))


JUDGE_PROMPT = """你是一个严格的知识库问答评审专家。请根据以下信息给回答打分。

问题：{question}
回答：{answer}
参考来源：{sources}
预期关键词：{expected}

评分维度（1-5分，5分最好，1分最差）：
1. relevance（相关性）：回答是否切题，有没有答非所问
2. accuracy（准确性）：回答内容是否正确，有没有事实错误
3. completeness（完整性）：回答是否覆盖了问题的所有方面
4. citation_accuracy（引用准确性）：回答是否正确引用了参考来源，有没有编造来源

请只输出 JSON，不要输出其他文字。格式如下：
{{"relevance": 4, "accuracy": 4, "completeness": 3, "citation_accuracy": 4, "overall": 4, "comment": "一句话评语"}}"""


def judge_answer(question: str, answer: str, sources: list[str], expected: str = "") -> dict:
    """LLM-as-judge：用大模型给回答质量打分。返回评分字典，失败返回 None。"""
    llm = LLMClient()
    if llm.provider == "mock":
        return None
    prompt = JUDGE_PROMPT.format(
        question=question[:500],
        answer=answer[:2000],
        sources=", ".join(sources[:5]) if sources else "无",
        expected=expected or "无",
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}])
        # 提取 JSON（兼容 LLM 输出前后有文字的情况）
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(raw[start:end + 1])
            return {
                "relevance": int(data.get("relevance", 0)),
                "accuracy": int(data.get("accuracy", 0)),
                "completeness": int(data.get("completeness", 0)),
                "citation_accuracy": int(data.get("citation_accuracy", 0)),
                "overall": int(data.get("overall", 0)),
                "comment": str(data.get("comment", ""))[:200],
            }
    except Exception:
        pass
    return None


def _source_list(result: dict) -> list[str]:
    return [s.get("source", "") for s in result.get("sources", []) if s.get("source")]


def evaluate_case(case: dict, mode: str = "graph", history: str = "", use_judge: bool = False) -> dict:
    t0 = time.time()
    llm = LLMClient()
    provider = llm.provider
    requires_real_model = bool(case.get("requires_real_model", False))
    mock_answer = (llm.mock_answer or "").strip()
    LLMClient.reset_usage()
    if provider == "mock" and requires_real_model:
        return {
            **case,
            "mode": mode,
            "skipped": True,
            "skip_reason": "mock provider",
            "latency_ms": int((time.time() - t0) * 1000),
            "rounds": 0,
            "from_web": False,
            "tool_result": "",
            "review_feedback": "",
            "top1": None,
            "sources": [],
            "hit_source": False,
            "contains": False,
            "expected_source_hit": False,
            "forbidden_source": False,
        }

    if mode == "single":
        result = ask_single(case["question"], topic=case.get("topic", ""))
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        meta = {"rounds": 1, "from_web": False, "tool_result": "", "review_feedback": ""}
    else:
        result = run_graph(case["question"], history=history)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        meta = result

    srcs = [s.get("source", "") for s in sources if s.get("source")]
    src_set = set(srcs)
    top1 = srcs[0] if srcs else None
    if answer.strip() == mock_answer:
        return {
            **case,
            "mode": mode,
            "skipped": True,
            "skip_reason": "mock answer fallback",
            "answer": answer,
            "sources": srcs,
            "top1": top1,
            "latency_ms": int((time.time() - t0) * 1000),
            "rounds": meta.get("rounds", 0),
            "from_web": meta.get("from_web", False),
            "tool_result": meta.get("tool_result", ""),
            "review_feedback": meta.get("review_feedback", ""),
            "hit_source": False,
            "contains": False,
            "expected_source_hit": False,
            "forbidden_source": False,
        }

    out = {
        **case,
        "mode": mode,
        "answer": answer,
        "sources": srcs,
        "top1": top1,
        "actual_domain": meta.get("topic", ""),
        "token_usage": LLMClient.get_usage(),
        "latency_ms": int((time.time() - t0) * 1000),
        "rounds": meta.get("rounds", 0),
        "from_web": meta.get("from_web", False),
        "tool_result": meta.get("tool_result", ""),
        "review_feedback": meta.get("review_feedback", ""),
        "hit_source": bool(
            (case.get("expected_source") and case["expected_source"] in src_set) or
            (case.get("expected_sources_any") and bool(set(case["expected_sources_any"]) & src_set))
        ),
        "has_source_expectation": bool(case.get("expected_source") or case.get("expected_sources_any")),
        "contains": bool(case.get("expected_contains") and case["expected_contains"] in answer),
        "source_mismatch": False,
        "forbidden_source": False,
        "domain_match": None,
        "need_tool_match": None,
    }
    if case.get("expected_domain"):
        out["domain_match"] = (meta.get("topic", "") == case["expected_domain"])
    if case.get("expected_need_tool") is not None:
        actual_need_tool = bool(meta.get("tool_result", "").strip())
        out["need_tool_match"] = (actual_need_tool == bool(case["expected_need_tool"]))
        out["actual_need_tool"] = actual_need_tool
    if case.get("expected_source"):
        out["source_mismatch"] = top1 != case["expected_source"]
    if case.get("forbid_sources_any"):
        forbidden = set(case["forbid_sources_any"])
        out["forbidden_source"] = bool(forbidden & src_set)
    if case.get("expected_sources_any"):
        allowed = set(case["expected_sources_any"])
        out["source_match_any"] = bool(allowed & src_set)
    if case.get("expected_tool_result_contains"):
        out["tool_result_match"] = bool(case["expected_tool_result_contains"] in (meta.get("tool_result", "") or ""))
    if case.get("expected_from_web") is not None:
        out["from_web_match"] = bool(meta.get("from_web", False)) == bool(case["expected_from_web"])
    if case.get("expect_overview") is not None:
        out["overview_match"] = bool(meta.get("is_overview", False)) == bool(case["expect_overview"])
    if use_judge:
        out["judge_score"] = judge_answer(
            case["question"], answer, srcs, case.get("expected_contains", "")
        )
    return out


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    active = [r for r in rows if not r.get("skipped")]
    skipped = [r for r in rows if r.get("skipped")]
    def pct(pred, seq=active):
        return sum(1 for r in seq if pred(r)) / len(seq) if seq else 0.0
    def pct_nonnull(pred, seq=active):
        vals = [r for r in seq if pred(r) is not None]
        return sum(1 for r in vals if r[pred.__name__ if hasattr(pred, '__name__') else 'x']) / len(vals) if vals else 0.0
    groups = defaultdict(list)
    for row in active:
        groups[row.get("kind", "other")].append(row)
    by_kind = {}
    for kind, items in groups.items():
        domain_vals = [r for r in items if r.get("domain_match") is not None]
        tool_vals = [r for r in items if r.get("need_tool_match") is not None]
        source_vals = [r for r in items if r.get("has_source_expectation")]
        by_kind[kind] = {
            "count": len(items),
            "source_hit_rate": sum(1 for r in source_vals if r["hit_source"]) / len(source_vals) if source_vals else None,
            "answer_hit_rate": sum(1 for r in items if r["contains"]) / len(items),
            "domain_match_rate": sum(1 for r in domain_vals if r["domain_match"]) / len(domain_vals) if domain_vals else None,
            "need_tool_match_rate": sum(1 for r in tool_vals if r["need_tool_match"]) / len(tool_vals) if tool_vals else None,
            "avg_latency_ms": round(statistics.mean(r["latency_ms"] for r in items), 1),
            "avg_rounds": round(statistics.mean(r.get("rounds", 0) for r in items), 2),
        }
    domain_all = [r for r in active if r.get("domain_match") is not None]
    tool_all = [r for r in active if r.get("need_tool_match") is not None]
    source_all = [r for r in active if r.get("has_source_expectation")]
    # Tool Agent 混淆矩阵
    tp = sum(1 for r in tool_all if r.get("expected_need_tool") and r.get("actual_need_tool"))
    fp = sum(1 for r in tool_all if not r.get("expected_need_tool") and r.get("actual_need_tool"))
    fn = sum(1 for r in tool_all if r.get("expected_need_tool") and not r.get("actual_need_tool"))
    tn = sum(1 for r in tool_all if not r.get("expected_need_tool") and not r.get("actual_need_tool"))
    # Token 统计
    all_tokens = [r.get("token_usage", {}) for r in active]
    avg_prompt = round(statistics.mean(t.get("prompt_tokens", 0) for t in all_tokens), 1) if all_tokens else 0
    avg_completion = round(statistics.mean(t.get("completion_tokens", 0) for t in all_tokens), 1) if all_tokens else 0
    avg_total = round(statistics.mean(t.get("total_tokens", 0) for t in all_tokens), 1) if all_tokens else 0
    total_calls = sum(t.get("calls", 0) for t in all_tokens)
    # Judge 评分统计
    judged = [r for r in active if r.get("judge_score")]
    judge_stats = None
    if judged:
        judge_stats = {
            "count": len(judged),
            "avg_relevance": round(statistics.mean(j["judge_score"]["relevance"] for j in judged), 2),
            "avg_accuracy": round(statistics.mean(j["judge_score"]["accuracy"] for j in judged), 2),
            "avg_completeness": round(statistics.mean(j["judge_score"]["completeness"] for j in judged), 2),
            "avg_citation_accuracy": round(statistics.mean(j["judge_score"]["citation_accuracy"] for j in judged), 2),
            "avg_overall": round(statistics.mean(j["judge_score"]["overall"] for j in judged), 2),
        }
    return {
        "total": total,
        "evaluated": len(active),
        "skipped": len(skipped),
        "source_hit_rate": sum(1 for r in source_all if r["hit_source"]) / len(source_all) if source_all else None,
        "answer_hit_rate": pct(lambda r: r["contains"]),
        "forbidden_source_rate": pct(lambda r: r["forbidden_source"]),
        "domain_match_rate": sum(1 for r in domain_all if r["domain_match"]) / len(domain_all) if domain_all else None,
        "need_tool_match_rate": sum(1 for r in tool_all if r["need_tool_match"]) / len(tool_all) if tool_all else None,
        "tool_confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn} if tool_all else None,
        "avg_prompt_tokens": avg_prompt,
        "avg_completion_tokens": avg_completion,
        "avg_total_tokens": avg_total,
        "total_llm_calls": total_calls,
        "judge_stats": judge_stats,
        "avg_latency_ms": round(statistics.mean(r["latency_ms"] for r in active), 1) if active else 0.0,
        "avg_rounds": round(statistics.mean(r.get("rounds", 0) for r in active), 2) if active else 0.0,
        "by_kind": by_kind,
    }


def write_report(summary: dict, rows: list[dict], mode: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(
        json.dumps({"summary": summary, "cases": rows, "mode": mode}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Agent Evaluation",
        "",
        f"- mode: {mode}",
        f"- total: {summary['total']}",
        f"- evaluated: {summary['evaluated']}",
        f"- skipped: {summary['skipped']}",
        f"- source_hit_rate: {summary['source_hit_rate']:.3f}" if summary['source_hit_rate'] is not None else "- source_hit_rate: N/A",
        f"- answer_hit_rate: {summary['answer_hit_rate']:.3f}",
        f"- forbidden_source_rate: {summary['forbidden_source_rate']:.3f}",
        f"- domain_match_rate: {summary['domain_match_rate']:.3f}" if summary['domain_match_rate'] is not None else "- domain_match_rate: N/A",
        f"- need_tool_match_rate: {summary['need_tool_match_rate']:.3f}" if summary['need_tool_match_rate'] is not None else "- need_tool_match_rate: N/A",
        f"- avg_latency_ms: {summary['avg_latency_ms']:.1f}",
        f"- avg_rounds: {summary['avg_rounds']:.2f}",
        f"- avg_prompt_tokens: {summary['avg_prompt_tokens']:.0f}",
        f"- avg_completion_tokens: {summary['avg_completion_tokens']:.0f}",
        f"- avg_total_tokens: {summary['avg_total_tokens']:.0f}",
        f"- total_llm_calls: {summary['total_llm_calls']}",
        "",
    ]
    # Judge 评分
    js = summary.get("judge_stats")
    if js:
        lines.extend([
            "## LLM-as-Judge 评分",
            "",
            f"- 评分数: {js['count']}",
            f"- 平均相关性(relevance): {js['avg_relevance']:.2f}/5",
            f"- 平均准确性(accuracy): {js['avg_accuracy']:.2f}/5",
            f"- 平均完整性(completeness): {js['avg_completeness']:.2f}/5",
            f"- 平均引用准确性(citation_accuracy): {js['avg_citation_accuracy']:.2f}/5",
            f"- 平均总分(overall): {js['avg_overall']:.2f}/5",
            "",
        ])
    # Tool Agent 混淆矩阵
    cm = summary.get("tool_confusion_matrix")
    if cm:
        lines.extend([
            "## Tool Agent 混淆矩阵",
            "",
            "|  | 预测需要工具 | 预测不需要 |",
            "|---|---:|---:|",
            f"| 实际需要 | {cm['tp']} (TP) | {cm['fn']} (FN) |",
            f"| 实际不需要 | {cm['fp']} (FP) | {cm['tn']} (TN) |",
            "",
            f"- 精确率(Precision): {cm['tp']/(cm['tp']+cm['fp']):.3f}" if (cm['tp']+cm['fp']) > 0 else "- 精确率(Precision): N/A",
            f"- 召回率(Recall): {cm['tp']/(cm['tp']+cm['fn']):.3f}" if (cm['tp']+cm['fn']) > 0 else "- 召回率(Recall): N/A",
            f"- 假阳性率: {cm['fp']/(cm['fp']+cm['tn']):.3f}" if (cm['fp']+cm['tn']) > 0 else "- 假阳性率: N/A",
            "",
        ])
    lines.extend([
        "| id | kind | expected_domain | actual_domain | domain_ok | top1 | source_hit | answer_hit | judge | rounds | latency_ms |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        if row.get("skipped"):
            lines.append(f"| {row['id']} | {row.get('kind','')} | skipped | - | - | - | 0 | 0 | - | 0 | {row['latency_ms']} |")
            continue
        exp_dom = row.get("expected_domain", "") or "-"
        act_dom = row.get("actual_domain", "") or "-"
        dom_ok = "1" if row.get("domain_match") else ("0" if row.get("domain_match") is not None else "-")
        judge_val = row.get("judge_score", {}).get("overall", "-") if row.get("judge_score") else "-"
        lines.append(
            f"| {row['id']} | {row.get('kind','')} | {exp_dom} | {act_dom} | {dom_ok} | "
            f"{row.get('top1') or ''} | "
            f"{int(row['hit_source'])} | {int(row['contains'])} | {judge_val} | "
            f"{row.get('rounds', 0)} | {row['latency_ms']} |"
        )
    lines.append("")
    # 失败案例详情（只有设置了预期值且未命中的才算失败）
    failures = [r for r in rows if not r.get("skipped") and (
        (r.get("expected_source") and not r["hit_source"]) or
        (r.get("expected_contains") and not r["contains"]) or
        r.get("domain_match") is False or
        r.get("forbidden_source")
    )]
    if failures:
        lines.append("## 失败案例详情")
        lines.append("")
        for r in failures:
            lines.append(f"### {r['id']}: {r['question']}")
            reasons = []
            if r.get("expected_source") and not r["hit_source"]:
                reasons.append(f"- 来源未命中：预期 `{r['expected_source']}`，实际 top1=`{r.get('top1') or '无'}`")
            if r.get("expected_contains") and not r["contains"]:
                reasons.append(f"- 回答未包含关键词：预期包含 `{r['expected_contains']}`")
            if r.get("domain_match") is False:
                reasons.append(f"- 领域判断错误：预期 `{r.get('expected_domain')}`，实际 `{r.get('actual_domain')}`")
            if r.get("forbidden_source"):
                reasons.append(f"- 命中禁止来源：{r.get('forbid_sources_any')}")
            lines.extend(reasons)
            lines.append("")
    lines.append("## By Kind")
    for kind, info in summary["by_kind"].items():
        src_str = f"source_hit_rate={info['source_hit_rate']:.3f}" if info['source_hit_rate'] is not None else "source_hit_rate=N/A"
        dom_str = f", domain_match={info['domain_match_rate']:.3f}" if info['domain_match_rate'] is not None else ""
        tool_str = f", need_tool_match={info['need_tool_match_rate']:.3f}" if info['need_tool_match_rate'] is not None else ""
        lines.append(
            f"- {kind}: {src_str}, "
            f"answer_hit_rate={info['answer_hit_rate']:.3f}{dom_str}{tool_str}, "
            f"avg_latency_ms={info['avg_latency_ms']:.1f}, avg_rounds={info['avg_rounds']:.2f}"
        )
    LATEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_history(summary: dict, rows: list[dict], mode: str):
    """保存评估结果到 history 目录，文件名带时间戳。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": ts,
        "mode": mode,
        "summary": summary,
        "rows": rows,
    }
    hist_file = HISTORY_DIR / f"eval_{ts}.json"
    hist_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return hist_file


def compare_history() -> str:
    """对比最近两次评估结果，返回对比报告文本。"""
    if not HISTORY_DIR.exists():
        return "暂无历史评估记录。"
    files = sorted(HISTORY_DIR.glob("eval_*.json"))
    if len(files) < 2:
        return f"只有 {len(files)} 次评估记录，至少需要 2 次才能对比。"
    prev = json.loads(files[-2].read_text(encoding="utf-8"))
    curr = json.loads(files[-1].read_text(encoding="utf-8"))
    ps, cs = prev["summary"], curr["summary"]
    metrics = [
        ("source_hit_rate", "来源命中率"),
        ("answer_hit_rate", "回答命中率"),
        ("domain_match_rate", "领域判断准确率"),
        ("need_tool_match_rate", "工具判断准确率"),
        ("avg_total_tokens", "平均Token"),
        ("avg_latency_ms", "平均延迟(ms)"),
    ]
    lines = [
        "# 评估历史对比",
        "",
        f"- 上一次: {prev['timestamp']} (mode={prev['mode']}, {ps['evaluated']}条)",
        f"- 本次:   {curr['timestamp']} (mode={curr['mode']}, {cs['evaluated']}条)",
        "",
        "| 指标 | 上一次 | 本次 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in metrics:
        pv = ps.get(key)
        cv = cs.get(key)
        if pv is None or cv is None:
            continue
        diff = cv - pv
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        if "rate" in key:
            lines.append(f"| {label} | {pv:.3f} | {cv:.3f} | {arrow} {diff:+.3f} |")
        else:
            lines.append(f"| {label} | {pv:.0f} | {cv:.0f} | {arrow} {diff:+.0f} |")
    # Judge 评分对比
    pjs, cjs = ps.get("judge_stats"), cs.get("judge_stats")
    if pjs and cjs:
        lines.extend([
            "",
            "## LLM-as-Judge 评分对比",
            "",
            f"| 维度 | 上一次 | 本次 | 变化 |",
            "|---|---:|---:|---:|",
        ])
        for key, label in [("avg_relevance", "相关性"), ("avg_accuracy", "准确性"),
                            ("avg_completeness", "完整性"), ("avg_citation_accuracy", "引用准确性"),
                            ("avg_overall", "总分")]:
            pv = pjs.get(key, 0)
            cv = cjs.get(key, 0)
            diff = cv - pv
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
            lines.append(f"| {label} | {pv:.2f} | {cv:.2f} | {arrow} {diff:+.2f} |")
    return "\n".join(lines) + "\n"


def tune_params(limit: int = 10) -> str:
    """自动调参：网格搜索 top_k 和 score_threshold 的最优组合。
    临时修改 CONFIG，运行评估，记录 source_hit_rate，最后恢复原始配置。"""
    from kb.config import CONFIG
    orig_top_k = CONFIG["recall"]["top_k"]
    orig_threshold = CONFIG["recall"]["score_threshold"]
    cases = load_cases()[:limit]
    grid_top_k = [2, 3, 4, 5, 6]
    grid_threshold = [0.30, 0.35, 0.40, 0.45, 0.50]
    results = []
    print(f"[tune] 网格搜索 {len(grid_top_k)}x{len(grid_threshold)}={len(grid_top_k)*len(grid_threshold)} 组参数，每组 {limit} 条用例...")
    for tk in grid_top_k:
        for th in grid_threshold:
            CONFIG["recall"]["top_k"] = tk
            CONFIG["recall"]["score_threshold"] = th
            rows = [evaluate_case(c, mode="graph") for c in cases]
            active = [r for r in rows if not r.get("skipped")]
            hit_rate = sum(1 for r in active if r["hit_source"]) / len(active) if active else 0
            ans_rate = sum(1 for r in active if r["contains"]) / len(active) if active else 0
            avg_lat = round(statistics.mean(r["latency_ms"] for r in active), 1) if active else 0
            results.append({"top_k": tk, "threshold": th, "hit_rate": hit_rate,
                            "ans_rate": ans_rate, "avg_latency": avg_lat})
            print(f"  top_k={tk}, threshold={th:.2f}: hit={hit_rate:.3f}, ans={ans_rate:.3f}, lat={avg_lat:.0f}ms")
    # 恢复原始配置
    CONFIG["recall"]["top_k"] = orig_top_k
    CONFIG["recall"]["score_threshold"] = orig_threshold
    # 找最优组合（按 hit_rate 排序，相同 hit_rate 按 latency 排序）
    best = max(results, key=lambda r: (r["hit_rate"], -r["avg_latency"]))
    # 生成报告
    lines = [
        "# 自动调参报告",
        "",
        f"- 搜索范围: top_k ∈ {grid_top_k}, score_threshold ∈ {grid_threshold}",
        f"- 每组用例数: {limit}",
        f"- 原始配置: top_k={orig_top_k}, score_threshold={orig_threshold}",
        "",
        "## 最优组合",
        "",
        f"- top_k = {best['top_k']}",
        f"- score_threshold = {best['threshold']:.2f}",
        f"- source_hit_rate = {best['hit_rate']:.3f}",
        f"- answer_hit_rate = {best['ans_rate']:.3f}",
        f"- avg_latency = {best['avg_latency']:.0f}ms",
        "",
        "## 完整结果（按 hit_rate 降序）",
        "",
        "| top_k | threshold | hit_rate | ans_rate | avg_latency |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: (-x["hit_rate"], x["avg_latency"])):
        marker = " ⭐" if r == best else ""
        lines.append(f"| {r['top_k']} | {r['threshold']:.2f} | {r['hit_rate']:.3f} | {r['ans_rate']:.3f} | {r['avg_latency']:.0f}ms |{marker}")
    report = "\n".join(lines) + "\n"
    # 保存报告
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tune_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent evaluation runner")
    parser.add_argument("--mode", choices=["graph", "single"], default="graph")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--category", type=str, default="",
                        help="只跑指定 kind 的用例（如 rag/tool/overview/web）")
    parser.add_argument("--judge", action="store_true",
                        help="启用 LLM-as-judge 打分（会增加 LLM 调用和耗时）")
    parser.add_argument("--compare", action="store_true",
                        help="只对比最近两次评估结果，不运行新评估")
    parser.add_argument("--no-history", action="store_true",
                        help="不保存本次评估到历史记录")
    parser.add_argument("--tune", action="store_true",
                        help="自动调参：网格搜索 top_k 和 score_threshold 的最优组合")
    parser.add_argument("--tune-limit", type=int, default=10,
                        help="自动调参时每组参数的用例数（默认10）")
    args = parser.parse_args()

    if args.compare:
        print(compare_history())
        return 0

    if args.tune:
        print(tune_params(limit=args.tune_limit))
        return 0

    cases = load_cases()
    if args.category:
        cases = [c for c in cases if c.get("kind") == args.category]
        print(f"[filter] category={args.category}, 匹配 {len(cases)} 条")
    if args.limit > 0:
        cases = cases[:args.limit]
    rows = [evaluate_case(case, mode=args.mode, use_judge=args.judge) for case in cases]
    summary = summarize(rows)
    write_report(summary, rows, args.mode)
    if not args.no_history:
        hist_file = save_history(summary, rows, args.mode)
        print(f"history saved: {hist_file.name}")

    print(f"mode={args.mode}")
    print(f"total={summary['total']}")
    print(f"evaluated={summary['evaluated']}")
    print(f"skipped={summary['skipped']}")
    print(f"source_hit_rate={summary['source_hit_rate']:.3f}" if summary['source_hit_rate'] is not None else "source_hit_rate=N/A")
    print(f"answer_hit_rate={summary['answer_hit_rate']:.3f}")
    print(f"forbidden_source_rate={summary['forbidden_source_rate']:.3f}")
    if summary['domain_match_rate'] is not None:
        print(f"domain_match_rate={summary['domain_match_rate']:.3f}")
    if summary['need_tool_match_rate'] is not None:
        print(f"need_tool_match_rate={summary['need_tool_match_rate']:.3f}")
    cm = summary.get("tool_confusion_matrix")
    if cm:
        print(f"tool_confusion: TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    print(f"avg_prompt_tokens={summary['avg_prompt_tokens']:.0f}")
    print(f"avg_completion_tokens={summary['avg_completion_tokens']:.0f}")
    print(f"avg_total_tokens={summary['avg_total_tokens']:.0f}")
    print(f"total_llm_calls={summary['total_llm_calls']}")
    js = summary.get("judge_stats")
    if js:
        print(f"judge_overall={js['avg_overall']:.2f}/5 (relevance={js['avg_relevance']:.2f}, accuracy={js['avg_accuracy']:.2f}, completeness={js['avg_completeness']:.2f}, citation={js['avg_citation_accuracy']:.2f})")
    print(f"avg_latency_ms={summary['avg_latency_ms']:.1f}")
    print(f"avg_rounds={summary['avg_rounds']:.2f}")
    if summary["skipped"]:
        print("note=mock provider detected; some real-model cases skipped")
    print(f"report={LATEST_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
