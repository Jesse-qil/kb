# -*- coding: utf-8 -*-
"""标签-文档相似度分布实测：为新「标签向量库复用」方案校准阈值。

口径与 tagger.embedding_tags 完全一致：
- 标签文本直接 embed_texts（不带指令前缀）
- 文档取前 1000 字符
- 余弦相似度（embed_texts 已归一化，内积即余弦）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb import embedding as E
from kb.ingestion.tagger import _DEFAULT_SCHEMA
from kb.config import KNOWLEDGE_DIR

# 1. 标签池（种子 = 默认候选表；tag_schema.json 存在则覆盖）
tags = [str(t) for t in _DEFAULT_SCHEMA["allowed_tags"] if t]
schema_path = KNOWLEDGE_DIR / "tag_schema.json"
if schema_path.exists():
    try:
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        if data.get("allowed_tags"):
            tags = [str(t) for t in data["allowed_tags"] if t]
    except Exception:
        pass
print(f"标签池: {len(tags)} 个")

# 2. 文档
docs = []
for p in sorted((KNOWLEDGE_DIR / "raw").rglob("*.md")):
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if not text.strip():
        continue
    docs.append((p.relative_to(KNOWLEDGE_DIR / "raw").as_posix(), text))
print(f"文档: {len(docs)} 篇")

# 3. 向量化
tag_vecs = E.embed_texts(tags)
doc_vecs = E.embed_texts([t[:1000] for _, t in docs])
print(f"模型: {('bge/hash' if E._use_fallback else 'bge-small-zh')}  向量维度: {len(tag_vecs[0])}")

# 4. 每篇文档 × 30 标签的相似度，取 Top3
rows = []
for (name, _), dvec in zip(docs, doc_vecs):
    scored = sorted(
        ((tags[i], sum(a * b for a, b in zip(dvec, tag_vecs[i]))) for i in range(len(tags))),
        key=lambda x: -x[1])
    rows.append({
        "doc": name,
        "top1": scored[0][0],
        "top1_sim": round(scored[0][1], 4),
        "top3": [(t, round(s, 4)) for t, s in scored[:3]],
    })

# 5. 分布统计
max_sims = sorted(r["top1_sim"] for r in rows)
n = len(max_sims)


def pct(p: float) -> float:
    return max_sims[min(n - 1, int(p * (n - 1)))]


stats = {"min": max_sims[0], "p10": pct(0.1), "p25": pct(0.25), "p50": pct(0.5),
         "p75": pct(0.75), "p90": pct(0.9), "max": max_sims[-1]}
print("\n=== 每篇与最优标签的最高相似度分布 ===")
print(f"min={stats['min']:.3f} p10={stats['p10']:.3f} p25={stats['p25']:.3f} "
      f"p50={stats['p50']:.3f} p75={stats['p75']:.3f} p90={stats['p90']:.3f} max={stats['max']:.3f}")

print("\n=== 各阈值下「免 LLM 直接复用」命中率 ===")
for T in (0.85, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3):
    hit = sum(1 for r in rows if r["top1_sim"] >= T)
    print(f"阈值 {T:.2f}: 命中 {hit}/{n} = {hit / n * 100:.1f}%")

print("\n=== 抽查 8 篇的 Top1-3 标签 ===")
step = max(1, n // 8)
for r in rows[::step][:8]:
    t3 = ", ".join(f"{t}({s:.2f})" for t, s in r["top3"])
    print(f"{r['doc'][:44]:<46} | {t3}")

# 6. 存结果
out = Path(__file__).resolve().parent / "tag_sim_probe_result.md"
lines = ["# 标签-文档相似度分布实测", "",
         f"- 标签池：{len(tags)} 个（默认候选表）",
         f"- 文档：{n} 篇 md（取前 1000 字符）",
         f"- 模型：bge-small-zh-v1.5（{'hash 降级' if E._use_fallback else '正常'}），口径与 tagger.embedding_tags 一致",
         "", "## 每篇最高相似度分布（与最优标签）", "| 统计量 | 值 |", "|---|---|"]
for k, v in stats.items():
    lines.append(f"| {k} | {v:.3f} |")
lines += ["", "## 阈值命中率（免 LLM 复用比例）", "| 阈值 | 命中篇数 | 命中率 |", "|---|---|---|"]
for T in (0.85, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3):
    hit = sum(1 for r in rows if r["top1_sim"] >= T)
    lines.append(f"| {T:.2f} | {hit}/{n} | {hit / n * 100:.1f}% |")
lines += ["", "## 全量明细", "", "| 文档 | Top1 标签 | 相似度 |", "|---|---|---|"]
for r in rows:
    lines.append(f"| {r['doc']} | {r['top1']} | {r['top1_sim']:.3f} |")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"\n结果已存: {out}")
