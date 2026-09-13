"""重排器（Reranker）：交叉编码器对 (query, 候选片段) 精细打分，重排混合检索结果。

为什么需要：双编码器（bge-small 向量检索）把 query 和文档各自编码成向量比相似度，
对"语义相近但表述不同"和"整本书型 PDF 系统性低分"两类情况区分力不足（README 遗留
瓶颈 agent_confusion 组）。交叉编码器把 query 和候选拼成一对输入整体打分，精度更高。

设计：
- 懒加载 + 进程内只尝试一次：模型没装 / 下载失败 / 加载异常 → 静默降级原排序，不影响主链路
- 只对候选池前 top_n 重排（全库 6786 片段逐一打分太慢，召回已由向量+BM25 保证）
- 重排分写 h["rerank_score"]，不覆盖原 score（下游阈值判断仍用融合分）
"""
from __future__ import annotations

try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except Exception:
    CrossEncoder = None
    _HAS_CROSS_ENCODER = False

from .config import rerank_cfg

_model = None
_load_tried = False


def _model_name() -> str:
    return rerank_cfg().get("model", "BAAI/bge-reranker-base")


def _load_model():
    """懒加载重排模型（进程内只尝试一次，失败即永久降级本进程）。"""
    global _model, _load_tried
    if _load_tried:
        return _model
    _load_tried = True
    if not _HAS_CROSS_ENCODER:
        print("[rerank] 无 sentence-transformers，已降级（跳过重排）")
        return None
    try:
        _model = CrossEncoder(_model_name())
        print(f"[rerank] 已加载模型 {_model_name()}")
    except Exception as e:
        print(f"[rerank] 模型加载失败（已降级，跳过重排）: {type(e).__name__}")
        _model = None
    return _model


def rerank(question: str, hits: list[dict]) -> list[dict]:
    """对候选片段重排：返回重排后的完整列表（长度不变，上层再截断）。

    排序策略：**加分微调而非颠覆排序**。交叉编码器对中文短文本打分普遍偏高且
    集中（0.85~0.99），直接按 rerank 分会把融合排序的正确结果打乱（实测 async_03
    融合分第 1 被打到第 7）。因此以池内最低分为基线，把 rerank 分差缩放为加分
    并入 score：融合排序仍主导，rerank 只对相邻项做精细化调整。

    不可用（无模型/打分失败）时原样返回，保证检索链路永不因重排中断。
    """
    if not hits or len(hits) <= 1:
        return hits
    model = _load_model()
    if model is None:
        return hits
    n = int(rerank_cfg().get("top_n", 30))
    pool = hits[:n]
    if len(pool) <= 1:
        return hits
    try:
        pairs = [(question, h["text"][:512]) for h in pool]
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        print(f"[rerank] 打分失败（已降级）: {type(e).__name__}")
        return hits

    try:
        base = min(float(s) for s in scores)
    except Exception:
        return hits
    weight = float(rerank_cfg().get("weight", 0.15))
    for h, s in zip(pool, scores):
        rs = round(float(s), 4)
        h["rerank_score"] = rs
        h["score"] = round(h["score"] + weight * (float(s) - base), 4)
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits
