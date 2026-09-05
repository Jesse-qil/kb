"""本地 Embedding：把文字变成向量。中文效果好、免费、不联网。
复用旧项目验证过的方案：bge-small-zh-v1.5 + 离线模式 + bge 检索指令前缀。

如果 sentence_transformers 不可用，就退回到一个纯本地的哈希向量器，
保证知识库在轻量环境里也能跑。"""
from __future__ import annotations

import hashlib
import os
import re

os.environ["HF_HUB_OFFLINE"] = "1"          # 离线：模型已缓存就不联网
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # 双保险
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from .config import embedding_cfg

_model = None
_use_fallback = SentenceTransformer is None

# bge 系列建议：检索 query 加指令前缀，效果更好
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_FALLBACK_DIM = 512


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if not norm:
        return vec
    return [v / norm for v in vec]


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_+\-]+|[\u4e00-\u9fff]{2,}", text)
    out = []
    for tok in tokens:
        out.append(tok)
        if len(tok) > 2 and re.fullmatch(r"[\u4e00-\u9fff]+", tok):
            out.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return out


def _fallback_encode(texts: list[str]) -> list[list[float]]:
    vecs = []
    for text in texts:
        vec = [0.0] * _FALLBACK_DIM
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % _FALLBACK_DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            weight = 1.0 + (len(tok) - 1) * 0.1
            vec[idx] += sign * weight
        vecs.append(_normalize(vec))
    return vecs


def get_model() -> SentenceTransformer | None:
    global _model, _use_fallback
    if _use_fallback:
        return None
    if _model is None:                       # 懒加载：只用一次就只加载一次
        try:
            _model = SentenceTransformer(embedding_cfg()["model_name"])
        except Exception:
            _use_fallback = True
            _model = None
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把多个文本批量转成向量（归一化后，余弦相似度 = 内积，检索更快）"""
    model = get_model()
    if model is None:
        return _fallback_encode(texts)
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """把用户问题转成向量（带 bge 指令前缀）"""
    return embed_texts([_QUERY_PREFIX + query])[0]


if __name__ == "__main__":
    print("\033[92m embedder.py 文件\033[0m")
    v = embed_texts(["你好", "hello"])
    print("向量维度:", len(v[0]))   # 预期 512（bge-small-zh 是 512 维）
    print("两个句子相似度:", round(sum(a * b for a, b in zip(v[0], v[1])), 4))
