"""本地 Embedding：把文字变成向量。中文效果好、免费、不联网。
复用旧项目验证过的方案：bge-small-zh-v1.5 + 离线模式 + bge 检索指令前缀。"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"          # 离线：模型已缓存就不联网
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # 双保险
from sentence_transformers import SentenceTransformer

from .config import embedding_cfg

_model = None

# bge 系列建议：检索 query 加指令前缀，效果更好
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:                       # 懒加载：只用一次就只加载一次
        _model = SentenceTransformer(embedding_cfg()["model_name"])
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把多个文本批量转成向量（归一化后，余弦相似度 = 内积，检索更快）"""
    vecs = get_model().encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def embed_query(query: str) -> list[float]:
    """把用户问题转成向量（带 bge 指令前缀）"""
    return embed_texts([_QUERY_PREFIX + query])[0]


if __name__ == "__main__":
    print("\033[92m embedder.py 文件\033[0m")
    v = embed_texts(["你好", "hello"])
    print("向量维度:", len(v[0]))   # 预期 512（bge-small-zh 是 512 维）
    print("两个句子相似度:", round(sum(a * b for a, b in zip(v[0], v[1])), 4))
