"""联网搜索：知识库没找到时的兜底。封装 duckduckgo_search。"""
try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


def web_search(query: str, top_k: int = 3) -> list[str]:
    """返回搜索结果的标题列表（免费无 key）。失败返回空列表，不抛异常。"""
    if DDGS is None:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=top_k))
        return [r["title"] for r in results]
    except Exception as e:
        print("联网搜索失败:", e)
        return []
