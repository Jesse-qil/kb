"""主题分类器：读文档开头内容，让 LLM 判断属于哪个主题目录。
放在 ingestion 层：分类是"文件进 raw/ 前"的决策步骤。
- 已有目录（raw 顶层）优先复用；都不合适 → LLM 新建简短主题名
- 分类失败 / mock 模式 → 返回 ""（前端显示"待定"，由人工填）"""
import json, re

from ..llm import LLMClient
from ..storage import doc_index
from ..config import raw_dir

_TOPIC_BAD = re.compile(r'[\\/:*?"<>|\s]+')   # 目录名非法字符


def _existing_topics() -> list[str]:
    """已有领域 = raw 顶层目录 + 台账里出现过的 topic（去重）。"""
    topics = []
    root = raw_dir()
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                topics.append(d.name)
    for info in doc_index.load().values():
        t = info.get("topic")
        if t and t not in topics:
            topics.append(t)
    return topics


def clean_topic(name: str) -> str:
    """清洗成合法目录名；空返回空串。"""
    name = _TOPIC_BAD.sub("_", (name or "")).strip("_ ")
    return name[:30]


def suggest_topic(text_head: str, existing: list[str] | None = None) -> str:
    """给一段文本，返回建议主题目录名；失败返回空串。"""
    existing = existing if existing is not None else _existing_topics()
    cand = "、".join(existing) if existing else "（暂无，请新建）"
    prompt = (
        "你是知识库归档员。根据文档开头内容判断它属于哪个主题。\n"
        f"可选已有主题：{cand}\n"
        "规则：1) 明显匹配已有主题就用它；2) 都不合适就新起一个简短主题名"
        "（2-6 字，如 Python基础、求职、AI-Agent）；3) 只输出 JSON 格式："
        '{"topic": "主题名"}，不要多余文字。\n\n文档开头：\n' + text_head[:1500]
    )
    try:
        out = LLMClient().chat([
            {"role": "system", "content": "你只输出 JSON。"},
            {"role": "user", "content": prompt},
        ])
        topic = json.loads(out).get("topic", "")
        topic = clean_topic(topic)
        return topic if topic else ""
    except Exception as e:
        print(f"[分类] 失败({type(e).__name__})，返回待定")
        return ""
