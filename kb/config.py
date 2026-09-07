"""配置中心：读 kb_config.yaml，提供统一的路径与参数常量。
设计：所有代码不写死路径，都从这里拿 → 改配置不用改代码。"""
from pathlib import Path
import json

try:
    import yaml
except Exception:
    yaml = None

# 项目根 = kb 包（本文件）的上一级
KB_ROOT = Path(__file__).resolve().parent.parent
# 知识库数据根（raw / index / vector_store / chunks 都在它下面）
KNOWLEDGE_DIR = KB_ROOT / "knowledge"

_DEFAULT_CONFIG = {
    "scan": {
        "raw_root": "raw",
        "compiled_root": "compiled_wiki",
        "enable_compiled_layer": False,
        "file_suffix": [".md", ".docx", ".pdf"],
        "exclude_dirs": [],
    },
    "split": {"chunk_size": 400, "chunk_overlap": 120},
    "embedding": {"model_name": "BAAI/bge-small-zh-v1.5"},
    "recall": {"top_k": 4, "score_threshold": 0.45,
               "bm25_boost": 0.10, "bm25_top_k": 10},
    "llm": {"provider": "auto", "temperature": 0.3},
    "server": {"host": "127.0.0.1", "port": 8000, "reload": True},
    "session": {"max_turns": 6},
    "agent": {
        "max_review_rounds": 3,
        "tool_max_rounds": 5,
        "query_top_k": 3,
        "tag_match_top_n": 3,
        "tag_match_threshold": 0.45,
    },
    "tagger": {
        "max_tags": 6,
        "retrieve_top_k": 5,
        "reuse_min": 0.60,
        "gray_min": 0.45,
    },
}


def load_config() -> dict:
    cfg_file = KB_ROOT / "kb_config.yaml"
    if yaml is None:
        # 轻量环境下没有 PyYAML 时，直接回退到内置默认配置。
        # 这能保证项目可启动；有需要再通过安装依赖启用 YAML 覆盖。
        return json.loads(json.dumps(_DEFAULT_CONFIG))
    with open(cfg_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = json.loads(json.dumps(_DEFAULT_CONFIG))
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


# 模块加载时读一次（改 yaml 后重启进程生效）
CONFIG = load_config()


def raw_dir() -> Path:
    return KNOWLEDGE_DIR / CONFIG["scan"]["raw_root"]


def index_file() -> Path:
    return KNOWLEDGE_DIR / "index" / "doc_index.json"



def pending_dir() -> Path:
    """待审查区：上传文件先落这里（在 raw 外，ingest 不会误扫）。"""
    return KNOWLEDGE_DIR / "pending"

def chroma_dir() -> Path:
    return KNOWLEDGE_DIR / "vector_store" / "chroma"


def chunks_file() -> Path:
    return KNOWLEDGE_DIR / "chunks" / "chunk_cache.jsonl"


def tag_schema_file() -> Path:
    return KNOWLEDGE_DIR / "tag_schema.json"


def split_cfg() -> dict:
    return CONFIG["split"]


def recall_cfg() -> dict:
    return CONFIG["recall"]


def embedding_cfg() -> dict:
    return CONFIG["embedding"]


def server_cfg() -> dict:
    return CONFIG["server"]


def session_cfg() -> dict:
    return CONFIG["session"]


def agent_cfg() -> dict:
    return CONFIG["agent"]


def tagger_cfg() -> dict:
    return CONFIG["tagger"]
