"""配置中心：读 kb_config.yaml，提供统一的路径与参数常量。
设计：所有代码不写死路径，都从这里拿 → 改配置不用改代码。"""
from pathlib import Path
import yaml

# 项目根 = kb 包（本文件）的上一级
KB_ROOT = Path(__file__).resolve().parent.parent
# 知识库数据根（raw / index / vector_store / chunks 都在它下面）
KNOWLEDGE_DIR = KB_ROOT / "knowledge"


def load_config() -> dict:
    with open(KB_ROOT / "kb_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 模块加载时读一次（改 yaml 后重启进程生效）
CONFIG = load_config()


def raw_dir() -> Path:
    return KNOWLEDGE_DIR / CONFIG["scan"]["raw_root"]


def index_file() -> Path:
    return KNOWLEDGE_DIR / "index" / "doc_index.json"


def chroma_dir() -> Path:
    return KNOWLEDGE_DIR / "vector_store" / "chroma"


def chunks_file() -> Path:
    return KNOWLEDGE_DIR / "chunks" / "chunk_cache.jsonl"


def split_cfg() -> dict:
    return CONFIG["split"]


def recall_cfg() -> dict:
    return CONFIG["recall"]


def embedding_cfg() -> dict:
    return CONFIG["embedding"]
