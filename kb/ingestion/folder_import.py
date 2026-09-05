"""文件夹导入：把本地一个文件夹里的知识文件批量收进知识库。
流程：遍历 → 按后缀白名单识别（.md/.docx/.pdf）→ 跳过垃圾目录 →
     内容 md5 去重 → 复制到 pending/（源文件不动）→ LLM 建议主题 →
     auto=1 直接入 raw/<topic>/ 并由调用方触发入库；auto=0 留待人工审查。
放在 ingestion 层：和上传走同一条 pending 链路，审查/去重逻辑零重复。"""
import hashlib
import re
import shutil
from pathlib import Path

from ..config import CONFIG, raw_dir, pending_dir
from . import pending
from .classifier import suggest_topic
from .splitter import read_head
from ..storage import doc_index

# 明显不含知识文件的目录，整条跳过（含所有子层）
_SKIP_DIRS = {".git", ".svn", "__pycache__", "node_modules", ".venv", "venv",
              ".idea", ".vscode", ".workbuddy", "dist", "build"}
_BAD_CHARS = re.compile(r'[\\/:*?"<>|\s]+')
_MAX_FILE_MB = 50          # 单文件上限（pdf/docx 超过这个数多半是书或二进制，先拒掉）


def _iter_knowledge_files(root: Path):
    """递归找出知识文件：后缀在白名单、不在垃圾目录、不超大小。"""
    suffixes = {s.lower() for s in CONFIG["scan"]["file_suffix"]}
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        parts = f.relative_to(root).parts
        # 任一层父目录命中垃圾目录 / 隐藏目录 → 整条排除
        if any(p in _SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        if f.name.startswith("."):
            continue
        if f.suffix.lower() not in suffixes:
            continue
        if f.stat().st_size > _MAX_FILE_MB * 1024 * 1024:
            continue
        yield f


def _flat_name(root: Path, f: Path) -> str:
    """子目录信息压进文件名，防不同子目录同名冲突：
    notes/python/tips.md → notes__python__tips.md"""
    rel = f.relative_to(root).with_suffix("")   # 先去掉后缀，最后统一补一次
    parts = [_BAD_CHARS.sub("_", p).strip("_") or "_" for p in rel.parts]
    name = "__".join(parts)
    return name[:150] + f.suffix.lower()


def _uniq_name(pending_name: str, fhash: str, topic: str) -> str:
    """pending 区或 raw 目标位已有同名（且内容不同）→ 文件名追加短 hash。"""
    stem, suf = pending_name.rsplit(".", 1)
    if (pending_dir() / pending_name).exists():
        return f"{stem}__{fhash[:6]}.{suf}"
    if (raw_dir() / topic / pending_name).exists():
        return f"{stem}__{fhash[:6]}.{suf}"
    return pending_name


def import_folder(folder: str, auto: bool = True, topic: str = "",
                  max_files: int = 500) -> dict:
    """导入一个本地文件夹。
    - auto=True: 自动分类入 raw（LLM 判主题；topic 参数可强制指定）
    - auto=False: 全部进待审查区，前端逐个确认
    源文件只读不删（复制而非移动）。返回 {total, imported, pending, duplicate, skipped, errors, details}"""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效文件夹: {folder}")
    # 防自导入：不能把知识库自身的数据目录再导一遍
    raw_root = raw_dir().resolve()
    if root == raw_root or root in raw_root.parents or raw_root in root.parents:
        raise ValueError("不能导入知识库自身的数据目录")

    pending_dir().mkdir(parents=True, exist_ok=True)
    summary = {"total": 0, "imported": 0, "pending": 0, "duplicate": 0,
               "skipped": 0, "errors": 0, "details": []}
    batch_hashes = set()   # 本批次内部去重（文件夹里同内容多份拷贝）

    for f in _iter_knowledge_files(root):
        if summary["total"] >= max_files:
            summary["skipped"] += 1
            continue
        summary["total"] += 1
        try:
            content = f.read_bytes()
            fhash = hashlib.md5(content).hexdigest()

            # ---- 去重：库里已有 / 待审查已有 / 本批已有 ----
            if doc_index.find_by_hash(fhash):
                summary["duplicate"] += 1
                summary["details"].append({"file": f.name, "status": "duplicate",
                                           "note": "库中已有相同内容"})
                continue
            if pending.find_by_hash(fhash):
                summary["duplicate"] += 1
                summary["details"].append({"file": f.name, "status": "duplicate",
                                           "note": "待审查区已有相同内容"})
                continue
            if fhash in batch_hashes:
                summary["duplicate"] += 1
                continue
            batch_hashes.add(fhash)

            # ---- 复制进 pending（先落这，approve 负责移到 raw）----
            name = _flat_name(root, f)
            text_head = read_head(f)
            t = (topic or "").strip() or suggest_topic(text_head)
            if auto and not t:
                t = "默认"
            # 自动打标签预览（多标签，入库时还会正式打一遍）
            from .tagger import suggest_tags
            tags = suggest_tags(text_head, title=f.name, topic=t)
            name = _uniq_name(name, fhash, t)
            shutil.copy2(f, pending_dir() / name)

            if auto:
                moved = pending.approve(name, t)
                summary["imported"] += 1
                summary["details"].append({"file": f.name, "status": "done",
                                           "topic": t, "tags": tags, "path": moved["path"]})
            else:
                pending.add(name, t, len(content), fhash, tags=tags)
                summary["pending"] += 1
                summary["details"].append({"file": f.name, "status": "pending",
                                           "suggest_topic": t, "tags": tags})
        except Exception as e:
            summary["errors"] += 1
            summary["details"].append({"file": f.name, "status": "error",
                                       "note": f"{type(e).__name__}: {e}"})

    print(f"[文件夹导入] {root.name}: 共{summary['total']} "
          f"入库{summary['imported']} 待审{summary['pending']} "
          f"重复{summary['duplicate']} 跳过{summary['skipped']} 错误{summary['errors']}")
    return summary
