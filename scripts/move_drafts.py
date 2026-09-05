"""半自动归位脚本：notes_draft 里的文件移到建议 topic。
用法：PY=$(cygpath -w "$(pwd)/.venv/Scripts/python.exe") && "$PY" scripts/move_drafts.py
特点：每条建议都要 y/N 确认；mv 后删旧台账条目 + 触发 ingest 重建。
"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, '.')

from kb.config import raw_dir
from kb.storage import doc_index

root = raw_dir()
draft = root / "notes_draft"

# 建议：文件 → 目标 topic
SUGGESTIONS = {
    "test_upload.md": "Python基础",
    "示例-装饰器.md": "Python基础",
}


def confirm(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False


def main():
    if not draft.exists():
        print(f"notes_draft 不存在: {draft}")
        return
    files = [f.name for f in sorted(draft.iterdir())
             if f.is_file() and f.suffix in (".md", ".docx", ".pdf")]
    if not files:
        print("notes_draft 空，无需归位")
        return

    print(f"=== 建议归位 ===")
    to_move = []
    for fname in files:
        target = SUGGESTIONS.get(fname, "")
        if target:
            print(f"  {fname:30s} → {target}/")
            to_move.append((fname, target))
        else:
            print(f"  {fname:30s} → [无建议，跳过]")

    if not to_move:
        print("无可归位文件")
        return

    print()
    if not confirm("执行归位？"):
        print("已取消")
        return

    # 1. mv 文件
    moved = []
    for fname, target in to_move:
        src = draft / fname
        dst = root / target / fname
        if dst.exists():
            print(f"  ⚠ 目标已存在，跳过: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  ✓ {fname} → {target}/")
        moved.append((fname, target))

    # 2. 删旧台账条目（让 ingest 重建）
    if moved:
        idx = doc_index.load()
        for fname, _ in moved:
            # 台账里存的是 "notes_draft/xxx.md" 路径
            old_key = f"notes_draft/{fname}"
            if old_key in idx:
                del idx[old_key]
                print(f"  ✓ 删台账: {old_key}")
        doc_index.save(idx)

    # 3. 触发 ingest 重建
    print()
    if confirm("现在触发 ingest 重建台账 + chroma？"):
        from kb.ingestion.pipeline import ingest
        result = ingest(verbose=False)
        print(f"\n=== ingest 完成 ===")
        print(f"  new: {result.get('new')}, changed: {result.get('changed')}")
        print(f"  chunks: {result.get('chunks')}, elapsed: {result.get('elapsed')}s")
    else:
        print("跳过 ingest。手动跑: python -m kb.cli ingest")


if __name__ == "__main__":
    main()
