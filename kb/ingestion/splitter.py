"""文档 → 片段：读取文件、按段落切块。只管"怎么切"，不管存哪。"""
from pathlib import Path


def read_file(path: Path) -> str:
    """读 .md 或 .docx 为纯文本（段落间空一行，兼容分块逻辑）。"""
    if path.suffix == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return path.read_text(encoding="utf-8")


def chunk_md(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按段落切分，合并成不超过 chunk_size 字、相邻重叠 overlap 字的片段。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= chunk_size:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
                cur = cur[-overlap:] + "\n\n" + p
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return chunks
