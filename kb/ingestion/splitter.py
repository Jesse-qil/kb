"""文档 → 片段：读取文件、按段落切块。只管"怎么切"，不管存哪。"""
from pathlib import Path


def read_file(path: Path) -> str:
    """读 .md / .docx / .pdf 为纯文本（段落间空一行，兼容分块逻辑）。"""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    if suffix == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def read_head(path: Path, limit: int = 1500) -> str:
    """读文件开头文本（分类器用，不用全读）。pdf 只提前几页。"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages[:3]:          # 只读前 3 页
            parts.append(page.extract_text() or "")
        return "\n\n".join(parts)[:limit]
    return read_file(path)[:limit]


def _read_pdf(path: Path) -> str:
    """用 pypdf 提取 PDF 全部文本（文本型 PDF；扫描图片版提不出字）。"""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


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
