"""Web 入口：启动服务后浏览器打开 http://127.0.0.1:8000
迁移自旧项目 app/main.py：API 保持一致（/api/ingest /api/stats /api/chat）。"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import File, UploadFile

from kb.config import KB_ROOT
from kb.ingestion.pipeline import ingest
from kb.storage.vector_store import list_contents
from kb.agents.graph import ask

app = FastAPI(title="kb-v2 个人知识库")

# 允许前端跨域（本机开发够用）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 托管前端页面（static 目录）
app.mount("/static", StaticFiles(directory=KB_ROOT / "static"), name="static")


class ChatRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return RedirectResponse(url="/static/index.html")


# 允许上传的后缀白名单（防乱传；纯本地个人用，够用即可）
_ALLOWED_SUFFIX = {".md", ".docx"}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), topic: str = "notes_draft"):
    """网页上传笔记：存进 knowledge/raw/<topic>/ 并自动增量入库。
    返回入库统计。topic 默认 notes_draft，前端可指定领域。"""
    # 安全：只收 .md/.docx；文件名清洗（去掉路径分隔符，防目录穿越）
    import re
    from kb.config import raw_dir
    from kb.ingestion.pipeline import ingest as run_ingest

    suffix = pathlib_suffix(file.filename or "")
    if suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(400, f"只支持 {'/'.join(sorted(_ALLOWED_SUFFIX))} 文件")
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", (file.filename or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
    if not safe_name.endswith(tuple(_ALLOWED_SUFFIX)):
        safe_name += suffix

    # 目标目录：raw/<topic>/（顶层目录 = 领域）
    dest_dir = raw_dir() / topic
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe_name

    # 落盘（分块写，支持稍大文件）
    content = await file.read()
    dest.write_bytes(content)
    print(f"[上传] {topic}/{safe_name} ({len(content)} 字节)")

    # 增量入库（hash 对比：同名同内容会自动跳过）
    return {"saved": f"{topic}/{safe_name}", "size": len(content), "ingest": run_ingest()}


def pathlib_suffix(name: str) -> str:
    """取文件名后缀（小写），无后缀返回空串。"""
    from pathlib import Path
    return Path(name).suffix.lower()


@app.post("/api/ingest")
def api_ingest():
    """扫描 raw/ 并增量导入。"""
    return ingest()


@app.get("/api/stats")
def api_stats():
    """库里有多少片段 + 各领域分布（兼容旧前端只读 chunks）。"""
    try:
        from kb.storage.vector_store import count
        contents = list_contents()
        return {"chunks": count(),
                "topics": {k: len(v) for k, v in contents.items()}}
    except Exception:
        return {"chunks": 0, "topics": {}}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    return ask(req.question)


if __name__ == "__main__":
    import uvicorn
    print("\033[92m kb-v2 启动中... 浏览器打开 http://127.0.0.1:8000\033[0m")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
