"""Web 入口：启动服务后浏览器打开 http://127.0.0.1:8000
迁移自旧项目 app/main.py：API 保持一致（/api/ingest /api/stats /api/chat）。"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import File, Form, UploadFile

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


class ReviewRequest(BaseModel):
    filename: str
    topic: str = ""


@app.get("/")
def index():
    return RedirectResponse(url="/static/index.html")


# 允许上传的后缀白名单（防乱传；纯本地个人用，够用即可）
_ALLOWED_SUFFIX = {".md", ".docx", ".pdf"}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), auto: str = Form("0")):
    """网页上传：默认进待审查区并 LLM 建议主题；auto=1 时直接分类入库。
    返回: {status: pending|done, filename, suggest_topic, ingest?...}"""
    import re
    from kb.config import pending_dir
    from kb.ingestion.pending import add as pending_add, approve as pending_approve
    from kb.ingestion.classifier import suggest_topic

    suffix = pathlib_suffix(file.filename or "")
    if suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(400, f"只支持 {'/'.join(sorted(_ALLOWED_SUFFIX))} 文件")
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", (file.filename or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
    if not safe_name.endswith(tuple(_ALLOWED_SUFFIX)):
        safe_name += suffix

    content = await file.read()

    # 文件先落 pending/（无论 auto 与否都先到这，auto 再继续走 approve）
    pdir = pending_dir()
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / safe_name).write_bytes(content)
    print(f"[上传] 收到 {safe_name} ({len(content)} 字节)")

    # LLM 建议主题：按后缀用对应解析器读文本开头（pdf 不能直接 decode）
    from kb.ingestion.splitter import read_head
    tmp_path = pdir / safe_name          # 文件已落盘
    text_head = read_head(tmp_path)
    topic = suggest_topic(text_head)
    print(f"[分类] 建议主题: {topic or '（待定）'}")

    # auto 模式：直接审查通过（移入 raw/<topic> + 入库）
    if auto in ("1", "true", "True"):
        if not topic:
            topic = "默认"
        try:
            moved = pending_approve(safe_name, topic)
        except Exception as e:
            raise HTTPException(500, f"自动入库失败: {e}")
        from kb.ingestion.pipeline import ingest as run_ingest
        stats = run_ingest()
        return {"status": "done", "filename": safe_name,
                "topic": topic, "path": moved["path"], "ingest": stats}

    # 默认：登记进待审查
    pending_add(safe_name, topic, len(content))
    return {"status": "pending", "filename": safe_name,
            "suggest_topic": topic, "size": len(content),
            "tip": "已进入待审查区，确认后才会入库"}


@app.get("/api/pending")
def api_pending():
    """列出待审查文件。"""
    from kb.ingestion.pending import list_all
    return {"pending": list_all()}


@app.post("/api/review")
async def api_review(req: ReviewRequest):
    """人工审查通过：移入 raw/<topic> 并入库。body: {filename, topic}"""
    from kb.ingestion.pipeline import ingest as run_ingest
    from kb.ingestion.pending import approve
    try:
        moved = approve(req.filename, req.topic)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    stats = run_ingest()
    return {"ok": True, "path": moved["path"], "ingest": stats}


@app.delete("/api/pending")
async def api_pending_delete(req: ReviewRequest):
    """丢弃一条待审查。body: {filename}（topic 可空）"""
    from kb.ingestion.pending import reject
    ok = reject(req.filename)
    if not ok:
        raise HTTPException(404, "记录不存在")
    return {"ok": True}


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
