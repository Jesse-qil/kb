"""Web 入口：启动服务后浏览器打开 http://127.0.0.1:8000
迁移自旧项目 app/main.py：API 保持一致（/api/ingest /api/stats /api/chat）。"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kb.config import KB_ROOT
from kb.kb_rag import ingest, list_contents
from kb.graph import ask

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


@app.post("/api/ingest")
def api_ingest():
    """扫描 raw/ 并增量导入。"""
    return ingest()


@app.get("/api/stats")
def api_stats():
    """库里有多少片段 + 各领域分布（兼容旧前端只读 chunks）。"""
    try:
        from kb.kb_rag import count
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
