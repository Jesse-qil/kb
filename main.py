"""Web 入口：启动服务后浏览器打开 http://127.0.0.1:8000
迁移自旧项目 app/main.py：API 保持一致（/api/ingest /api/stats /api/chat）。"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import File, Form, UploadFile

from kb.config import KB_ROOT
from kb.storage.vector_store import list_contents
from kb.agents.graph import ask

app = FastAPI(title="kb-v2 个人知识库")

# ---------- 后台入库任务状态（前端进度条轮询用） ----------
_INGEST = {"running": False, "stage": "", "percent": 0, "result": None, "error": None}
import threading


def _run_ingest_bg():
    """在后台线程执行 ingest，进度写入 _INGEST。"""
    from kb.ingestion.pipeline import ingest as run_ingest
    try:
        def cb(stage, pct):
            _INGEST["stage"] = stage
            _INGEST["percent"] = pct
        _INGEST["running"] = True
        _INGEST["error"] = None
        _INGEST["result"] = run_ingest(verbose=True, progress=cb)
    except Exception as e:
        _INGEST["error"] = str(e)
    finally:
        _INGEST["running"] = False
        _INGEST["percent"] = 100


def start_bg_ingest() -> bool:
    """若没在跑则启动后台入库，返回是否新启动。"""
    if _INGEST["running"]:
        return False
    t = threading.Thread(target=_run_ingest_bg, daemon=True)
    t.start()
    return True


@app.get("/api/ingest/progress")
def api_ingest_progress():
    """前端进度条轮询：{running, stage, percent, result?, error?}"""
    return {k: _INGEST[k] for k in ("running", "stage", "percent", "result", "error")}

# 允许前端跨域（本机开发够用）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 托管前端页面（static 目录）
app.mount("/static", StaticFiles(directory=KB_ROOT / "static"), name="static")


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"     # 前端生成，同一次会话用同一个


class ReviewRequest(BaseModel):
    filename: str
    topic: str = ""


@app.get("/")
def index():
    return RedirectResponse(url="/static/index.html")


# 允许上传的后缀白名单（防乱传；纯本地个人用，够用即可）
_ALLOWED_SUFFIX = {".md", ".docx", ".pdf"}


@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...), auto: str = Form("0")):
    """网页批量上传：一次收多个文件。每个文件：
    - 算内容 md5 → 库里/pending 已有同内容 → 标 duplicate 跳过
    - 否则默认进待审查 + LLM 建议主题；auto=1 直接分类移入 raw
    返回 {results: [{filename, status: pending|done|duplicate|error, ...}], ingest_started?}"""
    import hashlib, re
    from kb.config import pending_dir
    from kb.ingestion.pending import add as pending_add, approve as pending_approve, find_by_hash as pending_dup
    from kb.ingestion.classifier import suggest_topic
    from kb.ingestion.splitter import read_head
    from kb.storage.doc_index import find_by_hash as doc_dup

    results = []
    moved_any = False
    pdir = pending_dir()
    pdir.mkdir(parents=True, exist_ok=True)

    for file in files:
        name = file.filename or ""
        try:
            suffix = pathlib_suffix(name)
            if suffix not in _ALLOWED_SUFFIX:
                results.append({"filename": name, "status": "error",
                                "error": f"只支持 {'/'.join(sorted(_ALLOWED_SUFFIX))}"})
                continue
            safe_name = re.sub(r'[\\/:*?"<>|]', "_", name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])
            if not safe_name.endswith(tuple(_ALLOWED_SUFFIX)):
                safe_name += suffix

            content = await file.read()
            fhash = hashlib.md5(content).hexdigest()

            # ---- 内容级去重：台账已入库 or 待审查区已有同内容 ----
            dup_doc = doc_dup(fhash)
            if dup_doc:
                results.append({"filename": name, "status": "duplicate",
                                "error": f"库中已有相同内容：{dup_doc}"})
                continue
            dup_pending = pending_dup(fhash)
            if dup_pending:
                results.append({"filename": name, "status": "duplicate",
                                "error": f"待审查区已有相同内容：{dup_pending}"})
                continue

            # 落 pending/（auto 也先落这，approve 负责移动）
            (pdir / safe_name).write_bytes(content)
            print(f"[上传] 收到 {safe_name} ({len(content)} 字节)")

            # LLM 建议主题（按后缀解析读开头；pdf 不能直接 decode）
            text_head = read_head(pdir / safe_name)
            topic = suggest_topic(text_head)

            if auto in ("1", "true", "True"):
                if not topic:
                    topic = "默认"
                moved = pending_approve(safe_name, topic)
                moved_any = True
                results.append({"filename": safe_name, "status": "done",
                                "topic": topic, "path": moved["path"]})
            else:
                pending_add(safe_name, topic, len(content), fhash)
                results.append({"filename": safe_name, "status": "pending",
                                "suggest_topic": topic, "size": len(content)})
        except Exception as e:
            results.append({"filename": name, "status": "error", "error": f"{type(e).__name__}: {e}"})

    resp = {"results": results}
    if moved_any:
        start_bg_ingest()          # auto 有移动才触发一次后台入库
        resp["ingest_started"] = True
    return resp


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
    start_bg_ingest()
    return {"ok": True, "path": moved["path"], "ingest_started": True}


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
    """扫描 raw/ 并增量导入（后台执行，前端轮询进度）。"""
    started = start_bg_ingest()
    return {"started": started, "running": _INGEST["running"]}


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
    from kb import session
    # 取该会话的历史拼文本 → 传给 graph
    history = session.get_text(req.session_id)
    result = ask(req.question, history=history)
    # 记入历史（用户问的 + 小齐答的）
    session.append(req.session_id, "user", req.question)
    session.append(req.session_id, "assistant", result.get("answer", ""))
    return result


if __name__ == "__main__":
    import uvicorn
    print("\033[92m kb-v2 启动中... 浏览器打开 http://127.0.0.1:8000\033[0m")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
