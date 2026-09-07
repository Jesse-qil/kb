"""Web 入口：启动服务后浏览器打开 http://127.0.0.1:8000
迁移自旧项目 app/main.py：API 保持一致（/api/ingest /api/stats /api/chat）。"""
import threading
import time
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
import json
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import File, Form, UploadFile

from kb.config import KB_ROOT
from kb.storage.vector_store import list_contents
from kb.agents.graph import ask

app = FastAPI(title="kb-v2 个人知识库")

# ---------- 后台入库任务状态（前端进度条轮询用） ----------
# 进度回调签名：progress(stage: str, percent: float, *, current="", total=0, done=0)
# main.py 在 _INGEST 上自动维护 elapsed/eta，前端只读字段。
_INGEST: dict = {
    "running": False,
    "stage": "",
    "percent": 0,
    "current": "",       # 当前操作对象（如文件名 / "向量化 64/180"）
    "done": 0,
    "total": 0,
    "elapsed": 0,        # 已用秒
    "eta": 0,            # 预计剩余秒（按当前 percent 推算）
    "started_at": 0.0,   # 内部用：开始时间戳
    "result": None,
    "error": None,
}
_INGEST_LOCK = threading.Lock()  # 守护 _INGEST 写入原子性，前端轮询读一致

# ---------- 评估后台运行状态 ----------
_EVAL_STATE = {
    "running": False,
    "stage": "",
    "percent": 0,
    "started_at": 0.0,
    "elapsed": 0,
    "result": None,
    "error": None,
    "output": "",
}
_EVAL_LOCK = threading.Lock()


def _safe_raw_relpath(raw_subpath: str) -> Path:
    from kb.config import raw_dir
    base = raw_dir().resolve()
    rel = Path(raw_subpath.strip("/\\"))
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise HTTPException(400, "非法文件路径") from e
    return target


def _report_ingest(stage: str, percent: float, *, current: str = "",
                    total: int = 0, done: int = 0) -> None:
    """入库进度回调：把每一步的状态写进 _INGEST。供 pipeline.py 调用。"""
    now = time.time()
    with _INGEST_LOCK:
        if not _INGEST["started_at"]:
            _INGEST["started_at"] = now
        elapsed = int(now - _INGEST["started_at"])
        pct = max(0.0, min(100.0, float(percent)))
        # 已用 > 0 且 pct > 1 时，按线性外推估算剩余
        if pct >= 1 and elapsed > 0:
            eta = int(elapsed * (100 - pct) / pct)
        else:
            eta = 0
        _INGEST.update({
            "stage": stage,
            "percent": int(pct),
            "current": current or "",
            "done": int(done) if done else _INGEST["done"],
            "total": int(total) if total else _INGEST["total"],
            "elapsed": elapsed,
            "eta": eta,
        })


def _run_ingest_bg():
    """在后台线程执行 ingest，进度写入 _INGEST。"""
    from kb.ingestion.pipeline import ingest as run_ingest
    with _INGEST_LOCK:
        _INGEST.update({
            "running": True, "stage": "启动", "percent": 0,
            "current": "", "done": 0, "total": 0,
            "elapsed": 0, "eta": 0, "started_at": time.time(),
            "result": None, "error": None,
        })
    try:
        result = run_ingest(verbose=True, progress=_report_ingest)
        with _INGEST_LOCK:
            _INGEST["result"] = result
            _INGEST["percent"] = 100
            _INGEST["stage"] = "完成"
            _INGEST["eta"] = 0
    except Exception as e:
        with _INGEST_LOCK:
            _INGEST["error"] = f"{type(e).__name__}: {e}"
            _INGEST["stage"] = "出错"
    finally:
        with _INGEST_LOCK:
            _INGEST["running"] = False


def start_bg_ingest() -> bool:
    """若没在跑则启动后台入库，返回是否新启动。"""
    if _INGEST["running"]:
        return False
    t = threading.Thread(target=_run_ingest_bg, daemon=True)
    t.start()
    return True


@app.get("/api/ingest/progress")
def api_ingest_progress():
    """前端进度条轮询：running/stage/percent/current/done/total/elapsed/eta/result/error"""
    with _INGEST_LOCK:
        return {k: _INGEST[k] for k in
                ("running", "stage", "percent", "current", "done", "total",
                 "elapsed", "eta", "result", "error")}

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


class SessionCreateRequest(BaseModel):
    title: str = "新会话"


class FolderImportRequest(BaseModel):
    path: str
    auto: bool = True
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
            # 自动打标签预览（多标签，入库时还会正式打一遍）
            from kb.ingestion.tagger import suggest_tags
            tags = suggest_tags(text_head, title=safe_name, topic=topic)

            if auto in ("1", "true", "True"):
                if not topic:
                    topic = "默认"
                moved = pending_approve(safe_name, topic)
                moved_any = True
                results.append({"filename": safe_name, "status": "done",
                                "topic": topic, "tags": tags, "path": moved["path"]})
            else:
                pending_add(safe_name, topic, len(content), fhash, tags=tags)
                results.append({"filename": safe_name, "status": "pending",
                                "suggest_topic": topic, "tags": tags, "size": len(content)})
        except Exception as e:
            results.append({"filename": name, "status": "error", "error": f"{type(e).__name__}: {e}"})

    resp = {"results": results}
    if moved_any:
        start_bg_ingest()          # auto 有移动才触发一次后台入库
        resp["ingest_started"] = True
    return resp


@app.post("/api/import/folder")
def api_import_folder(req: FolderImportRequest):
    """导入本地文件夹：自动识别 .md/.docx/.pdf，LLM 分类后批量入库。
    body: {path, auto=True, topic=""}；auto=False 时全部进待审查区。"""
    from kb.ingestion.folder_import import import_folder
    try:
        summary = import_folder(req.path, auto=req.auto, topic=req.topic)
    except ValueError as e:
        raise HTTPException(400, str(e))
    resp = {"summary": summary}
    if summary["imported"]:
        resp["ingest_started"] = start_bg_ingest()
    return resp


# ---------- 会话管理 ----------
@app.get("/api/sessions")
def api_sessions():
    """会话列表（前端多会话 UI 用）。"""
    from kb import session
    return {"sessions": session.list_sessions()}


@app.post("/api/sessions")
def api_session_create(req: SessionCreateRequest):
    """新建会话，返回新 session_id。"""
    from kb import session
    sid = session.create(req.title)
    return {"session_id": sid, "title": req.title}


@app.delete("/api/sessions/{session_id}")
def api_session_delete(session_id: str):
    """删除会话（含其历史）。"""
    from kb import session
    ok = session.delete(session_id)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return {"ok": True}


@app.get("/api/sessions/{session_id}/history")
def api_session_history(session_id: str):
    """某会话的完整历史（切换会话时前端加载回看）。"""
    from kb import session
    return {"session_id": session_id, "messages": session.get_history(session_id)}


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
    try:
        ok = reject(req.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "记录不存在")
    return {"ok": True}


def pathlib_suffix(name: str) -> str:
    """取文件名后缀（小写），无后缀返回空串。"""
    from pathlib import Path
    return Path(name).suffix.lower()


def _post_chat_housekeeping(session_id: str, question: str, answer: str,
                            session) -> None:
    """问答后的收尾（写会话 + 记忆提炼）放后台线程跑。

    记忆提炼是一次 LLM 调用（数秒），若在请求线程里同步跑，
    会拖住 SSE 连接关闭 / 阻塞 /api/chat 响应。后台执行 + 全程静默失败，
    不影响主对话链路。
    """
    def _work() -> None:
        try:
            session.append(session_id, "user", question)
            session.append(session_id, "assistant", answer)
        except Exception:
            pass
        try:
            from kb import memory
            from kb.llm import LLMClient
            memory.update_from_conversation(question, answer, LLMClient())
        except Exception:
            pass

    import threading
    threading.Thread(target=_work, daemon=True).start()


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
    # 记入历史（用户问的 + 小齐答的）——保持同步，避免连续提问时历史竞争
    session.append(req.session_id, "user", req.question)
    session.append(req.session_id, "assistant", result.get("answer", ""))
    # 回答后自动提炼用户画像写回 profile（后台线程，不拖慢响应；静默失败）
    if result.get("answer", "").strip():
        import threading

        def _extract() -> None:
            try:
                from kb import memory
                from kb.llm import LLMClient
                memory.update_from_conversation(
                    req.question, result.get("answer", ""), LLMClient()
                )
            except Exception:
                pass

        threading.Thread(target=_extract, daemon=True).start()
    return result


class ChatStreamRequest(BaseModel):
    question: str
    session_id: str = "s_demo_default"


@app.post("/api/chat/stream")
def api_chat_stream(req: ChatStreamRequest):
    """SSE 流式问答：前端按事件渲染（打字机效果）。
    事件流：{"type":"start"} → status* / delta* → {"type":"done"|"error"}
    """
    if not req.question.strip():
        raise HTTPException(400, "问题不能为空")
    from kb import session
    from kb.agents.graph import ask_stream
    history = session.get_text(req.session_id)
    final = {"answer": ""}

    def gen():
        yield "data: " + json.dumps({"type": "start"}, ensure_ascii=False) + "\n\n"
        try:
            for ev in ask_stream(req.question, history=history):
                if ev.get("type") == "done":
                    final["answer"] = ev.get("answer", "")
                yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps(
                {"type": "error", "content": f"{type(e).__name__}: {e}"},
                ensure_ascii=False) + "\n\n"
        # 出错时 answer 为空：不写会话、不提炼记忆，避免污染历史
        if final["answer"].strip():
            _post_chat_housekeeping(req.session_id, req.question,
                                    final["answer"], session)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# ===== 管理层接口 =====
@app.get("/api/manage/stats")
def api_manage_stats():
    """管理层详细统计：文档/片段/标签分布 + 文档列表。"""
    import os
    from kb.config import index_file
    from kb.storage.vector_store import count
    index_path = str(index_file())
    docs = {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except Exception:
        pass
    tag_count = {}
    topic_count = {}
    doc_list = []
    for path_, meta in docs.items():
        topic = meta.get("topic", "未分类")
        topic_count[topic] = topic_count.get(topic, 0) + 1
        for t in meta.get("tags", []):
            tag_count[t] = tag_count.get(t, 0) + 1
        doc_list.append({
            "path": path_,
            "filename": os.path.basename(path_),
            "topic": topic,
            "tags": meta.get("tags", []),
            "mtime": meta.get("mtime", 0),
            "id": meta.get("id", ""),
        })
    tag_store_count = len(tag_count)
    try:
        from kb.ingestion.tag_store import list_tags
        tag_store_count = len(list_tags())
    except Exception:
        pass
    doc_list.sort(key=lambda x: x["mtime"], reverse=True)
    top_tags = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:20]
    return {
        "total_docs": len(docs),
        "total_chunks": count(),
        "total_tags": tag_store_count,
        "topic_distribution": dict(sorted(topic_count.items(), key=lambda x: x[1], reverse=True)),
        "tag_top20": [{"name": k, "count": v} for k, v in top_tags],
        "docs": doc_list,
    }


@app.get("/api/docs")
def api_docs(q: str = ""):
    """文档列表，支持按文件名/标签搜索。"""
    import os
    from kb.config import index_file
    index_path = str(index_file())
    docs = {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
    except Exception:
        pass
    result = []
    q = (q or "").lower().strip()
    for path_, meta in docs.items():
        if q and q not in path_.lower() and q not in " ".join(meta.get("tags", [])).lower():
            continue
        result.append({
            "path": path_,
            "filename": os.path.basename(path_),
            "topic": meta.get("topic", "未分类"),
            "tags": meta.get("tags", []),
            "mtime": meta.get("mtime", 0),
        })
    result.sort(key=lambda x: x["mtime"], reverse=True)
    return {"docs": result, "total": len(result)}


@app.get("/api/tags")
def api_tags():
    """标签列表（从标签向量库读）。"""
    try:
        from kb.ingestion.tag_store import list_tags
        tags = list_tags()
        return {"tags": tags, "total": len(tags)}
    except Exception as e:
        return {"tags": [], "total": 0, "error": str(e)}



@app.delete("/api/docs/{doc_path:path}")
def api_delete_doc(doc_path: str, delete_file: bool = False):
    """删除文档：从向量库删片段 + 删台账条目，可选删除 raw 文件。
    doc_path 是相对路径（如 AI-Agent/07-AI基础总纲.md）。
    delete_file=true 时同时删除 knowledge/raw/ 下的原始文件。"""
    from kb.storage.doc_index import load, save
    from kb.storage.vector_store import delete_by_source

    doc_path = doc_path.strip("/\\")
    rows = load()
    if doc_path not in rows:
        return {"ok": False, "error": f"文档不存在: {doc_path}"}

    info = rows[doc_path]
    topic = info.get("topic", "")
    try:
        delete_by_source(doc_path, topic=topic)
    except Exception as e:
        return {"ok": False, "error": f"向量库删除失败: {e}"}

    del rows[doc_path]
    save(rows)

    raw_path = _safe_raw_relpath(doc_path)
    file_deleted = False
    if delete_file and raw_path.exists():
        try:
            raw_path.unlink()
            file_deleted = True
        except Exception as e:
            return {"ok": True, "warning": f"raw 文件删除失败: {e}",
                    "doc_path": doc_path, "file_deleted": False}

    return {"ok": True, "doc_path": doc_path, "topic": topic,
            "file_deleted": file_deleted, "remaining_docs": len(rows)}


@app.get("/api/manage/export")
def api_manage_export():
    """知识库快照导出：打包 chroma 向量库 + 台账 + 配置为 zip 下载。"""
    import io, os, zipfile
    from kb.config import KB_ROOT, chroma_dir
    from fastapi.responses import StreamingResponse

    chroma_path = chroma_dir()
    index_path = os.path.join(KB_ROOT, "index", "doc_index.json")
    config_path = os.path.join(KB_ROOT, "kb_config.yaml")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(chroma_path):
            for root, dirs, files in os.walk(chroma_path):
                for fname in files:
                    if fname.endswith(("-wal", "-shm")):
                        continue
                    fpath = os.path.join(root, fname)
                    arcname = os.path.join("chroma", os.path.relpath(fpath, chroma_path))
                    zf.write(fpath, arcname)
        if os.path.exists(index_path):
            zf.write(index_path, "doc_index.json")
        if os.path.exists(config_path):
            zf.write(config_path, "kb_config.yaml")
        zf.writestr("README.txt",
            "kb-v2 知识库快照导出\n"
            "恢复方法：\n"
            "1. chroma/ 目录覆盖到 knowledge/vector_store/chroma/\n"
            "2. doc_index.json 覆盖到 knowledge/index/doc_index.json\n"
            "3. kb_config.yaml 放到项目根目录\n")

    buf.seek(0)
    from datetime import datetime
    fname = f"kb_v2_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )



@app.delete("/api/tags/{tag}")
def api_delete_tag(tag: str):
    """删除一个标签（从标签库移除）。"""
    from kb.ingestion.tag_store import delete_tag
    ok = delete_tag(tag)
    if ok:
        return {"ok": True, "tag": tag, "message": f"标签「{tag}」已删除"}
    return {"ok": False, "error": f"标签「{tag}」不存在或删除失败"}


@app.post("/api/tags/merge")
async def api_merge_tag(request: Request):
    """合并标签：把 from 标签合并到 to 标签，删除 from。"""
    body = await request.json()
    from_tag = body.get("from", "").strip()
    to_tag = body.get("to", "").strip()
    if not from_tag or not to_tag:
        return {"ok": False, "error": "from 和 to 标签名不能为空"}
    if from_tag == to_tag:
        return {"ok": False, "error": "两个标签相同，无需合并"}
    from kb.ingestion.tag_store import merge_tag
    result = merge_tag(from_tag, to_tag)
    return result


@app.get("/api/docs/{doc_path:path}/content")
def api_doc_content(doc_path: str):
    """文档在线预览：返回 raw 文件内容。"""
    doc_path = doc_path.strip("/\\")
    raw_path = _safe_raw_relpath(doc_path)
    if not raw_path.exists():
        return {"ok": False, "error": f"文件不存在: {doc_path}"}
    try:
        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {"ok": True, "path": doc_path, "content": content,
                "size": len(content), "filename": os.path.basename(doc_path)}
    except Exception as e:
        return {"ok": False, "error": f"读取失败: {e}"}


@app.get("/api/sessions/{session_id}/export")
def api_session_export(session_id: str):
    """导出完整会话为 Markdown 文件下载。"""
    import os
    from kb.config import KB_ROOT
    from datetime import datetime
    from fastapi.responses import Response

    sess_dir = os.path.join(KB_ROOT, "sessions")
    sess_file = os.path.join(sess_dir, f"{session_id}.json")
    if not os.path.exists(sess_file):
        return {"ok": False, "error": "会话不存在"}
    try:
        import json
        with open(sess_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        title = data.get("title", "未命名会话")
        messages = data.get("messages", [])
        lines = [f"# {title}", "", f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"> 消息数：{len(messages)}", "---", ""]
        for i, m in enumerate(messages):
            role = m.get("role", "unknown")
            content = m.get("content", "")
            role_label = "🧑 用户" if role == "user" else "🤖 助手" if role == "assistant" else role
            lines.append(f"## {role_label}（第 {i+1} 条）")
            lines.append("")
            lines.append(content)
            lines.append("")
            sources = m.get("sources") or []
            if sources:
                lines.append("**参考来源：**")
                for src in sources:
                    s_path = src.get("source", "") if isinstance(src, dict) else str(src)
                    lines.append(f"- {s_path}")
                lines.append("")
            lines.append("---")
            lines.append("")
        md = "\n".join(lines)
        fname = f"session_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        return Response(content=md, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename={fname}"})
    except Exception as e:
        return {"ok": False, "error": f"导出失败: {e}"}


@app.post("/api/docs/export")
async def api_docs_export(request: Request):
    """批量导出文档：把选中的 raw 文件打包为 zip 下载。
    body: {paths: ["topic/file1.md", "topic/file2.md"]}"""
    import io, os, zipfile
    from kb.config import raw_dir
    from fastapi.responses import StreamingResponse

    body = await request.json()
    paths = body.get("paths", [])
    if not paths:
        return {"ok": False, "error": "未选择任何文档"}

    buf = io.BytesIO()
    exported = []
    skipped = []
    raw_base = str(raw_dir())
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            p = p.strip("/\\")
            try:
                raw_path = _safe_raw_relpath(p)
            except HTTPException:
                skipped.append(p)
                continue
            if raw_path.exists():
                zf.write(raw_path, p)
                exported.append(p)
            else:
                skipped.append(p)
        if skipped:
            zf.writestr("_skipped.txt",
                "以下文件未找到，已跳过：\n" + "\n".join(skipped))

    if not exported:
        return {"ok": False, "error": "所有文件均未找到"}

    buf.seek(0)
    from datetime import datetime
    fname = f"kb_docs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


# ---------- Agent 评估仪表盘 API ----------
@app.get("/api/eval/results")
def api_eval_results():
    """返回最新 Agent 评估结果 JSON。"""
    eval_json = KB_ROOT / "tests" / "results" / "agent_eval_latest.json"
    if not eval_json.exists():
        return {"ok": False, "error": "暂无评估结果，请先运行 tests/agent_eval.py"}
    try:
        import json as _json
        data = _json.loads(eval_json.read_text(encoding="utf-8"))
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": f"读取评估结果失败: {e}"}


@app.get("/api/eval/history")
def api_eval_history():
    """返回历史评估记录列表（按时间倒序）。"""
    hist_dir = KB_ROOT / "tests" / "results" / "history"
    if not hist_dir.exists():
        return {"ok": True, "files": []}
    files = sorted(hist_dir.glob("eval_*.json"), reverse=True)
    result = []
    for f in files[:20]:  # 最多返回20条
        try:
            import json as _json
            data = _json.loads(f.read_text(encoding="utf-8"))
            s = data.get("summary", {})
            result.append({
                "file": f.name,
                "timestamp": data.get("timestamp", f.stem.replace("eval_", "")),
                "mode": data.get("mode", "graph"),
                "total": s.get("total", 0),
                "source_hit_rate": s.get("source_hit_rate", 0),
                "answer_hit_rate": s.get("answer_hit_rate", 0),
                "domain_match_rate": s.get("domain_match_rate", 0),
                "need_tool_match_rate": s.get("need_tool_match_rate", 0),
                "avg_latency_ms": s.get("avg_latency_ms", 0),
                "avg_total_tokens": s.get("avg_total_tokens", 0),
            })
        except Exception:
            continue
    return {"ok": True, "files": result}


#  评估后台运行
def _run_eval_bg(limit: int = 0, use_judge: bool = False):
    """在后台线程运行评估脚本，进度写入 _EVAL_STATE。"""
    import subprocess
    import sys
    with _EVAL_LOCK:
        _EVAL_STATE.update({
            "running": True, "stage": "启动评估", "percent": 0,
            "started_at": time.time(), "elapsed": 0,
            "result": None, "error": None, "output": "",
        })
    try:
        cmd = [sys.executable, str(KB_ROOT / "tests" / "agent_eval.py")]
        if limit and limit > 0:
            cmd += ["--limit", str(limit)]
        if use_judge:
            cmd += ["--judge"]
        proc = subprocess.Popen(
            cmd, cwd=str(KB_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            # 解析进度：如 "Evaluating 15/50..."
            line_stripped = line.strip()
            if line_stripped.startswith("Evaluating"):
                try:
                    parts = line_stripped.split()
                    curr_total = parts[1].split("/")
                    curr = int(curr_total[0])
                    total = int(curr_total[1])
                    pct = int(curr / total * 100) if total > 0 else 0
                    with _EVAL_LOCK:
                        _EVAL_STATE["percent"] = pct
                        _EVAL_STATE["stage"] = f"评估中 {curr}/{total}"
                        _EVAL_STATE["elapsed"] = int(time.time() - _EVAL_STATE["started_at"])
                except Exception:
                    pass
        proc.wait()
        full_output = "".join(output_lines)
        with _EVAL_LOCK:
            _EVAL_STATE["output"] = full_output[-2000:]  # 保留最后2000字符
            _EVAL_STATE["percent"] = 100
            _EVAL_STATE["stage"] = "完成" if proc.returncode == 0 else f"退出码 {proc.returncode}"
            _EVAL_STATE["elapsed"] = int(time.time() - _EVAL_STATE["started_at"])
            if proc.returncode == 0:
                _EVAL_STATE["result"] = "评估完成，结果已更新"
            else:
                _EVAL_STATE["error"] = f"评估进程退出码 {proc.returncode}"
    except Exception as e:
        with _EVAL_LOCK:
            _EVAL_STATE["error"] = f"{type(e).__name__}: {e}"
            _EVAL_STATE["stage"] = "出错"
    finally:
        with _EVAL_LOCK:
            _EVAL_STATE["running"] = False


def start_bg_eval(limit: int = 0, use_judge: bool = False) -> bool:
    """若没在跑则启动后台评估，返回是否新启动。"""
    with _EVAL_LOCK:
        if _EVAL_STATE["running"]:
            return False
    t = threading.Thread(target=_run_eval_bg, args=(limit, use_judge), daemon=True)
    t.start()
    return True


@app.post("/api/eval/run")
async def api_eval_run(request: Request):
    """触发后台评估运行。可选参数：limit（用例数，0=全部）、use_judge（是否LLM打分）。"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    limit = int(body.get("limit", 0) or 0)
    use_judge = bool(body.get("use_judge", False))
    started = start_bg_eval(limit=limit, use_judge=use_judge)
    if started:
        return {"ok": True, "started": True, "message": f"评估已启动（{'全部50条' if limit == 0 else f'{limit}条'}），约15-20分钟"}
    else:
        return {"ok": False, "started": False, "message": "评估正在运行中，请稍候"}


@app.get("/api/eval/status")
def api_eval_status():
    """前端轮询评估运行状态。"""
    with _EVAL_LOCK:
        return {
            "running": _EVAL_STATE["running"],
            "stage": _EVAL_STATE["stage"],
            "percent": _EVAL_STATE["percent"],
            "elapsed": _EVAL_STATE["elapsed"],
            "result": _EVAL_STATE["result"],
            "error": _EVAL_STATE["error"],
        }


if __name__ == "__main__":
    import uvicorn
    from kb.config import server_cfg
    _srv = server_cfg()
    print(f"\033[92m kb-v2 启动中... 浏览器打开 http://{_srv['host']}:{_srv['port']}\033[0m")
    uvicorn.run("main:app", host=_srv["host"], port=_srv["port"])
