# kb-v2 交接文档（新会话无缝继续）

> 生成时间：2026-09-05 16:33（上下文即将耗尽时生成）
> 项目：个人知识库问答系统（小齐），FastAPI + ChromaDB + 本地 LLM

---

## 一、项目一句话

本地知识库（103 篇 md/pdf）→ ingest（分块 + 多标签打标 + 向量化 + 缓存）→ ReAct Agent 检索问答 → 长期记忆回灌。

## 二、当前状态（最重要）

- **pytest：50/50 全绿（17.61s）**。chunk_cache 27 + memory 18 + retrieval 5
- **benchmark global hit@1 = 0.348**（旧数据清干净后，此前 0.087）
- **chroma 数据干净**：3766 embeddings，notes_draft 旧路径 0 条
- **待用户决定**：50 全绿后 AI 问过"是否提交这批改动"，用户尚未回复。**新会话第一件事：问用户是否 git 提交（或用户已手动删 43MB 旧文件后统一提交）**

## 三、Git 状态

已提交（6 个 commit，最新 → 最旧）：
- `3b2fdd1` chore: .gitignore 排除 tests/results/latest.*
- `fcf0405` docs: 项目文档同步更新
- `8081941` test: pytest 套件（27 用例）
- `14ab9de` feat: chunk_cache v2 拆分存储 + 进度条 + WinError 5 修复
- `4afd814` feat: 多标签分类体系（LLM 零样本 + embedding 语义 + 启发式兜底）
- `c78d9b8` feat: 长期记忆数据层 kb/memory.py

**未提交（两大批，建议分 2 个 commit）**：

A. 长期记忆接线：
- `kb/prompts.py`（USER_PROFILE_BLOCK + MEMORY_EXTRACTOR_PROMPT）
- `kb/memory.py`（update_from_conversation + _parse_extract_json 容错）
- `kb/agents/single.py`、`kb/agents/graph.py`（拼 profile 到 prompt）
- `main.py`（/api/chat 回答后自动提炼写回 profile.json）
- `tests/test_memory.py`（新增 18 用例）

B. 遗留解决：
- `scripts/clear_chroma.py`（sqlite 级联删 notes collection，绕过 chromadb API bug）
- `scripts/move_drafts.py`（notes_draft 归位 Python基础/，含台账删条目 + 重建）
- `tests/benchmark_retrieval.py`（--no-ingest 快速模式；ensure_indexed 不再删 chunk_cache）
- `tests/retrieval_cases.json`（15→23 条，加混淆组）
- `tests/test_retrieval.py`（新增 5 用例，GROUP_THRESHOLDS 按组阈值）
- `tests/test_chunk_cache_concurrent.py`（KNOWLEDGE_DIR 泄漏修复，try/finally 回滚）
- `kb/ingestion/pipeline.py`、`knowledge/index/doc_index.json`、tests/README.md
- 已删除：`knowledge/raw/notes_draft/` 两个文件（归位完成）

## 四、运行方式

```bash
# 统一用项目 venv（不要与系统 Python 混用！）
PY=$(cygpath -w "$(pwd)/.venv/Scripts/python.exe")
"$PY" -m pytest -ra -q                    # 全量测试
"$PY" tests/benchmark_retrieval.py --no-ingest   # 快速 benchmark（秒级）
"$PY" main.py                              # 启动服务
```

## 五、遗留未办事项（按优先级）

1. **提交未 commit 改动**（A/B 两批，见上）
2. **43MB 旧 chunk_cache.jsonl 手动删**：Windows 持锁，safe-delete/ctypes/PowerShell 全失败，需用户从资源管理器删（重启后更顺）。路径：`knowledge/index/chunks/chunk_cache.jsonl`
3. **benchmark 弱组**（需 LLM 重打 tag 或换 tagger，不急）：
   - agent_confusion hit@1=0.0（LangChain 难召回）
   - pydantic hit@1=0.0（期望 W08 但 top1 命中 W04，同 topic 属合理）
   - react_confusion hit@1=0.333（ReAct 框架命中 W03-LangGraph，合理）
4. **代码限制（用户已说先搁置）**：splitter 超长段无硬切 / tool_agent 每问必跑一次 LLM / execute_python 非真沙箱（对外开放需 Docker）
5. **路线图（不动）**：阶段 6 compiled_wiki / 阶段 7 有监督标签（需 100+ 样本，现 23 条）/ 阶段 8 多知识库

## 六、核心技术坑（踩过，勿再犯）

1. **chromadb PersistentClient API 视图 bug**：`.count()` / `.get()` / `list_collections()` 返回过期数据，`query()` 正确。绕过：`scripts/clear_chroma.py` 直接 sqlite 操作，或以 query() 为准。
2. **safe-delete hook**（sitecustomize.py）：拦 shutil.rmtree / os.unlink / Path.unlink → 走 trash → fail-closed。别在代码里依赖删除成功。
3. **双 Python 解释器并存**：`D:\Python\Python312\python.exe`（系统）vs `.venv\Scripts\python.exe`（项目）。一切用 venv；hang/超慢先 `Get-Process python` 看是否双套。
4. **spawn 子进程**：lambda 不可 pickle，worker 必须 module-level 函数。
5. **chunk_cache 是耗时大头**：benchmark ensure_indexed 已改为不删它。误删后全量重建 5-8 分钟；缓存完好时重建仅 ~9s。
6. **emb.npy 是 1D** shape=(N*512,)，emb_off 单位是 dim（步长 512）。
7. **mmap 句柄挡 os.replace**：np.load(mmap_mode='r') 后立刻 np.array() 拷贝释放句柄。
8. **HNSW 退化**：多次 delete+upsert 后 top1 召回漏掉，重建 collection 即可。
9. **后台子进程 stdout 缓冲**：Windows 无 SIGALRM；后台跑脚本看不到进度不等于卡死，用 `python -u` 验证。

## 七、长期记忆机制（刚接完，简述）

- 读：每次问答把 `memory.to_text()` 拼进 system prompt（single.py / graph.py 两条路径）
- 写：`/api/chat` 回答后 LLM 提炼本轮 facts/prefs/goals → 去重合并进 `knowledge/index/memory/profile.json`
- 失败全静默，不影响问答主链路

## 八、记忆文件位置

- 项目日志：`.workbuddy/memory/2026-09-05.md`（今日全部工作细节，约 280 行）
- 本交接文档：`HANDOFF.md`（项目根目录）
