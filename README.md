# kb-v2 个人知识库（RAG + Agent）

**分层架构的个人知识库**：把 .md/.docx/.pdf 笔记丢进 `knowledge/raw/`，系统自动增量入库
（分块 → 向量化 → 登记台账），支持单 Agent 问答、LangGraph 多 Agent 协作、Function
Calling 工具调用、FastAPI Web 界面。检索采用**向量（bge-small-zh）+ BM25 关键词双路召回**，
Web 端含问答 / 知识库管理 / 评估仪表盘三个互通页面。

> 旧项目（D:\Python\Agent\个人知识库）保持不动，本仓库是从零按分层架构重写的版本。

## 架构分层

```
main.py (FastAPI Web API)        ← 接口层（服务员）
cli.py (命令行)                  ← 接口层
        │  只认识
        ▼
kb/agents/  (Agent 层：决策与回答)
├─ single.py   单 Agent：检索 → 拼上下文 → LLM 回答
├─ graph.py    多 Agent：Planner 规划 → 检索 → Answerer 回答 → Reviewer 审查打回
└─ tools.py    工具注册表 + Function Calling 循环（查库 / 算数 / 联网）
        │  调用
        ▼
kb/ingestion/  kb/storage/   (数据摄入层 + 存储层)
├─ ingestion/scanner.py   扫 raw/，hash 对比台账 → 识别 增/改/删/未变
├─ ingestion/splitter.py  长文 → 分块（400/120）
├─ ingestion/tagger.py    标签自治（LLM 零样本 + embedding 复用，0.60/0.45 三分层）
├─ ingestion/pipeline.py  编排：扫描→分块→打标→向量化→写库→登记台账
├─ storage/vector_store.py Chroma 向量库（向量 + BM25 双路召回）
└─ storage/doc_index.py    inode 台账 doc_index.json（编号/hash/tags）
        │  都用
        ▼
kb/embedding.py  kb/llm.py  kb/prompts.py  kb/memory.py  kb/web_search.py  kb/config.py
        （基础设施：向量化 / 多提供商LLM / prompt 集中 / 长期记忆 / 联网 / 配置）
```

**依赖方向只从上往下**：接口层 → Agent 层 → 数据层 → 基础设施。底层不知道上层的存在。

## 目录结构

```
kb-v2/
├─ main.py                # FastAPI：/api/chat /api/ingest /api/upload /api/manage/* /api/eval/results
├─ kb/                    # 代码包（上面分层）
├─ static/                # 前端三页（index 问答 / manage 管理 / eval_dashboard 评估，互通+动画）
├─ tests/                 # pytest 60/60 + benchmark + agent_eval
├─ scripts/               # 运维脚本（clear_chroma 等）
├─ knowledge/             # 数据层
│   ├─ raw/               #   原始笔记（多领域目录，唯一真源）
│   ├─ index/             #   doc_index.json 台账（自动生成）
│   ├─ vector_store/      #   Chroma 向量库（自动生成，可随时删）
│   ├─ chunks/  pending/  sessions/  memory/   # 缓存 / 待审查 / 会话 / 长期记忆
│   ├─ compiled_wiki/  export/  ml_data/  temp/  # 预留
├─ kb_config.yaml         # 知识库独立配置
├─ .env                   # API key（DEEPSEEK_API_KEY 等）
└─ requirements.txt
```

## 用法

```powershell
# 安装
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 启动 Web（浏览器打开 http://127.0.0.1:8011；务必用 venv，否则缺 openai 退化为 mock）
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8011

# 命令行
.\.venv\Scripts\python.exe -m kb.cli scan        # 看哪些笔记变了
.\.venv\Scripts\python.exe -m kb.cli ingest      # 增量入库
.\.venv\Scripts\python.exe -m kb.cli query "装饰器怎么写"   # 检索片段
.\.venv\Scripts\python.exe -m kb.cli ask "装饰器是什么"     # 单 Agent 问答
.\.venv\Scripts\python.exe -m kb.cli list        # 库里有什么
.\.venv\Scripts\python.exe -m kb.cli reset       # 清空重建

# 测试与评测
.\.venv\Scripts\python.exe -m pytest -ra -q                             # 60/60 全绿
.\.venv\Scripts\python.exe tests\benchmark_retrieval.py --no-ingest     # 检索 benchmark（秒级）
.\.venv\Scripts\python.exe tests\agent_eval.py                          # Agent 评测
```

## 配置 .env

复制 `.env.example` 为 `.env` 并填 key（配了 DeepSeek 等任意一家即自动启用）：

```
DEEPSEEK_API_KEY=sk-xxxx
```

不配 key 也能跑（mock 模式假回答），向量化用本地 bge 模型，全程免费不联网。

> ⚠️ **mock 检查清单**：填了 key 仍是 mock 时，先确认服务是用**项目 venv** 启动的
> （`.\.venv\Scripts\python.exe`）。`kb/llm.py` 在解释器里 import 不到 openai 时会强制
> 降级 mock，与 .env 无关；系统 Python 默认没有 openai。

## 检索配置（kb_config.yaml → recall）

```yaml
recall:
  top_k: 4            # 向量 top_k
  score_threshold: 0.45
  bm25_boost: 0.10    # BM25 关键词命中加分权重（0.05~0.15 同效）
  bm25_top_k: 10      # BM25 保底候选数（独立读取，勿用 top_k 覆盖）
```

混合检索效果（`benchmark_retrieval.py --no-ingest`）：filtered hit@1 0.609→0.783、
global hit@1 0.348→0.435；decorator 组 0 回退。

---

# 交接状态（新会话先读）

> 更新：2026-09-06 ｜ pytest 60/60 ｜ 混合检索已上线 ｜ 前端三页互通

## 当前状态

- **pytest 60/60 全绿**（chunk_cache 27 + memory 18 + retrieval 5 + hybrid_retrieval 10）
- **混合检索已上线**：filtered hit@1 0.783、MRR 0.804；global hit@1 0.435、MRR 0.514；decorator 组 0 回退；依赖 `jieba` / `rank-bm25` 已装进 venv
- **管理页导入改造已完成**：导入入口收敛为侧边栏唯一「📤 导入文件」；上传/入库进度改**弹窗**（白底黑字）；问答页上传/导入/重导按钮已迁移删除（保留待审查面板）
- **三页互通 + 切换动画已上线**：eval_dashboard.html 统一深色布局；common.js 导航淡出过渡
- **待用户确认**：git 是否提交这批改动（混合检索 + 前端改造 + 文档合并）

## 遗留事项（按优先级）

1. **git 提交**：混合检索 + 前端改造 + 文档更新（用户确认后分批 commit）
2. **agent_confusion 组 5 例瓶颈**（agent_03/04、react_03、langchain_01/02）：整本书型 PDF chunk 向量分系统性偏低，BM25 翻不越恒定分差（agent_03 差 0.007，PDF 已排 #2）。文档级聚合加分已实验证明错误并回退。建议：reranker（交叉编码器）或"仅整本 PDF 类"文档先验（需防误伤）
3. **老代码限制**（用户此前搁置）：splitter 超长段无硬切 / tool_agent 每问必跑一次 LLM / execute_python 非真沙箱（对外开放需 Docker）
4. **路线图**：阶段 6 compiled_wiki / 阶段 7 有监督标签（样本 100+）/ 阶段 8 多知识库

## 技术坑速查（勿再犯）

1. **chromadb API 视图 bug**：count()/get()/list_collections() 返回过期数据，query() 正确；清理直接操作 sqlite
2. **safe-delete hook**：拦 rmtree/unlink → 走 trash → fail-closed，别依赖删除成功
3. **双 Python 解释器**：一切用 venv；mock 提示 = 运行解释器缺 openai，与 .env 无关
4. **chunk_cache 是耗时大头**：benchmark 已改为不删它；误删全量重建 5-8 分钟
5. **HNSW 退化**：多次 delete+upsert 后 top1 漏召回，重建 collection 即可
6. **前端 DOM 访问必须判空**：缓存旧页+新 JS 会 null.style（manage.js 已用 el() 判空）
7. **加分无法翻越恒定分差 / 聚合加分防长文档碾压**：见开发记录八点五阶段六
8. **并发编辑以磁盘为准**：多工具同改一批文件时，Edit 前先 Read 现状

## 检索机制要点

- `kb/storage/vector_store.py`：`query(question, topic="", top_k=0, tags=None)` 返回 `{text,source,topic,heading,tags,score}`
- 双路：向量（bge 余弦，主）+ BM25（`_search_by_bm25` 保底 `bm25_top_k` 进候选池；`_rerank_by_bm25` 按"命中查询词数/总数 × boost"并入 score）
- 配置 `recall {top_k:4, score_threshold:0.45, bm25_boost:0.10, bm25_top_k:10}`（**bm25_top_k 独立读取**）
- `_HAS_BM25=False` 时静默降级纯向量；35 标签词注入 jieba 用户词典

## 前端结构（本轮改造后）

- 三页互通：index（问答）/ manage（管理）/ eval_dashboard（评估），common.js 导航淡出过渡
- 导入链路：manage 侧边栏「📤 导入文件」→ 隐藏 `#importFileInput`（multiple）→ XHR POST `/api/upload`（auto=1）→ 轮询 `/api/ingest/progress` → 弹窗显示上传/入库双进度条 + 结果列表 → `loadAll()` 刷新
- 弹窗 `#importModal`：白底黑字；`importBusy` 期间禁止关闭；结果 ✅入库/🔁重复/📋待审查/❌失败
- 后端接口零改动（/api/upload、/api/ingest、/api/ingest/progress、/api/docs/export 均既有）

## 文档索引

| 文档 | 内容 |
|---|---|
| `README.md` | 本文件：项目总览 + 交接状态 + 技术坑速查 |
| `项目文档.md` | 项目全貌三合一：计划书（架构/阶段/风险）+ P1 功能设计 + 理解指南（代码地图/机制/面试总结） |
| `开发记录.md` | 开发记录两合一：详细踩坑（含检索/评估/前端专项）+ 新手学习清单 34 条 |
| `开发记录-新手版.html` | 新手清单交互版（勾选/进度条/删除，双击浏览器打开） |
| `面试八股深挖.md` | 求职面试问答（十层深挖 + 自测清单） |
| `项目评价报告.html` | 项目评价（72/100，B+，八维雷达） |
