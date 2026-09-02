# kb-v2 个人知识库（RAG + Agent）

**分层架构的个人知识库**：把 .md/.docx 笔记丢进 `knowledge/raw/`，系统自动增量入库
（分块 → 向量化 → 登记台账），支持单 Agent 问答、LangGraph 多 Agent 协作、Function
Calling 工具调用、FastAPI Web 界面。

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
├─ ingestion/splitter.py  长文 → 分块
├─ ingestion/pipeline.py  编排：扫描→分块→向量化→写库→登记台账
├─ storage/vector_store.py  Chroma 向量库（存片段、按向量检索）
└─ storage/doc_index.py     inode 台账 doc_index.json（编号/hash/tags）
        │  都用
        ▼
kb/embedding.py  kb/llm.py  kb/prompts.py  kb/web_search.py  kb/config.py
        （基础设施：向量化 / 多提供商LLM / prompt 集中 / 联网 / 配置）
```

**依赖方向只从上往下**：接口层 → Agent 层 → 数据层 → 基础设施。底层不知道上层的存在。

## 目录结构

```
kb-v2/
├─ main.py                # FastAPI：/api/ingest /api/stats /api/chat + 前端
├─ kb/                    # 代码包（上面分层）
├─ knowledge/             # 数据层
│   ├─ raw/               #   原始笔记（notes_draft / reference / project_material）
│   ├─ index/             #   doc_index.json 台账（自动生成）
│   ├─ vector_store/      #   Chroma 向量库（自动生成）
│   ├─ chunks/  temp/ export/ ml_data/ compiled_wiki/   # 预留
├─ kb_config.yaml         # 知识库独立配置
└─ requirements.txt
```

## 用法

```powershell
# 安装
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 命令行
.\.venv\Scripts\python.exe -m kb.cli scan        # 看哪些笔记变了
.\.venv\Scripts\python.exe -m kb.cli ingest      # 增量入库
.\.venv\Scripts\python.exe -m kb.cli query "装饰器怎么写"   # 检索片段
.\.venv\Scripts\python.exe -m kb.cli ask "装饰器是什么"     # 单 Agent 问答
.\.venv\Scripts\python.exe -m kb.cli list        # 库里有什么
.\.venv\Scripts\python.exe -m kb.cli reset       # 清空重建

# Web（浏览器打开 http://127.0.0.1:8000）
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

## 配置 .env

复制 `.env.example` 为 `.env` 并填 key（配了 DeepSeek 等任意一家即自动启用）：

```
DEEPSEEK_API_KEY=sk-xxxx
```

不配 key 也能跑（mock 模式假回答），向量化用本地 bge 模型，全程免费不联网。
