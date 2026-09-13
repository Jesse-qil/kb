# tests 测试目录说明

## 目录结构

```
tests/
├── README.md            # 本文档
├── conftest.py          # pytest 全局 fixture（禁 rerank/KG、隔离 chunk_cache）
├── unit/                # 单元测试（pytest 自动收集，11 个 test_*.py）
├── eval/                # 评估框架
│   ├── benchmark_retrieval.py   # 检索 benchmark（23 例刁钻题）
│   ├── agent_eval.py            # Agent 端到端评测（50 例）
│   ├── retrieval_cases.json     # 检索用例（含 expected_source）
│   └── agent_eval_cases.json    # Agent 评测用例
└── results/             # 运行产物（gitignore，不入库）
```

> 说明：单元测试与评估框架分开，避免 tests/ 根目录混杂三种职责；
> `results/` 是每次运行生成的产物，已被 gitignore（保留 `.gitkeep` 占位）。

## 口径

- `hit@1`：首条命中是否就是目标文档
- `hit@3`：前三条里是否包含目标文档
- `MRR`：目标文档首次出现的倒数排名均值
- `filtered`：用 `case.topic` 调 query（topic 过滤路径）
- `global`：不传 topic（全库召回，测跨主题检索鲁棒性）

## 运行

```powershell
# 全量重建 + 跑 benchmark（默认冷启动，约 5-8 分钟）
python .\tests\eval\benchmark_retrieval.py

# 快速模式：复用现有 chroma + chunk_cache，秒级（用于回归检索代码路径）
python .\tests\eval\benchmark_retrieval.py --no-ingest

# Agent 评测：跑多 Agent 版本
python .\tests\eval\agent_eval.py

# Agent 评测：只跑单 Agent 基线
python .\tests\eval\agent_eval.py --mode single
```

## 产物

- `tests/results/latest.json`
- `tests/results/latest.md`
- `tests/results/agent_eval_latest.json`
- `tests/results/agent_eval_latest.md`

## 用例分组

| group | 含义 | 数量 |
|---|---|---|
| decorator / async / agent | 单主题基本召回 | 各 5 |
| react_confusion | ReAct 不应命中 React.js | 3 |
| agent_confusion | LangChain 不应命中 reference | 2 |
| vector_db | 向量数据库 不应命中 Python基础 | 2 |
| pydantic | Python类型系统 | 1 |

## 增加样本

往 `tests/eval/retrieval_cases.json` 里加一条：

```json
{
  "id": "my_case_01",
  "question": "...",
  "expected_source": "Topic/file.md",  // 相对 raw/ 的路径
  "topic": "Topic",                     // 用于 filtered 路径
  "group": "my_group",
  "note": "可选说明（混淆目标等）"
}
```

## 已知坑

- **chromadb 视图滞后**：`.count()` / `.get()` 可能返回 0，但 `query()` 能拿到数据。  
  实际数据以 sqlite 直查为准（`kb/vector_store/chroma/chroma.sqlite3`）
- **安全删除 hook**：项目根目录外用 shutil.rmtree 会走 safe-delete；如失败用 `scripts/clear_chroma.py` 走 sqlite DELETE
