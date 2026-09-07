# Retrieval Research

这里放知识库检索研究用的基准与结果。

## 口径

- `hit@1`：首条命中是否就是目标文档
- `hit@3`：前三条里是否包含目标文档
- `MRR`：目标文档首次出现的倒数排名均值
- `filtered`：用 `case.topic` 调 query（topic 过滤路径）
- `global`：不传 topic（全库召回，测跨主题检索鲁棒性）

## 运行

```powershell
# 全量重建 + 跑 benchmark（默认冷启动，约 5-8 分钟）
python .\tests\benchmark_retrieval.py

# 快速模式：复用现有 chroma + chunk_cache，秒级（用于回归检索代码路径）
python .\tests\benchmark_retrieval.py --no-ingest

# Agent 评测：跑多 Agent 版本
python .\tests\agent_eval.py

# Agent 评测：只跑单 Agent 基线
python .\tests\agent_eval.py --mode single
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

往 `tests/retrieval_cases.json` 里加一条：

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
