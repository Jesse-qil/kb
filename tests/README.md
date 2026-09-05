# Retrieval Research

这里放知识库检索研究用的基准与结果。

## 口径

- `hit@1`：首条命中是否就是目标文档
- `hit@3`：前三条里是否包含目标文档
- `MRR`：目标文档首次出现的倒数排名均值

## 运行

```powershell
python .\tests\benchmark_retrieval.py
```

## 产物

- `tests/results/latest.json`
- `tests/results/latest.md`

## 增加样本

直接往 `tests/retrieval_cases.json` 里加一条 `{question, expected_source}` 即可。
