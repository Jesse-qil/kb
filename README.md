# kb-v2 个人知识库（重构版）

按豆包方案重构的知识库项目：目录分层 + inode 台账 + 增量入库 + 标签预留。
旧项目（D:\Python\Agent\个人知识库）保持不动。

## 目录结构

\`\`\`
kb-v2/
├─ kb/                  # 代码包
│   ├─ config.py        # 读 kb_config.yaml，统一路径/参数
│   ├─ file_scanner.py  # 扫 raw/，hash 对比台账识别 增/改/删
│   ├─ doc_index.py     # inode 台账 doc_index.json 读写
│   ├─ embedder.py      # bge 向量（离线）
│   ├─ kb_rag.py        # 入库 + 检索核心
│   └─ cli.py           # 命令行入口
├─ knowledge/           # 数据层
│   ├─ raw/             # 原始笔记（notes_draft/reference/project_material）
│   ├─ index/           # doc_index.json 台账
│   ├─ vector_store/    # Chroma 向量库
│   ├─ chunks/          # 切片缓存（预留）
│   ├─ temp/ export/ ml_data/ compiled_wiki/
└─ kb_config.yaml       # 知识库独立配置
\`\`\`

## 用法

\`\`\`
python -m kb.cli scan      # 扫描 raw，看哪些变了
python -m kb.cli ingest    # 入库
python -m kb.cli query "装饰器怎么写"
python -m kb.cli list
python -m kb.cli reset     # 清空重建
\`\`\`
