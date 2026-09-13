"""构建知识图谱（命令行入口）。

用法：
  python scripts/build_graph.py             # 仅元数据层（文档-领域-标签，秒级，零 LLM）
  python scripts/build_graph.py --entities  # + LLM 实体三元组抽取（增量，只处理新增/变化文档）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb.kg.build import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="构建知识图谱")
    ap.add_argument("--entities", action="store_true",
                    help="额外做 LLM 实体三元组抽取（增量）")
    args = ap.parse_args()
    build(extract_entities=args.entities, verbose=True)


if __name__ == "__main__":
    main()
