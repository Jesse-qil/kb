# 标签-文档相似度分布实测

- 标签池：35 个（默认候选表）
- 文档：101 篇 md（取前 1000 字符）
- 模型：bge-small-zh-v1.5（正常），口径与 tagger.embedding_tags 一致

## 每篇最高相似度分布（与最优标签）
| 统计量 | 值 |
|---|---|
| min | 0.467 |
| p10 | 0.535 |
| p25 | 0.565 |
| p50 | 0.597 |
| p75 | 0.632 |
| p90 | 0.675 |
| max | 0.741 |

## 阈值命中率（免 LLM 复用比例）
| 阈值 | 命中篇数 | 命中率 |
|---|---|---|
| 0.85 | 0/101 | 0.0% |
| 0.70 | 4/101 | 4.0% |
| 0.60 | 47/101 | 46.5% |
| 0.55 | 81/101 | 80.2% |
| 0.50 | 98/101 | 97.0% |
| 0.45 | 101/101 | 100.0% |
| 0.40 | 101/101 | 100.0% |
| 0.35 | 101/101 | 100.0% |
| 0.30 | 101/101 | 100.0% |

## 全量明细

| 文档 | Top1 标签 | 相似度 |
|---|---|---|
| AI-Agent/07-AI基础总纲.md | Python基础 | 0.623 |
| AI-Agent/08-Agent总纲.md | Agent | 0.648 |
| AI-Agent/13-GitHub项目精选-Agent全景.md | Python基础 | 0.656 |
| AI-Agent/W01-Agent全景与5模式.md | Agent | 0.626 |
| AI-Agent/W01-LLM全景与API.md | LLM | 0.612 |
| AI-Agent/W02-Prompt工程进阶.md | Python基础 | 0.531 |
| AI-Agent/W02-手写Agent不用框架.md | Agent | 0.605 |
| AI-Agent/W03-Function-Calling与Tool-Use.md | 工具调用 | 0.631 |
| AI-Agent/W03-LangGraph入门.md | LangGraph | 0.614 |
| AI-Agent/W04-LangGraph进阶.md | LangGraph | 0.585 |
| AI-Agent/W05-MCP协议详解.md | Agent | 0.530 |
| AI-Agent/W05-评测与LLM-as-Judge.md | 工具调用 | 0.540 |
| AI-Agent/W06-Claude与OpenAI-Agent-SDK.md | Agent | 0.597 |
| AI-Agent/W07-多Agent协作.md | Agent | 0.698 |
| AI-Agent/W08-浏览器与Computer-Use.md | Agent | 0.618 |
| AI-Agent/W09-Agent评测.md | Agent | 0.672 |
| AI-Agent/W10-可观测性.md | Agent | 0.629 |
| AI-Agent/W11-LangGraph源码导读.md | Python | 0.598 |
| AI-Agent/W12-写mini-agent-sdk.md | Agent | 0.597 |
| K8s与监控/W06-K8s与监控.md | Web | 0.467 |
| project_material/06-真实毕业项目案例库.md | Python基础 | 0.585 |
| project_material/07-GitHub项目精选-灵感库.md | Python基础 | 0.600 |
| Python基础/test_upload.md | Python | 0.634 |
| Python基础/W02-数据结构.md | Python | 0.704 |
| Python基础/W03-函数与模块.md | Python | 0.679 |
| Python基础/W04-面向对象.md | Python | 0.677 |
| Python基础/W05-装饰器与上下文.md | Python | 0.676 |
| Python基础/W06-迭代器与生成器.md | Python | 0.645 |
| Python基础/W07-并发与异步.md | Python | 0.630 |
| Python基础/W08-类型系统与Pydantic.md | Python基础 | 0.688 |
| Python基础/W09-工程化与发布.md | Python基础 | 0.635 |
| Python基础/示例-装饰器.md | 装饰器 | 0.646 |
| reference/07-GitHub项目精选-数据库DevOps.md | Python基础 | 0.653 |
| 商业化/05-商业化与持续迭代.md | 项目 | 0.541 |
| 学习路线/00-如何高效学习编程.md | Python基础 | 0.638 |
| 学习路线/01-Python总纲.md | Python | 0.740 |
| 学习路线/01-选题与产品设计.md | 项目 | 0.542 |
| 学习路线/02-Git与GitHub入门.md | Python | 0.576 |
| 学习路线/02-JavaScript总纲.md | Python基础 | 0.632 |
| 学习路线/02-架构设计与技术选型.md | 数据结构 | 0.549 |
| 学习路线/03-HTML-CSS总纲.md | Web | 0.632 |
| 学习路线/03-MVP开发流程.md | 项目 | 0.510 |
| 学习路线/03-命令行终端基础.md | Python基础 | 0.594 |
| 学习路线/04-React总纲.md | 前端 | 0.584 |
| 学习路线/04-VSCode与AI辅助编程.md | Python | 0.593 |
| 学习路线/04-发布与冷启动.md | Web | 0.491 |
| 学习路线/05-GitHub项目精选-CSS与设计.md | Python基础 | 0.608 |
| 学习路线/05-GitHub项目精选-入门导航.md | Python基础 | 0.649 |
| 学习路线/05-后端总纲.md | FastAPI | 0.613 |
| 学习路线/06-数据库工程化总纲.md | Python基础 | 0.574 |
| 学习路线/06-费曼学习法与笔记系统.md | 笔记 | 0.580 |
| 学习路线/07-GitHub项目精选-LLM学习.md | Python基础 | 0.663 |
| 学习路线/07-反向工程别人的代码.md | Python基础 | 0.565 |
| 学习路线/09-GitHub项目精选-后端实战.md | Python基础 | 0.597 |
| 学习路线/09-毕业项目总纲.md | Python基础 | 0.568 |
| 学习路线/11-GitHub项目精选-JavaScript.md | Python基础 | 0.639 |
| 学习路线/11-GitHub项目精选-Python.md | Python | 0.701 |
| 学习路线/11-GitHub项目精选-React生态.md | Python基础 | 0.615 |
| 学习路线/W01-HTML5与可访问性.md | Web | 0.594 |
| 学习路线/W01-HTTP与REST.md | 前端 | 0.580 |
| 学习路线/W01-React入门与JSX.md | Web | 0.558 |
| 学习路线/W01-SQL与PostgreSQL.md | Python | 0.538 |
| 学习路线/W01-语法与类型.md | Python | 0.675 |
| 学习路线/W01-语法基础.md | Python | 0.741 |
| 学习路线/W02-CSS基础与Flex-Grid.md | Web | 0.535 |
| 学习路线/W02-FastAPI入门.md | FastAPI | 0.680 |
| 学习路线/W02-Hooks全集.md | Python | 0.515 |
| 学习路线/W02-函数与this.md | Python | 0.605 |
| 学习路线/W03-CSS进阶与动画.md | Web | 0.531 |
| 学习路线/W03-自定义Hook与Context.md | 函数 | 0.535 |
| 学习路线/W03-认证授权.md | Python | 0.526 |
| 学习路线/W03-闭包与作用域.md | Python | 0.567 |
| 学习路线/W04-RAG完整实现.md | RAG | 0.610 |
| 学习路线/W04-Tailwind与渲染原理.md | Python | 0.543 |
| 学习路线/W04-WebSocket与SSE.md | Web | 0.598 |
| 学习路线/W04-原型与继承.md | Python | 0.547 |
| 学习路线/W04-状态管理与数据获取.md | 前端 | 0.551 |
| 学习路线/W05-Node与Hono.md | 前端 | 0.573 |
| 学习路线/W05-异步与事件循环.md | 异步 | 0.619 |
| 学习路线/W05-路由与表单.md | Python基础 | 0.573 |
| 学习路线/W06-DOM与事件.md | Web | 0.584 |
| 学习路线/W06-Next.js全栈.md | 前端 | 0.584 |
| 学习路线/W06-从零实现GPT.md | Python基础 | 0.606 |
| 学习路线/W06-限流缓存日志.md | Python | 0.551 |
| 学习路线/W07-ES6进阶特性.md | Python基础 | 0.602 |
| 学习路线/W07-测试.md | Python基础 | 0.587 |
| 学习路线/W07-测试与质量.md | Python | 0.568 |
| 学习路线/W08-Docker部署.md | Python | 0.540 |
| 学习路线/W08-性能优化.md | Python | 0.515 |
| 学习路线/W08-模块系统与工程化.md | Python | 0.604 |
| 学习路线/W09-TypeScript完全指南.md | Python基础 | 0.569 |
| 学习路线/W09-手写mini-react-v1.md | Python基础 | 0.558 |
| 学习路线/W10-CPython源码导读.md | Python | 0.674 |
| 学习路线/W10-V8引擎与性能优化.md | Python | 0.558 |
| 学习路线/W10-手写mini-react-v2.md | 函数 | 0.569 |
| 学习路线/学习路线全景图.md | Python基础 | 0.617 |
| 数据库/W02-索引事务锁.md | Python | 0.567 |
| 数据库/W03-Redis全集.md | Python | 0.497 |
| 数据库/W04-ORM与迁移.md | Python | 0.567 |
| 数据库/W05-向量数据库与全文搜索.md | 向量检索 | 0.666 |
| 环境配置/01-环境配置完全指南.md | Python基础 | 0.565 |