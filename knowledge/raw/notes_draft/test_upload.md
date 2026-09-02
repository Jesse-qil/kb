# Python 异步编程入门

async/await 是 Python 写并发代码的方式。

## 关键概念
- 协程（coroutine）：用 async def 定义的函数，调用不立即执行，返回协程对象
- await：等待一个耗时操作完成，期间让出控制权给事件循环
- 事件循环（event loop）：调度所有协程的"总调度员"

## 例子
```python
import asyncio

async def hello():
    print("开始")
    await asyncio.sleep(1)
    print("结束")

asyncio.run(hello())
```

## 为什么用异步
网络请求、文件读写大多是 IO 等待，异步让等待期间去干别的，提高吞吐。
