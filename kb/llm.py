"""LLM 客户端：OpenAI 兼容接口统一封装，多提供商可切换（deepseek/openai/siliconflow/ollama/mock/auto）。
复用旧项目 app/llm.py 验证过的逻辑，配置改从 kb_config.yaml 读取。"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from .config import KB_ROOT, CONFIG

# 加载 .env（放 key，如 DEEPSEEK_API_KEY=sk-xxx；文件不存在也不报错）
load_dotenv(KB_ROOT / ".env")

# 各提供商的 OpenAI 兼容接口参数
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "env_key": "SILICONFLOW_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        "env_key": None,               # 本地模型不需要 key
    },
}


class LLMClient:
    # 类变量：所有实例共享的 token 消耗累加（评估用）
    _usage_accum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    @staticmethod
    def reset_usage():
        """重置 token 消耗累加器（评估每条用例前调用）。"""
        LLMClient._usage_accum = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    @staticmethod
    def get_usage() -> dict:
        """获取当前 token 消耗累加值。"""
        return dict(LLMClient._usage_accum)

    @classmethod
    def _add_usage(cls, resp):
        """从 API 响应提取 usage 并累加。"""
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        cls._usage_accum["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        cls._usage_accum["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        cls._usage_accum["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        cls._usage_accum["calls"] += 1

    def __init__(self):
        cfg = CONFIG["llm"]
        self.provider = cfg.get("provider", "auto")
        if self.provider == "auto":               # 自动识别
            self.provider = self._detect_provider()
        self.temperature = cfg.get("temperature", 0.3)
        self.mock_answer = cfg.get(
            "mock_answer",
            "【mock】还没配置真实大模型或缺少 openai 依赖：在 .env 填 API key 并安装依赖后重启即可得到真实回答。",
        )

        if self.provider == "mock" or OpenAI is None:
            self.client = None
            self.model = "mock"
            if self.provider != "mock":
                self.provider = "mock"
        else:
            info = PROVIDERS[self.provider]
            api_key = "ollama" if info["env_key"] is None else self._get_key(info["env_key"])
            self.client = OpenAI(api_key=api_key, base_url=info["base_url"])
            self.model = info["model"]

    def chat(self, messages: list[dict]) -> str:
        """普通对话：messages=[{role,content}...]，返回回答文本。"""
        if self.provider == "mock":
            return self.mock_answer
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature)
            self._add_usage(resp)
            return resp.choices[0].message.content
        except Exception:
            return self.mock_answer

    def chat_stream(self, messages: list[dict]):
        """流式对话：逐段 yield 增量文本（打字机效果用）。
        mock 或异常时一次性 yield 完整结果，保证调用方流程不中断。"""
        if self.provider == "mock":
            yield self.mock_answer
            return
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, stream=True)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    piece = chunk.choices[0].delta.content
                    if piece:
                        yield piece
        except Exception as e:
            yield self.mock_answer

    def chat_with_tools(self, messages: list[dict], tools: list[dict]):
        """带工具注册表的对话：返回完整 message 对象（可能含 .tool_calls）。
        mock 不支持工具，返回 None（调用方当作"没有工具调用"）。"""
        if self.provider == "mock":
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tools=tools)
            self._add_usage(resp)
            if not resp.choices:
                return None
            return resp.choices[0].message
        except Exception:
            return None

    @staticmethod
    def _get_key(env_name: str) -> str:
        key = os.getenv(env_name, "")
        if not key or key.startswith("sk-xxxx"):
            raise ValueError(f"缺少 API key：请在 kb-v2/.env 填 {env_name}，或把 provider 改成 mock")
        return key

    @staticmethod
    def _detect_provider() -> str:
        """自动识别：.env 配了哪家 key 就用哪家，都没有 → mock"""
        for name, info in PROVIDERS.items():
            if info["env_key"] and os.getenv(info["env_key"]):
                return name
        return "mock"


if __name__ == "__main__":
    print("\033[92m kb/llm.py 文件\033[0m")
    llm = LLMClient()
    print("当前 provider:", llm.provider)
    print(llm.chat([{"role": "user", "content": "你好"}]))
