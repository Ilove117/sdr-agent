"""运行时配置：模型提供方、阈值、路径。

优先从环境变量读取，其次从项目根目录 .env 读取。
可用 provider：deepseek / dashscope(通义千问) / openai。联网检索走 dashscope 原生接口。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 解析，避免引入额外依赖。"""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    note: str = ""


PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        "deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat",
        "中文推理强、成本低，作为主模型",
    ),
    "dashscope": ProviderConfig(
        "dashscope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
        "qwen-plus", "通义千问，兼容 OpenAI 协议；原生接口支持联网检索",
    ),
    "openai": ProviderConfig(
        "openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini",
        "备用（视网络可用性）",
    ),
}

DASHSCOPE_NATIVE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    kb_dir: Path = BASE_DIR / "app" / "kb"
    trace_dir: Path = BASE_DIR / "data" / "traces"
    audio_dir: Path = BASE_DIR / "data" / "audio"
    transcript_dir: Path = BASE_DIR / "data" / "transcripts"

    primary_provider: str = os.getenv("SDR_PRIMARY_PROVIDER", "deepseek")
    fallback_provider: str = os.getenv("SDR_FALLBACK_PROVIDER", "dashscope")
    model_overrides: dict[str, str] = field(default_factory=lambda: {
        "deepseek": os.getenv("SDR_MODEL_DEEPSEEK", PROVIDERS["deepseek"].default_model),
        "dashscope": os.getenv("SDR_MODEL_DASHSCOPE", PROVIDERS["dashscope"].default_model),
        "openai": os.getenv("SDR_MODEL_OPENAI", PROVIDERS["openai"].default_model),
    })

    request_timeout: float = float(os.getenv("SDR_REQUEST_TIMEOUT", "90"))
    max_retries: int = int(os.getenv("SDR_MAX_RETRIES", "2"))
    temperature: float = float(os.getenv("SDR_TEMPERATURE", "0.1"))

    enable_web_search: bool = os.getenv("SDR_ENABLE_WEB_SEARCH", "1") not in ("0", "false", "False")
    search_provider: str = os.getenv("SDR_SEARCH_PROVIDER", "dashscope")
    search_model: str = os.getenv("SDR_SEARCH_MODEL", "qwen-plus")
    search_providers: str = os.getenv("SDR_SEARCH_PROVIDERS", "dashscope,html")

    # ---- 语音转写（ASR）----
    asr_provider: str = os.getenv("SDR_ASR_PROVIDER", "dashscope")
    asr_model: str = os.getenv("SDR_ASR_MODEL", "paraformer-v2")
    asr_language: str = os.getenv("SDR_ASR_LANGUAGE", "zh")
    asr_diarization: bool = os.getenv("SDR_ASR_DIARIZATION", "1") not in ("0", "false", "False")
    asr_poll_timeout: float = float(os.getenv("SDR_ASR_POLL_TIMEOUT", "600"))

    # 低于该置信度进入人工复核队列
    review_confidence_threshold: float = float(os.getenv("SDR_REVIEW_THRESHOLD", "0.75"))
    # 关键字段低于该置信度也进入人工复核队列
    critical_field_threshold: float = float(os.getenv("SDR_CRITICAL_THRESHOLD", "0.70"))

    api_host: str = os.getenv("SDR_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("SDR_API_PORT", "8000"))

    def model_for(self, provider: str) -> str:
        return self.model_overrides.get(provider, PROVIDERS[provider].default_model)

    def api_key_for(self, provider: str) -> str:
        cfg = PROVIDERS.get(provider)
        return os.getenv(cfg.api_key_env, "") if cfg else ""


settings = Settings()
