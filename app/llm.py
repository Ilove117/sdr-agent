"""LLM 客户端封装。

- 统一 OpenAI 兼容协议，支持 deepseek / dashscope(通义千问) / openai
- 主模型失败自动降级到备用 provider（提升链路可用性）
- chat_json：强约束 JSON 输出 + 容错解析（去围栏 / 截取平衡括号 / 尾随修复）
- search：走通义千问原生接口的联网检索，用于「企业抬头还原」的外部取证
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from app.config import DASHSCOPE_NATIVE_URL, PROVIDERS, settings

logger = logging.getLogger(__name__)

_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """LLM 调用失败（重试与降级后仍不可用）。"""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    via_fallback: bool = False
    raw: dict[str, Any] | None = None

    def to_trace(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "via_fallback": self.via_fallback,
            "text_length": len(self.text),
            "preview": self.text[:200],
        }


# --------------------------------------------------------------------------- #
# JSON 容错解析
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出中稳健地取出 JSON 对象。"""
    if not text:
        raise LLMError("模型返回为空，无法解析 JSON")
    candidates: list[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    candidates.append(text.strip())
    # 截取第一个平衡的 {...}
    start = text.find("{")
    if start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
    for cand in candidates:
        if not cand:
            continue
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            # 常见尾随逗号修复
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict):
            return data
    raise LLMError(f"无法解析模型返回的 JSON：{text[:300]!r}")


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class LLMClient:
    def __init__(self, provider: str | None = None, *, allow_fallback: bool = True) -> None:
        self.primary = provider or settings.primary_provider
        self.allow_fallback = allow_fallback

    # ---------------- 基础 HTTP ----------------
    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any],
              *, timeout: float | None = None) -> dict[str, Any]:
        timeout = timeout or settings.request_timeout
        last_err: Exception | None = None
        for attempt in range(settings.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code in _RETRY_STATUS_CODES:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:400]}")
                return resp.json()
            except (requests.RequestException, requests.HTTPError, LLMError) as exc:
                last_err = exc
                if attempt < settings.max_retries:
                    time.sleep(1.2 * (attempt + 1))
        raise LLMError(f"请求失败：{last_err}")

    def _call_provider(self, provider: str, messages: list[dict[str, str]], *,
                       temperature: float, json_mode: bool, model: str | None) -> LLMResponse:
        cfg = PROVIDERS.get(provider)
        if cfg is None:
            raise LLMError(f"未知 provider：{provider}")
        api_key = settings.api_key_for(provider)
        if not api_key:
            raise LLMError(f"provider {provider} 缺少 API Key（环境变量 {cfg.api_key_env}）")

        use_model = model or settings.model_for(provider)
        payload: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.time()
        try:
            data = self._post(f"{cfg.base_url}/chat/completions",
                              {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                              payload)
        except LLMError:
            if not json_mode:
                raise
            # 个别模型不支持 response_format，去掉后重试一次
            payload.pop("response_format", None)
            data = self._post(f"{cfg.base_url}/chat/completions",
                              {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                              payload)
        latency = int((time.time() - started) * 1000)

        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{provider} 返回无 choices：{str(data)[:300]}")
        text = (choices[0].get("message") or {}).get("content") or ""
        return LLMResponse(text=text, provider=provider, model=use_model,
                           latency_ms=latency, raw=data)

    # ---------------- 对外接口 ----------------
    def chat(self, messages: list[dict[str, str]], *, temperature: float | None = None,
             json_mode: bool = False, provider: str | None = None,
             model: str | None = None) -> LLMResponse:
        temp = settings.temperature if temperature is None else temperature
        order = [provider or self.primary]
        if self.allow_fallback and settings.fallback_provider not in order:
            order.append(settings.fallback_provider)

        last_err: Exception | None = None
        for idx, prov in enumerate(order):
            try:
                resp = self._call_provider(prov, messages, temperature=temp,
                                           json_mode=json_mode, model=model)
                resp.via_fallback = idx > 0
                return resp
            except LLMError as exc:
                last_err = exc
                logger.warning("provider %s 调用失败：%s", prov, exc)
                continue
        raise LLMError(f"所有 provider 均失败：{last_err}")

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[dict[str, Any], LLMResponse]:
        resp = self.chat(messages, json_mode=True, **kwargs)
        try:
            return extract_json(resp.text), resp
        except LLMError:
            # 一次纠错重试：把原始输出回灌，要求只输出 JSON
            repair = messages + [
                {"role": "assistant", "content": resp.text[:1500]},
                {"role": "user", "content": "你的输出不是合法 JSON。请只输出一个合法 JSON 对象，不要任何解释、不要 markdown 围栏。"},
            ]
            resp2 = self.chat(repair, json_mode=True, **kwargs)
            return extract_json(resp2.text), resp2

    # ---------------- 联网检索 ----------------
    def search(self, query: str, *, model: str | None = None) -> tuple[str, list[dict[str, str]]]:
        """使用通义千问原生接口的 enable_search 做联网检索，返回 (回答正文, 引用来源)。"""
        api_key = settings.api_key_for(settings.search_provider)
        if not api_key:
            raise LLMError("联网检索需要 DASHSCOPE_API_KEY")
        payload = {
            "model": model or settings.search_model,
            "input": {"messages": [{"role": "user", "content": query}]},
            "parameters": {"result_format": "message", "enable_search": True},
        }
        data = self._post(DASHSCOPE_NATIVE_URL,
                          {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          payload, timeout=max(settings.request_timeout, 90))
        output = data.get("output") or {}
        choices = output.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        if not text:
            text = output.get("text") or ""
        sources: list[dict[str, str]] = []
        for item in (output.get("search_info") or {}).get("search_results") or []:
            sources.append({
                "title": str(item.get("title", "")),
                "url": str(item.get("url") or item.get("site_name") or ""),
                "snippet": str(item.get("content") or item.get("snippet") or "")[:200],
            })
        return text, sources
