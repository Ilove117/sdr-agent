"""联网检索通道（企业抬头还原的外部取证）。

为什么需要自建：企业抬头还原的准确率天花板取决于「能否拿到权威外部信息」，
所以检索通道必须可插拔、可降级，而不是绑死在某一个付费 API 上。

通道优先级（可配置 SDR_SEARCH_PROVIDERS）：
  1. dashscope —— 通义千问原生 enable_search，结果质量最好（需账户可用）
  2. html      —— 中文搜索引擎 HTML 检索（360/搜狗），best-effort，无额外依赖

任一通道失败都只记录错误并继续往下一跳，绝不阻塞主流程；
最终若全部失败，则抬头还原退化为「本地库 + LLM 推理」，并降低置信度进人工复核。
"""

from __future__ import annotations

import html as html_mod
import logging
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from app.config import settings
from app.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# 中文工商主体名模式（用于从检索片段里确定性地挖出候选全称）
LEGAL_NAME_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()·]{2,28}?"
    r"(?:股份有限公司|有限责任公司|有限公司|集团有限公司|集团)")

# 标签清洗：块级标签换空格，行内标签直接删除
_BLOCK_TAG_RE = re.compile(
    r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|section|article|h[1-6])[^>]*>", re.I)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


def _repair_cjk_spacing(text: str) -> str:
    """把「木里 木外家具启东有限公司」这类被标签切碎的中文重新粘回去。"""
    return re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", text)


@dataclass
class SearchHit:
    title: str
    url: str = ""
    snippet: str = ""


@dataclass
class SearchOutcome:
    hits: list[SearchHit] = field(default_factory=list)
    raw_text: str = ""
    provider: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.hits) and not self.error

    def context(self, limit: int = 5) -> str:
        parts = [f"- {h.title}\n  {h.snippet}" for h in self.hits[:limit]]
        if self.raw_text:
            parts.append("【网页正文摘要】" + self.raw_text[:1200])
        return "\n".join(parts)

    def legal_name_candidates(self) -> list[str]:
        """从检索结果中确定性抽取工商全称候选（不依赖大模型）。

        只保留标题/摘要中「包含口述核心词」的全称，避免把搜索结果里的
        无关公司（如搜索引擎推荐的其它企业）混进候选池。
        """
        names: list[str] = []
        for hit in self.hits:
            blob = hit.title + " " + hit.snippet
            for m in LEGAL_NAME_RE.finditer(blob):
                name = m.group(0).strip("（）()· ")
                if 4 <= len(name) <= 34 and name not in names:
                    names.append(name)
        return names[:10]


# --------------------------------------------------------------------------- #
# 通道实现
# --------------------------------------------------------------------------- #
class HtmlSearchProvider:
    """中文搜索引擎 HTML 检索（best-effort）。"""

    name = "html"
    ENGINES = (
        ("360", "https://www.so.com/s?q={q}"),
        ("sogou", "https://www.sogou.com/web?query={q}"),
    )

    def available(self) -> bool:
        return True

    @staticmethod
    def _clean(fragment: str, *, join_cjk: bool = True) -> str:
        """清洗 HTML 片段。

        join_cjk=True 时会把被标签切碎的中文重新粘连（企业名常被 <em> 高亮切碎），
        正文摘要则保留空格，避免整段文字粘成一坨不可读。
        """
        text = _BLOCK_TAG_RE.sub(" ", fragment or "")
        text = _ANY_TAG_RE.sub("", text)
        text = html_mod.unescape(text)
        text = re.sub(r"[ \t\u00a0]+", " ", text).strip()
        if join_cjk:
            text = _repair_cjk_spacing(text)
        return text

    @classmethod
    def _parse(cls, raw: str) -> list[SearchHit]:
        text = re.sub(r"<script.*?</script>|<style.*?</style>|<noscript.*?</noscript>",
                      " ", raw, flags=re.S | re.I)
        hits: list[SearchHit] = []
        for m in re.finditer(r"<h[23][^>]*>(.*?)</h[23]>", text, re.S | re.I):
            title = cls._clean(m.group(1))
            if len(title) < 4 or title in [h.title for h in hits]:
                continue
            seg = text[m.end(): m.end() + 3000]
            p = re.search(r"<p[^>]*>(.*?)</p>", seg, re.S | re.I)
            snippet = cls._clean(p.group(1)) if p else ""
            hits.append(SearchHit(title=title, snippet=snippet))
        return hits[:12]

    def search(self, query: str, *, limit: int = 8) -> SearchOutcome:
        last_error = None
        for engine, template in self.ENGINES:
            url = template.format(q=urllib.parse.quote(query))
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
                raw = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
            except Exception as exc:  # noqa: BLE001 - 网络层异常种类多，统一降级
                last_error = f"{engine}: {type(exc).__name__}: {exc}"
                continue
            hits = self._parse(raw)[:limit]
            plain = self._clean(raw, join_cjk=False)[:4000]
            if hits or plain:
                return SearchOutcome(hits=hits, raw_text=plain, provider=f"html/{engine}")
            last_error = f"{engine}: 页面无可解析内容"
        return SearchOutcome(error=last_error or "HTML 检索无可用引擎")


class DashScopeSearchProvider:
    """通义千问原生 enable_search。"""

    name = "dashscope"

    def __init__(self) -> None:
        self._llm: LLMClient | None = None

    def available(self) -> bool:
        return bool(settings.api_key_for("dashscope"))

    def search(self, query: str, *, limit: int = 8) -> SearchOutcome:
        self._llm = self._llm or LLMClient()
        try:
            text, sources = self._llm.search(query)
        except LLMError as exc:
            return SearchOutcome(error=f"dashscope: {exc}")
        hits = [SearchHit(title=s.get("title", ""), url=s.get("url", ""),
                          snippet=s.get("snippet", "")) for s in sources]
        return SearchOutcome(hits=hits, raw_text=text, provider="dashscope")


# --------------------------------------------------------------------------- #
# 门面
# --------------------------------------------------------------------------- #
class SearchService:
    def __init__(self, providers: list[str] | None = None) -> None:
        names = providers if providers is not None else [
            p.strip() for p in settings.search_providers.split(",") if p.strip()]
        registry = {"html": HtmlSearchProvider(), "dashscope": DashScopeSearchProvider()}
        self.providers = [registry[n] for n in names if n in registry]

    def search(self, query: str, *, limit: int = 8) -> SearchOutcome:
        errors: list[str] = []
        for provider in self.providers:
            if not provider.available():
                errors.append(f"{provider.name}: 不可用（缺少凭证）")
                continue
            outcome = provider.search(query, limit=limit)
            if outcome.ok or (outcome.hits and not outcome.error):
                if errors:
                    outcome.error = None  # 前序通道失败不影响本次成功
                return outcome
            errors.append(outcome.error or f"{provider.name}: 无结果")
        return SearchOutcome(error="；".join(errors) or "检索通道未配置")
