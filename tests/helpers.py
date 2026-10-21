"""测试替身：让 Agent 链路可以在完全离线的条件下被确定性验证。"""

from __future__ import annotations

from typing import Any

from app.llm import LLMResponse, extract_json
from app.search import SearchHit, SearchOutcome


class FakeLLM:
    """按调用顺序/内容返回预设 JSON 的假模型。

    - extraction_payload: 抽取阶段的返回
    - company_payload:    抬头推理阶段的返回（可为 None，表示模型不可用）
    - fail:               为 True 时抛 LLMError，用于验证降级路径
    """

    def __init__(self, extraction_payload: dict[str, Any] | None = None,
                 company_payload: dict[str, Any] | None = None,
                 *, fail: bool = False, fail_stage: str | None = None) -> None:
        self.extraction_payload = extraction_payload or {}
        self.company_payload = company_payload if company_payload is not None else {}
        self.fail = fail
        self.fail_stage = fail_stage
        self.calls: list[str] = []

    @staticmethod
    def _is_extraction(messages: list[dict[str, str]]) -> bool:
        joined = " ".join(m.get("content", "") for m in messages)
        return "intent_signals" in joined or "intent_recognition" in joined

    def chat_json(self, messages: list[dict[str, str]], **_: Any):
        stage = "extraction" if self._is_extraction(messages) else "company"
        self.calls.append(stage)
        if self.fail or (self.fail_stage and self.fail_stage == stage):
            from app.llm import LLMError
            raise LLMError(f"fake failure at {stage}")
        payload = self.extraction_payload if stage == "extraction" else self.company_payload
        resp = LLMResponse(text="{}", provider="fake", model="fake-1", latency_ms=1)
        return payload, resp

    def chat(self, messages, **kwargs):  # pragma: no cover - 接口完整性
        payload, resp = self.chat_json(messages, **kwargs)
        resp.text = str(payload)
        return resp

    def search(self, query: str, *, model: str | None = None):
        return "", []


class FakeSearch:
    """假检索服务，可注入命中结果或失败。"""

    def __init__(self, hits: list[dict[str, str]] | None = None,
                 raw_text: str = "", *, fail: bool = False) -> None:
        self.hits = hits or []
        self.raw_text = raw_text
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 8) -> SearchOutcome:
        self.queries.append(query)
        if self.fail:
            return SearchOutcome(error="fake search unavailable")
        return SearchOutcome(
            hits=[SearchHit(title=h.get("title", ""), url=h.get("url", ""),
                            snippet=h.get("snippet", "")) for h in self.hits],
            raw_text=self.raw_text, provider="fake")


def extraction_payload(**overrides: Any) -> dict[str, Any]:
    """一份结构合法的抽取返回，便于各测试按需覆写。"""
    base: dict[str, Any] = {
        "intent_recognition": {"is_sales_call": True, "call_intent": "new_inquiry",
                               "customer_attitude": "positive"},
        "intent_signals": {
            "budget": {"mentioned": True, "specificity": "range",
                       "value_text": "一年几万块", "quote": "一年几万块我们能接受"},
            "timeline": {"mentioned": True, "urgency": "near_term",
                         "value_text": "下个月", "quote": "最好下个月就能试一下"},
            "scale": {"designers": 15, "stores": None, "quote": "设计部大概 15 个人"},
            "pain_points": [{"pain_text": "出图慢", "quote": "出图慢"}],
            "active_engagement": [{"type": "agree_next_step", "quote": "行，你加这个手机号就行"}],
            "negative_signals": [],
            "decision_maker": {"is_decision_maker": True, "quote": "请问是张总吗"},
        },
        "business_info": {
            "company_mention": "木里木外",
            "company_mention_quote": '叫“木里木外”',
            "industry_text": "全屋定制",
            "area_text": "杭州",
            "demand_summary": "出图慢、与工厂对接易出错，希望下月试用参数化出图并对接生产",
        },
        "contact_info": {"contact_mention": "张总", "phone": "138xxxx1234"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


SAMPLE_DIALOGUE = """销售：您好，请问是张总吗？我是酷家乐的小王。
客户：是的，你讲。
销售：上次您在我们官网留了信息，想了解企业版设计软件，今天想跟您确认一下需求。
客户：对，我们这边是杭州那边一个做全屋定制的，叫“木里木外”，不知道你们听过没有。
销售：了解的，木里木外是高定品牌。那您这边大概多少人用？主要想解决什么问题？
客户：设计部大概 15 个人，现在用的是别的软件，出图慢，而且跟工厂对接经常出错。
销售：明白，那我们企业版可以支持参数化设计，一键出图，还能对接生产。您大概什么时候想上线？
客户：最好下个月就能试一下，预算的话，一年几万块我们能接受。
销售：好的张总，我加您微信，把方案发您。
客户：行，你加这个手机号就行，138xxxx1234。"""
