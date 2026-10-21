"""结构化线索模型（严格对齐 CRM SQL 字段要求）。

输出契约：
`LeadRecord.to_sql_json()` 只返回题目要求的 15 个字段；
诊断信息（证据、置信度拆解、链路追踪）走 `to_diagnostic()`，便于人工复核与评测。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.rules.sql_fields import normalize_intent_level, validate_lead

SQL_FIELDS = (
    "company_name", "industry", "area", "contact_name", "contact_title", "phone",
    "intent_level", "intent_reason", "demand_summary", "mendian", "shejishi",
    "call_time", "next_action", "confidence_score", "need_human_review",
)


class LeadRecord(BaseModel):
    company_name: str
    industry: str | None = None
    area: str | None = None
    contact_name: str | None = None
    contact_title: str | None = None
    phone: str | None = None
    intent_level: str = Field(description="high | medium | low")
    intent_reason: str = ""
    demand_summary: str | None = None
    mendian: int | None = None
    shejishi: int | None = None
    call_time: str | None = None
    next_action: str = ""
    confidence_score: float = 0.0
    need_human_review: bool = True

    @field_validator("intent_level")
    @classmethod
    def _validate_intent(cls, v: str) -> str:
        norm = normalize_intent_level(v)
        if norm is None:
            raise ValueError(f"intent_level 非法：{v!r}")
        return norm

    def to_sql_json(self) -> dict[str, Any]:
        """严格的 CRM 输出契约（字段与顺序固定）。"""
        return {k: getattr(self, k) for k in SQL_FIELDS}

    def to_diagnostic(self) -> dict[str, Any]:
        return self.model_dump()

    def to_pretty_json(self) -> str:
        return json.dumps(self.to_sql_json(), ensure_ascii=False, indent=2)


def new_call_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_next_action(level: str, lead: dict[str, Any]) -> str:
    """按意向等级与字段缺口确定性生成下一步动作（可复现，不靠模型自由发挥）。"""
    gaps: list[str] = []
    if not lead.get("phone"):
        gaps.append("补全联系方式")
    elif "*" in str(lead.get("phone")):
        gaps.append("补全完整手机号（当前为转写掩码）")
    if not lead.get("company_name") or lead.get("company_name") in ("未知", "未识别"):
        gaps.append("确认企业全称")
    if lead.get("shejishi") is None and lead.get("mendian") is None:
        gaps.append("确认设计师/门店规模")
    if not lead.get("area"):
        gaps.append("确认所在城市")

    if level == "high":
        base = "24 小时内电话回访，发送企业版方案与报价，约演示时间"
    elif level == "medium":
        base = "3 个工作日内回访推进，发送行业案例，挖掘预算与上线时间"
    else:
        base = "转入培育池，按周触达，发送内容型资料，暂不投入售前资源"

    if gaps:
        base += "；待补：" + "、".join(gaps)
    return base


def validate(record: LeadRecord) -> list[str]:
    return validate_lead(record.to_sql_json())
