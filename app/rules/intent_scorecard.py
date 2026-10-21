"""客户意向度评分卡（可解释、可复现、可人工复核）。

设计依据提示：「意向度识别建议设计评分卡：预算、时间、人数、痛点、主动询问等维度加权」。

分工：
- LLM 只负责「取证」——把分散在对话中的意向信号找出来，并给出原文引用；
- 本模块负责「评分」——确定性加权，保证同一批证据永远得到同一等级。
这样既利用了大模型的语义理解，又避免大模型拍脑袋定级、便于人工复核与回归测试。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

LEVEL_THRESHOLDS = {"high": 65.0, "medium": 36.0}  # >=65 高；>=36 中；其余低

# 各维度满分
MAX_BUDGET = 24.0
MAX_TIMELINE = 22.0
MAX_SCALE = 16.0
MAX_PAIN = 14.0
MAX_ACTIVE = 16.0
MAX_AUTHORITY = 8.0
MAX_QUALIFICATION = 6.0
MAX_PENALTY = 30.0


@dataclass
class SignalSet:
    """LLM 取证结果。所有字段均需可回溯到原文引用。"""

    budget: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    scale: dict[str, Any] = field(default_factory=dict)
    pain_points: list[dict[str, Any]] = field(default_factory=list)
    active_engagement: list[dict[str, Any]] = field(default_factory=list)
    negative_signals: list[dict[str, Any]] = field(default_factory=list)
    decision_maker: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SignalSet":
        raw = raw or {}

        def _dict(key: str) -> dict[str, Any]:
            v = raw.get(key)
            return v if isinstance(v, dict) else {}

        def _list(key: str) -> list[dict[str, Any]]:
            v = raw.get(key)
            if not isinstance(v, list):
                return []
            return [x for x in v if isinstance(x, dict)]

        return cls(
            budget=_dict("budget"),
            timeline=_dict("timeline"),
            scale=_dict("scale"),
            pain_points=_list("pain_points"),
            active_engagement=_list("active_engagement"),
            negative_signals=_list("negative_signals"),
            decision_maker=_dict("decision_maker"),
        )


@dataclass
class DimensionScore:
    key: str
    label: str
    score: float
    max_score: float
    evidence: list[str]

    @property
    def ratio(self) -> float:
        return self.score / self.max_score if self.max_score else 0.0


@dataclass
class IntentResult:
    level: str
    total_score: float
    reason: str
    dimensions: list[DimensionScore]
    penalty: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_level": self.level,
            "intent_score": round(self.total_score, 1),
            "intent_reason": self.reason,
            "intent_dimensions": [asdict(d) | {"ratio": round(d.ratio, 2)} for d in self.dimensions],
            "intent_penalty": round(self.penalty, 1),
        }


def _text_of(d: dict[str, Any], *keys: str) -> str:
    parts = [str(d.get(k, "") or "") for k in keys]
    return " ".join(p for p in parts if p).strip()


def _score_budget(b: dict[str, Any]) -> DimensionScore:
    evidence: list[str] = []
    score = 0.0
    if b.get("mentioned"):
        spec = str(b.get("specificity", "vague")).lower()
        value_text = _text_of(b, "value_text")
        quote = _text_of(b, "quote")
        score = {"clear": 24.0, "range": 18.0, "vague": 10.0}.get(spec, 8.0)
        if value_text:
            evidence.append(f"预算表述：{value_text}")
        if quote:
            evidence.append(f"原文：「{quote}」")
    else:
        evidence.append("未提及预算")
    return DimensionScore("budget", "预算明确度", score, MAX_BUDGET, evidence)


def _score_timeline(t: dict[str, Any]) -> DimensionScore:
    evidence: list[str] = []
    score = 0.0
    if t.get("mentioned"):
        urgency = str(t.get("urgency", "vague")).lower()
        score = {"immediate": 22.0, "near_term": 19.0, "vague": 9.0}.get(urgency, 7.0)
        vt = _text_of(t, "value_text")
        quote = _text_of(t, "quote")
        if vt:
            evidence.append(f"时间计划：{vt}")
        if quote:
            evidence.append(f"原文：「{quote}」")
    else:
        evidence.append("未提及上线/试用时间")
    return DimensionScore("timeline", "时间计划", score, MAX_TIMELINE, evidence)


def _score_scale(s: dict[str, Any], designers: int | None, stores: int | None) -> DimensionScore:
    """规模：设计师人数 + 门店数，规模越大决策价值越高（但授权人数不是核心）。"""
    n = designers if designers is not None else s.get("designers")
    m = stores if stores is not None else s.get("stores")
    evidence: list[str] = []
    score = 0.0
    if isinstance(n, (int, float)) and n > 0:
        if n >= 30:
            score += 11.0
        elif n >= 10:
            score += 9.0
        elif n >= 3:
            score += 5.0
        else:
            score += 2.0
        evidence.append(f"设计师 {int(n)} 人")
    if isinstance(m, (int, float)) and m > 0:
        score += 7.0 if m >= 5 else 4.0
        evidence.append(f"门店 {int(m)} 家")
    if not evidence:
        evidence.append("未提及人员/门店规模")
    return DimensionScore("scale", "团队规模", min(score, MAX_SCALE), MAX_SCALE, evidence)


def _score_pain(points: list[dict[str, Any]]) -> DimensionScore:
    evidence: list[str] = []
    score = 0.0
    seen: set[str] = set()
    for p in points:
        desc = _text_of(p, "pain_text", "description")
        if desc and desc not in seen:
            seen.add(desc)
    if seen:
        score = min(MAX_PAIN, 7.0 + 3.5 * (len(seen) - 1))
        evidence.append("痛点：" + "；".join(list(seen)[:4]))
    else:
        evidence.append("未表达明确痛点")
    return DimensionScore("pain", "痛点明确度", score, MAX_PAIN, evidence)


def _score_active(items: list[dict[str, Any]]) -> DimensionScore:
    evidence: list[str] = []
    weights = {
        "ask_pricing": 5.0, "ask_demo": 5.0, "ask_solution": 4.0, "ask_case": 3.0,
        "provide_contact": 4.0, "agree_next_step": 5.0, "accept_meeting": 4.0,
        "ask_timeline": 3.0, "compare_vendor": 2.0,
    }
    score = 0.0
    kinds: set[str] = set()
    for it in items:
        kind = str(it.get("type", "")).strip()
        if kind in kinds:
            continue
        kinds.add(kind)
        score += weights.get(kind, 2.0)
        shot = _text_of(it, "quote", "description")
        if shot:
            evidence.append(("主动行为：" if kind else "") + shot)
    score = min(score, MAX_ACTIVE)
    if not evidence:
        evidence.append("无主动推进动作")
    return DimensionScore("active", "主动互动", score, MAX_ACTIVE, evidence)


def _has_hard_negative(negatives: list[dict[str, Any]]) -> set[str]:
    return {str(n.get("type", "")) for n in negatives} & {
        "reject", "not_interested", "just_looking", "already_locked"}


def _determine_level(signals: SignalSet, designers: int | None, stores: int | None,
                     title: str | None, total: float) -> str:
    """等级判定：以「资格要素组合」为主，分数做边界修正。

    业务含义：
      高 —— 有明确需求，且预算/时间/决策权三项基本齐备（BANT 成立），可立即推进
      中 —— 有明确需求，但预算/时间/决策权缺项，属于可培育商机
      低 —— 无明确需求，或明确拒绝 / 随便问问 / 已有替代
    这样避免「单纯按分数切线」导致两类误判：
      - 有痛点有规模的正常商机因未提预算被打成低意向
      - 各维度都不突出但要素齐备的商机被低估
    """
    need = bool(signals.pain_points)
    budget = bool(signals.budget.get("mentioned"))
    timeline = bool(signals.timeline.get("mentioned"))
    authority = (signals.decision_maker.get("is_decision_maker") is True
                 or any(k in (title or "") for k in ("负责人", "总经理", "老板", "总监", "院长")))
    scale = bool((isinstance(designers, (int, float)) and designers >= 3)
                 or (isinstance(stores, (int, float)) and stores >= 1))

    # 「高意向」要求至少一项是具体的承诺，只含糊提到「有预算」「以后再说」不足以判高
    budget_solid = str(signals.budget.get("specificity", "")).lower() in ("clear", "range")
    timeline_solid = str(signals.timeline.get("urgency", "")).lower() in ("immediate", "near_term")

    qualification = sum([budget, timeline, authority])
    hard_negative = _has_hard_negative(signals.negative_signals)

    if hard_negative:
        # 明确拒绝/随便问问：即使有需求线索也不允许判高
        return "medium" if (need and qualification >= 2 and total >= 55) else "low"

    if need and authority and (budget_solid or timeline_solid):
        return "high"
    if need and (qualification >= 1 or scale):
        return "medium"
    if need or scale or qualification >= 1:
        return "medium"
    return "low"


def _score_authority(dm: dict[str, Any], title: str | None) -> DimensionScore:
    evidence: list[str] = []
    score = 0.0
    is_dm = dm.get("is_decision_maker")
    t = title or ""
    if is_dm is True:
        score = 8.0
        evidence.append(f"决策人：{_text_of(dm, 'quote') or t or '客户对接人为决策者'}")
    elif is_dm is False:
        score = 2.0
        evidence.append("对接人非决策人")
    elif any(k in t for k in ("负责人", "总经理", "老板", "总监", "院长")):
        score = 6.0
        evidence.append(f"职位信号：{t}")
    elif t:
        score = 3.0
        evidence.append(f"职位：{t}")
    return DimensionScore("authority", "决策权限", score, MAX_AUTHORITY, evidence)


def _score_qualification(signals: "SignalSet", designers: int | None,
                         stores: int | None, title: str | None) -> DimensionScore:
    """BANT 资格完整度：预算 / 决策权 / 需求 / 时间 四项齐备度。

    单维度得分高不代表商机成立；四项都有线索才是「可推进的高质量线索」，
    因此单独给一个组合维度，避免各维度都不突出但整体很完整的商机被低估。
    """
    checks = {
        "预算": bool(signals.budget.get("mentioned")),
        "决策权": signals.decision_maker.get("is_decision_maker") is True
                  or any(k in (title or "") for k in ("负责人", "总经理", "老板", "总监", "院长")),
        "需求": bool(signals.pain_points),
        "时间": bool(signals.timeline.get("mentioned")),
    }
    present = [k for k, v in checks.items() if v]
    score = round(MAX_QUALIFICATION * len(present) / len(checks), 2)
    missing = [k for k, v in checks.items() if not v]
    evidence = ["已具备：" + "、".join(present) if present else "BANT 四项均无",
                ]
    if missing:
        evidence.append("缺失：" + "、".join(missing))
    return DimensionScore("qualification", "资格完整度", score, MAX_QUALIFICATION, evidence)


def _compact(text: str, limit: int = 42) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= limit else t[:limit] + "…"


def _penalty(negatives: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """负向信号扣分。

    校准原则：区分「拒绝」与「暂时不急」。「预算还没批」「等总部审批」属于后者，
    扣分要轻，否则会把有明确痛点、规模可观的正常商机误判成低意向。
    """
    weights = {
        "just_looking": 10.0, "no_budget": 4.0, "already_locked": 6.0, "reject": 15.0,
        "brush_off": 8.0, "wrong_person": 5.0, "delay": 3.0, "not_interested": 15.0,
        "price_objection": 3.0,
    }
    total = 0.0
    evidence: list[str] = []
    for n in negatives:
        kind = str(n.get("type", "")).strip()
        total += weights.get(kind, 3.0)
        quote = _text_of(n, "quote", "description")
        if quote:
            evidence.append(f"负向信号：{quote}")
    return min(total, MAX_PENALTY), evidence


def score_intent(
    signals: SignalSet,
    designers: int | None = None,
    stores: int | None = None,
    contact_title: str | None = None,
) -> IntentResult:
    dims = [
        _score_budget(signals.budget),
        _score_timeline(signals.timeline),
        _score_scale(signals.scale, designers, stores),
        _score_pain(signals.pain_points),
        _score_active(signals.active_engagement),
        _score_authority(signals.decision_maker, contact_title),
        _score_qualification(signals, designers, stores, contact_title),
    ]
    penalty, penalty_evidence = _penalty(signals.negative_signals)

    raw = sum(d.score for d in dims) - penalty
    total = max(0.0, min(100.0, raw))

    if total >= LEVEL_THRESHOLDS["high"]:
        level = "high"
    elif total >= LEVEL_THRESHOLDS["medium"]:
        level = "medium"
    else:
        level = "low"

    # 规则层最终裁决：分数只用于同档排序与边界修正，等级由资格要素组合决定
    level = _determine_level(signals, designers, stores, contact_title, total)

    # 置信度：证据覆盖度 × 等级与阈值边界的距离
    coverage = sum(1 for d in dims if d.score > 0) / len(dims)
    nearest = min(abs(total - LEVEL_THRESHOLDS["high"]), abs(total - LEVEL_THRESHOLDS["medium"]))
    boundary_bonus = min(0.25, nearest / 40.0)
    confidence = round(min(0.97, 0.55 + 0.3 * coverage + boundary_bonus), 3)

    # 判断依据：等级 + 总分 + 命中维度的得分与关键证据，让审核人直接看到「为什么是这个等级」
    parts = [f"判定为{ {'high': '高', 'medium': '中', 'low': '低'}[level] }意向"
             f"（总分 {total:.1f}/100，阈值：高≥{LEVEL_THRESHOLDS['high']:.0f}，"
             f"中≥{LEVEL_THRESHOLDS['medium']:.0f}；等级由资格要素组合裁决）"]
    for d in dims:
        if d.score <= 0:
            continue
        snippet = _compact(d.evidence[0]) if d.evidence else ""
        parts.append(f"{d.label} {d.score:.0f}/{d.max_score:.0f}"
                     + (f"（{snippet}）" if snippet else ""))
    if penalty > 0:
        parts.append(f"负向扣分 -{penalty:.0f}")
        parts.extend(penalty_evidence)
    reason = "；".join(parts)

    return IntentResult(
        level=level,
        total_score=total,
        reason=reason,
        dimensions=dims,
        penalty=penalty,
        confidence=confidence,
    )
