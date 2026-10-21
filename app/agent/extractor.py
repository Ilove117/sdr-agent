"""信息抽取节点（工作流第 2 步）。

策略：LLM 负责「语义取证」，规则层负责「口径规整」。
- LLM 输出：意图识别结果 + 意向信号（附原文引用）+ 业务/联系人线索
- 规则层：手机号正则、数量抽取（中文数字归一）、行业词典、地区补全、敬称映射

这样既拿到了大模型对口语的理解，又保证同一输入在可规则化字段上完全可复现。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.agent import prompts
from app import dialogue as dl
from app.llm import LLMClient, LLMError
from app.rules import industry_lexicon as ilx
from app.rules import sql_fields as sf
from app.rules.intent_scorecard import SignalSet

logger = logging.getLogger(__name__)

# 数量抽取的单位模式与语境锚点
DESIGNER_UNIT = r"人|设计师|美工"
DESIGNER_CONTEXT = ("设计部", "设计师", "设计团队", "设计人员", "美工", "设计岗", "设计部门")
STORE_UNIT = r"门店|分店|专卖店|旗舰店|经销商|代理商|加盟店|店"
STORE_CONTEXT = ("门店", "分店", "专卖店", "旗舰店", "经销商", "代理商", "加盟", "店")

# 引用校验用的信号中文名
_QUOTE_LABELS = {
    "budget": "预算",
    "timeline": "时间计划",
    "scale": "规模",
    "pain_points": "痛点",
    "active_engagement": "主动互动",
    "negative_signals": "负向信号",
    "decision_maker": "决策权",
    "company_mention": "企业名称",
}
_WS_RE = re.compile(r"\s+")


def _norm_ws(value: Any) -> str:
    """归一化空白：任意连续空白压成一个空格并去首尾。"""
    return _WS_RE.sub(" ", str(value or "")).strip()


def _quote_problem(block: dict[str, Any], dialogue_norm: str) -> str:
    """返回空串表示通过，missing 表示无引用，mismatch 表示引用不在原文中。"""
    quote = block.get("quote")
    if not isinstance(quote, str) or not _norm_ws(quote):
        return "missing"
    return "" if _norm_ws(quote) in dialogue_norm else "mismatch"


def _warn_dropped(warnings: list[str], label: str, problem: str, quote: Any) -> None:
    if problem == "missing":
        warnings.append(f"LLM 取证缺少原文引用（{label}），已丢弃该信号")
        return
    text = _norm_ws(quote)
    snippet = text if len(text) <= 30 else text[:30] + "…"
    warnings.append(
        f"LLM 引用校验失败（{label}）：「{snippet}」未在通话记录中出现，已丢弃该信号")


def verify_quotes(raw: dict[str, Any], dialogue: str,
                  warnings: list[str]) -> dict[str, Any]:
    """确定性校验 LLM 返回的原文引用，丢弃无法回溯的信号。

    规则：quote 归一化空白后必须是通话记录的子串；断言了内容却没有 quote 的信号
    同样视为不可回溯。作用有两个：守住「结论可回溯」硬约束；拦截 prompt injection
    伪造的「原文」进入评分卡。
    """
    if not isinstance(raw, dict):
        return {}
    dialogue_norm = _norm_ws(dialogue)

    signals = raw.get("intent_signals")
    if isinstance(signals, dict):
        # 单值信号：预算 / 时间计划 / 决策权
        for key, asserted in (
            ("budget", lambda b: bool(b.get("mentioned"))),
            ("timeline", lambda b: bool(b.get("mentioned"))),
            ("decision_maker", lambda b: b.get("is_decision_maker") is not None),
        ):
            block = signals.get(key)
            if not isinstance(block, dict):
                continue
            problem = _quote_problem(block, dialogue_norm)
            if not problem or (problem == "missing" and not asserted(block)):
                continue
            _warn_dropped(warnings, _QUOTE_LABELS[key], problem, block.get("quote"))
            signals[key] = {}

        # 规模：只有给出了具体数值时才必须可回溯
        scale = signals.get("scale")
        if isinstance(scale, dict):
            problem = _quote_problem(scale, dialogue_norm)
            asserted = scale.get("designers") is not None or scale.get("stores") is not None
            if problem and (asserted or problem == "mismatch"):
                _warn_dropped(warnings, _QUOTE_LABELS["scale"], problem, scale.get("quote"))
                signals["scale"] = {}

        # 列表信号：逐条校验，只丢弃引用不成立的那一条
        for key in ("pain_points", "active_engagement", "negative_signals"):
            items = signals.get(key)
            if not isinstance(items, list):
                continue
            kept: list[dict[str, Any]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                problem = _quote_problem(item, dialogue_norm)
                if problem:
                    _warn_dropped(warnings, _QUOTE_LABELS[key], problem, item.get("quote"))
                else:
                    kept.append(item)
            signals[key] = kept

    # 企业口述名称同样必须有原文出处；不成立时清空，交给规则层从客户话轮重新找
    biz = raw.get("business_info")
    if isinstance(biz, dict) and str(biz.get("company_mention") or "").strip():
        problem = _quote_problem({"quote": biz.get("company_mention_quote")}, dialogue_norm)
        if problem:
            _warn_dropped(warnings, _QUOTE_LABELS["company_mention"], problem,
                          biz.get("company_mention_quote"))
            biz["company_mention"] = None
            biz["company_mention_quote"] = None

    return raw


@dataclass
class ExtractionResult:
    fields: dict[str, Any] = field(default_factory=dict)
    field_confidence: dict[str, float] = field(default_factory=dict)
    field_evidence: dict[str, list[str]] = field(default_factory=dict)
    signals: SignalSet = field(default_factory=SignalSet)
    intent_recognition: dict[str, Any] = field(default_factory=dict)
    raw_llm: dict[str, Any] = field(default_factory=dict)
    llm_trace: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def set(self, name: str, value: Any, confidence: float, evidence: str | None = None) -> None:
        self.fields[name] = value
        self.field_confidence[name] = round(float(confidence), 3)
        if evidence:
            self.field_evidence.setdefault(name, []).append(evidence)


class Extractor:
    """把一通通话记录转成「字段 + 意向信号 + 逐字段证据」。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, dialogue: str, *, call_time: str | None = None, trace=None) -> ExtractionResult:
        result = ExtractionResult()
        if not dialogue or not dialogue.strip():
            result.warnings.append("输入通话记录为空")
            result.set("company_name", "未识别", 0.05, "输入为空")
            result.set("intent_level", "low", 0.2, "输入为空，默认低意向")
            result.set("demand_summary", "", 0.0, "输入为空")
            return result

        # ---- 1) LLM 语义取证 -------------------------------------------------
        with (trace.span("extract.llm") if trace else _null_span()) as span:
            try:
                raw, resp = self.llm.chat_json(prompts.build_extraction_messages(dialogue))
                result.llm_trace = resp.to_trace()
                if span is not None:
                    span.set(**{"provider": resp.provider, "model": resp.model,
                                "latency_ms": resp.latency_ms, "via_fallback": resp.via_fallback})
            except LLMError as exc:
                logger.error("抽取调用失败：%s", exc)
                result.warnings.append(f"LLM 抽取失败：{exc}")
                raw = {}

        # 反幻觉闸门：引用无法回溯到原文的信号，不允许进入评分卡
        raw = verify_quotes(raw, dialogue, result.warnings)
        result.raw_llm = raw

        intent_recog = raw.get("intent_recognition") or {}
        result.intent_recognition = intent_recog if isinstance(intent_recog, dict) else {}
        signals_raw = raw.get("intent_signals") or {}
        result.signals = SignalSet.from_dict(signals_raw if isinstance(signals_raw, dict) else {})

        biz = raw.get("business_info") or {}
        contact = raw.get("contact_info") or {}
        biz = biz if isinstance(biz, dict) else {}
        contact = contact if isinstance(contact, dict) else {}

        # ---- 2) 规则层规整 ---------------------------------------------------
        # 行业/地区/规模属于「客户侧事实」，只在客户话轮内取证，避免采信销售话术
        cust = dl.customer_text(dialogue)
        self._fill_industry(result, dialogue, cust, biz)
        self._fill_area(result, cust, biz)
        self._fill_contact(result, dialogue, contact)
        self._fill_scale(result, cust, result.signals)
        self._fill_demand(result, biz)
        self._fill_call_time(result, call_time)

        if result.intent_recognition.get("is_sales_call") is False:
            result.warnings.append("该记录被判定为非销售通话（可能是误拨/售后/无效通话）")
        return result

    # ---------------- 各字段规整 ----------------
    def _fill_industry(self, r: ExtractionResult, dialogue: str, customer: str,
                       biz: dict[str, Any]) -> None:
        industry_text = str(biz.get("industry_text") or "").strip()
        industry, hits, conf = ilx.classify_industry(industry_text)
        evidence = f"行业词库命中（{industry_text}）：{'、'.join(hits)}" if industry else None
        if industry is None:
            industry, hits, conf = ilx.classify_industry(customer)
            evidence = f"行业词库命中客户话轮：{'、'.join(hits)}" if industry else None
        if industry is None:
            industry, hits, conf = ilx.classify_industry(dialogue)
            evidence = f"行业词库全文本命中：{'、'.join(hits)}" if industry else None
        if industry is None and industry_text:
            industry, conf = industry_text, 0.45
            evidence = f"词库未覆盖，沿用模型表述：{industry_text}"
        if industry:
            if ilx.is_core_target(industry):
                conf = min(0.95, conf + 0.05)
            r.set("industry", industry, conf, evidence)

    def _fill_area(self, r: ExtractionResult, customer: str, biz: dict[str, Any]) -> None:
        area, conf, evidence = sf.extract_area(customer)
        if area is None:
            area_text = str(biz.get("area_text") or "").strip()
            if area_text:
                area, conf, evidence = area_text, 0.5, f"正则未命中，沿用模型表述：{area_text}"
        if area:
            r.set("area", area, conf, evidence)

    def _fill_contact(self, r: ExtractionResult, dialogue: str, contact: dict[str, Any]) -> None:
        # 手机号：正则优先（确定性），模型仅兜底
        phone, pconf, pevidence = sf.extract_phone(dialogue)
        if phone is None:
            llm_phone = str(contact.get("phone") or "").strip()
            if llm_phone:
                phone, pconf, pevidence = sf.extract_phone(llm_phone)
                if phone:
                    pconf = min(pconf, 0.6)
                    pevidence = f"正则未命中，取自模型抽取：{llm_phone}"
        if phone:
            r.set("phone", phone, pconf, pevidence)
            if "*" in phone:
                r.warnings.append("手机号为掩码形式（转写脱敏），需人工核对完整号码")

        mention = str(contact.get("contact_mention") or "").strip()
        if mention:
            name, title, cconf = sf.split_contact(mention)
            if name:
                r.set("contact_name", name, cconf, f"从称呼「{mention}」还原姓氏")
                if len(name) == 1:
                    r.warnings.append(f"联系人仅有姓氏（{mention}），需人工补全全名")
            if title:
                r.set("contact_title", title, min(0.8, cconf + 0.1), f"敬称「{mention}」映射为「{title}」")

    def _fill_scale(self, r: ExtractionResult, customer: str, signals: SignalSet) -> None:
        designers, dconf, devidence = sf.extract_quantity(
            customer, DESIGNER_UNIT, context_keywords=DESIGNER_CONTEXT)
        stores, sconf, sevidence = sf.extract_quantity(
            customer, STORE_UNIT, context_keywords=STORE_CONTEXT)

        llm_designers = signals.scale.get("designers")
        llm_stores = signals.scale.get("stores")

        # 正则可复现 -> 优先；正则未命中才采信模型；两者冲突时保留正则并记录冲突
        if designers is not None:
            r.set("shejishi", designers, dconf, devidence)
            if isinstance(llm_designers, (int, float)) and int(llm_designers) != designers:
                r.warnings.append(
                    f"设计师人数取值冲突：正则={designers}，模型={int(llm_designers)}，已采信正则")
        elif isinstance(llm_designers, (int, float)) and llm_designers:
            r.set("shejishi", int(llm_designers), 0.7,
                  f"正则未命中，取自模型抽取：{signals.scale.get('quote') or ''}")

        if stores is not None:
            r.set("mendian", stores, sconf, sevidence)
            if isinstance(llm_stores, (int, float)) and int(llm_stores) != stores:
                r.warnings.append(
                    f"门店数取值冲突：正则={stores}，模型={int(llm_stores)}，已采信正则")
        elif isinstance(llm_stores, (int, float)) and llm_stores:
            r.set("mendian", int(llm_stores), 0.7,
                  f"正则未命中，取自模型抽取：{signals.scale.get('quote') or ''}")

    def _fill_demand(self, r: ExtractionResult, biz: dict[str, Any]) -> None:
        summary = str(biz.get("demand_summary") or "").strip()
        if summary:
            r.set("demand_summary", summary[:160], 0.8, "由 LLM 归纳，来源：对话中的痛点与期望")

    def _fill_call_time(self, r: ExtractionResult, call_time: str | None) -> None:
        if call_time:
            r.set("call_time", call_time, 0.95, "来自通话元数据")
        else:
            r.set("call_time", _now_str(), 0.55,
                  "未提供通话元数据，按处理时刻记录，需人工确认")


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _NullSpan:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _null_span() -> _NullSpan:
    return _NullSpan()
