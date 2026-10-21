"""AI SDR Agent 工作流编排（工作流 5 个阶段）。

阶段划分与职责：
  1. 意图识别   —— 判断是否销售通话 / 通话类型 / 客户态度，作为后续处理的闸门
  2. 信息抽取   —— LLM 语义取证 + 规则层口径规整（见 extractor）
  3. 企业抬头还原 —— 多路召回 + 交叉验证 + 融合打分（见 company_resolver）
  4. 意向评分   —— 信号 -> 评分卡 -> 等级 + 判断依据（见 intent_scorecard）
  5. 结构化输出 —— 置信度融合、复核闸门、严格 CRM JSON 契约

设计取舍：阶段 1 复用阶段 2 的模型调用产出的意图识别结果（避免为同一段文本重复付费），
但把它作为独立路由闸门使用——非销售通话会短路后续的联网检索与 LLM 裁决，降低成本。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent.company_resolver import CompanyResolution, CompanyResolver
from app.agent.extractor import ExtractionResult, Extractor
from app.config import settings
from app.llm import LLMClient
from app.rules.intent_scorecard import score_intent
from app.rules.sql_fields import FIELD_SPECS, validate_lead
from app.schema import LeadRecord, build_next_action
from app.tracing import Tracer

logger = logging.getLogger(__name__)

# 这些 match_type 说明还原过程不可靠，必须人工确认
REVIEW_MATCH_TYPES = {
    "none", "low_confidence", "inferred", "exact_alias_conflict", "ambiguous",
}


@dataclass
class PipelineOutput:
    lead: LeadRecord
    meta: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    input_dialogue: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"lead": self.lead.to_sql_json(), "meta": self.meta, "trace_id": self.trace_id}


class SdrPipeline:
    def __init__(self, llm: LLMClient | None = None,
                 extractor: Extractor | None = None,
                 resolver: CompanyResolver | None = None,
                 asr_service=None) -> None:
        self.llm = llm or LLMClient()
        self.extractor = extractor or Extractor(self.llm)
        self.resolver = resolver or CompanyResolver(self.llm)
        self.asr_service = asr_service

    # ------------------------------------------------------------------ #
    def run_audio(self, audio_path: str, *, call_id: str | None = None,
                  call_time: str | None = None, metadata: dict[str, Any] | None = None,
                  tracer: Tracer | None = None) -> PipelineOutput:
        """音频入口：先用 ASR 转写成带说话人角色的对话，再走同一条文本链路。"""
        from app.asr import AsrError, AsrService

        tracer = tracer or Tracer(settings.trace_dir)
        service = self.asr_service or AsrService()
        with tracer.span("stage.asr", audio_file=str(audio_path)) as span:
            try:
                asr = service.transcribe(audio_path)
            except AsrError:
                if span is not None:
                    span.set(stage_failed=True)
                raise
            if span is not None:
                span.set(provider=asr.provider, model=asr.model,
                         duration_ms=asr.duration_ms,
                         speakers=asr.speaker_count,
                         chars=len(asr.dialogue),
                         warnings=asr.warnings)

        output = self.run(asr.dialogue, call_id=call_id, call_time=call_time,
                          metadata=metadata, tracer=tracer)
        output.meta["asr"] = asr.to_meta()
        if asr.warnings:
            output.meta["warnings"] = list(dict.fromkeys(
                output.meta.get("warnings", []) + asr.warnings))
        return output

    # ------------------------------------------------------------------ #
    def run(self, dialogue: str, *, call_id: str | None = None,
            call_time: str | None = None, metadata: dict[str, Any] | None = None,
            tracer: Tracer | None = None) -> PipelineOutput:
        tracer = tracer or Tracer(settings.trace_dir)
        tracer.log("pipeline.start", call_id=call_id, dialogue_chars=len(dialogue or ""),
                   dialogue_preview=(dialogue or "")[:120])

        # ---- 阶段 1 & 2：意图识别 + 信息抽取 ------------------------------ #
        extraction = self.extractor.run(dialogue, call_time=call_time, trace=tracer)
        intent_recog = extraction.intent_recognition
        is_sales_call = intent_recog.get("is_sales_call", True)
        tracer.log("stage.intent_recognition",
                   is_sales_call=is_sales_call,
                   call_intent=intent_recog.get("call_intent"),
                   customer_attitude=intent_recog.get("customer_attitude"))
        tracer.log("stage.information_extraction",
                   size=len(extraction.raw_llm),
                   warnings=extraction.warnings,
                   signals=self._signal_summary(extraction))

        # ---- 阶段 3：企业抬头还原 ---------------------------------------- #
        company = self._resolve_company(dialogue, extraction, tracer,
                                        skip=is_sales_call is False)

        # ---- 阶段 4：意向评分 -------------------------------------------- #
        intent = score_intent(
            extraction.signals,
            designers=extraction.fields.get("shejishi"),
            stores=extraction.fields.get("mendian"),
            contact_title=extraction.fields.get("contact_title"),
        )
        tracer.log("stage.intent_scoring", level=intent.level,
                   score=round(intent.total_score, 1), confidence=intent.confidence,
                   dimensions={d.key: round(d.score, 1) for d in intent.dimensions},
                   penalty=intent.penalty)

        # ---- 阶段 5：结构化输出 ------------------------------------------ #
        result = self._assemble(dialogue, extraction, company, intent,
                                is_sales_call, call_id, tracer)
        return result

    # ------------------------------------------------------------------ #
    def _resolve_company(self, dialogue: str, extraction: ExtractionResult,
                         tracer: Tracer, *, skip: bool) -> CompanyResolution:
        biz = extraction.raw_llm.get("business_info") or {}
        biz = biz if isinstance(biz, dict) else {}
        mention = str(biz.get("company_mention") or "").strip()
        quote = str(biz.get("company_mention_quote") or "").strip()
        industry_text = str(biz.get("industry_text") or "").strip()
        area = str(extraction.fields.get("area") or "").strip()
        industry_hint = extraction.fields.get("industry")

        if skip:
            tracer.log("stage.company_resolution", skipped=True,
                       reason="非销售通话，跳过联网检索与 LLM 裁决")
            return CompanyResolution(
                company_name=mention or "未识别", confidence=0.1 if mention else 0.05,
                original_mention=mention, match_type="none",
                conflicts=["非销售通话，未做抬头还原"],
                reasoning="意图识别判定为无效/非销售通话，跳过还原以节省成本。")

        with tracer.span("stage.company_resolution", mention=mention, area=area,
                         industry_hint=industry_hint) as span:
            resolution = self.resolver.resolve(
                dialogue=dialogue, mention=mention, quote=quote,
                industry_text=industry_text, area=area, industry_hint=industry_hint,
                trace=tracer)
            if span is not None:
                span.set(company_name=resolution.company_name,
                         confidence=resolution.confidence,
                         match_type=resolution.match_type,
                         paths_used=resolution.paths_used,
                         conflicts=resolution.conflicts,
                         candidates=len(resolution.candidates))
        return resolution

    # ------------------------------------------------------------------ #
    def _assemble(self, dialogue: str, extraction: ExtractionResult,
                  company: CompanyResolution, intent, is_sales_call: bool,
                  call_id: str | None, tracer: Tracer) -> PipelineOutput:

        fields = dict(extraction.fields)
        conf = dict(extraction.field_confidence)
        evidence = {k: list(v) for k, v in extraction.field_evidence.items()}

        # 企业抬头 + 证据
        company_conf = company.confidence
        if is_sales_call is False:
            company_conf = min(company_conf, 0.1)
        fields["company_name"] = company.company_name
        conf["company_name"] = company_conf
        evidence["company_name"] = list(dict.fromkeys(
            company.evidence + [f"口述原称：{company.original_mention}" if company.original_mention else ""
                                ]))
        evidence["company_name"] = [e for e in evidence["company_name"] if e]

        # 意向等级 + 判断依据 + 证据
        fields["intent_level"] = intent.level
        fields["intent_reason"] = intent.reason
        conf["intent_level"] = intent.confidence
        evidence["intent_level"] = [
            f"{d.label} {d.score:.0f}/{d.max_score:.0f}：" + "；".join(d.evidence)
            for d in intent.dimensions if d.evidence
        ] + ([f"负向扣分 -{intent.penalty:.0f}"] if intent.penalty else [])

        # 建议下一步（确定性规则）
        fields["next_action"] = build_next_action(intent.level, fields)
        conf["next_action"] = 0.85
        evidence["next_action"] = ["按意向等级 + 字段缺口规则生成"]

        # ---- 置信度融合 ---------------------------------------------------- #
        weights = {spec.name: spec.weight for spec in FIELD_SPECS}
        base = sum(w * conf.get(name, 0.0) for name, w in weights.items()) / max(
            sum(weights.values()), 1e-6)

        review_reasons: list[str] = []
        penalties = 0.0
        if company.conflicts:
            penalties += min(0.12, 0.04 * len(company.conflicts))
            review_reasons.extend(company.conflicts)
        if company.match_type in REVIEW_MATCH_TYPES:
            review_reasons.append(f"企业抬头还原方式不可靠（match_type={company.match_type}）")
        # 软提示只降置信度、只进 warnings，不单独触发复核（如「仅有姓氏」）
        penalties += min(0.06, 0.02 * len(extraction.warnings))
        if is_sales_call is False:
            penalties += 0.10
            review_reasons.append("意图识别判定为非销售通话")

        confidence = round(max(0.05, min(0.98, base - penalties)), 3)

        critical_low = [
            spec.label for spec in FIELD_SPECS
            if spec.critical and conf.get(spec.name, 0.0) < settings.critical_field_threshold
        ]
        if critical_low:
            review_reasons.append("关键字段置信度不足：" + "、".join(critical_low))
        if confidence < settings.review_confidence_threshold:
            review_reasons.append(
                f"整体置信度 {confidence:.2f} 低于复核阈值 {settings.review_confidence_threshold:.2f}")

        need_review = bool(
            confidence < settings.review_confidence_threshold
            or critical_low
            or company.conflicts
            or company.match_type in REVIEW_MATCH_TYPES
            or is_sales_call is False
        )
        conf["confidence_score"] = confidence
        conf["need_human_review"] = 1.0

        lead = LeadRecord(
            company_name=fields.get("company_name") or "未识别",
            industry=fields.get("industry"),
            area=fields.get("area"),
            contact_name=fields.get("contact_name"),
            contact_title=fields.get("contact_title"),
            phone=fields.get("phone"),
            intent_level=fields.get("intent_level") or "low",
            intent_reason=fields.get("intent_reason") or "",
            demand_summary=fields.get("demand_summary"),
            mendian=fields.get("mendian"),
            shejishi=fields.get("shejishi"),
            call_time=fields.get("call_time"),
            next_action=fields.get("next_action") or "",
            confidence_score=confidence,
            need_human_review=need_review,
        )

        issues = validate_lead(lead.to_sql_json())
        if issues:
            review_reasons.extend(issues)
            tracer.log("validate.issues", issues=issues)

        review_reasons = list(dict.fromkeys(review_reasons))
        tracer.log("pipeline.done", confidence_score=confidence,
                   intent_level=lead.intent_level, company_name=lead.company_name,
                   need_human_review=need_review, review_reason_count=len(review_reasons))

        meta: dict[str, Any] = {
            "call_id": call_id,
            "input_dialogue": dialogue,
            "intent_recognition": extraction.intent_recognition,
            "intent_scoring": intent.to_dict(),
            "company_resolution": company.to_dict(),
            "field_confidence": conf,
            "field_evidence": evidence,
            "warnings": extraction.warnings,
            "review_reasons": review_reasons,
            "validation_issues": issues,
            "paths_used": company.paths_used,
            "llm_trace": extraction.llm_trace,
            "trace_id": tracer.trace_id,
            "pipeline_stages": tracer.step_names(),
        }

        return PipelineOutput(lead=lead, meta=meta, trace_id=tracer.trace_id,
                              input_dialogue=dialogue)

    @staticmethod
    def _signal_summary(extraction: ExtractionResult) -> dict[str, Any]:
        s = extraction.signals
        return {
            "budget": s.budget.get("specificity"),
            "timeline": s.timeline.get("urgency"),
            "designers": s.scale.get("designers"),
            "stores": s.scale.get("stores"),
            "pain_count": len(s.pain_points),
            "active_count": len(s.active_engagement),
            "negative_count": len(s.negative_signals),
        }
