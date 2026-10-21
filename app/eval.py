"""离线评测：用 golden set 量化「企业抬头识别」与「意向度识别」准确率。

指标定义（对应题目的功能重要程度排序）：
  - 企业抬头准确率：预测工商全称与标准答案完全一致；对白名单可接受值也算对
  - 意向等级准确率：high/medium/low 三分类完全一致
  - 弃权正确性：对无法核实的企业，系统是否选择「保留口述原称 + 人工复核」而不是硬猜
  - 自动通过率：无需人工干预即可直接进 CRM 的比例（效率指标）

注意：评测会真实调用大模型与检索通道，因此指标会随模型版本/检索质量波动，
这不影响工程结论——评测的价值在于「每次改动都能量化对比」。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.pipeline import SdrPipeline
from app.rules import company_rules as cr
from app.tracing import Tracer


@dataclass
class CaseResult:
    case_id: str
    scenario: str
    expected_company: str | None
    predicted_company: str
    company_ok: bool
    company_soft_ok: bool
    expected_intent: str
    predicted_intent: str
    intent_ok: bool
    need_review: bool
    confidence: float
    note: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def _company_match(predicted: str, expected: str | None,
                   acceptable: list[str] | None) -> tuple[bool, bool]:
    """返回 (严格匹配, 宽松匹配)。宽松匹配允许核心词一致（如少了「集团」）。"""
    if expected is None:
        allowed = acceptable or ["未识别"]
        return predicted in allowed, predicted in allowed
    if predicted == expected:
        return True, True
    if acceptable and predicted in acceptable:
        return True, True
    return False, cr.similarity(predicted, expected) >= 0.9


def run_eval(cases_path: Path, *, run_llm: bool = True) -> dict[str, Any]:
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    pipeline = SdrPipeline()

    results: list[CaseResult] = []
    for case in cases:
        tracer = Tracer(Path("data") / "traces")
        out = pipeline.run(case["dialogue"], call_id=case["id"],
                           call_time=case.get("call_time"), tracer=tracer)
        comp = out.meta["company_resolution"]

        company_ok, soft_ok = _company_match(
            out.lead.company_name, case.get("expected_company"),
            case.get("acceptable_companies"))
        intent_ok = out.lead.intent_level == case.get("expected_intent")

        note = ""
        if case.get("expect_human_review") is not None:
            want = bool(case["expect_human_review"])
            if want and not out.lead.need_human_review:
                note = "弃权失败：应进人工复核却自动通过"
            elif not want and out.lead.need_human_review:
                note = "过度复核：可自动通过的线索被拦下"

        results.append(CaseResult(
            case_id=case["id"], scenario=case.get("scenario", ""),
            expected_company=case.get("expected_company"),
            predicted_company=out.lead.company_name,
            company_ok=company_ok, company_soft_ok=soft_ok,
            expected_intent=case.get("expected_intent", ""),
            predicted_intent=out.lead.intent_level, intent_ok=intent_ok,
            need_review=out.lead.need_human_review,
            confidence=out.lead.confidence_score, note=note,
            details={"match_type": comp["match_type"], "conflicts": comp["conflicts"]},
        ))
        tracer.flush(summary={"case_id": case["id"], "company_ok": company_ok,
                              "intent_ok": intent_ok})

    total = len(results) or 1
    summary = {
        "cases": len(results),
        "company_accuracy": round(sum(r.company_ok for r in results) / total, 4),
        "company_soft_accuracy": round(sum(r.company_soft_ok for r in results) / total, 4),
        "intent_accuracy": round(sum(r.intent_ok for r in results) / total, 4),
        "auto_pass_rate": round(sum(not r.need_review for r in results) / total, 4),
        "avg_confidence": round(sum(r.confidence for r in results) / total, 4),
    }
    return {"summary": summary, "results": [r.__dict__ for r in results]}


def print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("=" * 104)
    print("AI SDR Agent 离线评测报告")
    print("=" * 104)
    header = f"{'用例':<10}{'场景':<26}{'预测企业抬头':<32}{'核对':<6}{'意向(期望/预测)':<16}{'复核'}"
    print(header)
    print("-" * 104)
    for r in report["results"]:
        mark = "✓" if r["company_ok"] else ("≈" if r["company_soft_ok"] else "✗")
        intent_mark = "✓" if r["intent_ok"] else "✗"
        name = r["predicted_company"]
        name = name if len(name) <= 30 else name[:29] + "…"
        intent = f"{r['expected_intent']}/{r['predicted_intent']} {intent_mark}"
        print(f"{r['case_id']:<10}{r['scenario'][:24]:<26}{name:<32}{mark:<6}{intent:<16}"
              f"{'是' if r['need_review'] else '否'}")
        for key in ("note",):
            if r[key]:
                print(f"{'':<10}! {r[key]}")
    print("-" * 104)
    print(f"企业抬头准确率（严格）：{s['company_accuracy']:.1%}"
          f"    宽松（容忍后缀差异）：{s['company_soft_accuracy']:.1%}")
    print(f"意向等级准确率：{s['intent_accuracy']:.1%}"
          f"    自动通过率：{s['auto_pass_rate']:.1%}"
          f"    平均置信度：{s['avg_confidence']:.3f}")
    print("=" * 104)
