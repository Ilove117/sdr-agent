"""企业抬头还原器（难点 1 的核心实现）。

问题：客户口语里说的是简称 / 品牌名 / 别称，甚至转写有误，而 CRM 需要的是工商全称。

方案：多路召回 + 交叉验证 + 融合打分，四路证据依次增强：
  P1 本地企业别名词典（精确/别名命中，可解释、零成本）
  P2 本地企业库模糊匹配（核心词相似度，抗转写错误）
  P3 联网检索（通义千问 enable_search，补齐词典未覆盖的企业）
  P4 LLM 推理（结合上下文 + 上方全部证据做最终裁决）

交叉验证（这是准确率的关键）：
  - 行业一致性：候选企业行业若与通话中提到的行业信号冲突 -> 重罚
  - 地区一致性：候选企业所在地若与客户口述地区冲突 -> 降权并把冲突暴露给人工复核
  - 多路一致性：多路独立命中同一主体 -> 加分
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent import prompts
from app.config import settings
from app import dialogue as dl
from app.feedback_memory import FeedbackMemory
from app.llm import LLMClient, LLMError, extract_json
from app.rules import company_rules as cr
from app.rules import industry_lexicon as ilx
from app.search import SearchService

logger = logging.getLogger(__name__)

# 各路证据的先验权重。注意：HTML 检索噪音较大，只作为「召回」来源，
# 先验明显低于词典/企业库；只有被 LLM 交叉验证后才可能成为最终结论。
_PATH_WEIGHTS = {
    "human_feedback": 0.96,  # 人工订正沉淀的别名，最高优先级
    "dictionary": 0.90,
    "kb_fuzzy": 0.75,
    "web_search": 0.72,
    "llm": 0.80,
}
FUZZY_ACCEPT = 0.62
FUZZY_CANDIDATE = 0.45

# 缺乏任何可交叉验证的元数据（注册地/行业）时的降权系数
UNVERIFIED_DISCOUNT = 0.85
# 自动定论门槛：低于该分且没有词典/企业库强证据时弃权，交由人工裁决
DECISIVE_CONFIDENCE = 0.70


# --------------------------------------------------------------------------- #
# 数据模型
# --------------------------------------------------------------------------- #
@dataclass
class CompanyCandidate:
    legal_name: str
    brand: str | None = None
    province: str | None = None
    city: str | None = None
    industry: str | None = None
    path: str = "llm"
    path_confidence: float = 0.0
    text_similarity: float = 1.0
    fused_score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "legal_name": self.legal_name, "brand": self.brand,
            "province": self.province, "city": self.city, "industry": self.industry,
            "path": self.path,
            "path_confidence": round(self.path_confidence, 3),
            "text_similarity": round(self.text_similarity, 3),
            "fused_score": round(self.fused_score, 3),
            "evidence": self.evidence,
        }


@dataclass
class CompanyResolution:
    company_name: str
    confidence: float
    original_mention: str
    match_type: str
    candidates: list[CompanyCandidate] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    paths_used: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_name": self.company_name,
            "confidence": round(self.confidence, 3),
            "original_mention": self.original_mention,
            "match_type": self.match_type,
            "paths_used": self.paths_used,
            "conflicts": self.conflicts,
            "evidence": self.evidence,
            "reasoning": self.reasoning,
            "candidates": [c.to_dict() for c in self.candidates],
        }


# --------------------------------------------------------------------------- #
# 本地企业库
# --------------------------------------------------------------------------- #
class EnterpriseKB:
    """加载 app/kb/enterprise_kb.json，提供别名精确匹配与核心词模糊匹配。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (settings.kb_dir / "enterprise_kb.json")
        self.entries: list[dict[str, Any]] = []
        self._alias_index: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            logger.warning("企业库不存在：%s", self.path)
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = data.get("entries", [])
        for entry in self.entries:
            for name in [entry.get("legal_name", ""), entry.get("brand", ""),
                         *(entry.get("aliases") or [])]:
                if not name:
                    continue
                self._alias_index.setdefault(cr.strip_suffix(name).lower(), entry)

    def alias_lookup(self, core: str) -> dict[str, Any] | None:
        return self._alias_index.get(core.lower())

    def fuzzy_search(self, core: str, limit: int = 5) -> list[tuple[float, dict[str, Any]]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self.entries:
            names = [entry.get("legal_name", ""), entry.get("brand", ""),
                     *(entry.get("aliases") or [])]
            best = max((cr.similarity(core, n) for n in names if n), default=0.0)
            if best >= FUZZY_CANDIDATE:
                scored.append((best, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]


# --------------------------------------------------------------------------- #
# 还原器
# --------------------------------------------------------------------------- #
class CompanyResolver:
    def __init__(self, llm: LLMClient | None = None, kb: EnterpriseKB | None = None,
                 search: SearchService | None = None,
                 memory: FeedbackMemory | None = None) -> None:
        self.llm = llm or LLMClient()
        self.kb = kb or EnterpriseKB()
        self.search = search or SearchService()
        self.memory = memory if memory is not None else FeedbackMemory(
            settings.data_dir / "feedback.jsonl")

    # ---------------- 各路召回 ----------------
    def _match_human_feedback(self, core: str) -> CompanyCandidate | None:
        """人工订正沉淀的别名记忆：审核人核过的结论优先于任何机器推断。"""
        learned = self.memory.lookup(core)
        if learned is None:
            return None
        return CompanyCandidate(
            legal_name=learned.legal_name, path="human_feedback",
            path_confidence=_PATH_WEIGHTS["human_feedback"], text_similarity=1.0,
            evidence=[f"人工复核沉淀别名：{core} -> {learned.legal_name}"
                      f"（审核人 {learned.reviewer}，累计 {learned.count} 次）"],
        )

    def _match_dictionary(self, core: str) -> CompanyCandidate | None:
        entry = self.kb.alias_lookup(core)
        if not entry:
            return None
        return CompanyCandidate(
            legal_name=entry["legal_name"], brand=entry.get("brand"),
            province=entry.get("province"), city=entry.get("city"),
            industry=entry.get("industry"), path="dictionary",
            path_confidence=_PATH_WEIGHTS["dictionary"], text_similarity=1.0,
            evidence=[f"企业别名词典精确命中：{core} -> {entry['legal_name']}"],
        )

    def _match_kb_fuzzy(self, core: str) -> list[CompanyCandidate]:
        out: list[CompanyCandidate] = []
        for sim, entry in self.kb.fuzzy_search(core):
            if sim < FUZZY_CANDIDATE:
                continue
            conf = round(_PATH_WEIGHTS["kb_fuzzy"] * sim, 3)
            out.append(CompanyCandidate(
                legal_name=entry["legal_name"], brand=entry.get("brand"),
                province=entry.get("province"), city=entry.get("city"),
                industry=entry.get("industry"), path="kb_fuzzy",
                path_confidence=conf, text_similarity=round(sim, 3),
                evidence=[f"本地企业库模糊匹配：{core} ≈ {entry['legal_name']}（相似度 {sim:.2f}）"],
            ))
        return out

    def _web_search(self, mention: str, area: str, industry_text: str,
                    industry_hint: str | None = None) -> tuple[str, list[dict[str, str]], list[CompanyCandidate]]:
        """返回 (检索上下文, 来源列表, 从检索结果确定性挖出的候选企业)。"""
        if not settings.enable_web_search or not mention:
            return "", [], []
        query = prompts.COMPANY_SEARCH_QUERY.format(
            mention=mention, area=area or "", industry=industry_text or "")
        outcome = self.search.search(query)
        if outcome.error and not outcome.hits:
            raise LLMError(outcome.error)

        candidates: list[CompanyCandidate] = []
        for name in outcome.legal_name_candidates():
            sim = cr.similarity(cr.strip_suffix(mention), name)
            if sim < 0.5:
                continue
            candidates.append(CompanyCandidate(
                legal_name=name, path="web_search",
                path_confidence=round(_PATH_WEIGHTS["web_search"] * (0.72 + 0.28 * sim), 3),
                text_similarity=round(sim, 3),
                evidence=[f"网页检索命中企业全称：{name}（名称相似度 {sim:.2f}）"],
            ))
        sources = [{"title": h.title, "url": h.url, "snippet": h.snippet}
                   for h in outcome.hits]
        return outcome.context(), sources, candidates

    def _llm_resolve(self, *, mention: str, quote: str, industry_text: str, area: str,
                     others: list[str], search_context: str) -> dict[str, Any]:
        messages = prompts.build_company_messages(
            mention=mention, quote=quote, industry_text=industry_text, area=area,
            other_mentions=others, search_context=search_context)
        data, _ = self.llm.chat_json(messages)
        return data

    # ---------------- 融合 ----------------
    @staticmethod
    def _region_conflict(city: str | None, province: str | None, area: str | None) -> str | None:
        if not area or not (city or province):
            return None
        area_city = area
        for known_city in (city, province):
            if not known_city:
                continue
            if known_city in area:
                return None
        # 城市级不一致判定（避免「上海市」vs「浙江省杭州市」这类误报）
        if city and city not in area and area not in city:
            return f"客户口述地区「{area}」与候选企业注册地「{city}」不一致"
        return None

    def resolve(self, *, dialogue: str, mention: str, quote: str, industry_text: str,
                area: str, industry_hint: str | None, trace=None) -> CompanyResolution:
        # 只在客户话轮内取证，避免把销售方自家品牌（「我是酷家乐的小王」）当成客户企业
        cust = dl.customer_text(dialogue)
        mentions = [m for m in cr.extract_mentions(cust) if not dl.is_vendor_name(m.raw)]
        if mention and dl.is_vendor_name(mention):
            mention, quote = "", ""
        if not mention and mentions:
            mention, quote = mentions[0].raw, cust[mentions[0].span[0]:mentions[0].span[1]]
        core = cr.strip_suffix(cr.clean_mention(mention)) if mention else ""

        if not core:
            if trace is not None:
                trace.log("company.resolve", skipped=True, reason="客户未提及企业名")
            return CompanyResolution(
                company_name="未识别", confidence=0.03, original_mention="", match_type="none",
                candidates=[], conflicts=[],
                paths_used=[], evidence=["客户话轮中未出现可识别的企业名称"],
                reasoning="客户全程未提供企业名，不具备还原条件，需坐席后续补充。")

        candidates: list[CompanyCandidate] = []
        paths_used: list[str] = []
        evidence: list[str] = []

        # P0 人工反馈记忆；P1 + P2 本地词典与本地企业库
        if core:
            learned = self._match_human_feedback(core)
            if learned:
                candidates.append(learned)
                paths_used.append("human_feedback")
                evidence.append(learned.evidence[0])
            hit = self._match_dictionary(core)
            if hit:
                candidates.append(hit)
                paths_used.append("dictionary")
                evidence.append(hit.evidence[0])
            fuzzy = self._match_kb_fuzzy(core)
            if fuzzy:
                candidates.extend(fuzzy)
                paths_used.append("kb_fuzzy")
                evidence.extend(fuzzy[0].evidence)

        best_local = max(candidates, key=lambda c: c.path_confidence) if candidates else None
        need_external = best_local is None or best_local.path_confidence < 0.85

        # P3 联网检索（词典未高置信命中时才发起，控制成本与延迟）
        search_text, search_sources, search_candidates = "", [], []
        if need_external:
            try:
                search_text, search_sources, search_candidates = self._web_search(
                    mention, area, industry_text, industry_hint)
            except LLMError as exc:
                logger.warning("联网检索失败：%s", exc)
                evidence.append(f"联网检索通道异常：{exc}")
            if trace is not None:
                trace.log("company.web_search", used=bool(search_text),
                          sources=len(search_sources), mention=mention,
                          candidates=len(search_candidates))
            if search_text:
                evidence.append("已发起联网检索取证")
            if search_candidates:
                candidates.extend(search_candidates)
                paths_used.append("web_search")

        # P4 LLM 裁决
        llm_output: dict[str, Any] = {}
        llm_error: str | None = None
        if need_external or not candidates:
            search_context = search_text
            if search_sources:
                search_context += "\n" + "\n".join(
                    f"- {s.get('title')} {s.get('url')}\n  {s.get('snippet')}"
                    for s in search_sources[:5])
            try:
                llm_output = self._llm_resolve(
                    mention=mention, quote=quote, industry_text=industry_text, area=area,
                    others=[m.raw for m in mentions if m.raw != mention],
                    search_context=search_context)
                paths_used.append("llm")
            except LLMError as exc:
                llm_error = str(exc)
                logger.warning("LLM 抬头推理失败：%s", exc)

        for cand in (llm_output.get("candidates") or []):
            name = str(cand.get("legal_name") or "").strip()
            if not name:
                continue
            candidates.append(CompanyCandidate(
                legal_name=name,
                path="llm",
                path_confidence=min(float(cand.get("confidence", 0.6) or 0.6),
                                    _PATH_WEIGHTS["llm"]),
                text_similarity=cr.similarity(core, name) if core else 0.5,
                evidence=[f"LLM 推理候选：{cand.get('reason', '')}"],
            ))
        best_llm = (llm_output.get("best") or {}) if isinstance(llm_output, dict) else {}
        if best_llm.get("legal_name"):
            candidates.append(CompanyCandidate(
                legal_name=str(best_llm["legal_name"]).strip(),
                brand=best_llm.get("brand"), province=best_llm.get("province"),
                city=best_llm.get("city"), industry=best_llm.get("industry"),
                path="llm",
                path_confidence=min(float(best_llm.get("confidence", 0.7) or 0.7),
                                    _PATH_WEIGHTS["llm"]),
                text_similarity=cr.similarity(core, str(best_llm["legal_name"])) if core else 0.5,
                evidence=[f"LLM 裁决主体：match_type={best_llm.get('match_type')}"],
            ))

        # 去重合并（同名取最高分，汇总证据）
        merged: dict[str, CompanyCandidate] = {}
        for cand in candidates:
            key = cr.strip_suffix(cand.legal_name).lower()
            if key in merged:
                cur = merged[key]
                cur.path_confidence = max(cur.path_confidence, cand.path_confidence)
                cur.evidence = list(dict.fromkeys(cur.evidence + cand.evidence))
                cur.province = cur.province or cand.province
                cur.city = cur.city or cand.city
                cur.industry = cur.industry or cand.industry
                cur.path = cur.path if cur.path == cand.path else f"{cur.path}+{cand.path}"
            else:
                merged[key] = cand
        candidates = sorted(merged.values(), key=lambda c: c.path_confidence, reverse=True)

        conflicts: list[str] = []
        for item in (llm_output.get("conflicts") or []):
            if isinstance(item, str) and item.strip():
                conflicts.append(item.strip())

        # 交叉验证：行业 / 地区一致性 + 证据完整度 + 多路一致度，得到融合分
        # 说明：人工复核沉淀的结论已经过人工确认，视为权威，不参与机器交叉验证的降权。
        scored: list[CompanyCandidate] = []
        for cand in candidates:
            if cand.path == "human_feedback":
                scored.append(CompanyCandidate(**{**vars(cand),
                                                  "fused_score": round(cand.path_confidence, 4)}))
                continue
            score = cand.path_confidence
            # 文本相似度只在「非精确命中」时作为惩罚项（避免别名命中被误伤）
            if cand.text_similarity < 0.62 and cand.path != "dictionary":
                score *= 0.75 + 0.5 * cand.text_similarity
            # 证据完整度：既无注册地也无行业信息的候选无法交叉验证，降权
            if not (cand.city or cand.province) and not cand.industry:
                score *= UNVERIFIED_DISCOUNT
            # 候选企业的行业：优先用词典/企业库里的人工标注，其次从全称推断（strict 模式忽略泛化词）
            cand_industry = cand.industry or ilx.classify_industry(cand.legal_name, strict=True)[0]
            if industry_hint and cand_industry and cand_industry != industry_hint:
                if ilx.classify_industry(cand.legal_name, strict=True)[0] != industry_hint:
                    score *= 0.78
                    conflicts.append(
                        f"行业不一致：口述信号「{industry_hint}」vs 候选企业「{cand_industry}」")
                else:
                    score *= 0.95
            rc = self._region_conflict(cand.city, cand.province, area)
            if rc:
                score *= 0.68
                # 客户口述地区弱于工商注册地，但必须暴露冲突供人工确认
                conflicts.append(rc)
            # 多路一致加成
            if "+" in cand.path:
                score = min(1.0, score * 1.06)
            scored.append(CompanyCandidate(**{**vars(cand), "fused_score": round(score, 4)}))

        scored.sort(key=lambda c: c.fused_score, reverse=True)
        scored_candidates = scored

        conflicts = list(dict.fromkeys(conflicts))
        if llm_error:
            evidence.append(f"LLM 推理通道异常：{llm_error}")

        if not scored_candidates:
            return CompanyResolution(
                company_name=mention or "未识别",
                confidence=0.15 if mention else 0.05,
                original_mention=mention or "",
                match_type="none",
                candidates=[], conflicts=conflicts,
                paths_used=list(dict.fromkeys(paths_used)), evidence=evidence,
                reasoning="未命中词典/企业库，且联网与 LLM 均未给出可用主体。",
            )

        top = scored_candidates[0]
        top_score = top.fused_score
        has_strong_local = "dictionary" in top.path or "kb_fuzzy" in top.path

        # 弃权规则：没有词典/企业库强证据，且检索/推理给出的主体置信度不足时，
        # 不猜、不定论，保留客户口述原称并把候选池交人工裁决。
        if top_score < DECISIVE_CONFIDENCE and not has_strong_local:
            abstain_note = (
                f"候选主体置信度不足（{top_score:.2f} < {DECISIVE_CONFIDENCE}），"
                f"为避免污染 CRM 不做自动裁决，已保留口述原称并列出 {len(scored_candidates)} 个候选供人工选择")
            if trace is not None:
                trace.log("company.resolve", abstained=True, top_score=round(top_score, 3),
                          candidates=[c.legal_name for c in scored_candidates[:5]])
            return CompanyResolution(
                company_name=mention, confidence=round(max(0.15, top_score * 0.7), 3),
                original_mention=mention, match_type="ambiguous",
                candidates=scored_candidates[:5],
                conflicts=list(dict.fromkeys(conflicts + [abstain_note])),
                paths_used=list(dict.fromkeys(paths_used)),
                evidence=list(dict.fromkeys(evidence + [abstain_note])),
                reasoning=str(llm_output.get("reasoning", ""))[:600],
            )

        if "human_feedback" in top.path:
            match_type = "human_feedback"
        elif "dictionary" in top.path:
            # 词典精确/别名命中是最强证据，存在冲突时也保留主体但明确标记
            match_type = "exact_alias_conflict" if conflicts else "exact_alias"
        elif "kb_fuzzy" in top.path:
            match_type = "fuzzy"
        elif "web_search" in top.path or "llm" in top.path:
            match_type = "web_search" if search_text else "inferred"
        else:
            match_type = "low_confidence"

        # 融合多路一致度提升置信度
        if any(p in top.path for p in ("dictionary", "kb_fuzzy")) and search_text and \
                any("llm" in c.path for c in scored_candidates[:2]):
            top_score = min(1.0, top_score * 1.05)
            evidence.append("本地库与联网/LLM 结果一致，置信度提升")

        if top.province or top.city:
            evidence.append(f"候选企业注册地：{top.province or ''}{top.city or ''}")

        return CompanyResolution(
            company_name=top.legal_name,
            confidence=round(min(0.97, max(0.05, top_score)), 3),
            original_mention=mention or cr.clean_mention(top.legal_name),
            match_type=match_type,
            candidates=scored_candidates[:5],
            conflicts=conflicts,
            paths_used=list(dict.fromkeys(paths_used)),
            evidence=list(dict.fromkeys(evidence)),
            reasoning=str(llm_output.get("reasoning", ""))[:600],
        )
