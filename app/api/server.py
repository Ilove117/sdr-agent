"""Web 工作台后端 API（FastAPI）。

面向两类使用者：
  1. 业务用户 —— 工作台：上传录音或粘贴文本，拿到结构化线索
  2. 复核人   —— 线索库：审核、修正、批量订正、反馈沉淀

接口只有 5 组动作，刻意保持轻量：
  看（列表/详情/统计）· 分析（文本/音频）· 查任务（job 状态/真实阶段）· 改（单条/批量复核）· 导出

音频转写 + 抽取是长任务：分析接口只负责校验入参、入队并返回 job_id，
真正执行在 JobRegistry 的后台线程里，前端轮询 /api/jobs/{id} 获取阶段与结果。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.agent.pipeline import SdrPipeline
from app.api.jobs import JobRegistry
from app.asr import SUPPORTED_AUDIO_EXT, AsrError
from app.config import PROVIDERS, settings
from app.storage import LeadStore
from app.schema import SQL_FIELDS
from app.tracing import Tracer

logger = logging.getLogger(__name__)

app = FastAPI(title="AI SDR Agent · 通话线索工作台", version="1.2.0")
store = LeadStore(settings.data_dir)
pipeline = SdrPipeline()
jobs = JobRegistry()

CONSOLE_HTML = Path(__file__).resolve().parent.parent / "ui" / "console.html"


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class ReviewRequest(BaseModel):
    changes: dict[str, Any] = Field(default_factory=dict)
    reviewer: str = "reviewer"
    note: str | None = None
    status: str | None = None


class BatchReviewRequest(ReviewRequest):
    ids: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    dialogue: str
    call_id: str | None = None
    call_time: str | None = None
    save: bool = True


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def console() -> HTMLResponse:
    if not CONSOLE_HTML.exists():
        raise HTTPException(500, f"控制台页面缺失：{CONSOLE_HTML}")
    # 控制台是单文件页面，禁止缓存，避免改完前端还要手动强刷
    return HTMLResponse(CONSOLE_HTML.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, max-age=0"})


# --------------------------------------------------------------------------- #
# 查询
# --------------------------------------------------------------------------- #
@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "sql_fields": list(SQL_FIELDS),
        "review_confidence_threshold": settings.review_confidence_threshold,
        "critical_field_threshold": settings.critical_field_threshold,
        "primary_provider": settings.primary_provider,
        "fallback_provider": settings.fallback_provider,
        "models": {k: settings.model_for(k) for k in PROVIDERS},
        "web_search_enabled": settings.enable_web_search,
        "search_providers": settings.search_providers,
        "asr": {
            "provider": settings.asr_provider,
            "model": settings.asr_model,
            "language": settings.asr_language,
            "diarization": settings.asr_diarization,
            "supported_ext": sorted(SUPPORTED_AUDIO_EXT),
        },
    }


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    return store.stats()


@app.get("/api/leads")
def list_leads(status: str = "all", level: str = "all", need_review_only: bool = False,
               query: str = "", limit: int = 200, offset: int = 0,
               with_dialogue: bool = False) -> dict[str, Any]:
    items = store.list_records(status=status, level=level, need_review_only=need_review_only,
                               query=query or None, limit=limit, offset=offset,
                               with_dialogue=with_dialogue)
    summaries = []
    for r in items:
        lead = r.get("lead") or {}
        meta = r.get("meta") or {}
        summaries.append({
            "id": r.get("id"),
            "created_at": r.get("created_at"),
            "call_time": r.get("call_time"),
            "company_name": lead.get("company_name"),
            "industry": lead.get("industry"),
            "area": lead.get("area"),
            "intent_level": lead.get("intent_level"),
            "confidence_score": lead.get("confidence_score"),
            "need_human_review": lead.get("need_human_review"),
            "review_status": store.review_status(r),
            "reviewer": (r.get("review") or {}).get("reviewer"),
            "review_reason_count": len(meta.get("review_reasons") or []),
            "first_reason": (meta.get("review_reasons") or [None])[0],
            "match_type": (meta.get("company_resolution") or {}).get("match_type"),
        })
        if with_dialogue:
            summaries[-1]["dialogue"] = r.get("dialogue")
    return {"total": len(summaries), "items": summaries}


@app.get("/api/leads/{lead_id}")
def get_lead(lead_id: str) -> dict[str, Any]:
    record = store.get(lead_id)
    if record is None:
        raise HTTPException(404, f"线索不存在：{lead_id}")
    return record


@app.get("/api/export")
def export(only_approved: bool = True) -> dict[str, Any]:
    rows = store.export_sql(only_approved=only_approved)
    return {"count": len(rows), "rows": rows}


@app.get("/api/kb/candidates")
def kb_candidates() -> dict[str, Any]:
    """人工订正沉淀出的「口述别称 -> 工商全称」弱标注，供企业库增补。"""
    items = store.kb_candidates()
    return {"count": len(items), "items": items}


# --------------------------------------------------------------------------- #
# 复核
# --------------------------------------------------------------------------- #
@app.post("/api/leads/{lead_id}/review")
def review_lead(lead_id: str, payload: ReviewRequest) -> dict[str, Any]:
    try:
        record = store.apply_review(lead_id, changes=payload.changes, reviewer=payload.reviewer,
                                    note=payload.note, status=payload.status)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if record is None:
        raise HTTPException(404, f"线索不存在：{lead_id}")
    return record


@app.post("/api/batch/review")
def batch_review(payload: BatchReviewRequest) -> dict[str, Any]:
    if not payload.ids:
        raise HTTPException(400, "ids 不能为空")
    try:
        return store.batch_review(payload.ids, changes=payload.changes, reviewer=payload.reviewer,
                                  note=payload.note, status=payload.status)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------- #
# 现场分析（异步任务：立即返回 job_id，前端轮询 /api/jobs/{id}）
# --------------------------------------------------------------------------- #
@app.post("/api/analyze", status_code=202)
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    if not payload.dialogue.strip():
        raise HTTPException(400, "dialogue 不能为空")
    job = jobs.submit(mode="text", runner=lambda tracer: _run_text_job(payload, tracer),
                      meta={"call_id": payload.call_id})
    return job.accepted()


@app.post("/api/analyze-audio", status_code=202)
def analyze_audio(file: UploadFile = File(...),
                  call_id: str | None = Form(None),
                  call_time: str | None = Form(None),
                  save: bool = Form(True)) -> dict[str, Any]:
    """上传录音后立即返回 job_id，转写与抽取在后台跑，避免 HTTP 请求挂死。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXT:
        raise HTTPException(
            400, f"不支持的音频格式：{suffix or '(无扩展名)'}，"
                 f"支持 {', '.join(sorted(SUPPORTED_AUDIO_EXT))}")

    stem = Path(file.filename or "audio").stem
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in "-_")[:60] or "audio"
    target = settings.audio_dir / f"{stem}{suffix}"
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh, length=1024 * 1024)
    finally:
        file.file.close()

    rid = call_id or stem
    job = jobs.submit(
        mode="audio",
        runner=lambda tracer: _run_audio_job(target, rid, call_time, save, tracer),
        meta={"call_id": rid, "audio_file": str(target)},
    )
    return job.accepted()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """返回任务状态、真实阶段（来自 Tracer）与最终结果。"""
    snapshot = jobs.snapshot(job_id)
    if snapshot is None:
        raise HTTPException(404, f"任务不存在或已过期：{job_id}")
    return snapshot


def _run_text_job(payload: AnalyzeRequest, tracer: Tracer) -> dict[str, Any]:
    output = pipeline.run(payload.dialogue, call_id=payload.call_id,
                          call_time=payload.call_time, tracer=tracer)
    rid = payload.call_id
    if payload.save:
        record = store.save(lead=output.lead, meta=output.meta,
                            dialogue=payload.dialogue, call_id=payload.call_id)
        rid = record["id"]
    result = output.to_dict()
    result["id"] = rid
    return result


def _run_audio_job(target: Path, rid: str, call_time: str | None, save: bool,
                   tracer: Tracer) -> dict[str, Any]:
    try:
        output = pipeline.run_audio(str(target), call_id=rid, call_time=call_time,
                                    tracer=tracer)
    except AsrError as exc:
        raise RuntimeError(f"音频转写失败：{exc}") from exc

    if save:
        record = store.save(lead=output.lead, meta=output.meta,
                            dialogue=output.input_dialogue, call_id=rid)
        rid = record["id"]
    result = output.to_dict()
    result["id"] = rid
    return result


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
