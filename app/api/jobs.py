"""后台任务注册表（异步长请求）。

音频转写 + 抽取最长可能持续数分钟。同步挂在 HTTP 请求上，一旦代理超时或
浏览器休眠，结果就白丢了。这里把任务丢到后台线程，接口只返回 job_id；
前端轮询 `/api/jobs/{job_id}` 获取真实阶段（来自 Tracer）与最终结果。

阶段不做定时器假推进：completed 阶段完全由 tracer 已落下的 event 推导，
最后一个 event 之后的阶段才是 active。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.config import settings
from app.tracing import Tracer

logger = logging.getLogger(__name__)

# 阶段展示顺序：audio 比 text 多一个 ASR 阶段
STAGE_LABELS = {
    "asr": "语音转写",
    "extract": "意图识别与信息抽取",
    "company": "企业抬头还原",
    "scoring": "意向评分",
    "output": "结构化输出",
}

STAGE_ORDER = {
    "audio": ("asr", "extract", "company", "scoring", "output"),
    "text": ("extract", "company", "scoring", "output"),
}

# tracer event -> 该 event 落下时「已完成」的最后一个阶段
# 说明：extract.llm / company.* 是阶段内部事件，只能说明上一阶段已完成，
# 不能说明当前阶段完成；只有 stage.* 汇总事件才算完成。
EVENT_STAGE = {
    "stage.asr": "asr",
    "extract.llm": "extract",
    "stage.intent_recognition": "extract",
    "stage.information_extraction": "extract",
    "company.resolve": "extract",
    "company.web_search": "extract",
    "stage.company_resolution": "company",
    "stage.intent_scoring": "scoring",
    "pipeline.done": "output",
}


@dataclass
class Job:
    id: str
    mode: str
    created_at: float
    status: str = "queued"          # queued / running / succeeded / failed
    tracer: Tracer | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def accepted(self) -> dict[str, Any]:
        """提交接口的即时响应：只透出 job_id 与当前状态。"""
        return {"job_id": self.id, "status": self.status, "mode": self.mode}


class JobRegistry:
    """进程内任务表 + 固定大小线程池。"""

    def __init__(self, max_workers: int = 4, ttl_seconds: int = 2 * 3600,
                 trace_dir: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        self._trace_dir = trace_dir or settings.trace_dir
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="sdr-job")

    # ------------------------------------------------------------------ #
    def submit(self, *, mode: str, runner: Callable[[Tracer], dict[str, Any]],
               meta: dict[str, Any] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], mode=mode, created_at=time.time(),
                  meta=meta or {})
        with self._lock:
            self._prune_locked()
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, runner)
        return job

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            status = job.status
            tracer = job.tracer
            result = job.result
            error = job.error
            created_at = job.created_at
            started_at = job.started_at
            finished_at = job.finished_at
            mode = job.mode
            meta = dict(job.meta)

        stages, current, progress = _describe_stages(mode, status, tracer)
        elapsed = (finished_at or time.time()) - (started_at or created_at)
        payload: dict[str, Any] = {
            "job_id": job_id,
            "mode": mode,
            "status": status,
            "stage": current,
            "stage_label": STAGE_LABELS.get(current or "", ""),
            "stages": stages,
            "progress": progress,
            "elapsed_ms": max(0, int(elapsed * 1000)),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.localtime(created_at)),
            "result": result,
            "error": error,
            "trace_id": tracer.trace_id if tracer else None,
        }
        payload.update(meta)
        return payload

    def prune(self) -> int:
        with self._lock:
            return self._prune_locked()

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait)

    # ------------------------------------------------------------------ #
    def _run(self, job: Job, runner: Callable[[Tracer], dict[str, Any]]) -> None:
        tracer = Tracer(self._trace_dir)
        with self._lock:
            job.tracer = tracer
            job.status = "running"
            job.started_at = time.time()
        try:
            result = runner(tracer)
        except Exception as exc:  # noqa: BLE001 - 任务失败信息要回给前端
            logger.exception("后台任务失败：job_id=%s", job.id)
            with self._lock:
                job.status = "failed"
                job.error = str(exc) or type(exc).__name__
                job.finished_at = time.time()
        else:
            with self._lock:
                job.result = result
                job.status = "succeeded"
                job.finished_at = time.time()
        self._flush(job, tracer)

    @staticmethod
    def _flush(job: Job, tracer: Tracer) -> None:
        try:
            tracer.flush(summary={"job_id": job.id, "mode": job.mode,
                                  "status": job.status,
                                  "elapsed_ms": int(((job.finished_at or time.time())
                                                     - (job.started_at or job.created_at))
                                                    * 1000)})
        except OSError:
            logger.warning("trace 落盘失败：job_id=%s", job.id, exc_info=True)

    def _prune_locked(self) -> int:
        deadline = time.time() - self._ttl_seconds
        stale = [jid for jid, job in self._jobs.items()
                 if job.status in ("succeeded", "failed")
                 and (job.finished_at or job.created_at) < deadline]
        for jid in stale:
            del self._jobs[jid]
        return len(stale)


def _describe_stages(mode: str, status: str,
                     tracer: Tracer | None) -> tuple[list[dict[str, str]], str | None, float]:
    """把 tracer 事件映射为前端可直接渲染的阶段列表。"""
    order = STAGE_ORDER.get(mode, STAGE_ORDER["text"])
    steps = list(tracer.steps) if tracer else []

    completed = -1
    for step in steps:
        key = EVENT_STAGE.get(step.get("event"))
        if key in order:
            completed = max(completed, order.index(key))

    if status == "succeeded":
        completed = len(order) - 1
    elif status == "failed" and steps and steps[-1].get("status") == "error":
        # 失败发生在哪个阶段，就把该阶段标为 active，方便前端指出卡点
        failed_key = EVENT_STAGE.get(steps[-1].get("event"))
        if failed_key in order:
            completed = order.index(failed_key) - 1
    active = completed + 1
    if active >= len(order):
        active = -1

    stages = []
    for idx, key in enumerate(order):
        if idx <= completed:
            stage_status = "done"
        elif idx == active:
            stage_status = "active"
        else:
            stage_status = "pending"
        stages.append({"key": key, "label": STAGE_LABELS[key], "status": stage_status})

    current = order[active] if active >= 0 else None
    progress = 1.0 if status == "succeeded" else max(completed + 1, 0) / len(order)
    return stages, current, round(progress, 3)
