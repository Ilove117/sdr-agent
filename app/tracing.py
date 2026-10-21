"""执行追踪（observability）。

每次处理一通通话记录生成一个 trace_id，逐步记录每个 pipeline 节点的
输入/输出/耗时/状态，落盘 data/traces/*.jsonl，便于：
- 复盘「为什么这条线索被判成低意向」
- 定位企业抬头还原失败在哪一跳
- 沉淀评测样本
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class _Span:
    """上下文管理器，自动记录耗时与异常。"""

    def __init__(self, tracer: "Tracer", name: str, payload: dict[str, Any]) -> None:
        self.tracer = tracer
        self.name = name
        self.payload = payload
        self.outputs: dict[str, Any] = {}
        self.started = time.time()
        self.status = "ok"
        self.error: str | None = None

    def __enter__(self) -> "_Span":
        return self

    def set(self, **kwargs: Any) -> "_Span":
        self.outputs.update(kwargs)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.status = "error"
            self.error = f"{exc_type.__name__}: {exc}"
        self.tracer.log(
            self.name,
            status=self.status,
            duration_ms=int((time.time() - self.started) * 1000),
            inputs=self.payload or None,
            outputs=self.outputs or None,
            error=self.error,
        )
        return False  # 不吞异常


@dataclass
class Tracer:
    trace_dir: Path
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    steps: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "trace_id": self.trace_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "event": event,
            **payload,
        }
        self.steps.append(record)
        return record

    def span(self, name: str, **payload: Any) -> _Span:
        return _Span(self, name, payload)

    def step_names(self) -> list[str]:
        return [s["event"] for s in self.steps]

    def flush(self, *, summary: dict[str, Any] | None = None) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{self.trace_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for step in self.steps:
                fh.write(json.dumps(step, ensure_ascii=False) + "\n")
            if summary is not None:
                fh.write(json.dumps(
                    {"trace_id": self.trace_id, "event": "summary", **summary},
                    ensure_ascii=False) + "\n")
        return path
