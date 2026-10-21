"""后台任务（异步分析）测试：状态机、真实阶段映射、失败路径、HTTP 协议。"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.pipeline import SdrPipeline
from app.api.jobs import JobRegistry
from app.storage import LeadStore
from tests.helpers import SAMPLE_DIALOGUE, FakeLLM, extraction_payload


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestJobRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = JobRegistry(max_workers=2, trace_dir=Path(self.tmp.name))

    def tearDown(self):
        self.registry.shutdown(wait=True)
        self.tmp.cleanup()

    def test_text_job_exposes_real_stages_and_result(self):
        gate = threading.Event()

        def runner(tracer):
            tracer.log("pipeline.start")
            gate.wait(5)
            tracer.log("stage.intent_recognition")
            tracer.log("stage.information_extraction")
            tracer.log("stage.company_resolution")
            tracer.log("stage.intent_scoring")
            tracer.log("pipeline.done")
            return {"id": "call-1", "lead": {"company_name": "木里木外"}}

        job = self.registry.submit(mode="text", runner=runner)
        self.assertEqual(job.accepted()["job_id"], job.id)
        self.assertTrue(_wait(
            lambda: self.registry.snapshot(job.id)["status"] == "running"))

        snap = self.registry.snapshot(job.id)
        self.assertEqual(snap["stage"], "extract")
        self.assertEqual([s["label"] for s in snap["stages"]],
                         ["意图识别与信息抽取", "企业抬头还原", "意向评分", "结构化输出"])
        self.assertEqual(snap["stages"][0]["status"], "active")
        self.assertEqual(snap["stages"][1]["status"], "pending")

        gate.set()
        self.assertTrue(_wait(
            lambda: self.registry.snapshot(job.id)["status"] == "succeeded"))
        snap = self.registry.snapshot(job.id)
        self.assertEqual(snap["progress"], 1.0)
        self.assertTrue(all(s["status"] == "done" for s in snap["stages"]))
        self.assertEqual(snap["result"]["id"], "call-1")
        self.assertTrue(snap["trace_id"])

    def test_audio_job_marks_asr_done_before_extract(self):
        gate = threading.Event()

        def runner(tracer):
            tracer.log("stage.asr", duration_ms=1200, speakers=2)
            gate.wait(5)
            return {"id": "audio-1"}

        job = self.registry.submit(mode="audio", runner=runner)
        self.assertTrue(_wait(
            lambda: self.registry.snapshot(job.id)["stage"] == "extract"))

        snap = self.registry.snapshot(job.id)
        self.assertEqual(len(snap["stages"]), 5)
        self.assertEqual(snap["stages"][0]["status"], "done")
        self.assertEqual(snap["stage_label"], "意图识别与信息抽取")
        self.assertAlmostEqual(snap["progress"], 0.2)

        gate.set()
        self.assertTrue(_wait(
            lambda: self.registry.snapshot(job.id)["status"] == "succeeded"))

    def test_failed_job_reports_error_and_stage(self):
        def runner(tracer):
            with tracer.span("stage.asr"):
                raise RuntimeError("转写服务不可用")

        job = self.registry.submit(mode="audio", runner=runner)
        self.assertTrue(_wait(
            lambda: self.registry.snapshot(job.id)["status"] == "failed"))

        snap = self.registry.snapshot(job.id)
        self.assertEqual(snap["error"], "转写服务不可用")
        self.assertEqual(snap["stage"], "asr")
        self.assertEqual(snap["stages"][0]["status"], "active")
        self.assertIsNone(snap["result"])

    def test_unknown_job_returns_none(self):
        self.assertIsNone(self.registry.snapshot("not-exist"))

    def test_finished_job_expires_after_ttl(self):
        registry = JobRegistry(max_workers=1, ttl_seconds=0,
                               trace_dir=Path(self.tmp.name))
        try:
            job = registry.submit(mode="text", runner=lambda tracer: {})
            self.assertTrue(_wait(
                lambda: registry.snapshot(job.id)["status"] == "succeeded"))
            time.sleep(0.01)
            self.assertEqual(registry.prune(), 1)
            self.assertIsNone(registry.snapshot(job.id))
        finally:
            registry.shutdown(wait=True)


class TestAnalyzeApi(unittest.TestCase):
    """接口协议：POST 只返回 job_id，结果与阶段通过 GET /api/jobs/{id} 获取。"""

    def setUp(self):
        from app.api import server

        self.server = server
        self.tmp = tempfile.TemporaryDirectory()
        self._orig = (server.pipeline, server.store, server.jobs)
        server.store = LeadStore(Path(self.tmp.name))
        server.pipeline = SdrPipeline(llm=FakeLLM(extraction_payload()))
        server.jobs = JobRegistry(max_workers=2, trace_dir=Path(self.tmp.name))
        self.client = TestClient(server.app)

    def tearDown(self):
        self.server.jobs.shutdown(wait=True)
        (self.server.pipeline, self.server.store, self.server.jobs) = self._orig
        self.tmp.cleanup()

    def test_unknown_job_http_404(self):
        self.assertEqual(self.client.get("/api/jobs/nope").status_code, 404)

    def test_analyze_returns_job_id_then_result(self):
        resp = self.client.post("/api/analyze", json={
            "dialogue": SAMPLE_DIALOGUE, "call_id": "api-1", "save": True,
        })
        self.assertEqual(resp.status_code, 202)
        accepted = resp.json()
        self.assertIn(accepted["status"], ("queued", "running"))
        self.assertTrue(accepted["job_id"])

        snap = {}
        deadline = time.time() + 10
        while time.time() < deadline:
            snap = self.client.get(f"/api/jobs/{accepted['job_id']}").json()
            if snap["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.02)
        self.assertEqual(snap["status"], "succeeded")
        self.assertEqual(snap["result"]["id"], "api-1")
        self.assertEqual(snap["progress"], 1.0)

        record = self.server.store.get("api-1")
        self.assertIsNotNone(record)
        self.assertEqual(record["lead"]["company_name"], "上海木里实业发展有限公司")

    def test_empty_dialogue_rejected_before_job_creation(self):
        resp = self.client.post("/api/analyze", json={"dialogue": "   "})
        self.assertEqual(resp.status_code, 400)
