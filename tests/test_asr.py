"""音频转写层测试：说话人角色映射、对话格式化、文件校验、pipeline 集成。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.agent.pipeline import SdrPipeline
from app.asr import (AsrError, AsrResult, AsrSegment, AsrService, assign_roles,
                     format_dialogue)
from tests.helpers import FakeLLM, extraction_payload


class TestSpeakerRoleMapping(unittest.TestCase):
    def _segs(self, rows):
        return [AsrSegment(spk, i * 1000, i * 1000 + 900, text)
                for i, (spk, text) in enumerate(rows)]

    def test_two_speakers_sales_by_cues(self):
        segs = self._segs([
            (0, "您好，请问是张总吗？我是酷家乐的小王，想跟您确认一下需求。"),
            (1, "是的，我们这边是杭州做全屋定制的，预算一年几万块。"),
            (0, "好的，我给您发一份方案，安排演示。"),
        ])
        self.assertEqual(assign_roles(segs), {0: "销售", 1: "客户"})

    def test_two_speakers_reversed_order(self):
        # 客户先开口，但仍应按话术线索识别
        segs = self._segs([
            (7, "喂，你好。我们这边是苏州做橱柜的。"),
            (3, "孙经理您好，我是酷家乐的小李，给您发个方案。"),
        ])
        self.assertEqual(assign_roles(segs), {3: "销售", 7: "客户"})

    def test_single_speaker_not_forced_into_role(self):
        segs = self._segs([(0, "Hello world")])
        self.assertEqual(assign_roles(segs), {0: "说话人1"})

    def test_no_cues_first_speaker_is_sales(self):
        segs = self._segs([(5, "喂"), (9, "嗯")])
        self.assertEqual(assign_roles(segs), {5: "销售", 9: "客户"})

    def test_three_speakers_naming(self):
        segs = self._segs([
            (1, "我是酷家乐的小王，给您发方案。"),
            (2, "我们这边是杭州做定制的。"),
            (3, "我这边也听一下。"),
        ])
        roles = assign_roles(segs)
        self.assertEqual(roles[1], "销售")
        self.assertEqual(roles[2], "客户")
        self.assertEqual(roles[3], "客户2")

    def test_missing_speaker_ids(self):
        segs = [AsrSegment(None, 0, 100, "文本")]
        self.assertEqual(assign_roles(segs), {})


class TestDialogueFormatting(unittest.TestCase):
    def test_merge_consecutive_same_speaker(self):
        segs = [
            AsrSegment(0, 0, 500, "您好，"),
            AsrSegment(0, 500, 900, "我是酷家乐的小王。"),
            AsrSegment(1, 900, 1500, "你好。"),
            AsrSegment(0, 1500, 2000, "想确认一下需求。"),
        ]
        roles = {0: "销售", 1: "客户"}
        text = format_dialogue(segs, roles)
        self.assertEqual(text,
                         "销售：您好，我是酷家乐的小王。\n客户：你好。\n销售：想确认一下需求。")

    def test_empty_segments(self):
        self.assertEqual(format_dialogue([], {}), "")


class _FakeProvider:
    name = "fake"

    def __init__(self, result: AsrResult) -> None:
        self.result = result

    def available(self) -> bool:
        return True

    def transcribe(self, path: Path) -> AsrResult:
        return self.result


class TestAsrServiceValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):
        with self.assertRaises(AsrError):
            AsrService().transcribe(self.dir / "nope.mp3")

    def test_unsupported_extension(self):
        p = self.dir / "note.txt"
        p.write_text("x", encoding="utf-8")
        with self.assertRaises(AsrError) as ctx:
            AsrService().transcribe(p)
        self.assertIn("不支持的音频格式", str(ctx.exception))

    def test_empty_file(self):
        p = self.dir / "empty.mp3"
        p.write_bytes(b"")
        with self.assertRaises(AsrError) as ctx:
            AsrService().transcribe(p)
        self.assertIn("为空", str(ctx.exception))

    def test_fake_provider_end_to_end(self):
        p = self.dir / "call.mp3"
        p.write_bytes(b"fake-audio")
        result = AsrResult(
            dialogue="销售：您好。\n客户：你好。",
            segments=[AsrSegment(0, 0, 500, "您好。"), AsrSegment(1, 500, 900, "你好。")],
            provider="fake", model="fake-1", duration_ms=900,
            speaker_count=2, speaker_roles={0: "销售", 1: "客户"},
            audio_file="call.mp3",
        )
        service = AsrService(provider="fake", providers={"fake": _FakeProvider(result)})
        out = service.transcribe(p)
        self.assertEqual(out.dialogue, result.dialogue)
        self.assertEqual(out.to_meta()["speaker_count"], 2)


class TestPipelineAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = Path(self.tmp.name) / "call.mp3"
        self.audio.write_bytes(b"fake-audio")
        self.result = AsrResult(
            dialogue=("销售：您好，是张总吗？我是酷家乐的小王。\n"
                      "客户：是的，我们这边是杭州做全屋定制的，叫“木里木外”。\n"
                      "客户：设计部大概 15 个人，出图慢。\n"
                      "客户：最好下个月就能试一下，预算一年几万块能接受。"),
            segments=[], provider="fake", model="fake-1", duration_ms=12000,
            speaker_count=2, speaker_roles={0: "销售", 1: "客户"},
            audio_file="call.mp3", warnings=[],
        )
        self.pipeline = SdrPipeline(
            llm=FakeLLM(extraction_payload()),
            asr_service=AsrService(provider="fake",
                                   providers={"fake": _FakeProvider(self.result)}),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_audio_goes_through_same_pipeline(self):
        out = self.pipeline.run_audio(str(self.audio), call_id="a1",
                                      call_time="2026-10-21 10:00:00")
        self.assertEqual(out.lead.company_name, "上海木里实业发展有限公司")
        self.assertEqual(out.lead.shejishi, 15)
        self.assertEqual(out.lead.intent_level, "high")
        self.assertEqual(out.meta["asr"]["provider"], "fake")
        self.assertIn("stage.asr", out.meta["pipeline_stages"])
        self.assertIn("客户：是的", out.input_dialogue)

    def test_asr_warning_merged_into_meta(self):
        self.result.warnings = ["未识别出多个说话人"]
        out = self.pipeline.run_audio(str(self.audio), call_id="a2")
        self.assertIn("未识别出多个说话人", out.meta["warnings"])


if __name__ == "__main__":
    unittest.main()
