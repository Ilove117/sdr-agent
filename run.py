#!/usr/bin/env python
"""AI SDR Agent 命令行入口。

用法示例：
  python run.py analyze --file data/samples/sample_calls.json          # 批量分析样本
  python run.py analyze --file data/samples/sample_calls.json --only call-001 --verbose
  python run.py analyze --text "销售：您好……"                          # 单条文本
  python run.py serve                                                  # 启动人工干预管理台
  python run.py stats                                                  # 线索库统计
  python run.py selfcheck                                              # 规则层自检（不调用大模型）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent.pipeline import SdrPipeline  # noqa: E402
from app.config import settings  # noqa: E402
from app.storage import LeadStore  # noqa: E402
from app.tracing import Tracer  # noqa: E402


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_calls(file: str) -> list[dict]:
    data = json.loads(Path(file).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("calls", [])
    return data


def cmd_analyze(args: argparse.Namespace) -> int:
    store = LeadStore(settings.data_dir)
    pipeline = SdrPipeline()
    verbose = args.verbose

    calls: list[dict] = []
    if args.audio:
        audio = Path(args.audio).expanduser()
        calls = [{"id": args.call_id or audio.stem, "audio": str(audio),
                  "scenario": "音频输入"}]
    elif args.audio_dir:
        audio_ext = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
        folder = Path(args.audio_dir).expanduser()
        if not folder.is_dir():
            print(f"音频目录不存在：{folder}", file=sys.stderr)
            return 1
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in audio_ext)
        calls = [{"id": p.stem, "audio": str(p), "scenario": "音频输入"} for p in files]
    elif args.text:
        calls = [{"id": args.call_id or "cli-input", "dialogue": args.text}]
    else:
        calls = _load_calls(args.file)
    if args.only:
        wanted = set(args.only.split(","))
        calls = [c for c in calls if c.get("id") in wanted]
    if not calls:
        print("没有可分析的记录", file=sys.stderr)
        return 1

    summary_rows = []
    for call in calls:
        dialogue = call.get("dialogue") or call.get("text") or ""
        tracer = Tracer(settings.trace_dir)
        try:
            if call.get("audio"):
                output = pipeline.run_audio(call["audio"], call_id=call.get("id"),
                                           call_time=call.get("call_time"), tracer=tracer)
                dialogue = output.input_dialogue
            else:
                output = pipeline.run(dialogue, call_id=call.get("id"),
                                     call_time=call.get("call_time"), tracer=tracer)
        except Exception as exc:  # 单条失败不阻断批次
            print(f"[{call.get('id')}] 处理失败：{type(exc).__name__}: {exc}",
                  file=sys.stderr)
            tracer.flush(summary={"call_id": call.get("id"), "error": str(exc)})
            continue

        store.save(lead=output.lead, meta=output.meta, dialogue=dialogue,
                   call_id=call.get("id"))
        tracer.flush(summary={"call_id": call.get("id"),
                              "confidence_score": output.lead.confidence_score,
                              "intent_level": output.lead.intent_level,
                              "need_human_review": output.lead.need_human_review})

        print("=" * 78)
        print(f"通话记录 {call.get('id')}  |  场景：{call.get('scenario', '-')}")
        print("-" * 78)
        asr_meta = output.meta.get("asr")
        if asr_meta:
            print(f"-- 转写：{asr_meta['provider']}/{asr_meta['model']}  "
                  f"时长 {(asr_meta['duration_ms'] or 0) / 1000:.1f}s  "
                  f"说话人 {asr_meta['speaker_count']}  "
                  f"角色 {asr_meta['speaker_roles']}")
        print(output.lead.to_pretty_json())
        comp = output.meta["company_resolution"]
        print(f"-- 抬头还原：{comp['match_type']}  置信度 {comp['confidence']}  "
              f"路径 {'->'.join(comp['paths_used']) or '无'}")
        if comp["conflicts"]:
            print(f"-- 冲突：{'；'.join(comp['conflicts'])}")
        print(f"-- 需人工复核：{output.lead.need_human_review}  "
              f"原因 {output.meta['review_reasons']}")
        if verbose:
            print("-- 逐字段置信度：", json.dumps(output.meta["field_confidence"],
                                                ensure_ascii=False))
            print("-- 候选企业：", json.dumps(
                [c["legal_name"] for c in comp["candidates"]], ensure_ascii=False))
        summary_rows.append({
            "id": call.get("id"), "company": output.lead.company_name,
            "intent": output.lead.intent_level,
            "confidence": output.lead.confidence_score,
            "review": output.lead.need_human_review,
        })

    print("=" * 78)
    print("批次汇总")
    print(f"{'通话':<12}{'企业抬头':<34}{'意向':<8}{'置信':<8}{'复核'}")
    for row in summary_rows:
        print(f"{row['id']:<12}{row['company']:<34}{row['intent']:<8}"
              f"{row['confidence']:<8}{row['review']}")
    print(f"\n线索库：{store.stats()}")
    print(f"追踪文件：{settings.trace_dir}")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """只做语音转写，不跑线索抽取；便于先检查转写质量。"""
    from app.asr import AsrError, AsrService

    service = AsrService()
    try:
        result = service.transcribe(args.audio)
    except AsrError as exc:
        print(f"转写失败：{exc}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"音频：{result.audio_file}  |  {result.provider}/{result.model}")
    print(f"时长：{(result.duration_ms or 0) / 1000:.1f}s  |  "
          f"说话人：{result.speaker_count}  |  角色：{result.speaker_roles}")
    print("=" * 78)
    print(result.dialogue)
    if result.warnings:
        print("\n提示：" + "；".join(result.warnings))

    if args.json:
        out = settings.data_dir / "transcripts" / f"{Path(args.audio).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "audio_file": result.audio_file,
            "provider": result.provider,
            "model": result.model,
            "duration_ms": result.duration_ms,
            "speaker_count": result.speaker_count,
            "speaker_roles": {str(k): v for k, v in result.speaker_roles.items()},
            "dialogue": result.dialogue,
            "segments": [s.to_dict() for s in result.segments],
            "warnings": result.warnings,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入：{out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    print(f"人工干预管理台： http://{settings.api_host}:{settings.api_port}")
    uvicorn.run("app.api.server:app", host=settings.api_host,
                port=args.port or settings.api_port, reload=False)
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    store = LeadStore(settings.data_dir)
    print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_selfcheck(_: argparse.Namespace) -> int:
    """规则层离线自检：不依赖大模型，验证口径与评分卡。"""
    from app.rules import company_rules as cr
    from app.rules import intent_scorecard as sc
    from app.rules import sql_fields as sf
    from app.rules.intent_scorecard import SignalSet

    failures: list[str] = []

    def check(label: str, actual, expected) -> None:
        if actual != expected:
            failures.append(f"{label}: 期望 {expected!r}，实际 {actual!r}")

    # 数量抽取（含中文数字、语序兼容、抗干扰）
    check("设计师人数(语序后置)", sf.extract_quantity(
        "设计部大概 15 个人", r"人|设计师", ("设计部",))[0], 15)
    check("设计师人数(中文数字)", sf.extract_quantity(
        "设计师二十人", r"人|设计师", ("设计师",))[0], 20)
    check("无关数字不误伤", sf.extract_quantity(
        "想了解企业版设计软件，今天想跟您确认一下需求", r"人|设计师", ("设计部", "设计师"))[0], None)
    check("门店数", sf.extract_quantity(
        "佛山这边三家门店", r"门店|店", ("门店", "店"))[0], 3)

    # 手机号
    check("手机号掩码", sf.extract_phone("加我 138xxxx1234")[0], "138****1234")
    check("手机号明文", sf.extract_phone("加我 13812345678")[0], "13812345678")

    # 抬头别称
    check("别称还原", "木里木外家居" in cr.alias_variants("木里木外"), True)
    check("相似度自反", cr.similarity("木里木外", "木里木外"), 1.0)

    # 意向评分卡
    high = sc.score_intent(SignalSet.from_dict({
        "budget": {"mentioned": True, "specificity": "clear", "value_text": "一年几万"},
        "timeline": {"mentioned": True, "urgency": "near_term"},
        "pain_points": [{"pain_text": "出图慢"}, {"pain_text": "工厂对接出错"}],
        "active_engagement": [{"type": "provide_contact"}, {"type": "agree_next_step"}],
        "decision_maker": {"is_decision_maker": True},
    }), designers=15, contact_title="负责人/总经理")
    check("高意向判定", high.level, "high")

    low = sc.score_intent(SignalSet.from_dict({
        "negative_signals": [{"type": "just_looking"}, {"type": "reject"}],
    }))
    check("低意向判定", low.level, "low")

    print("自检结果：" + ("全部通过" if not failures else f"{len(failures)} 项失败"))
    for f in failures:
        print("  -", f)
    return 0 if not failures else 1


def cmd_eval(args: argparse.Namespace) -> int:
    from app.eval import print_report, run_eval
    path = Path(args.file)
    if not path.is_absolute():
        path = settings.base_dir / path
    report = run_eval(path)
    print_report(report)
    if args.json:
        out = settings.data_dir / "eval_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"报告已写入：{out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI SDR Agent")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印详细诊断信息")
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="分析通话记录并入库")
    p_an.add_argument("--file", default=str(settings.data_dir / "samples" / "sample_calls.json"))
    p_an.add_argument("--text", help="直接传入单条通话文本")
    p_an.add_argument("--audio", help="直接传入单个音频文件（mp3/wav/m4a 等）")
    p_an.add_argument("--audio-dir", help="批量转写并分析目录下的音频文件")
    p_an.add_argument("--call-id", help="配合 --text / --audio 使用，指定记录 ID")
    p_an.add_argument("--only", help="仅处理指定 ID，逗号分隔")
    p_an.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                      help="打印逐字段置信度与候选企业")
    p_an.set_defaults(func=cmd_analyze)

    p_tr = sub.add_parser("transcribe", help="仅做语音转写，输出带说话人角色的对话文本")
    p_tr.add_argument("--audio", required=True, help="音频文件路径")
    p_tr.add_argument("--json", action="store_true",
                      help="同时写出 data/transcripts/<文件名>.json（含逐句时间戳）")
    p_tr.set_defaults(func=cmd_transcribe)

    p_sv = sub.add_parser("serve", help="启动人工干预管理台")
    p_sv.add_argument("--port", type=int)
    p_sv.set_defaults(func=cmd_serve)

    sub.add_parser("stats", help="线索库统计").set_defaults(func=cmd_stats)
    sub.add_parser("selfcheck", help="规则层离线自检").set_defaults(func=cmd_selfcheck)

    p_ev = sub.add_parser("eval", help="用 golden set 评测抬头/意向准确率")
    p_ev.add_argument("--file", default="data/eval/golden_cases.json")
    p_ev.add_argument("--json", action="store_true", help="同时落盘 JSON 报告")
    p_ev.set_defaults(func=cmd_eval)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
