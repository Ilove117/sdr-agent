"""语音转写（ASR）层：把通话录音转成带说话人角色的对话文本。

设计目标：去掉「人工听录音再手打转写」这一步，让上游可以直接喂 mp3。

为什么不是简单调用一次 ASR：
1. 后续 Agent 的取证边界依赖「销售 / 客户」话轮（否则会把我方品牌当成客户企业）。
   录音多为单声道混音，必须依赖说话人分离（diarization）才能还原话轮。
2. 录音是本地文件，而云端 ASR 只接受公网 URL，需要一条临时存储上传通道。
3. 转写本身会失败（文件损坏、时长超限、欠费、网络抖动），需要明确的重试与错误语义。

当前实现：阿里云百炼 paraformer-v2
- 支持说话人分离，输出 `sentences[].speaker_id`
- 本地文件通过官方临时存储上传，得到 `oss://` 临时 URL
- 调用时需带 `X-DashScope-OssResourceResolve: enable`
- 仅单声道支持分离；建议时长 ≤ 2 小时
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/api/v1"

SUPPORTED_AUDIO_EXT = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"})
MAX_AUDIO_BYTES = 1024 * 1024 * 1024  # 官方临时存储单文件上限 1GB

# 销售话术线索词，用于把说话人映射到「销售 / 客户」
SALES_CUES: dict[str, float] = {
    "酷家乐": 3.0, "我是": 2.0, "我这边是": 2.0, "给您": 2.0, "您这边": 2.0,
    "方案": 2.0, "演示": 2.0, "报价": 2.0, "加您微信": 3.0, "加个微信": 3.0,
    "试用": 2.0, "企业版": 3.0, "回访": 2.0, "官网": 2.0, "留了信息": 2.0,
    "留过信息": 2.0, "参数化": 2.0, "一键出图": 2.0, "对接生产": 2.0,
    "发您": 2.0, "发给您": 2.0, "安排": 1.0, "方便": 1.0, "了解": 1.0,
    "想跟您": 2.0, "感谢": 1.0, "请问": 1.5,
}
CUSTOMER_CUES: dict[str, float] = {
    "我们这边是": 2.0, "我们是做": 3.0, "我们公司": 1.5, "我们门店": 2.0,
    "我们设计": 2.0, "我们叫": 2.0, "预算": 2.0, "多少钱": 2.0, "暂时": 2.0,
    "考虑": 1.0, "已经有": 2.0, "用别的": 2.0, "不知道你们": 2.0,
    "下个月": 1.0, "门店": 1.0, "设计师": 1.0, "出图慢": 2.0,
}


class AsrError(RuntimeError):
    """转写失败（文件、网络、服务或超时）。"""


@dataclass
class AsrSegment:
    speaker_id: int | None
    begin_ms: int | None
    end_ms: int | None
    text: str

    def to_dict(self) -> dict:
        return {"speaker_id": self.speaker_id, "begin_ms": self.begin_ms,
                "end_ms": self.end_ms, "text": self.text}


@dataclass
class AsrResult:
    dialogue: str
    segments: list[AsrSegment] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    duration_ms: int | None = None
    speaker_count: int = 0
    speaker_roles: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    audio_file: str = ""
    raw_transcript: dict | None = None

    def to_meta(self) -> dict:
        """写入线索 meta 的诊断信息（不落原始逐词时间戳，控制体积）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "audio_file": self.audio_file,
            "duration_ms": self.duration_ms,
            "speaker_count": self.speaker_count,
            "speaker_roles": {str(k): v for k, v in self.speaker_roles.items()},
            "segment_count": len(self.segments),
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# 说话人 -> 角色
# --------------------------------------------------------------------------- #
def _cue_score(text: str, cues: dict[str, float]) -> float:
    return sum(w for term, w in cues.items() if term in text)


def assign_roles(segments: list[AsrSegment]) -> dict[int, str]:
    """把 ASR 的 speaker_id 映射为「销售 / 客户」。

    规则：
    1. 单说话人（多为分离失败）：标记为「说话人1」，不臆造销售/客户，
       下游 `customer_text` 会退回全文，避免把内容全部丢掉。
    2. 多说话人：按销售话术得分减客户话术得分排序，最高者判为销售。
    3. 全部得分为 0：默认第一个开口的是销售（SDR 外呼场景）。

    这是启发式，不是语义理解；它的作用是在进入 Agent 前给出一个可用的话轮边界，
    不追求 100% 正确——真正的业务判断仍由下游规则层与人工复核兜底。
    """
    speakers = sorted({s.speaker_id for s in segments if s.speaker_id is not None})
    if not speakers:
        return {}
    if len(speakers) == 1:
        return {speakers[0]: "说话人1"}

    scores: dict[int, float] = {spk: 0.0 for spk in speakers}
    for seg in segments:
        if seg.speaker_id is None:
            continue
        scores[seg.speaker_id] += (_cue_score(seg.text, SALES_CUES)
                                   - _cue_score(seg.text, CUSTOMER_CUES))

    best = max(scores, key=lambda k: scores[k])
    if scores[best] <= 0:
        # 无线索：以最早出现的说话人为销售
        first = min((s for s in segments if s.speaker_id is not None),
                    key=lambda s: (s.begin_ms or 0)).speaker_id
        best = first if first is not None else speakers[0]

    roles = {best: "销售"}
    customer_idx = 0
    for spk in speakers:
        if spk == best:
            continue
        customer_idx += 1
        roles[spk] = "客户" if customer_idx == 1 else f"客户{customer_idx}"
    return roles


def format_dialogue(segments: list[AsrSegment], roles: dict[int, str]) -> str:
    """把句子序列合并成 `说话人：内容` 的逐行对话，连续同一人合并为一行。"""
    lines: list[str] = []
    cur_role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if cur_role and buf:
            text = "".join(buf).strip()
            if text:
                lines.append(f"{cur_role}：{text}")

    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        role = roles.get(seg.speaker_id, "说话人1") if seg.speaker_id is not None else "说话人1"
        if role != cur_role:
            flush()
            cur_role, buf = role, [text]
        else:
            buf.append(text)
    flush()
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Provider：阿里云百炼 paraformer
# --------------------------------------------------------------------------- #
class DashScopeAsrProvider:
    name = "dashscope"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 language: str | None = None, diarization: bool | None = None,
                 poll_timeout: float | None = None) -> None:
        self.api_key = api_key or settings.api_key_for("dashscope")
        self.model = model or settings.asr_model
        self.language = language or settings.asr_language
        self.diarization = settings.asr_diarization if diarization is None else diarization
        self.poll_timeout = poll_timeout or settings.asr_poll_timeout

    def available(self) -> bool:
        return bool(self.api_key)

    # ---------------- 上传 ----------------
    def _get_policy(self) -> dict:
        url = f"{DASHSCOPE_BASE}/uploads"
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params={"action": "getPolicy", "model": self.model},
                              headers={"Authorization": f"Bearer {self.api_key}"})
        if resp.status_code != 200:
            raise AsrError(f"获取上传凭证失败：HTTP {resp.status_code} {resp.text[:200]}")
        return resp.json()["data"]

    def _upload(self, path: Path, attempts: int = 3) -> str:
        """上传到百炼临时存储，返回 `oss://` 临时 URL（有效期 48 小时）。"""
        policy = self._get_policy()
        key = f"{policy['upload_dir']}/{path.name}"
        form = {
            "key": key,
            "success_action_status": "200",
            "policy": policy["policy"],
            "signature": policy["signature"],
            "OSSAccessKeyId": policy["oss_access_key_id"],
            "x-oss-object-acl": policy.get("x_oss_object_acl", "private"),
            "x-oss-forbid-overwrite": str(policy.get("x_oss_forbid_overwrite", "true")).lower(),
        }
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                with path.open("rb") as fh, httpx.Client(timeout=120) as client:
                    resp = client.post(
                        policy["upload_host"], data=form,
                        files={"file": (path.name, fh, "application/octet-stream")})
                if resp.status_code in (200, 204):
                    return f"oss://{key}"
                last = AsrError(f"上传失败：HTTP {resp.status_code} {resp.text[:200]}")
            except Exception as exc:  # noqa: BLE001 - 网络层异常种类多
                last = exc
            time.sleep(1.5 * (attempt + 1))
        raise AsrError(f"音频上传失败：{last}")

    # ---------------- 提交与轮询 ----------------
    def _submit(self, file_url: str) -> str:
        params: dict = {"language": self.language}
        if self.diarization:
            params["diarization_enabled"] = True
        body = {"model": self.model, "input": {"file_urls": [file_url]},
                "parameters": params}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            # 使用 oss:// 临时 URL 时必须带该头，否则服务端无法解析文件
            "X-DashScope-OssResourceResolve": "enable",
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{DASHSCOPE_BASE}/services/audio/asr/transcription",
                               headers=headers, json=body)
        if resp.status_code != 200:
            raise AsrError(f"提交转写任务失败：HTTP {resp.status_code} {resp.text[:300]}")
        task_id = ((resp.json().get("output") or {}).get("task_id"))
        if not task_id:
            raise AsrError(f"提交转写任务未返回 task_id：{resp.text[:300]}")
        return task_id

    def _poll(self, task_id: str) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "X-DashScope-OssResourceResolve": "enable"}
        deadline = time.time() + self.poll_timeout
        interval = 2.0
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=30) as client:
                    resp = client.get(f"{DASHSCOPE_BASE}/tasks/{task_id}", headers=headers)
                data = resp.json().get("output") or {}
                status = data.get("task_status")
                if status == "SUCCEEDED":
                    return data
                if status == "FAILED":
                    raise AsrError(
                        f"转写任务失败：{data.get('code')} {data.get('message') or ''}".strip())
            except AsrError:
                raise
            except Exception as exc:  # noqa: BLE001 - 轮询网络抖动可忽略
                logger.warning("轮询转写任务异常：%s", exc)
            time.sleep(interval)
            interval = min(interval * 1.3, 6.0)
        raise AsrError(f"转写超时（>{self.poll_timeout:.0f}s），task_id={task_id}")

    # ---------------- 解析 ----------------
    @staticmethod
    def _parse(transcript: dict) -> tuple[list[AsrSegment], int | None]:
        segments: list[AsrSegment] = []
        for channel in transcript.get("transcripts") or []:
            for sent in channel.get("sentences") or []:
                text = str(sent.get("text") or "").strip()
                if not text:
                    continue
                segments.append(AsrSegment(
                    speaker_id=sent.get("speaker_id"),
                    begin_ms=sent.get("begin_time"),
                    end_ms=sent.get("end_time"),
                    text=text,
                ))
            if not segments and channel.get("text"):
                segments.append(AsrSegment(None, None, None, str(channel["text"]).strip()))
        duration = ((transcript.get("properties") or {})
                    .get("original_duration_in_milliseconds"))
        segments.sort(key=lambda s: (s.begin_ms if s.begin_ms is not None else 0))
        return segments, duration

    def transcribe(self, path: Path) -> AsrResult:
        file_url = self._upload(path)
        task_id = self._submit(file_url)
        output = self._poll(task_id)
        results = output.get("results") or []
        if not results:
            raise AsrError("转写任务成功但没有结果")
        transcription_url = results[0].get("transcription_url")
        if not transcription_url:
            raise AsrError(f"转写结果缺少 transcription_url：{str(results[0])[:300]}")
        with httpx.Client(timeout=60) as client:
            resp = client.get(transcription_url)
        if resp.status_code != 200:
            raise AsrError(f"下载转写结果失败：HTTP {resp.status_code}")
        transcript = resp.json()

        segments, duration = self._parse(transcript)
        if not segments:
            raise AsrError("转写结果为空，可能是纯静音或音频损坏")

        roles = assign_roles(segments)
        dialogue = format_dialogue(segments, roles)
        warnings: list[str] = []
        if len(roles) <= 1 and self.diarization:
            warnings.append("未识别出多个说话人，已按单一说话人处理，话轮边界仅供参考")
        if duration and duration > 2 * 3600 * 1000:
            warnings.append("音频超过 2 小时，说话人分离准确率可能下降")

        return AsrResult(
            dialogue=dialogue, segments=segments, provider=self.name, model=self.model,
            duration_ms=duration, speaker_count=len(roles), speaker_roles=roles,
            warnings=warnings, audio_file=path.name, raw_transcript=transcript,
        )


# --------------------------------------------------------------------------- #
# 门面
# --------------------------------------------------------------------------- #
class AsrService:
    def __init__(self, provider: str | None = None,
                 providers: dict[str, object] | None = None) -> None:
        self.provider_name = (provider or settings.asr_provider).lower()
        self._providers = providers or {DashScopeAsrProvider.name: DashScopeAsrProvider()}

    def validate(self, audio_path: str | Path) -> Path:
        path = Path(audio_path).expanduser()
        if not path.exists() or not path.is_file():
            raise AsrError(f"音频文件不存在：{path}")
        if path.suffix.lower() not in SUPPORTED_AUDIO_EXT:
            raise AsrError(
                f"不支持的音频格式：{path.suffix or '(无扩展名)'}，"
                f"支持 {', '.join(sorted(SUPPORTED_AUDIO_EXT))}")
        size = path.stat().st_size
        if size == 0:
            raise AsrError(f"音频文件为空：{path}")
        if size > MAX_AUDIO_BYTES:
            raise AsrError(f"音频文件超过 1GB：{size / 1024 / 1024:.0f}MB")
        return path

    def available_providers(self) -> list[str]:
        return [name for name, p in self._providers.items() if p.available()]

    def transcribe(self, audio_path: str | Path) -> AsrResult:
        path = self.validate(audio_path)
        provider = self._providers.get(self.provider_name)
        if provider is None:
            raise AsrError(f"未知的 ASR 通道：{self.provider_name}")
        if not provider.available():
            raise AsrError(
                f"ASR 通道 {self.provider_name} 不可用：缺少 DASHSCOPE_API_KEY")
        return provider.transcribe(path)
