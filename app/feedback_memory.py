"""反馈闭环：把人工订正沉淀为「别称 -> 工商全称」别名记忆。

这是人工干预工具的价值出口：审核人每改一次企业抬头，系统就学一条别名，
下次再遇到同一口语表述（如「恒美居」）时可以直接确定性命中，不再依赖联网检索。

实现要点：
- 数据源是 append-only 的 data/feedback.jsonl，不额外维护状态表，避免双写不一致
- 按文件 mtime 做缓存，订正后立即生效，无需重启服务
- 同一别名出现多次时取最近一次订正结果（人工的最新判断优先）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.rules import company_rules as cr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnedAlias:
    alias_core: str
    legal_name: str
    reviewer: str
    at: str
    count: int = 1


class FeedbackMemory:
    def __init__(self, feedback_path: Path) -> None:
        self.path = Path(feedback_path)
        self._cache: dict[str, LearnedAlias] = {}
        self._mtime: float = -1.0

    def _load_if_changed(self) -> None:
        if not self.path.exists():
            if self._mtime != 0.0:
                self._cache, self._mtime = {}, 0.0
            return
        mtime = self.path.stat().st_mtime
        if mtime == self._mtime:
            return
        cache: dict[str, LearnedAlias] = {}
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    change = (event.get("changes") or {}).get("company_name")
                    if not isinstance(change, dict):
                        continue
                    legal = str(change.get("to") or "").strip()
                    if not legal:
                        continue
                    original = (event.get("original") or {}).get("original_mention")
                    # 订正前的值就是客户口述的别名；没有则退用订正前的公司名
                    alias = str(original or change.get("from") or "").strip()
                    core = cr.strip_suffix(cr.clean_mention(alias)) if alias else ""
                    if len(core) < 2 or core == cr.strip_suffix(legal):
                        continue
                    prev = cache.get(core)
                    cache[core] = LearnedAlias(
                        alias_core=core, legal_name=legal,
                        reviewer=str(event.get("reviewer") or "unknown"),
                        at=str(event.get("at") or ""),
                        count=(prev.count + 1) if prev else 1,
                    )
        except OSError as exc:
            logger.warning("读取反馈记忆失败：%s", exc)
            return
        self._cache, self._mtime = cache, mtime

    def lookup(self, core: str) -> LearnedAlias | None:
        if not core:
            return None
        self._load_if_changed()
        return self._cache.get(core.lower())

    def all(self) -> list[LearnedAlias]:
        self._load_if_changed()
        return sorted(self._cache.values(), key=lambda x: (-x.count, x.at), reverse=False)

    def __len__(self) -> int:
        self._load_if_changed()
        return len(self._cache)
