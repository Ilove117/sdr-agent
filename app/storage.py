"""线索存储与人工复核状态管理。

- data/leads.jsonl    线索全量记录（含原始通话、结构化字段、诊断元数据、复核状态）
- data/feedback.jsonl 人工订正事件（append-only），是「反馈闭环」的原料：
  可回灌为少样本示例、沉淀企业别名、驱动评分卡阈值回归。

并发模型：进程内加锁 + 临时文件原子替换，避免 Web 端并发写坏文件。
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from app.schema import SQL_FIELDS, LeadRecord

VALID_STATUS = ("pending", "approved", "corrected", "rejected")


class LeadStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "leads.jsonl"
        self.feedback_path = self.data_dir / "feedback.jsonl"
        self._lock = threading.RLock()
        self._records: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._load()

    # ---------------- 持久化 ----------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id"):
                    self._records[rec["id"]] = rec

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in self._records.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- 写入 ----------------
    def save(self, *, lead: LeadRecord, meta: dict[str, Any], dialogue: str,
             call_id: str | None = None, record_id: str | None = None) -> dict[str, Any]:
        rid = record_id or call_id or f"call-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
        with self._lock:
            existing = self._records.get(rid)
            record = {
                "id": rid,
                "created_at": existing.get("created_at") if existing else self._now(),
                "updated_at": self._now(),
                "call_time": lead.call_time,
                "dialogue": dialogue,
                "lead": lead.to_sql_json(),
                "meta": meta,
                "review": existing.get("review") if existing else {
                    "status": "pending" if lead.need_human_review else "auto_passed",
                    "reviewer": None,
                    "reviewed_at": None,
                    "corrected_fields": {},
                    "note": None,
                    "history": [],
                },
            }
            self._records[rid] = record
            self._flush()
            return record

    def _log_feedback(self, event: dict[str, Any]) -> None:
        with self.feedback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ---------------- 复核 ----------------
    def review_status(self, record: dict[str, Any]) -> str:
        return (record.get("review") or {}).get("status", "pending")

    def list_records(self, *, status: str | None = None, level: str | None = None,
                     need_review_only: bool = False, query: str | None = None,
                     sort: str = "created_at_desc", limit: int = 200,
                     offset: int = 0, with_dialogue: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._records.values())

        if need_review_only:
            items = [r for r in items if (r.get("lead") or {}).get("need_human_review")]
        if status and status != "all":
            items = [r for r in items if self.review_status(r) == status]
        if level and level != "all":
            items = [r for r in items if (r.get("lead") or {}).get("intent_level") == level]
        if query:
            q = query.strip().lower()
            items = [r for r in items
                     if q in str((r.get("lead") or {}).get("company_name", "")).lower()
                     or q in str((r.get("lead") or {}).get("intent_reason", "")).lower()
                     or q in str(r.get("dialogue", "")).lower()]

        reverse = sort.endswith("desc")
        if sort.startswith("confidence"):
            items.sort(key=lambda r: (r.get("lead") or {}).get("confidence_score", 0) or 0,
                       reverse=reverse)
        else:
            items.sort(key=lambda r: r.get("created_at", ""), reverse=reverse)

        page = items[offset: offset + limit]
        if with_dialogue:
            return page
        return [{k: v for k, v in r.items() if k != "dialogue"} for r in page]

    def get(self, record_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._records.get(record_id)

    def apply_review(self, record_id: str, *, changes: dict[str, Any] | None = None,
                     reviewer: str = "anonymous", note: str | None = None,
                     status: str | None = None) -> dict[str, Any] | None:
        """执行一次人工复核：可仅改状态，也可携带字段订正。"""
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return None

            lead = dict(record.get("lead") or {})
            changes = dict(changes or {})
            invalid = [k for k in changes if k not in SQL_FIELDS]
            if invalid:
                raise ValueError(f"非法字段：{invalid}")

            diff: dict[str, dict[str, Any]] = {}
            for field, new_value in changes.items():
                old_value = lead.get(field)
                if isinstance(new_value, str):
                    new_value = new_value.strip()
                    if field in ("mendian", "shejishi"):
                        new_value = new_value or None
                if field in ("mendian", "shejishi") and new_value is not None:
                    new_value = int(new_value)
                if new_value != old_value:
                    diff[field] = {"from": old_value, "to": new_value}
                    lead[field] = new_value

            if status is None:
                status = "corrected" if diff else "approved"
            if status not in VALID_STATUS:
                raise ValueError(f"非法复核状态：{status}")

            corrections = dict((record.get("review") or {}).get("corrected_fields") or {})
            corrections.update(diff)

            # 人工确认后即视为已解决，不再挂起复核标记
            lead["need_human_review"] = status == "pending"

            history = list((record.get("review") or {}).get("history") or [])
            history.append({
                "at": self._now(), "reviewer": reviewer, "action": status,
                "changes": diff, "note": note,
            })

            record["lead"] = lead
            record["updated_at"] = self._now()
            record["review"] = {
                "status": status,
                "reviewer": reviewer,
                "reviewed_at": self._now(),
                "corrected_fields": corrections,
                "note": note,
                "history": history,
            }
            self._records[record_id] = record
            self._flush()

            self._log_feedback({
                "at": record["updated_at"], "id": record_id, "reviewer": reviewer,
                "action": status, "changes": diff, "note": note,
                "original": dict(record.get("meta", {}).get("company_resolution") or {}),
                "intent_level": lead.get("intent_level"),
                "company_name": lead.get("company_name"),
                "dialogue": record.get("dialogue", "")[:1200],
            })
            return record

    def batch_review(self, ids: Iterable[str], *, changes: dict[str, Any] | None = None,
                     reviewer: str = "anonymous", note: str | None = None,
                     status: str | None = None) -> dict[str, Any]:
        """批量订正：如「批量把某批线索的意向等级改为 medium」。"""
        ok, failed = [], []
        for rid in ids:
            try:
                rec = self.apply_review(rid, changes=changes, reviewer=reviewer,
                                        note=note, status=status)
                (ok if rec else failed).append(rid)
            except (ValueError, TypeError) as exc:
                failed.append({"id": rid, "error": str(exc)})
        return {"updated": ok, "failed": failed, "count": len(ok)}

    # ---------------- 统计 / 导出 ----------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._records.values())
        total = len(items)
        by_level: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        by_status: dict[str, int] = {}
        conf_sum, review_count = 0.0, 0
        for r in items:
            lead = r.get("lead") or {}
            lvl = lead.get("intent_level")
            if lvl in by_level:
                by_level[lvl] += 1
            st = self.review_status(r)
            by_status[st] = by_status.get(st, 0) + 1
            conf_sum += float(lead.get("confidence_score") or 0)
            if lead.get("need_human_review"):
                review_count += 1
        return {
            "total": total,
            "pending_review": review_count,
            "avg_confidence": round(conf_sum / total, 3) if total else 0.0,
            "by_intent_level": by_level,
            "by_review_status": by_status,
        }

    def export_sql(self, *, only_approved: bool = True, limit: int = 1000) -> list[dict[str, Any]]:
        records = self.list_records(limit=limit, with_dialogue=False)
        out: list[dict[str, Any]] = []
        for r in records:
            if only_approved and self.review_status(r) == "pending":
                continue
            row = {"lead_id": r["id"]}
            row.update(r.get("lead") or {})
            out.append(row)
        return out

    def kb_candidates(self) -> list[dict[str, Any]]:
        """从人工订正中挖掘「企业别名 -> 工商全称」弱标注，供企业库增补。"""
        if not self.feedback_path.exists():
            return []
        seen: dict[str, dict[str, Any]] = {}
        with self.feedback_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                change = (ev.get("changes") or {}).get("company_name")
                if not change:
                    continue
                mention = (ev.get("original") or {}).get("original_mention")
                legal = change.get("to")
                if mention and legal and mention != legal:
                    seen[legal] = {
                        "legal_name": legal, "alias_mention": mention,
                        "reviewer": ev.get("reviewer"), "at": ev.get("at"),
                        "count": (seen.get(legal, {}) or {}).get("count", 0) + 1,
                    }
        return list(seen.values())
