"""通话文本结构解析：区分「销售」与「客户」话轮。

为什么需要：销售口中的公司名几乎总是自家公司（「我是酷家乐的小王」），
如果把它当成客户企业去检索，会直接把错误结论写进 CRM。
因此企业抬头、地区、规模这类「客户侧事实」只在客户话轮内取证。
"""

from __future__ import annotations

import re

SALES_KEYWORDS = ("销售", "客服", "坐席", "顾问", "销售顾问", "BD", "SDR")
CUSTOMER_KEYWORDS = ("客户", "顾客", "用户", "甲方", "买家")

_LINE_RE = re.compile(r"^(?P<speaker>[^\s：:]{1,8})[：:]\s*(?P<text>.*)$")

# 供应商/自家品牌，出现在客户话轮里也不应当作客户企业名
VENDOR_NAMES = frozenset({
    "酷家乐", "群核", "群核科技", "群核信息技术", "群核信息技术有限公司",
    "酷家乐云设计", "酷家乐企业版",
})


def split_turns(dialogue: str) -> list[tuple[str, str]]:
    """返回 [(角色, 文本)]，角色取 sales/customer/unknown。"""
    turns: list[tuple[str, str]] = []
    for raw in (dialogue or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            turns.append(("unknown", line))
            continue
        speaker, text = m.group("speaker"), m.group("text").strip()
        if any(k in speaker for k in CUSTOMER_KEYWORDS):
            role = "customer"
        elif any(k in speaker for k in SALES_KEYWORDS):
            role = "sales"
        else:
            role = "unknown"
        turns.append((role, text))
    return turns


def _joined(turns: list[tuple[str, str]], roles: tuple[str, ...]) -> str:
    return "\n".join(t for r, t in turns if r in roles)


def customer_text(dialogue: str) -> str:
    """客户话轮文本；若整段没有说话人标记，则退回全文（兼容单块输入）。"""
    turns = split_turns(dialogue)
    if not turns:
        return ""
    if not any(r != "unknown" for r, _ in turns):
        return dialogue
    text = _joined(turns, ("customer",))
    return text or dialogue


def sales_text(dialogue: str) -> str:
    turns = split_turns(dialogue)
    return _joined(turns, ("sales",))


def is_vendor_name(name: str) -> bool:
    if not name:
        return False
    core = name.strip()
    if core in VENDOR_NAMES:
        return True
    return any(core in v or v in core for v in VENDOR_NAMES if len(v) >= 3 and len(core) >= 3)
