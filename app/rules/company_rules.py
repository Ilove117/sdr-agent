"""企业抬头规整规则：从口语文本提取候选提及、生成别称变体、计算相似度。

这是「企业抬头识别」难点的基础设施：
客户说出口的往往不是工商全称（简称/品牌名/别称/转写错误），
所以需要先枚举「这句话可能指哪些写法」，再去企业库/联网检索里对齐。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# 工商主体后缀 / 常见行业后缀，用于剥离出「核心词」
SUFFIX_TOKENS: tuple[str, ...] = (
    "股份有限公司", "有限责任公司", "有限公司", "集团有限公司", "集团", "股份公司",
    "实业发展", "实业", "家居科技", "家居", "家具", "科技", "信息技术", "装饰工程",
    "装饰", "设计", "建材", "控股", "企业", "品牌", "公司", "旗舰店", "工厂",
)

# 需从提及中剔除的地域/修饰前缀
REGION_PREFIX = re.compile(
    r"^(?:我们|我们这|我这边|我们这边|这边|那边|本地|当地|"
    r"浙江|杭州|上海|江苏|南京|苏州|广东|广州|深圳|北京|天津|重庆|"
    r"成都|武汉|长沙|西安|郑州|济南|青岛|合肥|福州|厦门|宁波|佛山|东莞)*"
)

_SPEECH_NOISE = re.compile(r"^(?:一个|一家|做|是|叫|叫做|叫个|名字是|名字叫|这|那|的|吧|呢)+")

_QUOTED = re.compile(r"[“\"'‘「《]([^”\"'’」》]{2,30})[”\"'’」》]")
_CALLED = re.compile(r"(?:叫做|叫|名字是|名字叫|简称(?:是|为)?|别称(?:是|为)?|品牌(?:是|叫|为)?)\s*([\u4e00-\u9fa5A-Za-z0-9]{2,20})")
_LEGAL_FORM = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,24}?(?:股份有限公司|有限责任公司|有限公司|集团|实业有限公司|实业|控股))")
_BE_WE = re.compile(r"(?:我们是|我们这边是|我们公司是|公司叫|公司是|咱是|我是)\s*([\u4e00-\u9fa5A-Za-z0-9]{2,20})")


@dataclass
class CompanyMention:
    raw: str                     # 原文提及
    normalized_core: str         # 剥离后缀后的核心词
    pattern: str                 # 命中模式（quoted/called/legal_form/be_we）
    confidence: float            # 提及本身的可信度（不含后续对齐）
    span: tuple[int, int] = (0, 0)
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw, "normalized_core": self.normalized_core,
            "pattern": self.pattern, "confidence": round(self.confidence, 3),
            "span": list(self.span), "aliases": list(self.aliases),
        }


_PATTERN_CONFIDENCE = {
    "legal_form": 0.92,   # 已带工商后缀，几乎可直接用
    "quoted": 0.85,       # 客户主动说出的品牌名
    "called": 0.78,       # 「叫XXX」
    "be_we": 0.70,        # 「我们是XXX」
}


def strip_suffix(name: str) -> str:
    """剥离工商/行业后缀，得到核心词。"""
    core = name.strip()
    changed = True
    while changed:
        changed = False
        for tok in sorted(SUFFIX_TOKENS, key=len, reverse=True):
            if core.endswith(tok) and len(core) > len(tok):
                core = core[: -len(tok)]
                changed = True
    return core


def clean_mention(raw: str) -> str:
    """去掉口语句式噪声（「一个做…的」「叫」等）与地域前缀。"""
    s = raw.strip().strip("，。,.、;；:：!！?？ ")
    s = _SPEECH_NOISE.sub("", s)
    s = REGION_PREFIX.sub("", s)
    s = re.sub(r"(?:的|吧|呢|啊)$", "", s)
    return s.strip()


def alias_variants(name: str) -> tuple[str, ...]:
    """基于「核心词 + 常见后缀」枚举别称变体，覆盖「木里木外 -> 木里木外家居」这类别名。"""
    core = strip_suffix(name)
    if not core:
        return (name,)
    variants = {
        name, core,
        f"{core}家居", f"{core}家具", f"{core}家装",
        f"{core}有限公司", f"{core}家居有限公司", f"{core}股份有限公司",
        f"{core}集团", f"{core}实业", f"{core}实业有限公司",
        f"{core}高端定制", f"{core}全屋定制",
    }
    return tuple(sorted(v for v in variants if v))


def extract_mentions(text: str) -> list[CompanyMention]:
    """从通话文本中枚举企业提及，按置信度降序去重。"""
    found: dict[str, CompanyMention] = {}

    def _add(raw: str, pattern: str, span: tuple[int, int]) -> None:
        cleaned = clean_mention(raw)
        if len(cleaned) < 2 or not re.search(r"[\u4e00-\u9fa5A-Za-z]", cleaned):
            return
        # 过滤明显非企业名的短语
        if cleaned in {"产品", "方案", "软件", "需求", "价格", "预算", "企业版", "设计部"}:
            return
        core = strip_suffix(cleaned)
        if len(core) < 2:
            return
        conf = _PATTERN_CONFIDENCE.get(pattern, 0.6)
        if cleaned.endswith(("有限公司", "股份有限公司", "集团")):
            conf = max(conf, 0.92)
        prev = found.get(core)
        if prev and prev.confidence >= conf:
            return
        found[core] = CompanyMention(
            raw=cleaned, normalized_core=core, pattern=pattern,
            confidence=conf, span=span, aliases=alias_variants(cleaned),
        )

    for m in _QUOTED.finditer(text):
        _add(m.group(1), "quoted", m.span(1))
    for m in _LEGAL_FORM.finditer(text):
        _add(m.group(1), "legal_form", m.span(1))
    for m in _CALLED.finditer(text):
        _add(m.group(1), "called", m.span(1))
    for m in _BE_WE.finditer(text):
        _add(m.group(1), "be_we", m.span(1))

    return sorted(found.values(), key=lambda x: x.confidence, reverse=True)


def _bigrams(s: str) -> set[str]:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _lcs_len(a: str, b: str) -> int:
    """最长公共子序列长度（中文品牌常是工商全称的子序列，如「木里木外」⊂「上海木里…外」）。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, ch2 in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if ch == ch2 else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """中文企业名相似度。

    四种信号取最大值，覆盖不同的「简称 -> 全称」形态：
      1) 核心词完全相等                          -> 1.0
      2) 核心词互为包含（品牌 + 行业后缀）        -> 0.80~1.0
      3) 二元组 Jaccard 与序列相似度加权（抗转写错误）
      4) 核心词覆盖率（品牌字分散出现在全称中）    -> 0.45~0.85
    """
    if not a or not b:
        return 0.0
    na, nb = strip_suffix(a), strip_suffix(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    scores: list[float] = []
    if na in nb or nb in na:
        shorter, longer = sorted((len(na), len(nb)))
        scores.append(0.80 + 0.2 * (shorter / longer))

    jac = len(_bigrams(na) & _bigrams(nb)) / max(1, len(_bigrams(na) | _bigrams(nb)))
    seq = SequenceMatcher(None, na, nb).ratio()
    scores.append(0.35 * jac + 0.65 * seq)

    shorter_len = min(len(na), len(nb))
    lcs = _lcs_len(na, nb)
    if shorter_len >= 2 and lcs >= max(2, 0.5 * shorter_len):
        scores.append(0.45 + 0.40 * (lcs / shorter_len))

    return round(max(scores), 4)
