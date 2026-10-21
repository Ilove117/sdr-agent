"""SQL 字段抽取规则（唯一事实来源）。

本模块定义 CRM 侧 SQL 线索每个字段的抽取口径：
- 字段来源（哪些通道可以贡献该字段）
- 规整策略（口语/转写噪声 -> 规范值）
- 是否关键字段、置信度权重（用于最终 confidence_score 融合）

工作流只「取证」，本模块「定口径」，两者解耦，便于单独演进与回归测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# --------------------------------------------------------------------------- #
# 1. 字段级抽取规则
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    name: str            # JSON 字段名
    label: str           # 中文名
    dtype: str           # 目标类型
    sources: tuple[str, ...]  # 允许的抽取来源通道
    rule: str            # 抽取与规整口径
    critical: bool = False    # 关键字段：缺失/低置信直接触发人工复核
    weight: float = 0.0       # 参与 confidence_score 融合的权重（总和 = 1.0）
    nullable: bool = True     # 允许为 null


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="company_name", label="企业抬头", dtype="str",
        sources=("dialogue", "industry_lexicon", "enterprise_kb", "web_search", "llm_reasoning"),
        rule="按「企业抬头还原流水线」还原工商全称：优先命名词典/企业库，未命中则联网检索，"
             "仍无则保留口语原称并降低置信度；输出需保留 original_mention 以便人工比对。",
        critical=True, weight=0.22, nullable=False,
    ),
    FieldSpec(
        name="intent_level", label="客户意向等级", dtype="enum(high|medium|low)",
        sources=("llm_signal_extraction", "intent_scorecard"),
        rule="由「意向评分卡」计算：预算/时间/规模/痛点/主动互动/决策权/资格完整度 7 个正向维度"
             "加权得分，再按「资格要素组合」裁决等级（分数仅用于同档排序），负向信号按类型分级扣分；"
             "输出 intent_reason 作为判断依据。",
        critical=True, weight=0.20, nullable=False,
    ),
    FieldSpec(
        name="phone", label="手机号", dtype="str",
        sources=("regex",),
        rule="正则抽取 11 位手机号；转写为掩码（138xxxx1234）时规范为 138****1234 并降置信度。",
        weight=0.10,
    ),
    FieldSpec(
        name="industry", label="行业", dtype="str",
        sources=("industry_lexicon", "llm_reasoning"),
        rule="以行业词库命中为主（可解释），LLM 仅做词库未覆盖时的兜底归类。",
        weight=0.08,
    ),
    FieldSpec(
        name="area", label="地区", dtype="str",
        sources=("regex", "llm_reasoning"),
        rule="抽取客户所在地市并补全省份（如「杭州」->「浙江省杭州市」）。",
        weight=0.07,
    ),
    FieldSpec(
        name="demand_summary", label="客户需求", dtype="str",
        sources=("llm_reasoning",),
        rule="归纳痛点 + 期望能力，一句话概括，不超过 80 字，不做营销修饰。",
        weight=0.07,
    ),
    FieldSpec(
        name="contact_name", label="联系人", dtype="str",
        sources=("dialogue", "llm_reasoning"),
        rule="从称呼还原：仅「姓氏+敬称」时只输出姓氏并降置信度；有全名则输出全名。",
        weight=0.05,
    ),
    FieldSpec(
        name="contact_title", label="职位", dtype="str",
        sources=("honorific_map",),
        rule="敬称映射表：总->负责人/总经理，经理->经理，老板->企业负责人，店长->店长。",
        weight=0.05,
    ),
    FieldSpec(
        name="mendian", label="门店数", dtype="int",
        sources=("regex",),
        rule="关键词（门店/店/分店/经销商）邻近数量抽取，中文数字需转阿拉伯数字；未提及为 null。",
        weight=0.04,
    ),
    FieldSpec(
        name="shejishi", label="设计师人数", dtype="int",
        sources=("regex",),
        rule="关键词（设计师/设计部/设计团队）邻近数量抽取，中文数字转阿拉伯数字；未提及为 null。",
        weight=0.04,
    ),
    FieldSpec(
        name="call_time", label="通话时间", dtype="str",
        sources=("metadata", "llm_reasoning"),
        rule="优先取通话元数据；文本内出现相对时间（下个月）仅作为 time_signal 线索，不直接当通话时间。",
        weight=0.03,
    ),
    FieldSpec(
        name="next_action", label="建议下一步", dtype="str",
        sources=("rule_based",),
        rule="按意向等级 + 字段缺口确定性生成，保证可复现；不依赖模型自由发挥。",
        weight=0.05,
    ),
)

FIELD_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in FIELD_SPECS}

# 派生字段（非抽取得来，由规则/融合计算产生）
DERIVED_FIELDS = ("intent_reason", "confidence_score", "need_human_review")


# --------------------------------------------------------------------------- #
# 2. 归一化工具
# --------------------------------------------------------------------------- #
_MOBILE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_MASKED_MOBILE_RE = re.compile(r"(?<!\d)(1[3-9]\d)([*xX×]{2,6})(\d{4})(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)")

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "两": 2, "贰": 2,
              "三": 3, "叁": 3, "四": 4, "肆": 4, "五": 5, "伍": 5,
              "六": 6, "陆": 6, "七": 7, "柒": 7, "八": 8, "捌": 8, "九": 9, "玖": 9}
_CN_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000, "万": 10000}
_CN_NUM_CHARS = "".join(_CN_DIGITS) + "".join(_CN_UNITS)

CITY_TO_PROVINCE: dict[str, str] = {
    "杭州": "浙江省", "宁波": "浙江省", "温州": "浙江省", "嘉兴": "浙江省", "金华": "浙江省",
    "绍兴": "浙江省", "台州": "浙江省", "湖州": "浙江省",
    "上海": "", "北京": "", "天津": "", "重庆": "",
    "广州": "广东省", "深圳": "广东省", "佛山": "广东省", "东莞": "广东省", "中山": "广东省",
    "珠海": "广东省", "惠州": "广东省",
    "南京": "江苏省", "苏州": "江苏省", "无锡": "江苏省", "常州": "江苏省", "南通": "江苏省",
    "徐州": "江苏省",
    "成都": "四川省", "绵阳": "四川省",
    "武汉": "湖北省", "长沙": "湖南省", "郑州": "河南省", "西安": "陕西省",
    "济南": "山东省", "青岛": "山东省", "烟台": "山东省", "临沂": "山东省",
    "合肥": "安徽省", "福州": "福建省", "厦门": "福建省", "泉州": "福建省",
    "南昌": "江西省", "昆明": "云南省", "贵阳": "贵州省", "南宁": "广西壮族自治区",
    "沈阳": "辽宁省", "大连": "辽宁省", "哈尔滨": "黑龙江省", "长春": "吉林省",
    "石家庄": "河北省", "太原": "山西省", "兰州": "甘肃省", "乌鲁木齐": "新疆维吾尔自治区",
    "海口": "海南省", "呼和浩特": "内蒙古自治区",
}

# 敬称 -> 职位映射
HONORIFIC_TITLE: dict[str, str] = {
    "总": "负责人/总经理",
    "经理": "经理",
    "总监": "总监",
    "老板": "企业负责人",
    "店长": "店长",
    "院长": "院长",
    "所长": "所长",
    "主任": "主任",
    "主管": "主管",
    "设计师": "设计师",
}

_INTENT_SYNONYMS: dict[str, str] = {
    "high": "high", "高": "high", "高意向": "high", "强": "high", "强意向": "high",
    "a": "high", "hot": "high",
    "medium": "medium", "中": "medium", "中意向": "medium", "一般": "medium",
    "b": "medium", "warm": "medium",
    "low": "low", "低": "low", "低意向": "low", "弱意向": "low", "c": "low", "cold": "low",
}


def parse_cn_number(text: str) -> int | None:
    """中文数字转整数，支持到「万」。例：十五->15，一百二十->120，两->2。"""
    if not text:
        return None
    s = text.strip()
    if s.isdigit():
        return int(s)
    if not all(ch in _CN_NUM_CHARS for ch in s):
        return None
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
            number = 0
    return total + section + number


def extract_phone(text: str) -> tuple[str | None, float, str | None]:
    """抽取手机号，返回 (规范值, 字段置信度, 原始文本)。"""
    if not text:
        return None, 0.0, None
    m = _MOBILE_RE.search(text)
    if m:
        return m.group(1), 0.95, m.group(0)
    m = _MASKED_MOBILE_RE.search(text)
    if m:
        mask = "*" * len(m.group(2))
        return f"{m.group(1)}{mask}{m.group(3)}", 0.55, m.group(0)
    m = _LANDLINE_RE.search(text)
    if m:
        return m.group(1), 0.7, m.group(0)
    return None, 0.0, None


_NUM_EXPR = r"(?:\d{1,4}|[零一二两三四五六七八九十百千]{1,4})"


def extract_quantity(text: str, unit_regex: str, context_keywords: tuple[str, ...] = (),
                     window: int = 18, measures: str = "个名位家间台套") -> tuple[int | None, float, str | None]:
    """抽取「数量 + 量词 + 单位」并做语境校验。

    采用 unit 归一模式（而非盲目找数字），可有效避免把「一年几万块」「一下需求」
    这类无关数字误当成人数/门店数。语序兼容「15 个人」与「设计部大概 15 个人」；
    量词区兼容概数词「多」（二十多家门店 -> 20，取下界并降低置信度）。

    返回 (数值, 置信度, 证据片段)。
    """
    if not text:
        return None, 0.0, None
    pattern = re.compile(rf"(?P<n>{_NUM_EXPR})\s*[{measures}多]*\s*(?:{unit_regex})")
    best: tuple[int, int, float, str] | None = None
    for m in pattern.finditer(text):
        val = parse_cn_number(m.group("n"))
        if val is None or not (0 < val <= 100000):
            continue
        dist = 0
        if context_keywords:
            anchors = [mm.start() for kw in context_keywords
                       for mm in re.finditer(re.escape(kw), text)]
            if not anchors:
                continue
            dist = min(min(abs(m.start() - a), abs(m.end() - a)) for a in anchors)
            if dist > window:
                continue
        if best is None or dist < best[0]:
            head = max(0, m.start() - 8)
            conf = 0.9 if dist <= 8 else 0.75
            if "多" in m.group(0):
                conf -= 0.1  # 概数，取下界
            best = (dist, val, conf, text[head:m.end() + 2].strip())
    if best is None:
        return None, 0.0, None
    return best[1], best[2], best[3]


# 兼容旧命名
extract_count_near = extract_quantity


# 「X那边/X这边/X当地」兜底模式需排除的代词，否则「我们这边」会被当成地区
AREA_STOPWORDS = frozenset({
    "我们", "咱们", "他们", "你们", "咱", "我", "你", "他", "各位", "大家",
    "这边", "那边", "当地", "地区", "人家", "自己",
})


def extract_area(text: str) -> tuple[str | None, float, str | None]:
    """抽取地区并补全省份。

    只用两类高精度证据：城市词表、显式行政区划后缀（省/市/自治区）。
    刻意不做「X那边」这类模糊兜底——它会把「我们好莱客这边」误判成地区「们好莱客」，
    词表未覆盖的情况交给 LLM 的 area_text 兜底。
    """
    if not text:
        return None, 0.0, None
    for city, province in CITY_TO_PROVINCE.items():
        idx = text.find(city)
        if idx >= 0:
            full = f"{province}{city}" if province else city
            if province and city not in ("上海", "北京", "天津", "重庆"):
                full = f"{province}{city}市"
            return full, 0.8, text[max(0, idx - 6): idx + len(city) + 4]
    m = re.search(r"([\u4e00-\u9fa5]{2,8}?(?:省|市|自治区))", text)
    if m and m.group(1) not in AREA_STOPWORDS:
        return m.group(1), 0.7, m.group(0)
    return None, 0.0, None


def split_contact(mention: str) -> tuple[str | None, str | None, float]:
    """从「张总」类称呼还原 (联系人, 职位, 置信度)。"""
    if not mention:
        return None, None, 0.0
    s = mention.strip()
    for honor in sorted(HONORIFIC_TITLE, key=len, reverse=True):
        if s.endswith(honor):
            prefix = s[: -len(honor)].strip()
            title = HONORIFIC_TITLE[honor]
            if not prefix:
                return None, title, 0.4
            conf = 0.55 if len(prefix) == 1 else 0.75
            name = prefix if len(prefix) <= 3 else prefix[:1]
            return name, title, conf
    # 无敬称：整段视为姓名但置信度低
    return (s if 1 <= len(s) <= 4 else None), None, 0.4


def normalize_intent_level(raw: Any) -> str | None:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    return _INTENT_SYNONYMS.get(key)


def validate_lead(lead: dict[str, Any]) -> list[str]:
    """输出前校验：返回问题清单（空列表 = 通过）。"""
    issues: list[str] = []
    for spec in FIELD_SPECS:
        val = lead.get(spec.name)
        if spec.nullable:
            continue
        if val in (None, "", []):
            issues.append(f"必填字段缺失：{spec.label}({spec.name})")

    if lead.get("company_name") in ("未识别", "未知"):
        issues.append("企业抬头未能还原，需人工补录")

    level = lead.get("intent_level")
    if level not in ("high", "medium", "low"):
        issues.append(f"intent_level 非法：{level!r}")

    if lead.get("phone"):
        p = str(lead["phone"])
        if not (re.fullmatch(r"1[3-9]\d\*{2,6}\d{4}", p) or re.fullmatch(r"1[3-9]\d{9}", p)
                or re.fullmatch(r"0\d{2,3}-?\d{7,8}", p)):
            issues.append(f"phone 格式异常：{p!r}")

    for int_field in ("mendian", "shejishi"):
        v = lead.get(int_field)
        if v is not None and not isinstance(v, int):
            issues.append(f"{int_field} 应为整数：{v!r}")

    score = lead.get("confidence_score")
    if score is not None and not (0.0 <= float(score) <= 1.0):
        issues.append(f"confidence_score 越界：{score!r}")
    return issues
