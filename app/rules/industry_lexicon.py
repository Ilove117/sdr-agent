"""行业词库：口语表述 -> 规范行业。

用途有二：
1. 抽取 industry 字段（可解释、命中即高置信）；
2. 作为「企业抬头还原」的先验知识——限制了候选企业应属于的行业范围，
   用于对联网/LLM 返回的候选企业做一致性校验（跨行业候选降权）。
"""

from __future__ import annotations

# 规范行业 -> 口语/别称关键词（长词优先匹配，避免「定制家居」被「家居」抢走）
INDUSTRY_LEXICON: dict[str, tuple[str, ...]] = {
    "定制家居": (
        "全屋定制", "定制家居", "高端定制", "高定", "门墙柜", "全屋整装", "定制衣柜",
        "定制橱柜", "整木定制", "整木", "护墙板", "板式家具", "实木定制", "定制家具",
        "衣柜", "橱柜", "木门", "榻榻米", "家居定制", "定制柜",
    ),
    "家装设计": (
        "家装公司", "装修公司", "装饰公司", "室内设计", "设计工作室", "装修", "装饰",
        "家装", "硬装", "软装", "整装", "工装", "设计公司", "装修设计",
    ),
    "建材": (
        "建筑陶瓷", "陶瓷", "瓷砖", "卫浴", "地板", "木地板", "门窗", "系统门窗", "涂料",
        "五金", "石材", "吊顶", "集成吊顶", "墙纸", "墙布", "灯具", "照明", "家居建材",
        "板材", "生态板", "木饰面", "板材定制", "人造板", "石材加工",
    ),
    "家具制造": (
        "家具厂", "家具制造", "成品家具", "软体家具", "实木家具", "沙发", "床垫",
        "床具", "家具", "家居",
    ),
    "房地产": (
        "房地产", "地产", "置业", "楼盘", "开发商", "商业地产", "产业地产", "物业",
    ),
    "建筑设计": (
        "建筑设计", "设计院", "规划院", "勘察设计", "市政设计", "院所", "建筑规划",
    ),
    "商业展陈": (
        "展柜", "展陈", "商业空间", "陈列", "道具", "展厅设计", "店铺装修",
    ),
    "智能家居": (
        "智能家居", "全屋智能", "智能门锁", "智能家电",
    ),
    "办公家具": (
        "办公家具", "办公空间", "工位", "办公桌椅", "办公环境",
    ),
    "家电厨电": (
        "集成灶", "厨电", "厨房电器", "家电",
    ),
}

# 行业 -> 是否酷家乐核心目标行业（用于意向度「规模/匹配度」与候选企业行业校验）
CORE_TARGET_INDUSTRIES = frozenset({
    "定制家居", "家装设计", "建材", "家具制造", "商业展陈", "办公家具", "智能家居",
})

PRODUCT_TERMS = (
    "企业版", "参数化设计", "一键出图", "出图", "效果图", "渲染", "对接生产",
    "设计软件", "云设计", "SSO", "API", "酷家乐", "设计工具",
)

# 长词优先，保证「全屋定制」不会被「家居」抢占
_SORTED_KEYWORDS: list[tuple[str, str]] = sorted(
    ((kw, ind) for ind, kws in INDUSTRY_LEXICON.items() for kw in kws),
    key=lambda kv: len(kv[0]), reverse=True,
)


# 过于泛化的关键词：出现在工商全称里并不能说明行业
# （如「XX家居有限公司」可能做橱柜、板材、家具，需靠更具体的词判断）
GENERIC_KEYWORDS = frozenset({"家居", "家具", "设计", "装饰", "建材", "实业", "科技", "技术"})


def classify_industry(text: str, *, strict: bool = False) -> tuple[str | None, tuple[str, ...], float]:
    """关键词命中分类，返回 (规范行业, 命中词, 置信度)。

    打分与择优规则（保证可解释且稳定）：
      1) 每个行业累加其命中关键词的字符长度（长词权重更高）
      2) 长度相同时，优先「客户最先自我描述」的那个行业 ——
         因为「我们做板材的……想做整装方案」里，板材才是主营业务，整装是诉求
      3) strict=True 时忽略泛化词，用于校验工商全称所属行业（避免「XX家居有限公司」
         被一律判成家具制造）
    """
    if not text:
        return None, (), 0.0
    hits: list[str] = []
    scores: dict[str, float] = {}
    first_pos: dict[str, int] = {}
    for kw, ind in _SORTED_KEYWORDS:
        if strict and kw in GENERIC_KEYWORDS:
            continue
        idx = text.find(kw)
        if idx < 0:
            continue
        if kw not in hits:
            hits.append(kw)
        scores[ind] = scores.get(ind, 0.0) + len(kw)
        first_pos[ind] = min(first_pos.get(ind, idx), idx)
    if not scores:
        return None, (), 0.0

    def rank(ind: str) -> tuple[float, int]:
        return (scores[ind], -first_pos[ind])

    best = max(scores, key=rank)
    hit_terms = tuple(h for h in hits if h in INDUSTRY_LEXICON.get(best, ()))
    conf = min(0.95, 0.5 + 0.08 * len(hit_terms) + (0.15 if best in CORE_TARGET_INDUSTRIES else 0.0))
    return best, hit_terms, round(conf, 3)


def is_core_target(industry: str | None) -> bool:
    return bool(industry) and industry in CORE_TARGET_INDUSTRIES


def industry_of_keyword(term: str) -> str | None:
    for kw, ind in _SORTED_KEYWORDS:
        if kw == term:
            return ind
    return None
