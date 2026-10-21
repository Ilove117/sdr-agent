"""提示词层。

原则：LLM 只负责「取证」与「语义归一」，不负责「定级」与「定口径」。
因此每条提示都硬性要求：
1. 所有结论必须附原文引用（quote），杜绝幻觉；
2. 必须输出严格 JSON；
3. 不确定就显式给出 null / 低置信，而不是猜。
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# 1. 意图识别 + 信息抽取（一次调用产出全部语义证据）
# --------------------------------------------------------------------------- #
EXTRACTION_SYSTEM = """你是酷家乐销售通话质检与线索挖掘专家，服务对象是家居/建材行业 SaaS（云设计平台）。
你的任务：从一通销售通话的转写文本中，抽取业务事实与意向信号，输出严格 JSON。

硬性规则：
1) 只依据文本中真实出现的内容作答，禁止脑补。每一项判断都必须给出原文引用 quote（逐字摘录，不要改写）。
2) 文本中没有的信息，填 null 或空数组，不要编造。
3) 预算/时间/规模这类字段，必须区分「明确」与「模糊」：
   - 预算 specificity: clear=给出明确金额或区间且客户认可 / range=给出大概区间 / vague=只提预算没说数
   - 时间 urgency: immediate=已明确立即或本月内 / near_term=明确近期（下个月、近两个月内） / vague=只说"以后""再看看"
4) 负向信号必须如实记录（随便问问、已经有供应商、明确拒绝、预算不足、拖延、价格异议），不要因为通话气氛好就忽略。
5) demand_summary 用一句中文概括客户痛点+期望，不超过 60 字，不要营销话术。
6) 只能抽取「客户」说的话。「我是酷家乐的小王」这类销售自报家门属于我方公司，绝不能当作客户企业名。
7) 客户没有明确说出企业名、行业或地区时，对应字段填 null —— 宁缺勿猜。

输出 JSON 结构（字段必须齐全，不要增删顶层键）：
{
  "intent_recognition": {
    "is_sales_call": true,
    "call_intent": "new_inquiry|requirement_confirm|pricing|post_sale|invalid|other",
    "customer_attitude": "positive|neutral|negative"
  },
  "intent_signals": {
    "budget": {"mentioned": bool, "specificity": "clear|range|vague", "value_text": str|null, "quote": str|null},
    "timeline": {"mentioned": bool, "urgency": "immediate|near_term|vague", "value_text": str|null, "quote": str|null},
    "scale": {"designers": int|null, "stores": int|null, "quote": str|null},
    "pain_points": [{"pain_text": str, "quote": str}],
    "active_engagement": [{"type": "ask_pricing|ask_demo|ask_solution|ask_case|provide_contact|agree_next_step|accept_meeting|ask_timeline|compare_vendor", "quote": str}],
    "negative_signals": [{"type": "just_looking|no_budget|already_locked|reject|brush_off|wrong_person|delay|not_interested|price_objection", "quote": str}],
    "decision_maker": {"is_decision_maker": bool|null, "quote": str|null}
  },
  "business_info": {
    "company_mention": str|null,
    "company_mention_quote": str|null,
    "industry_text": str|null,
    "area_text": str|null,
    "demand_summary": str
  },
  "contact_info": {
    "contact_mention": str|null,
    "phone": str|null
  }
}
只输出 JSON，不要任何解释或 markdown 围栏。"""

EXTRACTION_USER = """【通话记录】
<<DIALOGUE>>

请按上述结构输出 JSON。"""

EXTRACTION_FEWSHOT_DIALOGUE = """销售：您好，请问是李总吗？我是酷家乐的小张。
客户：是的。我们这边在佛山做瓷砖的，叫"金鹏陶瓷"。
销售：好的，目前门店大概多少家？
客户：佛山这边三家，还在扩。
销售：那设计这块怎么做的？
客户：我们就用 CAD，画图太慢了，而且客户看不到效果，签单率低。
销售：我们企业版可以快速出效果图。什么时候想上线？
客户：先了解下吧，今年预算还没批，明年再说。
销售：明白，我先加您微信。
客户：行，13800001111。"""

EXTRACTION_FEWSHOT_OUTPUT = """{
  "intent_recognition": {"is_sales_call": true, "call_intent": "new_inquiry", "customer_attitude": "neutral"},
  "intent_signals": {
    "budget": {"mentioned": true, "specificity": "vague", "value_text": null, "quote": "今年预算还没批"},
    "timeline": {"mentioned": true, "urgency": "vague", "value_text": "明年再说", "quote": "今年预算还没批，明年再说"},
    "scale": {"designers": null, "stores": 3, "quote": "佛山这边三家，还在扩"},
    "pain_points": [
      {"pain_text": "CAD 画图太慢", "quote": "我们就用 CAD，画图太慢了"},
      {"pain_text": "客户看不到效果导致签单率低", "quote": "客户看不到效果，签单率低"}
    ],
    "active_engagement": [{"type": "provide_contact", "quote": "行，13800001111"}],
    "negative_signals": [
      {"type": "delay", "quote": "先了解下吧，今年预算还没批，明年再说"},
      {"type": "no_budget", "quote": "今年预算还没批"}
    ],
    "decision_maker": {"is_decision_maker": true, "quote": "请问是李总吗"}
  },
  "business_info": {
    "company_mention": "金鹏陶瓷",
    "company_mention_quote": "叫\\"金鹏陶瓷\\"",
    "industry_text": "瓷砖",
    "area_text": "佛山",
    "demand_summary": "用 CAD 出图慢且客户看不到效果导致签单率低，希望提升出图效率与展示效果"
  },
  "contact_info": {"contact_mention": "李总", "phone": "13800001111"}
}"""


def build_extraction_messages(dialogue: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"【示例通话记录】\n{EXTRACTION_FEWSHOT_DIALOGUE}"},
        {"role": "assistant", "content": EXTRACTION_FEWSHOT_OUTPUT},
        {"role": "user", "content": EXTRACTION_USER.replace("<<DIALOGUE>>", dialogue.strip())},
    ]


# --------------------------------------------------------------------------- #
# 2. 企业抬头还原（LLM 推理通道）
# --------------------------------------------------------------------------- #
COMPANY_SYSTEM = """你是中国企业工商信息核对专家，尤其熟悉家居、建材、装修、房地产行业的公司主体。
任务：把销售通话里客户口语说出的公司名，还原成最可能的工商注册全称。

判据优先级：
1) 客户自己说的名称 + 引号内的名字，是最强证据；
2) 结合通话中提到的地区、行业、品牌特征做交叉验证；
3) 若提供了「联网检索资料」，以检索资料为准，但必须与客户所说的行业/地区不冲突；
4) 如果客户口述名称与检索到的公司所在地区明显冲突，不要强行合并，而是标记 conflict。

必须输出严格 JSON：
{
  "best": {"legal_name": str, "brand": str|null, "province": str|null, "city": str|null,
           "industry": str|null, "match_type": "exact_alias|brand_to_legal|fuzzy|inferred|none",
           "confidence": 0.0-1.0},
  "candidates": [{"legal_name": str, "reason": str, "confidence": 0.0-1.0}],
  "conflicts": [str],
  "reasoning": str
}
无法确定时 best 填 {"legal_name": null, ...}，confidence 给 0.1 以下，并在 reasoning 说明原因。
只输出 JSON。"""

COMPANY_USER = """【客户口述公司名】<<MENTION>>（来源句式：<<QUOTE>>）
【通话中提及的行业表述】<<INDUSTRY_TEXT>>
【通话中提及的地区】<<AREA>>
【同通话中其他企业提及】<<OTHER_MENTIONS>>
【联网检索资料】
<<SEARCH_CONTEXT>>

请输出 JSON。"""


def build_company_messages(*, mention: str, quote: str, industry_text: str, area: str,
                           other_mentions: list[str], search_context: str) -> list[dict[str, str]]:
    user = (COMPANY_USER
            .replace("<<MENTION>>", mention or "(未明确)")
            .replace("<<QUOTE>>", quote or "(无)")
            .replace("<<INDUSTRY_TEXT>>", industry_text or "(未提及)")
            .replace("<<AREA>>", area or "(未提及)")
            .replace("<<OTHER_MENTIONS>>", "、".join(other_mentions) or "(无)")
            .replace("<<SEARCH_CONTEXT>>", search_context or "(无联网检索资料，请仅依据通话内容推理)"))
    return [{"role": "system", "content": COMPANY_SYSTEM}, {"role": "user", "content": user}]


COMPANY_SEARCH_QUERY = ("{mention} {area} {industry} 公司 工商注册全称 品牌 是哪家公司 "
                        "统一社会信用代码")
