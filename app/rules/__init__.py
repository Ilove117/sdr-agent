"""业务规则层：SQL 字段口径、行业词库、意向评分卡、企业抬头规整规则。

设计原则（对应提示「先定义清楚 SQL 字段的抽取规则，再设计 Agent 工作流」）：
本层是字段口径的唯一事实来源，Agent 工作流只负责「取证」，规则层负责「定口径」。
"""

from app.rules import company_rules, industry_lexicon, intent_scorecard, sql_fields

__all__ = ["sql_fields", "industry_lexicon", "intent_scorecard", "company_rules"]
