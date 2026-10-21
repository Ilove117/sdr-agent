"""规则层单元测试：字段口径、数值抽取、地区/手机号归一、评分卡。"""

from __future__ import annotations

import unittest

from app.rules import company_rules as cr
from app.rules import industry_lexicon as ilx
from app.rules import sql_fields as sf
from app.rules.intent_scorecard import SignalSet, score_intent


class TestQuantityExtraction(unittest.TestCase):
    def test_designer_count_after_context(self):
        # 语序「设计部大概 15 个人」：数字在关键词之后
        val, conf, _ = sf.extract_quantity("设计部大概 15 个人", r"人|设计师",
                                           context_keywords=("设计部", "设计师"))
        self.assertEqual(val, 15)
        self.assertGreaterEqual(conf, 0.7)

    def test_chinese_number(self):
        self.assertEqual(sf.extract_quantity("设计师二十人", r"人|设计师",
                                             context_keywords=("设计师",))[0], 20)

    def test_approx_number_takes_lower_bound(self):
        val, conf, _ = sf.extract_quantity("二十多家门店", r"门店|店",
                                           context_keywords=("门店", "店"))
        self.assertEqual(val, 20)
        self.assertLess(conf, 0.9)  # 概数降低置信度

    def test_no_false_positive_on_unrelated_numbers(self):
        # 「一下需求」「企业版」中的数字/字符不应被识别为人数
        val, _, _ = sf.extract_quantity(
            "想了解企业版设计软件，今天想跟您确认一下需求", r"人|设计师",
            context_keywords=("设计部", "设计师"))
        self.assertIsNone(val)

    def test_scale_does_not_confuse_budget_with_headcount(self):
        dialogue = "预算的话，一年几万块我们能接受"
        self.assertIsNone(sf.extract_quantity(dialogue, r"人|设计师",
                                              context_keywords=("设计部", "设计师"))[0])


class TestPhoneAndArea(unittest.TestCase):
    def test_masked_phone(self):
        self.assertEqual(sf.extract_phone("加我 138xxxx1234")[0], "138****1234")

    def test_plain_phone(self):
        self.assertEqual(sf.extract_phone("加我 13812345678")[0], "13812345678")

    def test_area_completed_with_province(self):
        self.assertEqual(sf.extract_area("我们这边是杭州那边一个做全屋定制的")[0],
                         "浙江省杭州市")

    def test_pronoun_is_not_area(self):
        # 「我们这边」不能被当成地区
        self.assertIsNone(sf.extract_area("我们这边已经有在用别的设计软件了")[0])


class TestCompanyRules(unittest.TestCase):
    def test_alias_variants_cover_brand_suffix(self):
        variants = cr.alias_variants("木里木外")
        self.assertIn("木里木外", variants)
        self.assertIn("木里木外家居", variants)

    def test_strip_suffix(self):
        self.assertEqual(cr.strip_suffix("木里木外家居有限公司"), "木里木外")

    def test_mentions_prefer_quoted_name(self):
        mentions = cr.extract_mentions('我们这边是做全屋定制的，叫“木里木外”')
        self.assertTrue(mentions)
        self.assertEqual(mentions[0].raw, "木里木外")
        self.assertGreaterEqual(mentions[0].confidence, 0.8)

    def test_similarity_self_and_contains(self):
        self.assertEqual(cr.similarity("木里木外", "木里木外"), 1.0)
        self.assertGreater(cr.similarity("木里木外", "上海木里实业发展有限公司"), 0.5)


class TestIndustry(unittest.TestCase):
    def test_full_house_custom(self):
        self.assertEqual(ilx.classify_industry("我们做全屋定制的")[0], "定制家居")

    def test_main_business_wins_over_secondary_need(self):
        # 「做板材的……想做整装方案」：主营业务是板材（建材），不是家装设计
        self.assertEqual(ilx.classify_industry("我们是在湖州做板材的，现在想用设计工具做整装方案")[0],
                         "建材")


class TestIntentScorecard(unittest.TestCase):
    def test_high_intent(self):
        result = score_intent(SignalSet.from_dict({
            "budget": {"mentioned": True, "specificity": "clear", "value_text": "一年几万"},
            "timeline": {"mentioned": True, "urgency": "near_term"},
            "pain_points": [{"pain_text": "出图慢"}, {"pain_text": "对接出错"}],
            "active_engagement": [{"type": "provide_contact"}, {"type": "agree_next_step"}],
            "decision_maker": {"is_decision_maker": True},
        }), designers=15, contact_title="负责人/总经理")
        self.assertEqual(result.level, "high")
        self.assertGreater(result.total_score, 65)

    def test_low_intent_on_rejection(self):
        result = score_intent(SignalSet.from_dict({
            "negative_signals": [{"type": "just_looking", "quote": "我就随便问问"},
                                 {"type": "reject", "quote": "不用了不用了"}],
        }))
        self.assertEqual(result.level, "low")

    def test_delay_alone_does_not_collapse_to_low(self):
        # 有明确痛点、有规模、对接人是决策者，只是时间未定、预算未批 -> 中意向
        result = score_intent(SignalSet.from_dict({
            "budget": {"mentioned": True, "specificity": "vague"},
            "timeline": {"mentioned": True, "urgency": "vague"},
            "pain_points": [{"pain_text": "出图慢"}],
            "active_engagement": [{"type": "provide_contact"}, {"type": "agree_next_step"}],
            "negative_signals": [{"type": "delay"}, {"type": "no_budget"}],
            "decision_maker": {"is_decision_maker": True},
        }), designers=20, stores=20)
        self.assertEqual(result.level, "medium")

    def test_need_without_budget_is_medium_not_low(self):
        # 首次触达常见形态：有痛点、有规模、留了联系方式，但还没聊预算和时间
        result = score_intent(SignalSet.from_dict({
            "timeline": {"mentioned": False}, "budget": {"mentioned": False},
            "pain_points": [{"pain_text": "老软件效率低"}, {"pain_text": "改图麻烦"}],
            "active_engagement": [{"type": "provide_contact"}],
            "decision_maker": {"is_decision_maker": False},
        }), designers=20, contact_title="经理")
        self.assertEqual(result.level, "medium")

    def test_qualified_but_no_pain_is_not_high(self):
        # 预算 + 时间 + 决策权齐备，但客户没说出任何痛点 -> 不能判高意向
        result = score_intent(SignalSet.from_dict({
            "budget": {"mentioned": True, "specificity": "clear", "value_text": "20万"},
            "timeline": {"mentioned": True, "urgency": "immediate"},
            "decision_maker": {"is_decision_maker": True},
        }), designers=30, contact_title="负责人/总经理")
        self.assertNotEqual(result.level, "high")

    def test_reason_contains_evidence(self):
        result = score_intent(SignalSet.from_dict({
            "budget": {"mentioned": True, "specificity": "range", "value_text": "几万块"},
        }), designers=10)
        self.assertIn("预算", result.reason)
        self.assertIn("几万块", result.reason)


class TestValidateLead(unittest.TestCase):
    def _base(self, **overrides):
        lead = {
            "company_name": "上海木里实业发展有限公司", "industry": "定制家居",
            "area": "浙江省杭州市", "contact_name": "张", "contact_title": "负责人/总经理",
            "phone": "138****1234", "intent_level": "high", "intent_reason": "…",
            "demand_summary": "…", "mendian": None, "shejishi": 15,
            "call_time": "2026-10-14 10:32:00", "next_action": "…",
            "confidence_score": 0.8, "need_human_review": False,
        }
        lead.update(overrides)
        return lead

    def test_valid_record_passes(self):
        self.assertEqual(sf.validate_lead(self._base()), [])

    def test_invalid_intent_level(self):
        issues = sf.validate_lead(self._base(intent_level="很高"))
        self.assertTrue(any("intent_level" in i for i in issues))

    def test_unrecognized_company_flagged(self):
        issues = sf.validate_lead(self._base(company_name="未识别"))
        self.assertTrue(any("企业抬头" in i for i in issues))

    def test_confidence_out_of_range(self):
        issues = sf.validate_lead(self._base(confidence_score=1.5))
        self.assertTrue(any("confidence_score" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
