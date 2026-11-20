# -*- coding: utf-8 -*-
"""
小V野外助教 —— 最小单元测试
作者：周荣发 / 2026210209 / 地理科学262 / 滁州学院
版本：1.0.0

只用标准库 unittest，不依赖 pytest。
运行：python -m unittest discover -s tests -v
"""
import csv
import datetime
import json
import os
import sys
import tempfile
import unittest

# 把上级目录（02_CODE_源代码）加入 sys.path，便于导入被测模块
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)          # 02_CODE_源代码
PKG_DIR = os.path.dirname(CODE_DIR)       # 2026210209周荣发地理科学262
CFG_DIR = os.path.join(PKG_DIR, "03_CONFIG_配置规则")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import hard_check          # noqa: E402
import rule_engine         # noqa: E402
import model_chain         # noqa: E402


# ---------------------------------------------------------------- 测试夹具
def _write_csv(path, header, rows, encoding="utf-8-sig"):
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


HEADER = ["record_id", "sample_no", "longitude", "latitude", "species",
          "cover", "slope_aspect", "survey_date", "altitude"]

GOOD_ROW = ["T-001", "Q01", "118.3125", "32.2871", "马尾松",
            "85", "东南", "2026-10-12", "182.5"]

FIELD_TYPES = {
    "record_id": "str", "sample_no": "str", "longitude": "float",
    "latitude": "float", "species": "str", "cover": "float",
    "slope_aspect": "str", "survey_date": "date", "altitude": "float",
}


class TestHardCheck(unittest.TestCase):
    """第一关：格式硬校验 F01–F08"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _path(self, name="t.csv"):
        return os.path.join(self.tmpdir, name)

    def test_clean_row_passes(self):
        p = self._path()
        _write_csv(p, HEADER, [GOOD_ROW])
        issues, records, meta = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertEqual(len(issues), 0, f"干净行不应报错，实际：{issues}")
        self.assertEqual(len(records), 1)
        self.assertEqual(meta["total_rows"], 1)

    def test_f01_missing_field(self):
        """F01：缺字段（表头少一列）"""
        p = self._path()
        bad_header = HEADER[:-1]  # 去掉 altitude
        _write_csv(p, bad_header, [GOOD_ROW[:-1]])
        issues, _, _ = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertTrue(any(i["rule_id"] == "F01" for i in issues),
                        f"应报 F01，实际：{[i['rule_id'] for i in issues]}")

    def test_f02_empty_value(self):
        """F02：必填字段为空"""
        p = self._path()
        row = list(GOOD_ROW)
        row[4] = ""  # species 空
        _write_csv(p, HEADER, [row])
        issues, _, _ = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertTrue(any(i["rule_id"] == "F02" and i["field"] == "species"
                            for i in issues),
                        f"应报 F02/species，实际：{[(i['rule_id'], i['field']) for i in issues]}")

    def test_f03_type_error(self):
        """F03：类型不可解析（cover 写成 abc）"""
        p = self._path()
        row = list(GOOD_ROW)
        row[5] = "abc"
        _write_csv(p, HEADER, [row])
        issues, _, _ = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertTrue(any(i["rule_id"] == "F03" and i["field"] == "cover"
                            for i in issues),
                        f"应报 F03/cover，实际：{[(i['rule_id'], i['field']) for i in issues]}")

    def test_f04_bad_date(self):
        """F04：日期格式错"""
        p = self._path()
        row = list(GOOD_ROW)
        row[7] = "2026年10月12日"
        _write_csv(p, HEADER, [row])
        issues, _, _ = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertTrue(any(i["rule_id"] == "F04" for i in issues),
                        f"应报 F04，实际：{[i['rule_id'] for i in issues]}")

    def test_f06_duplicate_header(self):
        """F06：表头重复"""
        p = self._path()
        dup_header = list(HEADER)
        dup_header[1] = "record_id"  # sample_no 改成 record_id
        _write_csv(p, dup_header, [GOOD_ROW])
        issues, _, _ = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertTrue(any(i["rule_id"] == "F06" for i in issues),
                        f"应报 F06，实际：{[i['rule_id'] for i in issues]}")

    def test_f08_empty_file(self):
        """F08：文件为空（只有表头）"""
        p = self._path()
        _write_csv(p, HEADER, [])
        issues, records, _ = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertTrue(any(i["rule_id"] == "F08" for i in issues),
                        f"应报 F08，实际：{[i['rule_id'] for i in issues]}")
        self.assertEqual(len(records), 0)

    def test_gbk_encoding(self):
        """F05 反向验证：GBK 编码应能正常读取，不报 F05"""
        p = self._path("gbk.csv")
        _write_csv(p, HEADER, [GOOD_ROW], encoding="gbk")
        issues, records, meta = hard_check.hard_check(p, HEADER, FIELD_TYPES)
        self.assertEqual(len(records), 1)
        self.assertFalse(any(i["rule_id"] == "F05" for i in issues))
        self.assertIn(meta["encoding"].lower(), ("gbk", "gb18030"))

    def test_issue_id_stable(self):
        """issue_id 必须是稳定哈希（同输入同输出）"""
        a = hard_check.make_issue_id("T-001", "cover", "R01", "rule")
        b = hard_check.make_issue_id("T-001", "cover", "R01", "rule")
        c = hard_check.make_issue_id("T-001", "cover", "R02", "rule")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 8)


class TestRuleEngine(unittest.TestCase):
    """第二关：规则引擎 R01–R13"""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.rule_cfg = self._load_rule_config()
        self.vocab = rule_engine.load_vocab(CFG_DIR)

    def _load_rule_config(self):
        p = os.path.join(CFG_DIR, "rule_config.json")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _records(self, **overrides):
        """构造一条「第一关已通过」的记录（含 _record_id / _row，日期已解析为 date）。"""
        rec = {
            "_record_id": "T-001", "_row": 2,
            "record_id": "T-001", "sample_no": "Q01",
            "longitude": 118.3125, "latitude": 32.2871,
            "species": "马尾松", "cover": 85.0,
            "slope_aspect": "东南", "survey_date": datetime.date(2026, 10, 12),
            "altitude": 182.5,
        }
        rec.update(overrides)
        return [rec]

    def _run(self, records, enable_l2=False):
        return rule_engine.rule_engine(
            records, self.rule_cfg, self.vocab,
            enable_l2=enable_l2, today=datetime.date(2026, 11, 3))

    def test_clean_record_no_issue(self):
        issues, meta = self._run(self._records())
        self.assertEqual(len(issues), 0, f"干净记录不应报错：{issues}")
        self.assertIn("rule_version", meta)

    def test_r01_cover_out_of_range(self):
        issues, _ = self._run(self._records(cover=168.0))
        self.assertTrue(any(i["rule_id"] == "R01" for i in issues))

    def test_r02_lon_lat_swapped(self):
        """R02：经纬度写反 → 合并为 1 条问题（field 为 longitude,latitude）"""
        issues, _ = self._run(self._records(longitude=32.2895, latitude=118.3150))
        r02 = [i for i in issues if i["rule_id"] == "R02"]
        self.assertEqual(len(r02), 1, "经纬度写反应合并为 1 条问题")
        self.assertEqual(r02[0]["field"], "longitude,latitude")
        self.assertEqual(r02[0]["severity"], "错误")

    def test_r03_future_date(self):
        issues, _ = self._run(self._records(survey_date=datetime.date(2027, 3, 15)))
        self.assertTrue(any(i["rule_id"] == "R03" for i in issues),
                        f"应报 R03，实际：{[i['rule_id'] for i in issues]}")

    def test_r04_too_early_date(self):
        issues, _ = self._run(self._records(survey_date=datetime.date(2015, 5, 1)))
        self.assertTrue(any(i["rule_id"] == "R04" for i in issues),
                        f"应报 R04，实际：{[i['rule_id'] for i in issues]}")

    def test_r05_duplicate_sample_no(self):
        recs = self._records() + self._records(
            _record_id="T-002", _row=3, record_id="T-002")
        issues, _ = self._run(recs)
        self.assertTrue(any(i["rule_id"] == "R05" for i in issues))

    def test_r06_duplicate_record_id(self):
        recs = self._records() + self._records(_row=3, sample_no="Q02")
        issues, _ = self._run(recs)
        self.assertTrue(any(i["rule_id"] == "R06" for i in issues),
                        f"应报 R06，实际：{[i['rule_id'] for i in issues]}")

    def test_r07_altitude_out_of_range(self):
        issues, _ = self._run(self._records(altitude=3500.0))
        self.assertTrue(any(i["rule_id"] == "R07" for i in issues))

    def test_r11_bad_slope_aspect(self):
        issues, _ = self._run(self._records(slope_aspect="朝东南"))
        self.assertTrue(any(i["rule_id"] == "R11" for i in issues))

    def test_r12_species_alias(self):
        issues, _ = self._run(self._records(species="马尾松树"))
        r12 = [i for i in issues if i["rule_id"] == "R12"]
        self.assertTrue(r12, "别名应命中 R12")
        self.assertEqual(r12[0]["severity"], "提示")

    def test_r13_species_not_in_list(self):
        issues, _ = self._run(self._records(species="完全不存在的植物XYZ"))
        r13 = [i for i in issues if i["rule_id"] == "R13"]
        self.assertTrue(r13, "名录外物种应命中 R13")
        self.assertEqual(r13[0]["severity"], "提示")

    def test_severity_floor_enforced(self):
        """severity_floor：R01 最低「错误」，配置里写「提示」也要被抬回「错误」"""
        cfg = json.loads(json.dumps(self.rule_cfg))
        for r in cfg["rules"]:
            if r["id"] == "R01":
                r["severity"] = "提示"
        issues, _ = rule_engine.rule_engine(
            self._records(cover=168.0), cfg, self.vocab,
            enable_l2=False, today=datetime.date(2026, 11, 3))
        r01 = [i for i in issues if i["rule_id"] == "R01"]
        self.assertTrue(r01)
        self.assertEqual(r01[0]["severity"], "错误",
                         "severity_floor 未生效：R01 被降级了")

    def test_l2_disabled_by_default(self):
        _, meta = self._run(self._records())
        self.assertFalse(meta["l2_enabled"])
        self.assertIn("R09", meta["disabled_rule_set"])


class TestModelChain(unittest.TestCase):
    """第三关：AI 补充提示（不联网，只测约束逻辑）"""

    def test_ai_cannot_override_rule(self):
        """AI 输出的「错误」级必须被强制降为「提示」"""
        ai_raw = [{
            "record_id": "T-001", "field": "cover", "rule_id": "AI-01",
            "severity": "错误", "confidence": 0.9,
            "message": "AI 认为盖度有问题", "suggestion": "复核",
        }]
        rule_issues = []
        main, appendix, meta = model_chain._postprocess(
            ai_raw, rule_issues, {"ai_policy": {"low_confidence_threshold": 0.6,
                                                "suppress_duplicate_with_rule": True}})
        all_ai = main + appendix
        self.assertTrue(all_ai)
        for i in all_ai:
            self.assertEqual(i["severity"], "提示",
                             "AI 不得输出「错误」或「警告」级")
            self.assertFalse(i["ai_can_override"])

    def test_low_confidence_filtered(self):
        """置信度低于阈值的提示应被过滤到附录"""
        ai_raw = [{
            "record_id": "T-001", "field": "species", "rule_id": "AI-02",
            "severity": "提示", "confidence": 0.3,
            "message": "低置信度提示", "suggestion": "参考",
        }]
        main, appendix, meta = model_chain._postprocess(
            ai_raw, [], {"ai_policy": {"low_confidence_threshold": 0.6,
                                       "suppress_duplicate_with_rule": True}})
        self.assertEqual(len(main), 0, "低置信度不应进主清单")
        self.assertEqual(len(appendix), 1, "低置信度应进附录")
        self.assertEqual(meta["low_confidence_count"], 1)

    def test_duplicate_suppressed(self):
        """AI 提示与规则结论重复时应被抑制"""
        rule_issues = [{
            "record_id": "T-001", "field": "cover", "rule_id": "R01",
            "severity": "错误", "source": "rule",
        }]
        ai_raw = [{
            "record_id": "T-001", "field": "cover", "rule_id": "AI-03",
            "severity": "提示", "confidence": 0.9,
            "message": "盖度异常", "suggestion": "复核",
        }]
        main, appendix, meta = model_chain._postprocess(
            ai_raw, rule_issues,
            {"ai_policy": {"low_confidence_threshold": 0.6,
                           "suppress_duplicate_with_rule": True}})
        self.assertEqual(len(main) + len(appendix), 0,
                         "与规则重复的 AI 提示应被抑制")
        self.assertEqual(meta["suppressed_duplicates"], 1)

    def test_offline_mode_skips_ai(self):
        """offline 模式：ai_status 必须是 offline_skipped，不假装跑过"""
        main, appendix, meta = model_chain.model_chain(
            [{"record_id": "T-001"}], [], {"model_policy": {}}, mode="offline")
        self.assertEqual(meta["ai_status"], "offline_skipped")
        self.assertFalse(meta["ai_enabled"])
        self.assertEqual(len(main), 0)


class TestConfigIntegrity(unittest.TestCase):
    """配置完整性：防止有人把 AI 权限改大"""

    def setUp(self):
        self.cfg_dir = CFG_DIR

    def _load(self, name):
        with open(os.path.join(self.cfg_dir, name), "r", encoding="utf-8") as f:
            return json.load(f)

    def test_ai_cannot_override_rule_is_false(self):
        cfg = self._load("config.json")
        self.assertIs(cfg["ai_policy"]["ai_can_override_rule"], False,
                      "ai_can_override_rule 必须为 false")

    def test_ai_max_severity_is_hint(self):
        cfg = self._load("config.json")
        self.assertEqual(cfg["ai_policy"]["ai_max_severity"], "提示")

    def test_host_is_localhost_only(self):
        cfg = self._load("config.json")
        self.assertIn(cfg["server"]["host"], ("127.0.0.1", "localhost"))

    def test_cover_max_in_legal_range(self):
        rc = self._load("rule_config.json")
        lo, hi = rc["thresholds"]["cover_max_legal_range"]
        self.assertTrue(lo <= rc["thresholds"]["cover_max"] <= hi,
                        f"cover_max={rc['thresholds']['cover_max']} 超出合法区间 [{lo},{hi}]")

    def test_all_rules_have_required_keys(self):
        rc = self._load("rule_config.json")
        need = {"id", "name", "field", "level", "severity", "message", "suggestion"}
        for r in rc["rules"]:
            missing = need - set(r.keys())
            self.assertFalse(missing, f"规则 {r.get('id')} 缺字段：{missing}")

    def test_severity_floor_covers_all_rules(self):
        rc = self._load("rule_config.json")
        floor = rc["severity_floor"]
        for r in rc["rules"]:
            self.assertIn(r["id"], floor, f"规则 {r['id']} 未定义 severity_floor")


if __name__ == "__main__":
    unittest.main(verbosity=2)
