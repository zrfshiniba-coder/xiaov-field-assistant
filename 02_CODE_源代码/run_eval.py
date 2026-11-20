# -*- coding: utf-8 -*-
"""
小V野外助教 · 跑分脚本
========================
作者：周荣发
学号：2026210209
班级：地理科学262
学校：滁州学院
日期：2026-11-03

版权声明：本文件为周荣发（2026210209）原创作品，仅用于滁州学院2026级
新生人工智能创新应用比赛提交与评审。未经作者书面许可，不得用于商业用途。

用法
----
    python run_eval.py                      # 在线模式（需 DASHSCOPE_API_KEY）
    python run_eval.py --offline            # 离线模式：只跑规则引擎，仍生成教师复核材料
    python run_eval.py --demo-ai            # 演示模式：加载本地 AI 缓存
    python run_eval.py --no-l2              # 禁用 L2 外部地理增强规则

输出（写入 output/ 目录）
------------------------
    metrics.json          指标（唯一事实来源）
    teacher_review.csv    教师复核表
    summary_report.csv    共性错误统计
    问题清单.json          全部问题
    运行日志.txt           运行摘要
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
PKG = BASE.parent

sys.path.insert(0, str(BASE))

from hard_check import hard_check  # noqa: E402
from model_chain import model_chain  # noqa: E402
from rule_engine import load_rule_config, load_vocab, rule_engine  # noqa: E402

__author__ = "周荣发"
__student_id__ = "2026210209"
__version__ = "1.0.0"

SCHEMA_VERSION = "1.0"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _load_config() -> dict:
    return json.loads((PKG / "03_CONFIG_配置规则" / "config.json").read_text(encoding="utf-8"))


def _validate_config(cfg: dict, rule_cfg: dict) -> list[str]:
    """
    启动校验（对应 config.schema.json 的关键约束）。
    返回错误列表，空列表表示通过。
    """
    errors: list[str] = []

    # 必填字段
    for key in ("schema_version", "app", "server", "data", "model_policy", "output", "ai_policy"):
        if key not in cfg:
            errors.append(f"config.json 缺少必填段：{key}")

    # 硬规则不可关
    if cfg.get("ai_policy", {}).get("ai_can_override_rule") is not False:
        errors.append("ai_policy.ai_can_override_rule 必须为 false（AI 不可覆盖规则）")

    # 严重度不可降级
    if cfg.get("ai_policy", {}).get("ai_max_severity") != "提示":
        errors.append("ai_policy.ai_max_severity 必须为「提示」")

    # 阈值合法区间
    th = rule_cfg.get("thresholds", {})
    legal = th.get("cover_max_legal_range", [50, 100])
    cover_max = th.get("cover_max")
    if cover_max is None or not (legal[0] <= cover_max <= legal[1]):
        errors.append(
            f"thresholds.cover_max={cover_max} 超出合法区间 {legal}（防止绕过硬规则）"
        )

    # 硬规则严重度不得低于 floor
    order = {"提示": 1, "警告": 2, "错误": 3}
    floor = rule_cfg.get("severity_floor", {})
    for rule in rule_cfg.get("rules", []):
        rid = rule["id"]
        if rid in floor:
            if order.get(rule.get("severity", "提示"), 0) < order.get(floor[rid], 0):
                errors.append(f"规则 {rid} 严重度 {rule['severity']} 低于 floor {floor[rid]}")

    # 服务绑定本机
    host = cfg.get("server", {}).get("host")
    if host not in ("127.0.0.1", "localhost"):
        errors.append(f"server.host={host} 必须为 127.0.0.1（不监听局域网）")

    return errors


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


#: 禁止写入的目录（防止现场运行覆盖随包正式证据）
PROTECTED_DIRS = [
    (PKG / "02_CODE_源代码" / "expected_output").resolve(),
    (PKG / "00_评审入口").resolve(),
    (PKG / "01_DOCS_方案文档").resolve(),
    (PKG / "03_CONFIG_配置规则").resolve(),
    (PKG / "04_DATA_样本与金标准").resolve(),
]


def _check_outdir_safe(out_dir: Path) -> str | None:
    """
    输出目录安全检查：不允许写进随包正式证据目录，也不允许写进包根。

    返回错误说明（None 表示安全）。
    评委现场运行 `python run_eval.py` 时，默认写 `02_CODE_源代码/output/`，
    不会覆盖 `expected_output/` 里的正式证据，也不会污染包根。
    """
    resolved = out_dir.resolve()
    if resolved == PKG.resolve():
        return f"输出目录不能是包根目录：{resolved}（会污染提交包结构）"
    for prot in PROTECTED_DIRS:
        if resolved == prot or prot in resolved.parents:
            return f"输出目录不能位于受保护目录内：{prot}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="小V野外助教 · 跑分脚本")
    ap.add_argument("--csv", default=str(PKG / "04_DATA_样本与金标准" / "samples" / "all_samples.csv"),
                    help="待评测 CSV 路径")
    ap.add_argument("--gt", default=str(PKG / "04_DATA_样本与金标准" / "ground_truth.csv"),
                    help="金标准 CSV 路径")
    ap.add_argument("--offline", action="store_true", help="离线模式：跳过第三关 AI")
    ap.add_argument("--demo-ai", action="store_true", help="演示模式：加载本地 AI 缓存")
    ap.add_argument("--no-l2", action="store_true", help="禁用 L2 外部地理增强规则")
    ap.add_argument("--out", "--outdir", dest="out", default=str(BASE / "output"),
                    help="输出目录（--outdir 为同义别名；默认 02_CODE_源代码/output/）")
    ap.add_argument("--include-hint-in-review", action="store_true",
                    help="把「提示」级问题也写进 teacher_review.csv（默认只写错误/警告）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    unsafe = _check_outdir_safe(out_dir)
    if unsafe:
        print(f"[错误] {unsafe}")
        print("       请改用 --outdir 指定一个独立目录，例如：--outdir ./output_run")
        return 3
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config()
    rule_cfg = load_rule_config(PKG / "03_CONFIG_配置规则" / "rule_config.json")

    # ---- 启动校验
    cfg_errors = _validate_config(cfg, rule_cfg)
    if cfg_errors:
        print("[配置校验失败]")
        for e in cfg_errors:
            print("  -", e)
        return 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[错误] 样本文件不存在：{csv_path}")
        return 1

    # ---- 第一关
    issues_1, records, meta_1 = hard_check(
        csv_path, cfg["data"]["required_fields"], cfg["data"]["field_types"]
    )

    # ---- 第二关
    vocab = load_vocab(PKG / "03_CONFIG_配置规则")
    issues_2, meta_2 = rule_engine(
        records, rule_cfg, vocab, enable_l2=not args.no_l2
    )

    rule_issues = issues_1 + issues_2

    # 词表降级告警：非空说明有规则未真正生效，指标不可直接采信
    vocab_warnings = meta_2.get("vocab_warnings", []) or []
    for w in vocab_warnings:
        print(f"[告警] {w}")

    # ---- 第三关
    if args.offline:
        mode = "offline"
    elif args.demo_ai:
        mode = "demo_ai"
    else:
        mode = "online"

    ai_main, ai_appendix, meta_3 = model_chain(
        records, rule_issues, cfg, mode=mode,
        cache_path=BASE / "ai_sample_cache.json",
    )

    all_issues = rule_issues + ai_main

    # ---- 指标计算（口径：错误+警告 计入召回/精确率；提示不计）
    gt_path = Path(args.gt)
    gt_map: dict[tuple[str, str], str] = {}
    if gt_path.exists():
        with gt_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # 兼容两种列名：label（旧）/ severity（现行 ground_truth.csv）
            cols = [c.strip() for c in (reader.fieldnames or [])]
            sev_col = "label" if "label" in cols else ("severity" if "severity" in cols else None)
            if sev_col is None:
                print("[错误] ground_truth.csv 缺少级别列（label 或 severity）")
                return 1
            for row in reader:
                gt_map[(row["record_id"].strip(), row["field"].strip())] = row[sev_col].strip()

    def _key(it: dict) -> tuple[str, str]:
        return (it["record_id"], it["field"])

    def _prf(tp: int, fp: int, fn: int) -> dict:
        r = round(tp / (tp + fn), 4) if (tp + fn) else 0.0
        p = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
        f = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
        return {"recall": r, "precision": p, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    # ============ 口径一：问题级（issue-level）============
    # 一个 (record_id, field) 算一个「问题单元」。同一字段被多条规则命中时只算一次，
    # 避免「一条记录被 3 条规则同时命中 → 精确率被稀释」的失真。
    pred_keys = {_key(it) for it in all_issues if it["severity"] in ("错误", "警告")}
    gt_keys = {k for k, v in gt_map.items() if v in ("错误", "警告")}
    issue_level = _prf(
        len(pred_keys & gt_keys),
        len(pred_keys - gt_keys),
        len(gt_keys - pred_keys),
    )

    # ============ 口径二：记录级（record-level）============
    # 一条记录只要有一个「错误/警告」即计为正例。这是**主指标**，
    # 因为教师实际关心的是「这条记录要不要返工」，而不是「这条记录有几个问题」。
    pred_recs = {it["record_id"] for it in all_issues if it["severity"] in ("错误", "警告")}
    gt_recs = {rid for (rid, _f), v in gt_map.items() if v in ("错误", "警告")}
    record_level = _prf(
        len(pred_recs & gt_recs),
        len(pred_recs - gt_recs),
        len(gt_recs - pred_recs),
    )

    # 主指标 = 记录级；辅助指标 = 问题级
    recall = record_level["recall"]
    precision = record_level["precision"]
    f1 = record_level["f1"]
    tp, fp, fn = record_level["tp"], record_level["fp"], record_level["fn"]

    # 消融：规则 only（同样按记录级口径，与主指标可比）
    rule_only_recs = {it["record_id"] for it in rule_issues if it["severity"] in ("错误", "警告")}
    rule_only = _prf(
        len(rule_only_recs & gt_recs),
        len(rule_only_recs - gt_recs),
        len(gt_recs - rule_only_recs),
    )
    recall_r, precision_r = rule_only["recall"], rule_only["precision"]

    hint_count = len([it for it in all_issues if it["severity"] == "提示"])

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "rule_version": rule_cfg.get("version"),
        "config_hash": _sha256_of_file(PKG / "03_CONFIG_配置规则" / "config.json"),
        "sample_set_hash": _sha256_of_file(csv_path),
        "mode": mode,
        "ai_enabled": meta_3["ai_enabled"],
        "ai_status": meta_3["ai_status"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        # 复现性：日期类规则（R03/R04）使用的基准日，来自 rule_config.json。
        # 同一份包在任何机器、任何日期运行，该值不变 → 结果不变。
        "reference_date": rule_cfg.get("reference_date"),
        "enabled_rule_set": meta_2["enabled_rule_set"],
        "disabled_rule_set": meta_2["disabled_rule_set"],
        "l2_enabled": meta_2["l2_enabled"],
        "vocab_warnings": vocab_warnings,
        "sample": {
            "file": csv_path.name,
            "total_rows": meta_1["total_rows"],
            "passed_rows": meta_1["passed_rows"],
            "failed_rows": meta_1["failed_rows"],
            "encoding": meta_1["encoding"],
        },
        # ---- 主指标：记录级（一条记录有任一错误/警告即算正例）
        "record_level_metrics": {
            "stat_unit": "记录（record_id）",
            "is_primary": True,
            "recall": record_level["recall"],
            "precision": record_level["precision"],
            "f1": record_level["f1"],
            "tp": record_level["tp"],
            "fp": record_level["fp"],
            "fn": record_level["fn"],
            "gt_positive_records": len(gt_recs),
            "pred_positive_records": len(pred_recs),
        },
        # ---- 辅助指标：问题级（record_id + field 为一个问题单元）
        "issue_level_metrics": {
            "stat_unit": "问题单元（record_id + field）",
            "is_primary": False,
            "recall": issue_level["recall"],
            "precision": issue_level["precision"],
            "f1": issue_level["f1"],
            "tp": issue_level["tp"],
            "fp": issue_level["fp"],
            "fn": issue_level["fn"],
            "hint_count": hint_count,
        },
        # ---- 兼容旧字段：等价于 record_level_metrics（主指标）
        "metrics": {
            "stat_unit": "记录（record_id）",
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "hint_count": hint_count,
        },
        "ablation": {
            "rule_only": {"recall": recall_r, "precision": precision_r,
                          "tp": rule_only["tp"], "fp": rule_only["fp"],
                          "fn": rule_only["fn"], "hint_count": 0},
            "rule_plus_ai": {"recall": recall, "precision": precision,
                             "tp": tp, "fp": fp, "fn": fn, "hint_count": hint_count},
        },
        "ai_meta": {
            "suppressed_duplicates": meta_3["suppressed_duplicates"],
            "low_confidence_count": meta_3["low_confidence_count"],
            "model": meta_3["model"],
        },
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 问题清单
    (out_dir / "问题清单.json").write_text(
        json.dumps(all_issues, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 教师复核表
    # 设计说明（回应评审意见 16/17）：
    #   1. 默认只写「错误 / 警告」级问题；「提示」级属于 AI 补充建议，
    #      不进入教师复核清单（否则教师工作量被无意义放大）。
    #      需要时用 --include-hint-in-review 打开。
    #   2. 字段名用 teacher_proposed_value 而非 final_value：
    #      教师填的是「建议改成什么」，**不直接改原始数据**，
    #      原始数据的修改必须回到采集端（correction_source 记录来源）。
    #      这样避免「教师一填就等于系统自动改数据」的误解。
    review_rows = []
    for it in all_issues:
        if not args.include_hint_in_review and it["severity"] == "提示":
            continue
        review_rows.append({
            "issue_id": it["issue_id"],
            "record_id": it["record_id"],
            "row": it["row"],
            "field": it["field"],
            "rule_id": it["rule_id"],
            "source": it["source"],
            "severity": it["severity"],
            "confidence": it["confidence"],
            "ai_suggestion": it["suggestion"] if it["source"] == "ai" else "",
            "review_status": "pending",
            "reviewer": "",
            "review_time": "",
            "teacher_note": "",
            "teacher_proposed_value": "",
            "correction_source": "",
        })
    _write_csv(
        out_dir / "teacher_review.csv", review_rows,
        ["issue_id", "record_id", "row", "field", "rule_id", "source", "severity",
         "confidence", "ai_suggestion", "review_status", "reviewer", "review_time",
         "teacher_note", "teacher_proposed_value", "correction_source"],
    )

    # ---- 共性错误统计
    by_rule = Counter(it["rule_id"] for it in all_issues)
    by_field = Counter(it["field"] for it in all_issues)
    by_sev = Counter(it["severity"] for it in all_issues)
    by_sample = Counter(it["record_id"] for it in all_issues)

    summary_rows = []
    for rid, cnt in by_rule.most_common():
        summary_rows.append({"维度": "按规则", "取值": rid, "数量": cnt})
    for fld, cnt in by_field.most_common():
        summary_rows.append({"维度": "按字段", "取值": fld, "数量": cnt})
    for sev, cnt in by_sev.most_common():
        summary_rows.append({"维度": "按严重度", "取值": sev, "数量": cnt})
    for rec, cnt in by_sample.most_common(10):
        summary_rows.append({"维度": "按记录(Top10)", "取值": rec, "数量": cnt})
    for rid, cnt in by_rule.most_common(5):
        summary_rows.append({"维度": "Top5高频问题", "取值": rid, "数量": cnt})
    summary_rows.append({"维度": "误报标记", "取值": "false_positive", "数量": 0})

    _write_csv(out_dir / "summary_report.csv", summary_rows, ["维度", "取值", "数量"])

    # ---- 运行日志
    log = [
        f"运行时间：{metrics['generated_at']}",
        f"模式：{mode}（ai_enabled={meta_3['ai_enabled']}, ai_status={meta_3['ai_status']}）",
        f"样本：{csv_path.name}（{meta_1['total_rows']} 行，通过 {meta_1['passed_rows']}，失败 {meta_1['failed_rows']}）",
        f"第一关问题：{len(issues_1)}",
        f"第二关问题：{len(issues_2)}",
        f"第三关提示：{len(ai_main)}（附录 {len(ai_appendix)}，抑制重复 {meta_3['suppressed_duplicates']}）",
        f"召回率：{recall:.2%}  精确率：{precision:.2%}  F1：{f1:.2%}",
        f"规则 only：召回 {recall_r:.2%} / 精确 {precision_r:.2%}",
        f"启用规则：{', '.join(meta_2['enabled_rule_set']) or '（无）'}",
        f"禁用规则：{', '.join(meta_2['disabled_rule_set']) or '（无）'}",
    ]
    (out_dir / "运行日志.txt").write_text("\n".join(log), encoding="utf-8")

    print("\n".join(log))
    print(f"\n输出目录：{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
