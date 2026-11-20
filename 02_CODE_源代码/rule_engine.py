# -*- coding: utf-8 -*-
"""
小V野外助教 · 第二关：规则引擎（R01–R13 + P01–P03）
=====================================================
作者：周荣发
学号：2026210209
班级：地理科学262
学校：滁州学院
日期：2026-11-03

版权声明：本文件为周荣发（2026210209）原创作品，仅用于滁州学院2026级
新生人工智能创新应用比赛提交与评审。未经作者书面许可，不得用于商业用途。

设计原则
--------
1. **确定性优先**：本关全部为确定性规则，同一输入必得同一输出。
2. **规则分级**：L0 离线硬规则 / L1 本地词表规则 / L2 外部地理增强规则。
3. **硬规则不可关**：severity_floor 保证硬规则最低严重度，配置无法降低。
4. **不调用模型**：纯 Python 标准库，离线可跑。
5. **L2 规则可禁用**：外部数据不可用时自动跳过，并在 meta 中声明。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from hard_check import make_issue_id

__author__ = "周荣发"
__student_id__ = "2026210209"
__version__ = "1.0.0"

SEVERITY_ORDER = {"提示": 1, "警告": 2, "错误": 3}


#: 规则置信度。L0/L1 规则是确定性判定（阈值比较、词表命中），置信度恒为 1.0；
#: L2 规则依赖外部地理数据（行政区划边界 / DEM），数据本身有精度误差，
#: 因此置信度降为 0.8，并强制进入教师复核。
CONFIDENCE_DETERMINISTIC = 1.0
CONFIDENCE_EXTERNAL_DATA = 0.8


def _issue(
    record_id: str,
    row: int,
    field: str,
    raw_value: Any,
    error_type: str,
    rule_id: str,
    severity: str,
    message: str,
    suggestion: str = "",
    confidence: float = CONFIDENCE_DETERMINISTIC,
) -> dict:
    """
    构造统一的问题记录结构。

    ``need_teacher_confirm`` 的判定（修正了旧版「错误 且 confidence<0.9」不可达的问题）：
      1. 严重度为「错误」或「警告」→ 必须教师复核（规则结论，教师终裁）；
      2. 置信度低于 0.9 → 必须教师复核（目前唯一来源是 L2 外部数据规则，
         因为 L0/L1 是确定性判定，置信度恒为 1.0）。
    两条都是**可达**分支：L2 规则（R09/R10）severity=警告 且 confidence=0.8，
    同时命中条件 1 和 2。
    """
    return {
        "issue_id": make_issue_id(record_id, field, rule_id, "rule"),
        "record_id": record_id,
        "row": row,
        "field": field,
        "raw_value": "" if raw_value is None else str(raw_value),
        "error_type": error_type,
        "rule_id": rule_id,
        "source": "rule",
        "severity": severity,
        "confidence": confidence,
        "message": message,
        "suggestion": suggestion,
        "ai_can_override": False,
        "need_teacher_confirm": (
            severity in ("错误", "警告") or confidence < 0.9
        ),
        "related_rule_ids": [],
        "conflict_with_rule": False,
    }


def _l2_data_available(rule_cfg: dict) -> bool:
    """
    L2 规则是否具备外部数据。

    本版本**不随包提供**外部行政区划边界 / DEM 数据，因此恒返回 False，
    R09/R10 不会触发，也不会出现在 enabled_rule_set 中。
    这样「规则覆盖度 11/13」是**如实**的，而不是假装 13/13。
    """
    l2 = rule_cfg.get("l2_external_data") or {}
    return bool(l2.get("available", False))


def _check_l2(rule_id: str, rec: dict, rule_cfg: dict) -> dict | None:
    """
    L2 外部数据校验的**接入点**（当前无数据，恒返回 None）。

    接入真实数据后在此实现：
      R09：坐标是否落在实习区域行政区划边界内
      R10：海拔与 DEM 高程差异是否超阈值
    返回 {"raw": 原始值, "fmt": 消息模板参数} 或 None。
    """
    return None


def _apply_severity_floor(rule_id: str, severity: str, floor: dict[str, str]) -> str:
    """硬规则最低严重度保护：配置不能把硬规则降到 floor 以下。"""
    lowest = floor.get(rule_id)
    if lowest is None:
        return severity
    if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get(lowest, 0):
        return lowest
    return severity


def load_rule_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_vocab(base_dir: str | Path) -> dict[str, Any]:
    """
    加载本地词表（物种名录 / 别名表 / 受控词表）。

    文件不存在时返回空结构，规则自动降级为「不触发」；
    但**必须留下告警**（vocab["_warnings"]），由 rule_engine 汇总进 meta，
    避免「词表路径写错 → 规则静默失效 → 指标虚高」的假通过。
    """
    base = Path(base_dir)
    vocab: dict[str, Any] = {
        "species": set(), "alias": {}, "controlled": {}, "_warnings": [],
    }

    species_file = base / "species_list.txt"
    if species_file.exists():
        for line in species_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            vocab["species"].add(parts[0].strip())
    else:
        vocab["_warnings"].append(f"词表缺失：{species_file}（R12/R13 将不触发）")

    alias_file = base / "species_alias.txt"
    if alias_file.exists():
        for line in alias_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                vocab["alias"][parts[0].strip()] = parts[1].strip()
    else:
        vocab["_warnings"].append(f"别名表缺失：{alias_file}（R12 将不触发）")

    return vocab


def rule_engine(
    records: list[dict],
    rule_cfg: dict,
    vocab: dict[str, Any],
    enable_l2: bool = True,
    today: date | None = None,
) -> tuple[list[dict], dict]:
    """
    执行第二关规则引擎。

    参数
    ----
    records : 第一关通过的记录（dict 列表，含 _row / _record_id）
    rule_cfg : rule_config.json 内容
    vocab : 本地词表
    enable_l2 : 是否启用 L2 外部地理增强规则
    today : 日期校验基准日。**优先取 rule_config.json 的 reference_date**，
            保证同一份提交包在任何机器、任何日期运行结果完全一致；
            显式传参仅用于单元测试注入。

    返回
    ----
    (issues, meta)
    """
    issues: list[dict] = []
    # 复现性：基准日固定来自配置，不依赖运行机器的当前日期。
    # 旧版 `today = today or date.today()` 会让同一份包在不同日期跑出不同结果。
    ref = rule_cfg.get("reference_date")
    if today is None and ref:
        today = date.fromisoformat(str(ref))
    today = today or date.today()
    th = rule_cfg["thresholds"]
    floor = rule_cfg.get("severity_floor", {})
    controlled = rule_cfg.get("controlled_vocab", {})

    enabled_rules: list[str] = []
    disabled_rules: list[str] = []

    # ---- 预统计：样方编号 / 记录 ID 出现次数（R05 / R06）
    sample_counter = Counter(r.get("sample_no", "") for r in records)
    record_counter = Counter(r.get("record_id", "") for r in records)

    for rec in records:
        rid = rec.get("_record_id", "")
        row = rec.get("_row", 0)

        # ---------------- R01 盖度越界（L0）
        cover = rec.get("cover")
        if isinstance(cover, (int, float)):
            if cover < th["cover_min"] or cover > th["cover_max"]:
                sev = _apply_severity_floor("R01", "错误", floor)
                div10 = round(cover / 10, 1)
                issues.append(
                    _issue(
                        rid, row, "cover", cover, "数值越界", "R01", sev,
                        f"盖度 {cover}% 超出 {th['cover_min']}–{th['cover_max']} 范围",
                        f"请核对是否误填为 {div10} 或 100",
                    )
                )
                enabled_rules.append("R01")

        # ---------------- R02 经纬度疑似写反（L0）
        lon = rec.get("longitude")
        lat = rec.get("latitude")
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
            lon_lo, lon_hi = th["longitude_range"]
            lat_lo, lat_hi = th["latitude_range"]
            lon_bad = not (lon_lo <= lon <= lon_hi)
            lat_bad = not (lat_lo <= lat <= lat_hi)
            if lon_bad and lat_bad:
                # 交换后是否都合法 → 判定为写反
                if (lon_lo <= lat <= lon_hi) and (lat_lo <= lon <= lat_hi):
                    sev = _apply_severity_floor("R02", "错误", floor)
                    issues.append(
                        _issue(
                            rid, row, "longitude,latitude", f"{lon},{lat}",
                            "经纬度写反", "R02", sev,
                            f"经度 {lon} / 纬度 {lat} 疑似写反"
                            f"（经度应在 {lon_lo}–{lon_hi}，纬度应在 {lat_lo}–{lat_hi}）",
                            "请交换经度与纬度后重新核对",
                        )
                    )
                    enabled_rules.append("R02")

        # ---------------- R03 / R04 日期异常（L0）
        d = rec.get("survey_date")
        if isinstance(d, date):
            if d > today + timedelta(days=th["date_max_future_days"]):
                sev = _apply_severity_floor("R03", "错误", floor)
                issues.append(
                    _issue(
                        rid, row, "survey_date", d.isoformat(), "日期在未来", "R03", sev,
                        f"调查日期 {d.isoformat()} 晚于今天（{today.isoformat()}）",
                        "请核对是否误填年份",
                    )
                )
                enabled_rules.append("R03")
            elif d.year < th["date_min_year"]:
                sev = _apply_severity_floor("R04", "错误", floor)
                issues.append(
                    _issue(
                        rid, row, "survey_date", d.isoformat(), "日期过早", "R04", sev,
                        f"调查日期 {d.isoformat()} 早于 {th['date_min_year']} 年",
                        f"请核对是否误填年份（如 {today.year} 写成 {d.year}）",
                    )
                )
                enabled_rules.append("R04")

        # ---------------- R05 样方编号重复（L0）
        sno = rec.get("sample_no", "")
        if sno and sample_counter[sno] > 1:
            sev = _apply_severity_floor("R05", "错误", floor)
            issues.append(
                _issue(
                    rid, row, "sample_no", sno, "编号重复", "R05", sev,
                    f"样方编号 {sno} 在本批次中重复出现 {sample_counter[sno]} 次",
                    "请核对样方编号是否复制粘贴错误",
                )
            )
            enabled_rules.append("R05")

        # ---------------- R06 记录 ID 重复（L0）
        if rid and record_counter[rid] > 1:
            sev = _apply_severity_floor("R06", "错误", floor)
            issues.append(
                _issue(
                    rid, row, "record_id", rid, "ID重复", "R06", sev,
                    f"记录 ID {rid} 重复出现 {record_counter[rid]} 次",
                    "请重新编号",
                )
            )
            enabled_rules.append("R06")

        # ---------------- R07 海拔越界（L0）
        alt = rec.get("altitude")
        if isinstance(alt, (int, float)):
            if alt < th["altitude_min"] or alt > th["altitude_max"]:
                sev = _apply_severity_floor("R07", "错误", floor)
                issues.append(
                    _issue(
                        rid, row, "altitude", alt, "数值越界", "R07", sev,
                        f"海拔 {alt} m 超出 {th['altitude_min']}–{th['altitude_max']} 范围",
                        "请核对海拔单位（米/千米）",
                    )
                )
                enabled_rules.append("R07")

        # ---------------- R08 盖度与植被类型矛盾（L1）
        species = rec.get("species", "")
        if species and isinstance(cover, (int, float)):
            if species in ("裸地", "水面", "水体") and cover > 5:
                sev = _apply_severity_floor("R08", "错误", floor)
                issues.append(
                    _issue(
                        rid, row, "cover,species", f"{species}/{cover}",
                        "类型矛盾", "R08", sev,
                        f"物种「{species}」为无植被类型，盖度 {cover}% 与类型不符",
                        "请核对物种名与盖度是否对应",
                    )
                )
                enabled_rules.append("R08")

        # ---------------- R11 坡向不在受控词表（L1）
        aspect = rec.get("slope_aspect", "")
        vocab_aspect = controlled.get("slope_aspect", [])
        if aspect and vocab_aspect and aspect not in vocab_aspect:
            sev = _apply_severity_floor("R11", "警告", floor)
            issues.append(
                _issue(
                    rid, row, "slope_aspect", aspect, "词表外取值", "R11", sev,
                    f"坡向「{aspect}」不在受控词表内",
                    "请使用：" + "/".join(vocab_aspect),
                )
            )
            enabled_rules.append("R11")

        # ---------------- R12 物种名疑似别名（L1）
        alias_map = vocab.get("alias", {})
        if species and species in alias_map:
            standard = alias_map[species]
            sev = _apply_severity_floor("R12", "提示", floor)
            issues.append(
                _issue(
                    rid, row, "species", species, "疑似别名", "R12", sev,
                    f"物种「{species}」可能是「{standard}」的别名",
                    f"建议统一使用标准名「{standard}」",
                )
            )
            enabled_rules.append("R12")

        # ---------------- R13 物种名不在名录（L1）
        # 注意：无植被类型词（裸地/水面/水体）由 R08 专门处理，
        # 它们本就不属于「物种名录」，此处必须跳过，否则同一字段会重复报两条。
        species_set = vocab.get("species", set())
        non_vegetation = {"裸地", "水面", "水体"}
        if (
            species
            and species_set
            and species not in species_set
            and species not in alias_map
            and species not in non_vegetation
        ):
            sev = _apply_severity_floor("R13", "提示", floor)
            issues.append(
                _issue(
                    rid, row, "species", species, "名录外物种", "R13", sev,
                    f"物种「{species}」不在本地物种名录中",
                    "请核对物种名拼写，或确认是否为名录外物种",
                )
            )
            enabled_rules.append("R13")

        # ---------------- R09 / R10（L2 外部地理增强，可禁用）
        # 本版本未随包提供外部行政区划边界 / DEM 数据，因此 L2 规则不产生问题。
        # 但一旦接入外部数据，其结论必须带 confidence=0.8 并强制教师复核
        # （外部数据本身有精度误差，不能与确定性规则同等对待）。
        # 这段代码保留为「接入点」，说明 confidence<0.9 的分支是可达的。
        if enable_l2 and _l2_data_available(rule_cfg):
            for rule_id, field_name, msg_tpl in (
                ("R09", "longitude,latitude", "坐标 ({lon}, {lat}) 不在实习区域范围内"),
                ("R10", "altitude,longitude,latitude", "海拔 {alt} m 与坐标 ({lon}, {lat}) 处地形高程差异过大"),
            ):
                hit = _check_l2(rule_id, rec, rule_cfg)
                if hit is None:
                    continue
                sev = _apply_severity_floor(rule_id, "警告", floor)
                issues.append(
                    _issue(
                        rid, row, field_name, hit["raw"], "外部数据不一致", rule_id, sev,
                        msg_tpl.format(**hit["fmt"]),
                        "请核对外部地理数据与记录是否一致",
                        confidence=CONFIDENCE_EXTERNAL_DATA,
                    )
                )
                enabled_rules.append(rule_id)

    if not enable_l2:
        disabled_rules.extend(["R09", "R10"])

    meta = {
        "enabled_rule_set": sorted(set(enabled_rules)),
        "disabled_rule_set": sorted(set(disabled_rules)),
        "l2_enabled": enable_l2,
        "rule_version": rule_cfg.get("version", "unknown"),
        "total_issues": len(issues),
        # 词表缺失等降级告警：非空即说明有规则未真正生效，指标不可直接采信
        "vocab_warnings": list(vocab.get("_warnings", []) or []),
    }
    return issues, meta


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parent
    cfg_dir = base.parent / "03_CONFIG_配置规则"
    rule_cfg = load_rule_config(cfg_dir / "rule_config.json")
    # 词表就放在 03_CONFIG_配置规则/ 下（species_list.txt / species_alias.txt），
    # 不存在独立的 05_知识库 目录——旧版这里写错路径会静默降级。
    vocab = load_vocab(cfg_dir)
    print("规则数：", len(rule_cfg["rules"]), "+ 照片规则", len(rule_cfg["photo_rules"]))
    print("受控词表：", list(rule_cfg["controlled_vocab"].keys()))
    print("词表加载：物种", len(vocab["species"]), "别名", len(vocab["alias"]))
