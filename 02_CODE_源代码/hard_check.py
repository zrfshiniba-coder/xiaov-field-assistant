# -*- coding: utf-8 -*-
"""
小V野外助教 · 第一关：格式硬校验（F01–F08）
================================================
作者：周荣发
学号：2026210209
班级：地理科学262
学校：滁州学院
日期：2026-11-03

版权声明：本文件为周荣发（2026210209）原创作品，仅用于滁州学院2026级
新生人工智能创新应用比赛提交与评审。未经作者书面许可，不得用于商业用途。

设计原则
--------
1. 本关**只做确定性校验**：字段是否存在、类型是否可解析、是否为空。
2. 不做任何业务判断（业务判断在第二关 rule_engine.py）。
3. 不调用任何模型，纯 Python 标准库，离线可跑。
4. 输出统一结构，供后续关卡与教师复核使用。
"""

from __future__ import annotations

import csv
import hashlib
import io
from datetime import date, datetime
from pathlib import Path
from typing import Any

__author__ = "周荣发"
__student_id__ = "2026210209"
__version__ = "1.0.0"

# ---------------------------------------------------------------- 错误码定义
# F01 缺字段 / F02 空值 / F03 类型不可解析 / F04 日期格式错
# F05 编码异常 / F06 表头重复 / F07 行长度不符 / F08 文件为空
ERROR_CODES = {
    "F01": "缺少必填字段",
    "F02": "必填字段为空",
    "F03": "类型不可解析",
    "F04": "日期格式错误",
    "F05": "文件编码异常",
    "F06": "表头字段重复",
    "F07": "行列数与表头不符",
    "F08": "文件为空或无数据行",
}

DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d")


def make_issue_id(record_id: str, field: str, rule_id: str, source: str = "rule") -> str:
    """生成稳定问题 ID：record_id + field + rule_id + source 的哈希前 8 位。"""
    raw = f"{record_id}|{field}|{rule_id}|{source}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


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
) -> dict:
    """构造统一的问题记录结构。"""
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
        "confidence": 1.0,
        "message": message,
        "suggestion": suggestion,
        "ai_can_override": False,
        "need_teacher_confirm": False,
        "related_rule_ids": [],
        "conflict_with_rule": False,
    }


def _parse_float(value: str) -> float | None:
    """尝试解析浮点数，失败返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_date(value: str) -> date | None:
    """尝试按多种格式解析日期，失败返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def read_csv_text(path: str | Path) -> tuple[str | None, str | None]:
    """
    读取 CSV 文本，自动尝试 utf-8-sig / utf-8 / gbk。
    返回 (文本, 错误信息)。成功时错误信息为 None。
    """
    p = Path(path)
    if not p.exists():
        return None, f"文件不存在：{p}"
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc), None
        except UnicodeDecodeError:
            continue
    return None, "文件编码无法识别（已尝试 utf-8-sig / utf-8 / gbk / gb18030）"


def hard_check(
    csv_path: str | Path,
    required_fields: list[str],
    field_types: dict[str, str],
) -> tuple[list[dict], list[dict], dict]:
    """
    执行第一关格式硬校验。

    参数
    ----
    csv_path : CSV 文件路径
    required_fields : 必填字段列表（来自 config.json）
    field_types : 字段类型映射（str / float / date）

    返回
    ----
    (issues, records, meta)
      issues  : 问题清单（F01–F08）
      records : 通过格式校验的记录（dict 列表，值已按类型转换）
      meta    : 元信息（总行数 / 通过数 / 失败数 / 编码）
    """
    issues: list[dict] = []
    records: list[dict] = []
    meta = {
        "total_rows": 0,
        "passed_rows": 0,
        "failed_rows": 0,
        "encoding": "",
        "file": str(csv_path),
    }

    text, err = read_csv_text(csv_path)
    if err is not None:
        issues.append(
            _issue(
                record_id="FILE",
                row=0,
                field="",
                raw_value="",
                error_type=ERROR_CODES["F05"],
                rule_id="F05",
                severity="错误",
                message=err,
                suggestion="请用 Excel 另存为「CSV UTF-8（逗号分隔）」后重试",
            )
        )
        return issues, records, meta

    # 探测编码（用于 meta 记录）
    raw = Path(csv_path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            raw.decode(enc)
            meta["encoding"] = enc
            break
        except UnicodeDecodeError:
            continue

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        issues.append(
            _issue(
                record_id="FILE",
                row=0,
                field="",
                raw_value="",
                error_type=ERROR_CODES["F08"],
                rule_id="F08",
                severity="错误",
                message="文件为空，没有表头",
                suggestion="请确认上传的是含表头的 CSV 文件",
            )
        )
        return issues, records, meta

    header = [h.strip() for h in header]

    # F06 表头重复
    seen: dict[str, int] = {}
    for idx, name in enumerate(header):
        if name in seen:
            issues.append(
                _issue(
                    record_id="FILE",
                    row=0,
                    field=name,
                    raw_value=name,
                    error_type=ERROR_CODES["F06"],
                    rule_id="F06",
                    severity="错误",
                    message=f"表头字段「{name}」重复出现（第 {seen[name] + 1} 列与第 {idx + 1} 列）",
                    suggestion="请删除重复列后重新上传",
                )
            )
        else:
            seen[name] = idx

    # F01 缺字段
    missing = [f for f in required_fields if f not in header]
    for f in missing:
        issues.append(
            _issue(
                record_id="FILE",
                row=0,
                field=f,
                raw_value="",
                error_type=ERROR_CODES["F01"],
                rule_id="F01",
                severity="错误",
                message=f"缺少必填字段「{f}」",
                suggestion=f"请在表头中加入「{f}」列",
            )
        )

    if missing:
        # 缺字段时无法继续逐行校验
        return issues, records, meta

    col_index = {name: i for i, name in enumerate(header)}
    expected_len = len(header)

    for line_no, row in enumerate(reader, start=2):  # 第 1 行是表头
        if not any(cell.strip() for cell in row):
            continue  # 跳过空行
        meta["total_rows"] += 1

        # F07 行长度不符
        if len(row) != expected_len:
            issues.append(
                _issue(
                    record_id=f"ROW{line_no}",
                    row=line_no,
                    field="",
                    raw_value=",".join(row),
                    error_type=ERROR_CODES["F07"],
                    rule_id="F07",
                    severity="错误",
                    message=f"该行有 {len(row)} 列，表头有 {expected_len} 列",
                    suggestion="请检查是否有未转义的逗号或缺失的单元格",
                )
            )
            meta["failed_rows"] += 1
            continue

        record_id = row[col_index["record_id"]].strip() or f"ROW{line_no}"
        row_ok = True
        parsed: dict[str, Any] = {"_row": line_no, "_record_id": record_id}

        for field in required_fields:
            raw_value = row[col_index[field]]
            text_value = raw_value.strip()

            # F02 空值
            if text_value == "":
                issues.append(
                    _issue(
                        record_id=record_id,
                        row=line_no,
                        field=field,
                        raw_value=raw_value,
                        error_type=ERROR_CODES["F02"],
                        rule_id="F02",
                        severity="错误",
                        message=f"必填字段「{field}」为空",
                        suggestion="请补填该字段",
                    )
                )
                row_ok = False
                continue

            ftype = field_types.get(field, "str")

            # F03 类型不可解析
            if ftype == "float":
                num = _parse_float(text_value)
                if num is None:
                    issues.append(
                        _issue(
                            record_id=record_id,
                            row=line_no,
                            field=field,
                            raw_value=raw_value,
                            error_type=ERROR_CODES["F03"],
                            rule_id="F03",
                            severity="错误",
                            message=f"字段「{field}」的值「{text_value}」不是有效数字",
                            suggestion="请填写纯数字，不要带单位或文字",
                        )
                    )
                    row_ok = False
                    continue
                parsed[field] = num

            elif ftype == "date":
                d = _parse_date(text_value)
                if d is None:
                    issues.append(
                        _issue(
                            record_id=record_id,
                            row=line_no,
                            field=field,
                            raw_value=raw_value,
                            error_type=ERROR_CODES["F04"],
                            rule_id="F04",
                            severity="错误",
                            message=f"字段「{field}」的日期「{text_value}」格式无法识别",
                            suggestion="请使用 YYYY-MM-DD 格式，如 2026-10-15",
                        )
                    )
                    row_ok = False
                    continue
                parsed[field] = d

            else:
                parsed[field] = text_value

        if row_ok:
            records.append(parsed)
            meta["passed_rows"] += 1
        else:
            meta["failed_rows"] += 1

    if meta["total_rows"] == 0:
        issues.append(
            _issue(
                record_id="FILE",
                row=0,
                field="",
                raw_value="",
                error_type=ERROR_CODES["F08"],
                rule_id="F08",
                severity="错误",
                message="文件只有表头，没有数据行",
                suggestion="请确认数据已填写",
            )
        )

    return issues, records, meta


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("用法：python hard_check.py <csv路径>")
        sys.exit(1)

    cfg_path = Path(__file__).resolve().parent.parent / "03_CONFIG_配置规则" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    iss, recs, mt = hard_check(
        sys.argv[1],
        cfg["data"]["required_fields"],
        cfg["data"]["field_types"],
    )
    print(f"总行数 {mt['total_rows']} / 通过 {mt['passed_rows']} / 失败 {mt['failed_rows']}")
    print(f"问题数 {len(iss)}")
    for it in iss[:20]:
        print(f"  [{it['rule_id']}] 行{it['row']} {it['field']}: {it['message']}")
