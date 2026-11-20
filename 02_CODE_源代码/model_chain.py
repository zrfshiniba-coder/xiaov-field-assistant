# -*- coding: utf-8 -*-
"""
小V野外助教 · 第三关：AI 补充提示（不覆盖规则）
==================================================
作者：周荣发
学号：2026210209
班级：地理科学262
学校：滁州学院
日期：2026-11-03

版权声明：本文件为周荣发（2026210209）原创作品，仅用于滁州学院2026级
新生人工智能创新应用比赛提交与评审。未经作者书面许可，不得用于商业用途。

设计原则（核心）
----------------
1. **AI 只能新增「提示级」内容**，不能修改、降级或覆盖规则结论。
2. **AI 提示必须绑定 record_id + field + 证据来源**，不生成泛化建议。
3. **与规则结论重复的 AI 提示自动抑制**。
4. **低置信度（<0.6）AI 提示不进主清单**，只进可选附录。
5. **三种模式**：
   - online  ：真实调用模型
   - offline ：跳过第三关（ai_status=offline_skipped）
   - demo_ai ：加载本地缓存 ai_sample_cache.json（标注「离线演示缓存」）
6. 模型名只写在 config.json，本文件不含任何硬编码模型名。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hard_check import make_issue_id

__author__ = "周荣发"
__student_id__ = "2026210209"
__version__ = "1.0.0"

AI_MAX_SEVERITY = "提示"
LOW_CONFIDENCE = 0.6

SYSTEM_PROMPT = """你是地理野外实习数据质检助手。你的任务是**补充提示**，不是判定错误。

严格规则：
1. 你只能输出「提示」级建议，不能输出「错误」或「警告」级判定。
2. 你不能推翻任何已有规则结论。
3. 每条提示必须绑定具体的 record_id 和 field。
4. 只输出 JSON 数组，不要任何解释文字。
5. 如果没有可补充的提示，输出空数组 []。

输出格式（每条）：
{"record_id": "S001", "field": "species", "message": "提示内容", "suggestion": "建议", "confidence": 0.8}
"""


def _ai_issue(
    record_id: str,
    row: int,
    field: str,
    message: str,
    suggestion: str,
    confidence: float,
    related_rule_ids: list[str] | None = None,
) -> dict:
    """构造 AI 提示记录（强制 severity=提示）。"""
    return {
        "issue_id": make_issue_id(record_id, field, "AI", "ai"),
        "record_id": record_id,
        "row": row,
        "field": field,
        "raw_value": "",
        "error_type": "AI补充提示",
        "rule_id": "AI",
        "source": "ai",
        "severity": AI_MAX_SEVERITY,
        "confidence": round(float(confidence), 2),
        "message": message,
        "suggestion": suggestion,
        "ai_can_override": False,
        "need_teacher_confirm": False,
        "related_rule_ids": related_rule_ids or [],
        "conflict_with_rule": False,
    }


def _suppress_duplicates(ai_issues: list[dict], rule_issues: list[dict]) -> tuple[list[dict], int]:
    """
    抑制与规则结论重复的 AI 提示。
    判据：同一 record_id + field 已有规则问题 → 该 AI 提示被抑制。
    返回 (保留的 AI 提示, 抑制数量)。
    """
    rule_keys = {(it["record_id"], it["field"]) for it in rule_issues}
    kept: list[dict] = []
    suppressed = 0
    for it in ai_issues:
        key = (it["record_id"], it["field"])
        if key in rule_keys:
            suppressed += 1
            continue
        kept.append(it)
    return kept, suppressed


def _filter_low_confidence(ai_issues: list[dict]) -> tuple[list[dict], list[dict]]:
    """低置信度提示分离到附录。返回 (主清单, 附录)。"""
    main: list[dict] = []
    appendix: list[dict] = []
    for it in ai_issues:
        if it["confidence"] < LOW_CONFIDENCE:
            appendix.append(it)
        else:
            main.append(it)
    return main, appendix


def _call_model(records: list[dict], role_cfg: dict, api_key: str) -> list[dict]:
    """
    真实调用模型（OpenAI 兼容接口）。
    失败时抛异常，由上层决定降级策略。
    """
    import requests  # 延迟导入，离线模式不需要

    payload_records = [
        {
            "record_id": r.get("_record_id"),
            "species": r.get("species"),
            "cover": r.get("cover"),
            "slope_aspect": r.get("slope_aspect"),
            "altitude": r.get("altitude"),
        }
        for r in records
    ]

    body = {
        "model": role_cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "以下是已通过格式校验与规则校验的记录，请补充提示：\n"
                + json.dumps(payload_records, ensure_ascii=False),
            },
        ],
        "temperature": role_cfg.get("temperature", 0.3),
        "reasoning_effort": role_cfg.get("reasoning_effort", "max"),
    }

    resp = requests.post(
        role_cfg["base_url"].rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=role_cfg.get("timeout", 120),
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()

    # 容错：剥离可能的 ```json 包裹
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)
    return data if isinstance(data, list) else []


def _load_demo_cache(cache_path: Path) -> list[dict]:
    """加载离线演示缓存。"""
    if not cache_path.exists():
        return []
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _postprocess(
    ai_raw: list[dict],
    rule_issues: list[dict],
    config: dict,
) -> tuple[list[dict], list[dict], dict]:
    """
    对「模型原始输出」做统一后处理（可独立单测，不依赖网络）。

    处理顺序（顺序不可颠倒）：
      1. 强制降级：任何 severity 一律改为「提示」，ai_can_override 一律 False
      2. 去重抑制：与规则结论同 record_id + field 的 AI 提示被抑制
      3. 低置信度分离：confidence < 阈值 → 附录，不进主清单

    参数
    ----
    ai_raw : 模型原始输出（或 demo 缓存），每条含 record_id/field/message/suggestion/confidence
    rule_issues : 第一关 + 第二关已产生的问题
    config : config.json 内容（读取 ai_policy）

    返回
    ----
    (main_ai_issues, appendix_ai_issues, meta)
    """
    policy = (config or {}).get("ai_policy", {}) or {}
    threshold = float(policy.get("low_confidence_threshold", LOW_CONFIDENCE))
    suppress = bool(policy.get("suppress_duplicate_with_rule", True))

    meta = {
        "suppressed_duplicates": 0,
        "low_confidence_count": 0,
        "forced_downgrade_count": 0,
    }

    # ---- 1. 强制降级为「提示」级 ----
    normalized: list[dict] = []
    for item in ai_raw or []:
        rid = str(item.get("record_id", "") or "")
        field = str(item.get("field", "") or "")
        if not rid or not field:
            continue  # 缺关键字段 → 丢弃（防幻觉）
        raw_sev = str(item.get("severity", AI_MAX_SEVERITY) or AI_MAX_SEVERITY)
        if raw_sev != AI_MAX_SEVERITY:
            meta["forced_downgrade_count"] += 1
        try:
            conf = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        normalized.append(
            {
                "issue_id": make_issue_id(rid, field, "AI", "ai"),
                "record_id": rid,
                "row": int(item.get("row", 0) or 0),
                "field": field,
                "raw_value": str(item.get("raw_value", "") or ""),
                "error_type": "AI补充提示",
                "rule_id": "AI",
                "source": "ai",
                "severity": AI_MAX_SEVERITY,   # 强制
                "confidence": round(conf, 2),
                "message": str(item.get("message", "") or ""),
                "suggestion": str(item.get("suggestion", "") or ""),
                "ai_can_override": False,      # 强制
                "need_teacher_confirm": False,
                "related_rule_ids": list(item.get("related_rule_ids", []) or []),
                "conflict_with_rule": False,
            }
        )

    # ---- 2. 去重抑制 ----
    if suppress:
        normalized, suppressed = _suppress_duplicates(normalized, rule_issues)
        meta["suppressed_duplicates"] = suppressed

    # ---- 3. 低置信度分离 ----
    main: list[dict] = []
    appendix: list[dict] = []
    for it in normalized:
        if it["confidence"] < threshold:
            appendix.append(it)
        else:
            main.append(it)
    meta["low_confidence_count"] = len(appendix)

    return main, appendix, meta


def model_chain(
    records: list[dict],
    rule_issues: list[dict],
    config: dict,
    mode: str = "online",
    cache_path: str | Path | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """
    执行第三关 AI 补充提示。

    参数
    ----
    records : 第一关通过的记录
    rule_issues : 第一关 + 第二关已产生的问题（用于去重）
    config : config.json 内容
    mode : "online" / "offline" / "demo_ai"
    cache_path : demo_ai 模式的缓存文件路径

    返回
    ----
    (main_ai_issues, appendix_ai_issues, meta)
    """
    meta = {
        "mode": mode,
        "ai_enabled": False,
        "ai_status": "",
        "model": "",
        "suppressed_duplicates": 0,
        "low_confidence_count": 0,
    }

    if mode == "offline":
        meta["ai_status"] = "offline_skipped"
        return [], [], meta

    raw: list[dict] = []

    if mode == "demo_ai":
        cp = Path(cache_path) if cache_path else Path(__file__).parent / "ai_sample_cache.json"
        raw = _load_demo_cache(cp)
        meta["ai_status"] = "demo_cache_loaded"
        meta["ai_enabled"] = True
        meta["model"] = "demo_cache（离线演示缓存，非实时模型调用）"
    else:
        role_cfg = config["model_policy"]["roles"]["reviewer"]
        api_key = os.environ.get(config["model_policy"].get("api_key_env", "DASHSCOPE_API_KEY"), "")
        if not api_key:
            # 无 key → 自动降级为离线
            meta["ai_status"] = "no_api_key_offline_fallback"
            return [], [], meta
        try:
            raw = _call_model(records, role_cfg, api_key)
            meta["ai_status"] = "online_ok"
            meta["ai_enabled"] = True
            meta["model"] = role_cfg["model"]
        except Exception as exc:  # noqa: BLE001
            meta["ai_status"] = f"online_failed_offline_fallback: {type(exc).__name__}"
            # DEMO-DIAG: 把真实异常写入文件，便于定位（仅 _demo_pkg 副本）
            try:
                import traceback as _tb
                _detail = "%s: %s\n%s" % (type(exc).__name__, exc, _tb.format_exc())
                _resp = getattr(exc, 'response', None)
                if _resp is not None:
                    _detail += "\nHTTP %s\nBODY: %s" % (
                        getattr(_resp, 'status_code', '?'),
                        (getattr(_resp, 'text', '') or '')[:800])
                with open(Path(__file__).parent / "_ai_error.log", "a",
                          encoding="utf-8") as _f:
                    _f.write("=== %s\n%s\n" % (meta.get("model", ""), _detail))
            except Exception:
                pass
            return [], [], meta

    # 过滤模型幻觉出的不存在记录（row 回填）
    row_map = {r.get("_record_id"): r.get("_row", 0) for r in records}
    filtered_raw: list[dict] = []
    for item in raw:
        rid = item.get("record_id", "")
        if rid not in row_map:
            continue  # 模型幻觉出的不存在记录 → 丢弃
        item = dict(item)
        item["row"] = row_map[rid]
        filtered_raw.append(item)

    # 统一后处理：强制降级 → 去重抑制 → 低置信度分离
    main, appendix, post_meta = _postprocess(filtered_raw, rule_issues, config)
    meta["suppressed_duplicates"] = post_meta["suppressed_duplicates"]
    meta["low_confidence_count"] = post_meta["low_confidence_count"]
    meta["forced_downgrade_count"] = post_meta["forced_downgrade_count"]

    return main, appendix, meta


if __name__ == "__main__":
    print("model_chain.py · 第三关 AI 补充提示")
    print("模式：online / offline / demo_ai")
    print("原则：AI 只能新增提示级内容，不能覆盖规则结论")
