# -*- coding: utf-8 -*-
"""
小V野外助教 · 视觉录入层（计算机视觉入口）
=============================================
作者：周荣发
学号：2026210209
班级：地理科学262
学校：滁州学院
日期：2026-11-09（v2，吸收 qwen 第 1 轮评审）

版权声明：本文件为周荣发（2026210209）原创作品，仅用于滁州学院2026级
新生人工智能创新应用比赛提交与评审。未经作者书面许可，不得用于商业用途。

解决的问题（对应比赛通知「须结合计算机视觉」）
------------------------------------------------
野外实习现场没有电脑，学生用手机拍下纸质记录表。本模块调用视觉模型，
把「一张照片」变成「结构化记录 + 字段级置信度」，再交给四道关校验链。
这是本作品「计算机视觉」能力的落点：**只做识别，不做判定**——
识别结果仍要过第一关格式硬校验、第二关规则引擎，CV 无权定对错。

v2 相对 v1 的改动（吸收 qwen 第 1 轮 5 条核心意见）
--------------------------------------------------
1. 范围从 rule_config.json 的 thresholds 读取，不写死在提示词里（可迁移）。
2. 增加字段级置信度 + source_type + status + need_review 接口契约。
3. 防幻觉优先级明确：看不清输出空串 优先于 范围约束，禁止模型为符合范围而编造。
4. vision / text 模型角色分离：视觉必须用多模态模型，纯文本模型不得作 vision 角色。
5. 支持一图多条记录：输出 records 数组，无法确定行归属时标记 need_review。

用法
-----
    python vision_import.py "照片路径.jpg"                # 单张识别
    python vision_import.py "照片1.jpg" "照片2.jpg"       # 多张识别
    python vision_import.py "照片.jpg" --prompt-v1        # 用粗糙提示词（A/B 对比）
    python vision_import.py "照片.jpg" --out result.json  # 结果存文件
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

__author__ = "周荣发"
__student_id__ = "2026210209"
__version__ = "1.1.0"

BASE = Path(__file__).resolve().parent
PKG = BASE.parent

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# 四道关需要的 9 个字段（与 config.json data.required_fields 一致）
FIELDS = [
    "record_id", "sample_no", "longitude", "latitude", "species",
    "cover", "slope_aspect", "survey_date", "altitude",
]

# ============================================================
# 配置读取（阈值从 rule_config.json 读，不写死在提示词）
# ============================================================
def _load_rule_config() -> dict:
    p = PKG / "03_CONFIG_配置规则" / "rule_config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _get_thresholds() -> dict:
    return _load_rule_config().get("thresholds", {})


def _load_config() -> dict:
    cfg_path = PKG / "03_CONFIG_配置规则" / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _get_vision_role(cfg: dict) -> dict:
    """读 config.json 的 vision 角色（必须多模态模型）。"""
    role = cfg.get("model_policy", {}).get("roles", {}).get("vision", {})
    if role:
        return role
    return {
        "provider": "dashscope",
        "model": "qwen3-vl-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "timeout": 180,
    }


def _get_api_key(cfg: dict) -> str:
    env = cfg.get("model_policy", {}).get("api_key_env", "DASHSCOPE_API_KEY")
    return os.environ.get(env, "").strip()


# ============================================================
# 提示词（V1 粗糙版 / V2 优化版，范围引用配置值）
# ============================================================
def _build_prompt(use_v1: bool = False) -> str:
    if use_v1:
        return "识别这张图里的野外实习记录内容。"

    th = _get_thresholds()
    lon = th.get("longitude_range", [117.0, 119.5])
    lat = th.get("latitude_range", [31.5, 33.5])
    alt = [th.get("altitude_min", 0), th.get("altitude_max", 2000)]
    cover = [th.get("cover_min", 0), th.get("cover_max", 100)]
    date_min = th.get("date_min_year", 2020)

    return (
        "你是地理野外实习记录识别助手。请从这张野外实习记录表照片中，"
        "识别所有记录，严格输出一个 JSON 对象（不要任何解释文字、不要 markdown 代码块）：\n"
        "\n"
        "{\n"
        '  "records": [\n'
        "    {\n"
        '      "record_id": "记录编号，如 LY-001",\n'
        '      "sample_no": "样方编号，如 Q01",\n'
        f'      "longitude": "经度数值，东经，范围 {lon[0]}~{lon[1]}",\n'
        f'      "latitude": "纬度数值，北纬，范围 {lat[0]}~{lat[1]}",\n'
        '      "species": "物种名，中文，用规范名不用俗名",\n'
        f'      "cover": "盖度数值，百分比，范围 {cover[0]}~{cover[1]}",\n'
        '      "slope_aspect": "坡向，只能取 东南/南/西南/西/西北/北/东北/东 之一",\n'
        f'      "survey_date": "调查日期，格式 YYYY-MM-DD，年份不早于 {date_min}",\n'
        f'      "altitude": "海拔数值，米，范围 {alt[0]}~{alt[1]}",\n'
        '      "confidence": "该条记录整体置信度 0~1"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "识别规则（按优先级）：\n"
        "1. 【最高优先级】看不清、不确定的字段，一律输出空字符串 \"\"，严禁编造。"
        "即使你知道该字段大概率落在某个范围内，也不得推测补全。\n"
        "2. 数字按十进制小数输出，不要保留度分秒符号。\n"
        "3. 坡向统一用标准 8 方位词；遇到「东南偏南」这类非标准写法，"
        "若不能确定归入哪个标准方位，输出空串，不要强行改写。\n"
        "4. 一张照片可能含多条记录：输出 records 数组。无法确定行归属时，"
        "把整张照片的字段集中输出为一条，并在 confidence 取低值。\n"
        "5. 只输出一个 JSON 对象本身，前后不要有任何文字。"
    )


# ============================================================
# 图片 → data URL
# ============================================================
def image_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图片不存在：{path}")
    ext = p.suffix.lower()
    if ext not in IMAGE_EXTS:
        raise ValueError(f"不支持的图片格式：{ext}（支持 {', '.join(sorted(IMAGE_EXTS))}）")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ============================================================
# API 调用（OpenAI 兼容协议，带重试 + 超时）
# ============================================================
def _api_call(base_url: str, api_key: str, model: str, messages: list,
              timeout: int = 180, retries: int = 3) -> str:
    import time
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "reasoning_effort": "high",
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"视觉模型调用失败（重试 {retries} 次）：{last_err}")


# ============================================================
# 解析模型输出（容错：剥 markdown、找第一个 JSON）
# ============================================================
def _parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型输出不含 JSON 对象：{text[:200]}")
    return json.loads(s[start:end + 1])


# ============================================================
# 接口契约：把模型原始输出 → 带置信度的标准记录
# ============================================================
def _normalize_records(parsed: dict, image_path: str, model_name: str,
                       prompt_version: str) -> dict:
    """输出接口契约，供第一关直接消费。"""
    raw_records = parsed.get("records", [parsed])  # 兼容单对象
    if not isinstance(raw_records, list):
        raw_records = [raw_records]

    records = []
    for r in raw_records:
        if not isinstance(r, dict):
            continue
        fields = {}
        need_review = []
        for f in FIELDS:
            v = str(r.get(f, "")).strip()
            fields[f] = v
            if not v:
                need_review.append(f)  # 空串字段 → 待复核
        # 整体置信度：模型给的值，缺失默认 0
        conf = r.get("confidence", 0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        records.append({
            "record": fields,
            "field_confidence": {"overall": round(conf, 2)},
            "source_type": "vision",
            "raw_image_ref": image_path,
            "recognition_status": "need_review" if need_review else "recognized",
            "need_review_fields": need_review,
            "model_name": model_name,
            "prompt_version": prompt_version,
        })

    return {
        "records": records,
        "image": image_path,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }


# ============================================================
# 主识别
# ============================================================
def recognize(image_path: str, use_v1: bool = False) -> dict:
    """识别一张照片，返回接口契约 dict。"""
    cfg = _load_config()
    role = _get_vision_role(cfg)
    api_key = _get_api_key(cfg)
    if not api_key:
        raise RuntimeError(
            "缺少 API Key：请设置环境变量 "
            + cfg.get("model_policy", {}).get("api_key_env", "DASHSCOPE_API_KEY")
        )

    data_url = image_to_data_url(image_path)
    prompt = _build_prompt(use_v1=use_v1)
    prompt_version = "v1" if use_v1 else "v2"
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    raw = _api_call(role.get("base_url"), api_key, role.get("model"), messages,
                    timeout=role.get("timeout", 180))
    obj = json.loads(raw)
    content = obj["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    parsed = _parse_json(content)

    return _normalize_records(parsed, image_path, role.get("model"), prompt_version)


def main() -> int:
    ap = argparse.ArgumentParser(description="小V野外助教 · 视觉录入层")
    ap.add_argument("images", nargs="+", help="图片路径（可多张）")
    ap.add_argument("--prompt-v1", action="store_true", help="用粗糙提示词（A/B 对比）")
    ap.add_argument("--out", help="结果 JSON 输出路径")
    args = ap.parse_args()

    results = []
    for img in args.images:
        print(f"\n[识别] {img}")
        rec = recognize(img, use_v1=args.prompt_v1)
        results.append(rec)
        for r in rec["records"]:
            for f in FIELDS:
                v = r["record"].get(f, "")
                print(f"    {f:<14}: {v if v else '（空）'}")
            print(f"    置信度        : {r['field_confidence']['overall']}")
            print(f"    状态          : {r['recognition_status']}")
            if r["need_review_fields"]:
                print(f"    待复核字段    : {', '.join(r['need_review_fields'])}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[输出] {args.out}")

    print("\n[说明] 视觉层只做识别，不判定对错；空串字段标记 need_review，"
          "识别结果需继续送第一关格式硬校验 + 第二关规则引擎。")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
