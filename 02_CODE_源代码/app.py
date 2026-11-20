# -*- coding: utf-8 -*-
"""
小V野外助教 · Flask 表单服务
==============================
作者：周荣发
学号：2026210209
班级：地理科学262
学校：滁州学院
日期：2026-11-03

版权声明：本文件为周荣发（2026210209）原创作品，仅用于滁州学院2026级
新生人工智能创新应用比赛提交与评审。未经作者书面许可，不得用于商业用途。

安全边界
--------
1. 只绑定 127.0.0.1，不监听局域网。
2. 只接受 .csv 文件，大小 ≤10MB。
3. 上传文件名安全化，防路径穿越。
4. 不执行上传文件，只解析文本。
5. 日志不含原始数据，只含行号/字段名/规则编号。
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
PKG = BASE.parent
sys.path.insert(0, str(BASE))

from hard_check import hard_check  # noqa: E402
from model_chain import model_chain  # noqa: E402
from rule_engine import load_rule_config, load_vocab, rule_engine  # noqa: E402

__author__ = "周荣发"
__student_id__ = "2026210209"
__version__ = "1.0.0"

CFG = json.loads((PKG / "03_CONFIG_配置规则" / "config.json").read_text(encoding="utf-8"))
RULE_CFG = load_rule_config(PKG / "03_CONFIG_配置规则" / "rule_config.json")
VOCAB = load_vocab(PKG / "05_知识库")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "xiaov-field-assistant-local-only")
app.config["MAX_CONTENT_LENGTH"] = CFG["server"]["max_upload_mb"] * 1024 * 1024

ALLOWED_EXT = set(CFG["server"]["allowed_ext"])

# 最近一次结果（单机单用户，内存保存即可）
LAST_RESULT: dict = {}


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        app_name=CFG["app"]["name"],
        subtitle=CFG["app"]["subtitle"],
        author=CFG["app"]["author"],
        student_id=CFG["app"]["student_id"],
        klass=CFG["app"]["class"],
        school=CFG["app"]["school"],
        max_mb=CFG["server"]["max_upload_mb"],
    )


@app.route("/upload", methods=["POST"])
def upload():
    global LAST_RESULT

    file = request.files.get("csv_file")
    if file is None or file.filename == "":
        flash("请选择要上传的 CSV 文件")
        return redirect(url_for("index"))

    if not _allowed(file.filename):
        flash(f"只支持 {', '.join(sorted(ALLOWED_EXT))} 文件")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename) or "upload.csv"
    if not safe_name.lower().endswith(".csv"):
        safe_name = "upload.csv"

    # 写入临时文件（不执行，只解析）
    tmp_dir = Path(tempfile.mkdtemp(prefix="xiaov_"))
    tmp_path = tmp_dir / safe_name
    file.save(str(tmp_path))

    # 第一关
    issues_1, records, meta_1 = hard_check(
        tmp_path, CFG["data"]["required_fields"], CFG["data"]["field_types"]
    )

    # 第二关
    issues_2, meta_2 = rule_engine(records, RULE_CFG, VOCAB, enable_l2=True)

    # 第三关（无 key 自动降级离线）
    ai_main, ai_appendix, meta_3 = model_chain(
        records, issues_1 + issues_2, CFG, mode="online",
        cache_path=BASE / "ai_sample_cache.json",
    )

    all_issues = issues_1 + issues_2 + ai_main

    by_sev = {"错误": 0, "警告": 0, "提示": 0}
    for it in all_issues:
        by_sev[it["severity"]] = by_sev.get(it["severity"], 0) + 1

    LAST_RESULT = {
        "file": safe_name,
        "meta_1": meta_1,
        "meta_2": meta_2,
        "meta_3": meta_3,
        "issues": all_issues,
        "by_sev": by_sev,
    }

    return render_template(
        "result.html",
        app_name=CFG["app"]["name"],
        author=CFG["app"]["author"],
        student_id=CFG["app"]["student_id"],
        klass=CFG["app"]["class"],
        school=CFG["app"]["school"],
        result=LAST_RESULT,
    )


@app.route("/download/<kind>", methods=["GET"])
def download(kind: str):
    if not LAST_RESULT:
        flash("还没有结果，请先上传文件")
        return redirect(url_for("index"))

    if kind == "json":
        payload = json.dumps(LAST_RESULT["issues"], ensure_ascii=False, indent=2)
        buf = io.BytesIO(payload.encode("utf-8"))
        buf.seek(0)
        return send_file(buf, mimetype="application/json",
                         as_attachment=True, download_name="问题清单.json")

    if kind == "csv":
        import csv as _csv

        buf = io.StringIO()
        w = _csv.writer(buf)
        w.writerow(["issue_id", "record_id", "row", "field", "rule_id",
                    "source", "severity", "message", "suggestion"])
        for it in LAST_RESULT["issues"]:
            w.writerow([it["issue_id"], it["record_id"], it["row"], it["field"],
                        it["rule_id"], it["source"], it["severity"],
                        it["message"], it["suggestion"]])
        data = buf.getvalue().encode("utf-8-sig")
        return send_file(io.BytesIO(data), mimetype="text/csv",
                         as_attachment=True, download_name="问题清单.csv")

    flash("未知下载类型")
    return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(_e):
    flash(f"文件超过 {CFG['server']['max_upload_mb']}MB 限制")
    return redirect(url_for("index"))


if __name__ == "__main__":
    host = CFG["server"]["host"]
    port = int(os.environ.get("PORT", CFG["server"]["port"]))
    print(f"{CFG['app']['name']} 启动中：http://{host}:{port}")
    print(f"作者：{CFG['app']['author']} / {CFG['app']['student_id']} / {CFG['app']['class']}")
    app.run(host=host, port=port, debug=CFG["server"]["debug"])
