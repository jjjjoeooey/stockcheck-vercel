"""Vercel Python 入口：复用本地打分逻辑，单票查询。"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request  # noqa: E402

from stockcheck.data_io import fetch_price_em, fetch_quote_em, resolve_stock  # noqa: E402
from stockcheck.score import score_stock  # noqa: E402

PUBLIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
app = Flask(__name__, static_folder=PUBLIC, static_url_path="")


@app.get("/")
def home():
    return app.send_static_file("hud.html")


@app.get("/api/index")
def api_get():
    action = request.args.get("action", "health")

    if action == "health":
        return jsonify(ok=True, service="stockcheck", runtime="vercel")

    if action == "resolve":
        try:
            return jsonify(ok=True, **resolve_stock(request.args.get("q", "")))
        except Exception as exc:  # noqa: BLE001
            return jsonify(ok=False, error=str(exc)), 404

    if action == "stock":
        code = re.sub(r"\D", "", request.args.get("code", ""))[-6:].zfill(6)
        if not code or code == "000000":
            return jsonify(ok=False, error="缺少股票代码"), 400
        try:
            df = fetch_price_em(code, days=600)
            quote = {}
            try:
                quote = fetch_quote_em(code)
            except Exception:  # noqa: BLE001
                pass
            fund = {k: quote.get(k) for k in ("pe", "pb", "market_cap") if quote.get(k) is not None}
            res = score_stock(df, fundamentals=fund or None, code=code,
                              name=str(quote.get("name") or ""))
            payload = res.to_dict()
            payload["source"] = "eastmoney"
            return jsonify(ok=True, result=payload)
        except Exception as exc:  # noqa: BLE001
            return jsonify(ok=False, error=str(exc)), 400

    return jsonify(ok=False, error=f"unknown action: {action}"), 404


@app.post("/api/index")
def api_post():
    action = request.args.get("action", "")
    if action == "sync":
        return jsonify(ok=True, synced=0)
    return jsonify(ok=False, error="unknown action"), 404
