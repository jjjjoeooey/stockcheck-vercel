"""本地 Web 平台服务（零第三方依赖，仅标准库 + pandas）。

启动：
    python -m stockcheck.server --port 8766
然后浏览器打开 http://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import os
import re
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
from html import escape

import numpy as np
import pandas as pd

from .data_io import (
    fetch_board_stocks,
    resolve_stock,
    fetch_boards,
    fetch_market_snapshot,
    fetch_price_em,
    fetch_quote_em,
    load_fundamentals,
    load_price_file,
    normalize_price_frame,
    read_table,
)
from .demo_data import build_demo
from .indicators import compute_boll, compute_kdj
from .model import DEFAULT_WEIGHTS, score_stock  # noqa: F401
from .score import quick_score
from .report import render_text  # noqa: F401
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MAX_BODY = 40 * 1024 * 1024  # 40MB
PASSWORD = os.environ.get("SC_PASSWORD", "").strip()
READ_TOKEN = os.environ.get("SC_READ_TOKEN", "").strip()
AUTH_COOKIE = "sc_auth"

FUND_KEYS = ("pe", "pb", "roe", "profit_growth", "revenue_growth", "debt_ratio", "gross_margin", "market_cap")


def _json_safe(obj):
    """numpy / NaN 转成合法 JSON。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def series_to_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for row in df.itertuples(index=False):
        out.append(
            {
                "date": str(row.date)[:10],
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume) if np.isfinite(row.volume) else 0.0,
                "amount": float(row.amount) if np.isfinite(row.amount) else 0.0,
                "turnover": float(row.turnover) if np.isfinite(row.turnover) else None,
            }
        )
    return out


def records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce").astype(float)
    return df


def resample_ohlcv(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """日线 -> 周线/月线。period: day / week / month。"""
    if period not in ("week", "month"):
        return df
    rule = "W-FRI" if period == "week" else "ME"
    x = (
        df.set_index("date")
        .resample(rule)
        .agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum", "amount": "sum", "turnover": "sum",
        })
        .dropna(subset=["close"])
        .reset_index()
    )
    return x


def chart_payload(df: pd.DataFrame, n: int = 160, period: str = "day") -> dict:
    df = resample_ohlcv(df, period)
    b = compute_boll(df)
    k = compute_kdj(df)
    m = min(n, len(df))
    start = len(df) - m
    tail = df.iloc[start:]

    def cut(series):
        arr = np.asarray(series)
        return [None if not np.isfinite(v) else round(float(v), 3) for v in arr]

    return {
        "dates": [str(d)[:10] for d in tail["date"]],
        "open": cut(tail["open"]),
        "high": cut(tail["high"]),
        "low": cut(tail["low"]),
        "close": cut(tail["close"]),
        "volume": cut(tail["volume"]),
        "turnover": cut(tail["turnover"]),
        "upper": cut(b["upper_series"].iloc[start:]),
        "mid": cut(b["mid_series"].iloc[start:]),
        "lower": cut(b["lower_series"].iloc[start:]),
        "k": cut(k["k_series"].iloc[start:]),
        "d": cut(k["d_series"].iloc[start:]),
        "j": cut((3.0 * k["k_series"] - 2.0 * k["d_series"]).iloc[start:]),
    }


def _fund_dict_row(row) -> dict:
    out = {}
    for key in FUND_KEYS:
        v = row.get(key)
        if v is not None and pd.notna(v):
            out[key] = float(v)
    return out




def similar_patterns(store: Store, code: str, k: int = 8, window: int = 60, step: int = 3) -> dict:
    """在已入库股票的历史里，找与目标最近 60 日走势最像的片段，并统计之后的表现。"""
    target = store.get(code)
    if not target:
        raise ValueError(f"没有 {code} 的数据")
    tdf = records_to_df(target["series"])
    if len(tdf) < window + 1:
        raise ValueError("目标股票历史不足 60 根")

    tclose = tdf["close"].to_numpy(dtype=float)[-window:]
    tvec = tclose / tclose[0] - 1.0
    tnorm = float(np.linalg.norm(tvec)) or 1.0

    matches = []
    for item in store.all_raw():
        if item["code"] == code:
            continue
        df = records_to_df(item["series"])
        closes = df["close"].to_numpy(dtype=float)
        n = len(closes)
        for end in range(window - 1, n - 21, step):
            seg = closes[end - window + 1: end + 1]
            base = seg[0] or 1.0
            v = seg / base - 1.0
            denom = tnorm * (float(np.linalg.norm(v)) or 1.0)
            if denom <= 0:
                continue
            sim = float(np.dot(tvec, v) / denom)
            fwd = {}
            for d in (5, 10, 20):
                fwd[d] = float(closes[end + d] / closes[end] - 1.0)
            matches.append({
                "code": item["code"],
                "name": item["name"] or "",
                "start": str(df["date"].iloc[end - window + 1])[:10],
                "end": str(df["date"].iloc[end])[:10],
                "sim": round(sim, 4),
                "fwd5": round(fwd[5] * 100, 2),
                "fwd10": round(fwd[10] * 100, 2),
                "fwd20": round(fwd[20] * 100, 2),
            })

    matches.sort(key=lambda x: x["sim"], reverse=True)
    top = matches[: max(1, min(k, 50))]
    n = len(top)

    def stats(key):
        vals = [m[key] for m in top]
        return {
            "avg": round(float(np.mean(vals)), 2) if vals else None,
            "win_rate": round(float(np.mean([v > 0 for v in vals])) * 100, 1) if vals else None,
        }

    return {
        "code": code,
        "window": window,
        "n": n,
        "matches": top,
        "stats": {"fwd5": stats("fwd5"), "fwd10": stats("fwd10"), "fwd20": stats("fwd20")},
        "note": "相似度基于 60 日归一化价格路径的余弦相似度；之后涨跌只是历史统计，不是预测。",
    }


class Api:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.upload_dir = data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(str(data_dir / "platform.db"))
        settings = self.store.get_settings()
        stored_w = settings.get("weights")
        if not stored_w:
            self.store.set_settings(weights=dict(DEFAULT_WEIGHTS))
        else:
            # 维度不全或合计不是 100 -> 用新的 100 分默认权重
            if set(stored_w) != set(DEFAULT_WEIGHTS) or abs(sum(stored_w.values()) - 100.0) > 0.1:
                self.store.set_settings(weights=dict(DEFAULT_WEIGHTS))
        if self.store.count():
            # 一次性迁移：早期入库的数据没写 source，统一标记为东财
            for item in self.store.all():
                if not item["result"].get("source"):
                    r = item["result"]
                    r["source"] = "eastmoney"
                    self.store.update_result(item["code"], r)
            try:
                self.rescore_all()
            except Exception:  # noqa: BLE001 - 旧数据重算失败不影响启动
                pass

    # ---------- utils ----------
    def weights(self) -> dict:
        w = self.store.get_settings().get("weights") or dict(DEFAULT_WEIGHTS)
        return {k: float(w.get(k, v)) for k, v in DEFAULT_WEIGHTS.items()}

    def fundamentals_map(self) -> dict:
        return self.store.get_settings().get("fundamentals") or {}

    def rescore_all(self, weights: dict | None = None) -> int:
        w = weights or self.weights()
        n = 0
        for item in self.store.all_raw():
            df = records_to_df(item["series"])
            res = score_stock(df, fundamentals=item["fundamentals"], weights=w,
                              code=item["code"], name=item["name"] or "")
            payload = res.to_dict()
            old = self.store.get(item["code"])
            if old:
                for key in ("source", "synced_at"):
                    if old["result"].get(key):
                        payload[key] = old["result"][key]
            self.store.update_result(item["code"], payload)
            n += 1
        return n

    def ingest_file(self, raw: bytes, filename: str, weights: dict,
                    fund_map: dict, default_code: str = "", name: str = "",
                    source: str = "file") -> dict:
        safe = re.sub(r"[^\w.\-]+", "_", filename)[-80:] or "upload.csv"
        path = self.upload_dir / f"{int(time.time() * 1000)}_{safe}"
        path.write_bytes(raw)

        df = load_price_file(str(path))
        code = default_code or self._guess_code(filename, df)
        if not code:
            raise ValueError("无法识别股票代码：请把文件名写成 600519.csv，或加一列「代码」")
        names = self.store.get_settings().get("names") or {}
        name = name or names.get(code) or str(df.attrs.get("name") or "").strip()

        fund = fund_map.get(code) or None
        res = score_stock(df, fundamentals=fund, weights=weights, code=code, name=name)
        payload = res.to_dict()
        payload["source"] = source
        self.store.upsert(code, name, series_to_records(df), payload, fund)
        return {"ok": True, "code": code, "name": name, "total": res.total,
                "file": filename, "bars": len(df)}

    @staticmethod
    def _guess_code(filename: str, df: pd.DataFrame) -> str:
        m = re.search(r"(\d{6})", filename)
        if m:
            return m.group(1)
        code = str(df.attrs.get("code") or "")
        m = re.search(r"(\d{6})", code)
        return m.group(1) if m else ""

    def sync_code(self, code: str, days: int = 600) -> dict:
        """联网同步一只股票的真实前复权日线。"""
        df = fetch_price_em(code, days=days)
        quote: dict = {}
        try:
            quote = fetch_quote_em(code)
        except Exception:  # noqa: BLE001 - 估值拿不到不影响行情
            pass

        code = re.sub(r"\D", "", str(code))[-6:].zfill(6)
        fund: dict = {}
        user_fund = self.fundamentals_map().get(code)
        if user_fund:
            fund.update(user_fund)
        for k in ("pe", "pb", "market_cap"):
            if quote.get(k) is not None:
                fund[k] = quote[k]

        name = str(quote.get("name") or df.attrs.get("name") or "")
        res = score_stock(df, fundamentals=fund or None, weights=self.weights(),
                          code=code, name=name)
        payload = res.to_dict()
        payload["source"] = "eastmoney"
        payload["synced_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.store.upsert(code, name, series_to_records(df), payload, fund or None)
        return {"ok": True, "code": code, "name": name, "total": res.total,
                "close": float(df["close"].iloc[-1]), "date": str(df["date"].iloc[-1])[:10],
                "bars": len(df), "pe": fund.get("pe"), "pb": fund.get("pb")}

    def ingest_fundamentals(self, raw: bytes, filename: str) -> dict:
        safe = re.sub(r"[^\w.\-]+", "_", filename)[-60:] or "fund.csv"
        path = self.upload_dir / f"fund_{int(time.time() * 1000)}_{safe}"
        path.write_bytes(raw)
        table = load_fundamentals(str(path))
        fmap = self.fundamentals_map()
        names = self.store.get_settings().get("names") or {}
        applied = 0
        for _, row in table.iterrows():
            code = str(row["code"])
            fmap[code] = _fund_dict_row(row)
            nm = str(row.get("name") or "").strip()
            if nm:
                names[code] = nm
                self.store.set_name(code, nm)
            applied += 1
        self.store.set_settings(fundamentals=fmap, names=names)

        # 同步到已入库股票 + 重算
        w = self.weights()
        for item in self.store.all_raw():
            if item["code"] in fmap:
                self.store.set_fundamentals(item["code"], fmap[item["code"]])
        self.rescore_all(w)
        return {"ok": True, "applied": applied, "codes": list(fmap.keys())[:50]}


class Handler(BaseHTTPRequestHandler):
    api: Api
    protocol_version = "HTTP/1.1"

    # ---------- auth ----------
    def authed(self) -> bool:
        if not PASSWORD:
            return True
        # 供 Scriptable / 小组件使用的只读 token
        if READ_TOKEN:
            qs = parse_qs(urlparse(self.path).query)
            if (qs.get("token") or [""])[0] and hmac.compare_digest((qs.get("token") or [""])[0], READ_TOKEN):
                return True
        raw = self.headers.get("Cookie") or ""
        m = re.search(r"(?:^|;\s*)" + AUTH_COOKIE + r"=([0-9a-f]{64})", raw)
        if not m:
            return False
        return hmac.compare_digest(m.group(1), hashlib.sha256(PASSWORD.encode()).hexdigest())

    def redirect(self, loc: str, cookie: str = ""):
        self.send_response(302)
        self.send_header("Location", loc)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def login_page(self, error: bool = False, next_path: str = "/"):
        html = """<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>StockCheck</title>
<style>body{margin:0;background:#0a101c;color:#e6edf6;font-family:-apple-system,"PingFang SC",sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh}
.box{width:300px;background:#111c2e;border:1px solid rgba(148,163,184,.16);border-radius:16px;padding:24px}
h1{font-size:18px;margin:0 0 6px}p{color:#8ea0b8;font-size:12px;margin:0 0 18px}
input{width:100%;box-sizing:border-box;background:#0b1322;border:1px solid #22314a;border-radius:10px;
padding:11px;color:#e6edf6;font-size:14px;outline:none}
button{width:100%;margin-top:12px;background:linear-gradient(135deg,#22d3ee,#34d399);border:none;
border-radius:10px;padding:11px;font-weight:700;color:#04121a;font-size:14px}
.err{color:#f87171;font-size:12px;margin-top:10px}</style></head>
<body><form class="box" method="post" action="/login">
<h1>StockCheck</h1><p>请输入访问口令</p>
<input type="password" name="password" autofocus autocomplete="current-password">
<button type="submit">进入</button>""" + ('<div class="err">口令不对</div>' if error else '') + """</form></body></html>"""
        safe_next = next_path if next_path.startswith("/") and not next_path.startswith("//") else "/"
        html = html.replace('action="/login">',
                            'action="/login"><input type="hidden" name="next" value="' + escape(safe_next) + '">')
        body = html.encode("utf-8")
        self._send(200, body, "text/html; charset=utf-8")

    def handle_login(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
        form = parse_qs(raw)
        pwd = (form.get("password") or [""])[0]
        nxt = (form.get("next") or ["/"])[0]
        if not (nxt.startswith("/") and not nxt.startswith("//")):
            nxt = "/"
        if PASSWORD and hmac.compare_digest(pwd, PASSWORD):
            token = hashlib.sha256(PASSWORD.encode()).hexdigest()
            cookie = f"{AUTH_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=604800"
            return self.redirect(nxt, cookie)
        return self.login_page(error=True, next_path=nxt)

    # ---------- responses ----------
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def json(self, code: int, obj):
        body = json.dumps(_json_safe(obj), ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def err(self, code: int, msg: str):
        self.json(code, {"ok": False, "error": msg})

    def body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("请求体过大（上限 40MB）")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def log_message(self, fmt, *args):  # 安静一点
        if "/api/" in (args[0] if args else ""):
            print(f"[api] {args[0]}")

    # ---------- routing ----------
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                return self.json(200, {"ok": True, "version": "1.0"})
            if path == "/login":
                qs = parse_qs(urlparse(self.path).query)
                return self.login_page(next_path=(qs.get("next") or ["/"])[0])
            if path == "/logout":
                return self.redirect("/login", f"{AUTH_COOKIE}=; Path=/; Max-Age=0")
            if not self.authed() and path not in ("/favicon.ico", "/icon-192.png", "/icon-512.png", "/manifest.webmanifest"):
                if path.startswith("/api/"):
                    return self.err(401, "需要登录")
                nxt = self.path if self.path.startswith("/") else "/"
                return self.redirect("/login?next=" + quote(nxt, safe=""))

            if path == "/api/state":
                stocks = self.api.store.all()
                return self.json(200, {
                    "ok": True,
                    "weights": self.api.weights(),
                    "stocks": [s["result"] | {"updated_at": s["updated_at"]} for s in stocks],
                    "count": len(stocks),
                    "fundamental_count": len(self.api.fundamentals_map()),
                })
            if path == "/api/market":
                force = "refresh=1" in (self.path or "")
                rows = fetch_market_snapshot(force=force)
                deep = self.api.store.codes()
                for r in rows:
                    r["quick"] = round(quick_score(r), 1)
                    r["deep"] = r["code"] in deep
                rows.sort(key=lambda x: x["quick"], reverse=True)
                return self.json(200, {
                    "ok": True, "count": len(rows), "stocks": rows,
                    "updated_at": time.strftime("%H:%M:%S"),
                })

            if path == "/api/resolve":
                qs = parse_qs(urlparse(self.path).query)
                q = (qs.get("q") or [""])[0]
                try:
                    return self.json(200, {"ok": True, **resolve_stock(q)})
                except Exception as exc:  # noqa: BLE001
                    return self.err(404, str(exc))

            if path == "/api/boards":
                qs = parse_qs(urlparse(self.path).query)
                kind = (qs.get("type") or ["industry"])[0]
                if kind not in ("industry", "concept", "region"):
                    kind = "industry"
                boards = fetch_boards(kind)
                return self.json(200, {"ok": True, "type": kind, "count": len(boards),
                                       "boards": boards})

            if path.startswith("/api/board/"):
                board_code = path.rsplit("/", 1)[-1]
                qs = parse_qs(urlparse(self.path).query)
                kind = (qs.get("type") or ["industry"])[0]
                if kind not in ("industry", "concept", "region"):
                    kind = "industry"
                members = fetch_board_stocks(board_code, kind)
                stored = {x["code"]: x["result"] for x in self.api.store.all()}
                for m in members:
                    m["quick"] = round(quick_score(m), 1)
                    r = stored.get(m["code"])
                    m["deep"] = bool(r)
                    m["total"] = r.get("total") if r else None
                board = {"code": board_code, "name": "", "pct": None}
                try:
                    for b in fetch_boards(kind):
                        if b["code"] == board_code:
                            board.update(b)
                            break
                except Exception:  # noqa: BLE001
                    pass
                return self.json(200, {"ok": True, "board": board,
                                       "count": len(members), "stocks": members})

            if path.startswith("/api/quote/"):
                code = path.rsplit("/", 1)[-1]
                try:
                    q = fetch_quote_em(code)
                except Exception as exc:  # noqa: BLE001
                    return self.err(502, f"取行情失败：{exc}")
                return self.json(200, {"ok": True, "quote": q, "at": time.strftime("%H:%M:%S")})

            if path.startswith("/api/similar/"):
                code = path.rsplit("/", 1)[-1]
                return self.json(200, {"ok": True, **similar_patterns(self.api.store, code)})

            if path.startswith("/api/stock/"):
                code = path.rsplit("/", 1)[-1]
                item = self.api.store.get(code)
                if not item:
                    return self.err(404, f"没有 {code} 的数据")
                df = records_to_df(item["series"])
                qs = parse_qs(urlparse(self.path).query)
                period = (qs.get("period") or ["day"])[0]
                if period not in ("day", "week", "month"):
                    period = "day"
                return self.json(200, {
                    "ok": True,
                    "period": period,
                    "result": item["result"],
                    "chart": chart_payload(df, period=period),
                    "fundamentals": item["fundamentals"],
                    "bars": len(df),
                    "updated_at": item["updated_at"],
                })
            if path == "/api/export.csv":
                return self.export_csv()
            if path == "/api/export.json":
                return self.json(200, {"ok": True, "stocks": [s["result"] for s in self.api.store.all()]})
            if path in ("/", "/index.html"):
                return self.static("index.html")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            return self.static(path.lstrip("/"))
        except FileNotFoundError:
            return self.err(404, "Not found")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.err(500, f"服务器错误：{exc}")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/login":
                return self.handle_login()
            if not self.authed():
                return self.err(401, "需要登录")
            body = self.body_json()
            if path == "/api/upload":
                files = body.get("files") or []
                if not files:
                    return self.err(400, "没有收到文件")
                weights = self.api.weights()
                fund_map = self.api.fundamentals_map()
                results = []
                for f in files:
                    try:
                        raw = base64.b64decode(f.get("data", ""), validate=False)
                        results.append(self.api.ingest_file(
                            raw, f.get("name", "upload.csv"), weights, fund_map,
                            default_code=str(f.get("code") or ""),
                        ))
                    except Exception as exc:  # noqa: BLE001
                        results.append({"ok": False, "file": f.get("name"), "error": str(exc)})
                return self.json(200, {"ok": True, "results": results,
                                       "count": self.api.store.count()})

            if path == "/api/paste":
                text = str(body.get("text") or "")
                code = str(body.get("code") or "").strip()
                if not text.strip():
                    return self.err(400, "粘贴内容为空")
                raw = text.encode("utf-8")
                try:
                    r = self.api.ingest_file(
                        raw, f"{code or 'paste'}.csv", self.api.weights(),
                        self.api.fundamentals_map(), default_code=code, source="paste")
                except Exception as exc:  # noqa: BLE001
                    return self.err(400, str(exc))
                return self.json(200, {"ok": True, "result": r})

            if path == "/api/fundamentals":
                files = body.get("files") or []
                if not files:
                    return self.err(400, "没有收到文件")
                out = []
                for f in files:
                    raw = base64.b64decode(f.get("data", ""), validate=False)
                    try:
                        out.append(self.api.ingest_fundamentals(raw, f.get("name", "fund.csv")))
                    except Exception as exc:  # noqa: BLE001
                        out.append({"ok": False, "file": f.get("name"), "error": str(exc)})
                return self.json(200, {"ok": True, "results": out})

            if path == "/api/weights":
                w = body.get("weights") or {}
                merged = self.api.weights()
                for k in DEFAULT_WEIGHTS:
                    if k in w:
                        merged[k] = max(0.0, float(w[k]))
                total = sum(float(v) for v in merged.values() if float(v) > 0)
                if total <= 0:
                    return self.err(400, "权重之和必须大于 0")
                merged = {k: round(float(v) * 100.0 / total, 2) for k, v in merged.items()}
                self.api.store.set_settings(weights=merged)
                n = self.api.rescore_all(merged)
                return self.json(200, {"ok": True, "weights": merged, "rescored": n})

            if path == "/api/demo":
                tmp = Path(tempfile.mkdtemp(prefix="sc_demo_"))
                build_demo(str(tmp))
                weights = self.api.weights()
                fmap: dict = {}
                fund_file = tmp / "fundamentals.csv"
                if fund_file.exists():
                    table = load_fundamentals(str(fund_file))
                    for _, row in table.iterrows():
                        fmap[str(row["code"])] = _fund_dict_row(row)
                pool_names = {}
                pool_file = tmp / "pool.csv"
                if pool_file.exists():
                    pool_df = read_table(str(pool_file))
                    pool_names = {str(r["代码"]): str(r["名称"]) for _, r in pool_df.iterrows()}
                out = []
                for p in sorted(tmp.glob("[0-9]*.csv")):
                    code = p.stem
                    out.append(self.api.ingest_file(
                        p.read_bytes(), p.name, weights, fmap,
                        default_code=code, name=pool_names.get(code, "")))
                return self.json(200, {"ok": True, "results": out, "count": self.api.store.count()})

            if path == "/api/deepen":
                items = body.get("items") or []
                if not items:
                    return self.err(400, "请提供要深度扫描的股票")
                if len(items) > 200:
                    return self.err(400, "一次最多深度扫描 200 只")
                weights = self.api.weights()
                fund_map = self.api.fundamentals_map()

                def work(it):
                    code = re.sub(r"\D", "", str(it.get("code") or ""))[-6:].zfill(6)
                    try:
                        df = fetch_price_em(code)
                        fund = {k: it[k] for k in ("pe", "pb", "market_cap") if it.get(k) is not None}
                        if code in fund_map:
                            fund.update(fund_map[code])
                        res = score_stock(df, fundamentals=fund or None, weights=weights,
                                          code=code, name=str(it.get("name") or ""))
                        payload = res.to_dict()
                        payload["source"] = "market"
                        self.api.store.upsert(code, str(it.get("name") or ""),
                                              series_to_records(df), payload, fund or None)
                        return {"ok": True, "code": code, "name": it.get("name"),
                                "total": res.total, "close": float(df["close"].iloc[-1])}
                    except Exception as exc:  # noqa: BLE001
                        return {"ok": False, "code": code, "error": str(exc)}

                with ThreadPoolExecutor(max_workers=6) as ex:
                    out = list(ex.map(work, items))
                ok = [r for r in out if r["ok"]]
                ok.sort(key=lambda x: (x["total"] is None, -(x["total"] or 0)))
                return self.json(200, {"ok": True, "results": out, "scanned": len(ok),
                                       "count": self.api.store.count()})

            if path == "/api/sync":
                raw_codes = body.get("codes") or []
                if isinstance(raw_codes, str):
                    raw_codes = re.split(r"[\s,，;；]+", raw_codes)
                codes = [c for c in raw_codes if str(c).strip()]
                if not codes:
                    return self.err(400, "请输入至少一个股票代码")
                if len(codes) > 50:
                    return self.err(400, "一次最多同步 50 只，避免触发限频")
                out = []
                for c in codes:
                    try:
                        out.append(self.api.sync_code(str(c), days=int(body.get("days") or 600)))
                    except Exception as exc:  # noqa: BLE001
                        out.append({"ok": False, "code": str(c), "error": str(exc)})
                return self.json(200, {"ok": True, "results": out,
                                       "count": self.api.store.count()})

            if path == "/api/clear":
                self.api.store.clear()
                return self.json(200, {"ok": True})

            if path == "/api/delete":
                code = str(body.get("code") or "")
                return self.json(200, {"ok": self.api.store.delete(code)})

            return self.err(404, "接口不存在")
        except ValueError as exc:
            return self.err(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self.err(500, f"服务器错误：{exc}")

    # ---------- static ----------
    def static(self, rel: str):
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            return self.err(404, "Not found")
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".webmanifest": "application/manifest+json",
            ".json": "application/json",
            ".txt": "text/plain; charset=utf-8",
            ".png": "image/png",
        }.get(target.suffix.lower(), "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)

    def export_csv(self):
        rows = []
        for s in self.api.store.all():
            r = s["result"]
            row = {"代码": r["code"], "名称": r["name"], "日期": r["date"], "收盘": r["close"],
                   "总分": r["total"], "覆盖度": r["coverage"], "标签": "/".join(r["tags"]),
                   "结论": r["conclusion"]}
            for d in r["dims"]:
                row[d["label"]] = d["score"]
            rows.append(row)
        buf = io.BytesIO()
        pd.DataFrame(rows).to_csv(buf, index=False, encoding="utf-8-sig")
        body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="stockcheck_rank.csv"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(host: str = "127.0.0.1", port: int = 8766, data_dir: Path = DATA_DIR) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"api": Api(Path(data_dir))})
    return ThreadingHTTPServer((host, port), handler)


def lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return ""


def serve(host: str = "127.0.0.1", port: int = 8766, data_dir: Path = DATA_DIR):
    httpd = build_server(host, port, data_dir)
    actual = httpd.server_address[1]
    print(f"stockcheck 平台已启动： http://127.0.0.1:{actual}")
    if host in ("0.0.0.0", "::"):
        ip = lan_ip()
        if ip:
            print(f"同一 Wi-Fi 下的 iPhone 访问： http://{ip}:{actual}")
    print("按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(description="stockcheck 本地平台")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8766")))
    args = p.parse_args(argv)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
