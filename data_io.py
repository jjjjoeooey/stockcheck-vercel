"""数据输入层：读取指南针导出的 CSV/TXT/XLS，或（可选）走 akshare。

设计原则
--------
1. 文件优先：指南针没有公开 API，本地是加密的 QMI 格式，所以最稳的方式是
   在软件里把 K 线/报价列表「复制/导出」成 CSV 或 XLS，再由这里读入。
2. 列名宽容：中英文、带单位、带百分号、代码带市场前缀都能自动识别。
3. akshare 只作为兜底，导入失败不影响主流程。
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import urllib.request
from urllib.parse import quote
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 通用清洗
# --------------------------------------------------------------------------- #

_UNITS = {"万": 1e4, "亿": 1e8, "千": 1e3}

_PCT_ALIASES = {
    "换手率": "turnover",
    "换手": "turnover",
    "换手率(%)": "turnover",
    "turnover": "turnover",
    "turn": "turnover",
    "换手(自由流通)": "turnover",
}


def to_number(x) -> Optional[float]:
    """把 '1,234.5万'、'3.21%'、'--'、'' 之类的单元格转成 float。"""
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return float(x)
    s = str(x).strip().replace(",", "").replace("，", "")
    if s in {"", "-", "--", "—", "None", "nan", "NaN", "null"}:
        return None
    scale = 1.0
    for unit, mult in _UNITS.items():
        if s.endswith(unit):
            scale = mult
            s = s[: -len(unit)]
            break
    s = s.replace("%", "").replace("％", "")
    try:
        return float(s) * scale
    except ValueError:
        return None


def _norm_key(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(s).strip().lower())


# 价格表列名 -> 标准列
_PRICE_ALIASES: dict[str, str] = {
    # 日期
    "日期": "date", "date": "date", "时间": "date", "交易日": "date",
    "交易日期": "date", "trade_date": "date", "datetime": "date",
    # 开高低收
    "开盘": "open", "开盘价": "open", "open": "open", "o": "open",
    "最高": "high", "最高价": "high", "high": "high", "h": "high",
    "最低": "low", "最低价": "low", "low": "low", "l": "low",
    "收盘": "close", "收盘价": "close", "close": "close", "c": "close",
    "前收盘": "pre_close", "昨收": "pre_close", "preclose": "pre_close",
    # 量额
    "成交量": "volume", "volume": "volume", "vol": "volume", "量": "volume",
    "成交额": "amount", "amount": "amount", "成交金额": "amount", "额": "amount",
    "换手率": "turnover", "换手": "turnover", "turnover": "turnover",
    "涨跌幅": "pct_chg", "涨幅": "pct_chg", "pct_chg": "pct_chg",
    "pctchange": "pct_chg", "change": "pct_chg",
    "涨跌额": "chg", "涨跌": "chg",
    "振幅": "amplitude",
    "代码": "code", "股票代码": "code", "证券代码": "code", "code": "code",
    "名称": "name", "股票名称": "name", "证券名称": "name", "name": "name",
}


def _detect_encoding(path: str) -> str:
    """猜编码：UTF-8-SIG -> UTF-8 -> GBK -> GB18030。"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "big5"):
        try:
            with open(path, "rb") as f:
                f.read(200_000).decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"


def read_table(path: str, dtype: str | None = "str", **kwargs) -> pd.DataFrame:
    """读 CSV / TXT / XLS(X)，自动判分隔符与编码。

    默认 dtype="str"：股票代码 000001 若按数值读会变成 1，必须先按文本读，
    数值列随后用 to_number 统一清洗。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(path, dtype=dtype, **kwargs)
    enc = _detect_encoding(path)
    # 先探测分隔符
    with open(path, "r", encoding=enc, errors="replace") as f:
        head = "".join(f.readline() for _ in range(5))
    sep = "\t" if head.count("\t") >= head.count(",") else ","
    if head.count(";") > max(head.count(","), head.count("\t")):
        sep = ";"
    return pd.read_csv(path, encoding=enc, sep=sep, dtype=dtype, **kwargs)


def parse_text_table(text: str) -> pd.DataFrame:
    """解析从行情软件复制出来的制表符/逗号分隔文本（可带表头）。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("粘贴内容为空")
    first = text.splitlines()[0]
    if "\t" in first:
        sep = "\t"
    elif "," in first:
        sep = ","
    elif ";" in first:
        sep = ";"
    else:
        sep = r"\s+"
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, engine="python")


def _rename_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        key = _norm_key(col)
        std = _PRICE_ALIASES.get(key)
        if std is None:
            # 去掉括号后的内容再试一次，如 "换手率(%)"
            key2 = re.sub(r"[（(].*?[)）]", "", str(col)).strip()
            std = _PRICE_ALIASES.get(_norm_key(key2))
        if std:
            mapping[col] = std
    df = df.rename(columns=mapping)
    # 同名列（例如同时有 "换手率" 和 "换手"）只保留第一个
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def normalize_price_frame(
    df: pd.DataFrame, code: str = "", name: str = ""
) -> pd.DataFrame:
    """把任意来源的日线表规整成标准列：date/open/high/low/close/volume/amount/turnover。

    - 缺 turnover 时置 NaN（无法算换手分，模型会自动降权）
    - 缺 amount 且无 volume 时抛错
    - 成交量单位：若数值极小（<1e4 且看起来像「手」）不做猜测，仅记录；
      指南针导出的是「手」，akshare 也是「手」，两者相对比例一致，打分只看相对量，故不换算。
    """
    if df is None or len(df) == 0:
        raise ValueError("价格数据为空")

    df = df.copy()
    # 有时第一行是标题/单位行，导致整列 object
    df = _rename_price_columns(df)

    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"缺少必需列 {sorted(missing)}；识别到的列: {list(df.columns)}。"
            "请把表头改成 日期/开盘/最高/最低/收盘/成交量/换手率 这类常见写法。"
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close"):
        out[c] = df[c].map(to_number).astype(float)
    for c in ("volume", "amount", "turnover"):
        out[c] = df[c].map(to_number).astype(float) if c in df.columns else np.nan

    out = out.dropna(subset=["date", "close", "high", "low"])
    out = out[out["close"] > 0]
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out = out.reset_index(drop=True)
    out.attrs["code"] = code or (str(df["code"].iloc[0]) if "code" in df.columns else "")
    out.attrs["name"] = name or (str(df["name"].iloc[0]) if "name" in df.columns else "")
    return out


def load_price_file(path: str, code: str = "", name: str = "") -> pd.DataFrame:
    """读取指南针导出的单只股票日线文件。"""
    return normalize_price_frame(read_table(path), code=code, name=name)


# --------------------------------------------------------------------------- #
# 批量股票池
# --------------------------------------------------------------------------- #


def load_universe(path: str, sep: Optional[str] = None) -> list[dict]:
    """读取股票池（代码/名称/可选数据文件路径）。

    支持两种形式：
    - 长表：每只股票一个文件，列 `文件,代码,名称`
    - 宽表：所有股票在同一张表，含 `代码` 列，可直接给 --wide 模式
    """
    df = read_table(path, sep=sep) if sep else read_table(path)
    df = _rename_price_columns(df)
    if "code" not in df.columns:
        raise ValueError("股票池文件缺少「代码」列")

    items: list[dict] = []
    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        if not code or code.lower() == "nan":
            continue
        # 统一成 6 位数字
        m = re.search(r"(\d{6})", code)
        code = m.group(1) if m else code
        code = code.zfill(6) if code.isdigit() else code
        items.append(
            {
                "code": code,
                "name": str(row.get("name", "")).strip(),
                "file": str(row.get("文件", "")).strip() if "文件" in df.columns else "",
            }
        )
    return items


# --------------------------------------------------------------------------- #
# 基本面表
# --------------------------------------------------------------------------- #

_FUND_ALIASES: dict[str, str] = {
    "代码": "code", "股票代码": "code", "code": "code",
    "名称": "name", "股票名称": "name", "name": "name",
    "市盈率": "pe", "市盈率(pe)": "pe", "pe": "pe", "pe_ttm": "pe", "pettm": "pe",
    "市净率": "pb", "pb": "pb", "pb_mrq": "pb",
    "净资产收益率": "roe", "roe": "roe", "加权净资产收益率": "roe", "roe(%)": "roe",
    "净利润增长率": "profit_growth", "净利润同比": "profit_growth",
    "归母净利润同比": "profit_growth", "profitgrowth": "profit_growth",
    "营收增长率": "revenue_growth", "营业收入同比": "revenue_growth",
    "营收同比": "revenue_growth", "revenuegrowth": "revenue_growth",
    "资产负债率": "debt_ratio", "负债率": "debt_ratio", "debtratio": "debt_ratio",
    "毛利率": "gross_margin", "grossmargin": "gross_margin",
    "总市值": "market_cap", "市值": "market_cap", "marketcap": "market_cap",
    "流通市值": "float_cap",
    "股息率": "dividend_yield", "股息率(%)": "dividend_yield",
    "行业": "industry",
}


def load_fundamentals(path: str) -> pd.DataFrame:
    """读取基本面表，标准列: code,name,pe,pb,roe,profit_growth,revenue_growth,debt_ratio,gross_margin"""
    df = read_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {}
    for col in df.columns:
        key = _norm_key(col)
        std = _FUND_ALIASES.get(key)
        if std is None:
            key2 = re.sub(r"[（(].*?[)）]", "", str(col)).strip()
            std = _FUND_ALIASES.get(_norm_key(key2))
        if std:
            mapping[col] = std
    df = df.rename(columns=mapping)
    df = df.loc[:, ~df.columns.duplicated()]
    if "code" not in df.columns:
        raise ValueError("基本面表缺少「代码」列")

    df["code"] = (
        df["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna(df["code"].astype(str))
    )
    for c in ("pe", "pb", "roe", "profit_growth", "revenue_growth", "debt_ratio", "gross_margin"):
        if c in df.columns:
            df[c] = df[c].map(to_number)
        else:
            df[c] = np.nan
    if "name" not in df.columns:
        df["name"] = ""
    return df


# --------------------------------------------------------------------------- #
# akshare 兜底（可选）
# --------------------------------------------------------------------------- #


def _need_akshare():
    try:
        import akshare as ak  # noqa: F401
        return ak
    except Exception as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "未安装 akshare。要么 pip install akshare，要么改用文件模式：\n"
            "  python -m stockcheck.cli --file 你的导出.csv --code 600519"
        ) from exc


def fetch_price_akshare(code: str, start: str = "20200101", end: str = "") -> pd.DataFrame:
    """用 akshare 取前复权日线（兜底通道）。"""
    ak = _need_akshare()
    symbol = re.sub(r"\D", "", code).zfill(6)
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    raw = ak.stock_zh_a_hist(
        symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq"
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f"akshare 未返回 {symbol} 的行情")
    raw = raw.rename(
        columns={
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
            "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover",
        }
    )
    return normalize_price_frame(raw, code=symbol)


def fetch_fundamentals_akshare(code: str) -> dict:
    """尽力而为地取几个基本面字段；取不到的字段留空。"""
    ak = _need_akshare()
    symbol = re.sub(r"\D", "", code).zfill(6)
    out: dict = {"code": symbol}
    try:
        info = ak.stock_individual_info_em(symbol=symbol)
        kv = dict(zip(info["item"].astype(str), info["value"]))
        out["name"] = str(kv.get("股票简称", ""))
        out["market_cap"] = to_number(kv.get("总市值"))
    except Exception:
        pass
    try:
        fin = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2023")
        if len(fin):
            row = fin.iloc[0]
            out["roe"] = to_number(row.get("净资产收益率(%)"))
            out["debt_ratio"] = to_number(row.get("资产负债率(%)"))
            out["gross_margin"] = to_number(row.get("销售毛利率(%)"))
            out["profit_growth"] = to_number(row.get("净利润增长率(%)"))
            out["revenue_growth"] = to_number(row.get("主营业务收入增长率(%)"))
    except Exception:
        pass
    return out

# --------------------------------------------------------------------------- #
# 东方财富：联网同步真实前复权行情（无需安装第三方库，无需 token）
# --------------------------------------------------------------------------- #

_EM_UT = "fa5fd1943c7b386f172d6893dbfba10b"
_EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
}
_EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_EM_QUOTE = "https://push2.eastmoney.com/api/qt/stock/get"


def em_secid(code: str) -> str:
    """代码 -> 东财 secid：沪市 1.xxxxxx，深市/北交所 0.xxxxxx。"""
    digits = re.sub(r"\D", "", str(code))
    code6 = digits[-6:].zfill(6) if digits else ""
    if not code6:
        raise ValueError(f"无法识别的股票代码：{code}")
    if code6.startswith(("6", "9", "5")):
        return f"1.{code6}"
    return f"0.{code6}"


def _http_json(url: str, timeout: float = 12.0) -> dict:
    req = urllib.request.Request(url, headers=_EM_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_em_klines(payload: dict, code: str = "", name: str = "") -> pd.DataFrame:
    """解析东财 K 线 JSON -> 标准日线表。"""
    data = (payload or {}).get("data") or {}
    lines = data.get("klines") or []
    if not lines:
        raise ValueError(f"东财未返回 {code} 的行情（代码可能不存在或已退市）")
    rows = []
    for line in lines:
        parts = str(line).split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6] if len(parts) > 6 else None,
                "turnover": parts[10] if len(parts) > 10 else None,
            }
        )
    df = pd.DataFrame(rows)
    df = normalize_price_frame(df, code=code or str(data.get("code", "")),
                               name=name or str(data.get("name", "")))
    # 前复权老数据可能出现 <=0 的价格，直接丢弃
    df = df[df["close"] > 0].reset_index(drop=True)
    if len(df) < 5:
        raise ValueError(f"{code} 可用日线不足（{len(df)} 根）")
    return df


def fetch_price_em(code: str, days: int = 600) -> pd.DataFrame:
    """取东财前复权日线（默认最近 600 根）。"""
    secid = em_secid(code)
    url = (
        f"{_EM_KLINE}?secid={secid}&ut={_EM_UT}"
        "&fields1=f1,f2,f3,f4,f5,f6"
        "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt=101&fqt=1&end=20500101&lmt={int(days)}"
    )
    payload = _http_json(url)
    return parse_em_klines(payload, code=re.sub(r"\D", "", str(code))[-6:].zfill(6))


def fetch_quote_em(code: str) -> dict:
    """取东财实时快照里的估值字段：PE(动)、PB、总市值、换手率。"""
    secid = em_secid(code)
    url = f"{_EM_QUOTE}?secid={secid}&ut={_EM_UT}&fields=f57,f58,f162,f167,f116,f117,f168"
    data = _http_json(url).get("data") or {}
    if not data:
        return {}

    def num(key, scale=1.0):
        v = data.get(key)
        if v in (None, "-", ""):
            return None
        try:
            return float(v) / scale
        except (TypeError, ValueError):
            return None

    out = {
        "name": data.get("f58") or "",
        "pe": num("f162", 100.0),
        "pb": num("f167", 100.0),
        "market_cap": num("f116"),
        "turnover_live": num("f168", 100.0),
    }
    return {k: v for k, v in out.items() if v not in (None, "") and not (k == "name" and not v)}

# --------------------------------------------------------------------------- #
# 全市场快照（东方财富 clist），一次拿全 A 股
# --------------------------------------------------------------------------- #

_CLIST = "https://push2.eastmoney.com/api/qt/clist/get"
_MARKET_CACHE: dict = {"ts": 0.0, "rows": []}
_MARKET_TTL = 45.0
_MARKET_FS = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"


def parse_market_payload(payload: dict) -> list[dict]:
    data = (payload or {}).get("data") or {}
    diff = data.get("diff") or []
    if isinstance(diff, dict):
        diff = [diff[k] for k in sorted(diff, key=lambda x: int(x))]
    out = []
    for d in diff:
        try:
            price = float(d.get("f2"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or price <= 0:
            continue
        out.append(
            {
                "code": str(d.get("f12") or "").zfill(6),
                "name": str(d.get("f14") or ""),
                "price": price,
                "pct": _f(d.get("f3")),
                "turnover": _f(d.get("f8")),
                "pe": _f(d.get("f9")),
                "pb": _f(d.get("f23")),
                "market_cap": _f(d.get("f20")),
                "volume_ratio": _f(d.get("f10")),
                "volume": _f(d.get("f5")),
                "amount": _f(d.get("f6")),
            }
        )
    return out


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _fetch_market_page(pn: int, pz: int = 100) -> dict:
    url = (
        f"{_CLIST}?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f3&fs={_MARKET_FS}"
        "&fields=f12,f14,f2,f3,f5,f8,f9,f10,f20,f23"
    )
    return _http_json(url, timeout=10.0)


def fetch_market_snapshot(force: bool = False) -> list[dict]:
    """全 A 股快照（约 5500 只），带 45 秒内存缓存。"""
    now = time.time()
    if not force and _MARKET_CACHE["rows"] and now - _MARKET_CACHE["ts"] < _MARKET_TTL:
        return _MARKET_CACHE["rows"]

    first = _fetch_market_page(1)
    total = int(((first.get("data") or {}).get("total")) or 0)
    rows = parse_market_payload(first)
    pages = min(80, (total + 99) // 100)

    from concurrent.futures import ThreadPoolExecutor

    def safe_page(pn: int):
        for _ in range(2):
            try:
                return _fetch_market_page(pn)
            except Exception:  # noqa: BLE001 - 单页失败不影响整体
                time.sleep(0.15)
        return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(safe_page, range(2, pages + 1)))
    for payload in results:
        if payload:
            rows.extend(parse_market_payload(payload))
    if not rows:
        raise RuntimeError("行情接口暂时不可用（DNS/网络异常），请稍后重试")

    seen, uniq = set(), []
    for r in rows:
        if r["code"] and r["code"] not in seen:
            seen.add(r["code"])
            uniq.append(r)

    _MARKET_CACHE["rows"] = uniq
    _MARKET_CACHE["ts"] = now
    return uniq

# --------------------------------------------------------------------------- #
# 板块（行业 / 概念 / 地域）与成分股
# --------------------------------------------------------------------------- #

_BOARD_FS = {"industry": "m:90+t:2", "concept": "m:90+t:3", "region": "m:90+t:1"}
_BOARD_CACHE: dict = {}
_BOARD_TTL = 300.0


def _clist(fs: str, fields: str, pn: int, pz: int = 100) -> dict:
    url = (
        f"{_CLIST}?pn={pn}&pz={pz}&po=1&np=1&fltt=2&invt=2&fid=f3&fs={fs}"
        f"&fields={fields}"
    )
    return _http_json(url, timeout=10.0)


def fetch_boards(kind: str = "industry") -> list[dict]:
    """板块列表（行业/概念/地域），5 分钟缓存。"""
    fs = _BOARD_FS.get(kind)
    if not fs:
        raise ValueError(f"未知板块类型：{kind}")
    now = time.time()
    hit = _BOARD_CACHE.get(kind)
    if hit and now - hit[0] < _BOARD_TTL:
        return hit[1]

    first = _clist(fs, "f12,f14,f3", 1)
    total = int(((first.get("data") or {}).get("total")) or 0)
    rows = _parse_board_list(first)
    from concurrent.futures import ThreadPoolExecutor

    pages = min(10, (total + 99) // 100)

    def safe(pn):
        try:
            return _clist(fs, "f12,f14,f3", pn)
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        for payload in ex.map(safe, range(2, pages + 1)):
            if payload:
                rows.extend(_parse_board_list(payload))
    rows.sort(key=lambda x: x.get("pct") if x.get("pct") is not None else -999, reverse=True)
    _BOARD_CACHE[kind] = (now, rows)
    return rows


def _parse_board_list(payload: dict) -> list[dict]:
    data = (payload or {}).get("data") or {}
    diff = data.get("diff") or []
    if isinstance(diff, dict):
        diff = [diff[k] for k in sorted(diff, key=lambda x: int(x))]
    out = []
    for d in diff:
        code = str(d.get("f12") or "")
        if not code.startswith("BK"):
            continue
        out.append({"code": code, "name": str(d.get("f14") or ""), "pct": _f(d.get("f3"))})
    return out


def fetch_board_stocks(board_code: str, kind: str = "industry") -> list[dict]:
    """某个板块的成分股快照。"""
    fields = "f12,f14,f2,f3,f5,f6,f8,f9,f10,f20,f23"
    first = _clist(f"b:{board_code}", fields, 1)
    total = int(((first.get("data") or {}).get("total")) or 0)
    rows = parse_market_payload(first)
    from concurrent.futures import ThreadPoolExecutor

    pages = min(6, (total + 99) // 100)

    def safe(pn):
        try:
            return _clist(f"b:{board_code}", fields, pn)
        except Exception:  # noqa: BLE001
            return None

    if pages > 1:
        with ThreadPoolExecutor(max_workers=5) as ex:
            for payload in ex.map(safe, range(2, pages + 1)):
                if payload:
                    rows.extend(parse_market_payload(payload))
    return rows


def find_board_name(kind: str, board_code: str) -> str:
    for b in fetch_boards(kind):
        if b["code"] == board_code:
            return b["name"]
    return board_code

def resolve_stock(text: str) -> dict:
    """代码或名称 -> {code, name}。名称走东财 suggest。"""
    text = (text or "").strip()
    m = re.search(r"(\d{6})", text)
    if m:
        return {"code": m.group(1), "name": ""}
    if not text:
        raise ValueError("请输入代码或名称")
    url = (
        "https://searchapi.eastmoney.com/api/suggest/get?input="
        + quote(text) + "&type=14&token=D43BF722C8E33BDC906FB84D85E326E8&count=5"
    )
    data = _http_json(url, timeout=6.0)
    items = (data.get("QuotationCodeTable") or {}).get("Data") or []
    for it in items:
        if it.get("Classify") == "AStock":
            return {"code": str(it.get("Code")), "name": str(it.get("Name") or "")}
    raise ValueError(f"没找到「{text}」对应的股票")
