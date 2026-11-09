"""技术指标计算：换手、KDJ、相对位置、布林（含新布林极限）、相对压力位。

所有函数输入为标准日线 DataFrame（date/open/high/low/close/volume/amount/turnover），
输出为「指标值 dict」，不打分；打分在 score.py 中统一完成，便于单独验证。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _piecewise(x: float, points: list[tuple[float, float]]) -> float:
    """按 (阈值, 分数) 折线插值；points 需按阈值升序。x 超出两端取端点分。"""
    if not np.isfinite(x):
        return float("nan")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    return float(np.interp(x, xs, ys))


# --------------------------------------------------------------------------- #
# 换手
# --------------------------------------------------------------------------- #


def compute_turnover(df: pd.DataFrame, lookback: int = 250) -> dict:
    """换手维度。

    口径
    ----
    level   : 最新交易日换手率(%)；若文件没有换手率列，退回「量比」近似
              （量比 = 当日成交量 / 前 20 日均量），并在 used_proxy 中标记。
    pct_250 : 当日换手在近 250 日换手分布中的分位（0~100）。
    vs_5d   : 当日换手 / 近 5 日平均换手；<0.8 表示缩量。
    """
    n = len(df)
    win = df.tail(min(lookback, n))

    used_proxy = False
    if df["turnover"].notna().sum() >= min(60, n // 2) and df["turnover"].tail(5).notna().any():
        turnover = df["turnover"].astype(float)
    else:
        used_proxy = True
        vol = df["volume"].astype(float)
        base = vol.rolling(20, min_periods=10).mean().shift(1)
        turnover = vol / base.replace(0, np.nan)

    t_win = turnover.tail(len(win)).dropna()
    level = float(turnover.iloc[-1]) if np.isfinite(turnover.iloc[-1]) else float("nan")

    if len(t_win) >= 40 and np.isfinite(level):
        pct = float((t_win <= level).mean() * 100.0)
    else:
        pct = float("nan")

    recent = turnover.tail(6).dropna()
    vs_5d = float("nan")
    if len(recent) >= 3:
        denom = float(recent.iloc[:-1].mean())
        if denom > 0:
            vs_5d = float(recent.iloc[-1] / denom)

    return {
        "level": level,
        "pct_250": pct,
        "vs_5d": vs_5d,
        "avg_20": float(t_win.tail(20).mean()) if len(t_win) else float("nan"),
        "used_proxy": used_proxy,
    }


# --------------------------------------------------------------------------- #
# KDJ
# --------------------------------------------------------------------------- #


def compute_kdj(
    df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3, lookback_cross: int = 20
) -> dict:
    """KDJ 及金叉判定（K/D 平滑系数 1/3，等价于通达信 KDJ(9,3,3)）。

    返回
    ----
    k, d, j           : 最新值
    state             : 'golden'（当日金叉）/ 'above'（K 在 D 上方但非当日）
                        / 'below'（K 在 D 下方）
    days_since_cross  : 最近一次金叉距今天数；无则 NaN
    k_at_cross, d_at_cross : 金叉当日 K/D 值（判断低位金叉）
    cross_strength    : 金叉日 K 相对前日 K 的抬升幅度
    j_oversold_before : 金叉前 J 是否曾 < 0（超卖修复，加分项）
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    llv = low.rolling(n, min_periods=n).min()
    hhv = high.rolling(n, min_periods=n).max()
    rng = (hhv - llv).replace(0, np.nan)
    rsv = ((close - llv) / rng * 100.0).fillna(50.0)
    k = rsv.ewm(alpha=1.0 / m1, adjust=False, min_periods=n).mean()
    d = k.ewm(alpha=1.0 / m2, adjust=False, min_periods=n).mean()
    j = 3.0 * k - 2.0 * d

    kv, dv, jv = float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])

    cross_up = (k > d) & (k.shift(1) <= d.shift(1))
    state = "above" if kv > dv else "below"
    days_since_cross = float("nan")
    k_at_cross = d_at_cross = cross_strength = float("nan")
    j_oversold_before = False

    idx = np.where(cross_up.to_numpy())[0]
    idx = idx[idx >= n + 1]
    if len(idx):
        last = int(idx[-1])
        gap = len(df) - 1 - last
        if gap == 0:
            state = "golden"
        if gap <= lookback_cross:
            days_since_cross = float(gap)
            k_at_cross = float(k.iloc[last])
            d_at_cross = float(d.iloc[last])
            k_prev = float(k.iloc[last - 1])
            cross_strength = k_at_cross - k_prev
            lo = max(0, last - 6)
            j_oversold_before = bool((j.iloc[lo:last] < 0).any())

    return {
        "k": kv,
        "d": dv,
        "j": jv,
        "state": state,
        "days_since_cross": days_since_cross,
        "k_at_cross": k_at_cross,
        "d_at_cross": d_at_cross,
        "cross_strength": cross_strength,
        "j_oversold_before": j_oversold_before,
        "k_series": k,
        "d_series": d,
    }


# --------------------------------------------------------------------------- #
# 布林 + 新布林极限
# --------------------------------------------------------------------------- #


def compute_boll(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> dict:
    close = df["close"].astype(float)
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std

    width = (upper - lower) / mid.replace(0, np.nan)
    b = (close - lower) / (upper - lower).replace(0, np.nan)

    return {
        "mid": float(mid.iloc[-1]),
        "upper": float(upper.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "pctb": float(b.iloc[-1]) if np.isfinite(b.iloc[-1]) else float("nan"),
        "width": float(width.iloc[-1]) if np.isfinite(width.iloc[-1]) else float("nan"),
        "width_series": width,
        "pctb_series": b,
        "mid_series": mid,
        "upper_series": upper,
        "lower_series": lower,
    }


def compute_boll_extreme(df: pd.DataFrame, boll: dict, lookback: int = 60) -> dict:
    """新布林极限：上轨突破力度 + 带宽扩张，两者共振才算「真突破」。

    breakout = (收盘 - 上轨) / (上轨 - 中轨)     上轨外为正，越大越极端
    width_expansion = 带宽 / 近 60 日带宽均值     >1 表示开口放大
    """
    close = df["close"].astype(float)
    upper = boll["upper_series"]
    mid = boll["mid_series"]
    width = boll["width_series"]

    half = (upper - mid).replace(0, np.nan)
    breakout = ((close - upper) / half).replace([np.inf, -np.inf], np.nan)
    hist = width.tail(min(lookback, len(width))).dropna()
    width_mean = float(hist.mean()) if len(hist) else float("nan")

    width_now = float(width.iloc[-1]) if np.isfinite(width.iloc[-1]) else float("nan")
    expansion = float(width_now / width_mean) if width_mean and width_mean > 0 else float("nan")

    # 上轨突破的历史分位（近 250 日），用于确认「极限」程度
    b = boll["pctb_series"].tail(min(250, len(close))).dropna()
    pctb_now = float(boll["pctb"]) if np.isfinite(boll["pctb"]) else float("nan")
    pctb_pct = float((b <= pctb_now).mean() * 100.0) if len(b) and np.isfinite(pctb_now) else float("nan")

    body_close_strong = bool(np.isfinite(pctb_now) and pctb_now >= 0.95)

    return {
        "breakout": float(breakout.iloc[-1]) if np.isfinite(breakout.iloc[-1]) else float("nan"),
        "width": width_now,
        "width_expansion": expansion,
        "pctb": pctb_now,
        "pctb_pct": pctb_pct,
        "close_strong": body_close_strong,
    }


# --------------------------------------------------------------------------- #
# 相对位置
# --------------------------------------------------------------------------- #


def compute_relpos(df: pd.DataFrame, window: int = 250) -> dict:
    """相对位置 = (收盘 - N 日最低) / (N 日最高 - N 日最低)，N 默认 250（约一年）。"""
    w = df.tail(min(window, len(df)))
    hi = float(w["high"].max())
    lo = float(w["low"].min())
    close = float(df["close"].iloc[-1])
    pos = (close - lo) / (hi - lo) if hi > lo else float("nan")
    return {
        "pos": float(pos),
        "hi": hi,
        "lo": lo,
        "close": close,
        "window_used": int(len(w)),
        "off_high_pct": (close / hi - 1.0) * 100.0 if hi > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# 相对压力位
# --------------------------------------------------------------------------- #


def _swing_highs(df: pd.DataFrame, left: int = 6, right: int = 6) -> list[float]:
    h = df["high"].to_numpy(dtype=float)
    idx = [
        i
        for i in range(left, len(h) - right)
        if h[i] == h[i - left : i + right + 1].max()
    ]
    return [float(h[i]) for i in idx]


def _volume_profile_resistance(df: pd.DataFrame, bins: int = 60) -> list[float]:
    """筹码/成交量分布：返回当前价上方的成交密集区价位（由近到远）。"""
    w = df.tail(min(250, len(df)))
    price = ((w["high"] + w["low"] + w["close"]) / 3.0).to_numpy(dtype=float)
    vol = w["volume"].to_numpy(dtype=float)
    if not np.isfinite(vol).any() or np.nanmax(vol) <= 0:
        return []
    lo, hi = float(w["low"].min()), float(w["high"].max())
    if not (hi > lo):
        return []
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist, _ = np.histogram(price, bins=edges, weights=np.nan_to_num(vol))
    hist = hist.astype(float)
    if hist.sum() <= 0:
        return []
    hist = hist / hist.sum()
    close = float(df["close"].iloc[-1])

    # 上方：从当前价往上累计到 50% 成交量，取该累积边界价作为主压力
    above = centers > close
    out: list[float] = []
    if above.any():
        csum = np.cumsum(hist[above])
        i = int(np.searchsorted(csum, 0.5))
        i = min(i, len(csum) - 1)
        out.append(float(centers[above][i]))
    return out


def compute_resistance(df: pd.DataFrame, relpos: dict) -> dict:
    """相对压力位。

    候选压力：
      R1 = 上方第一个成交量密集区（筹码峰），无则取近 250 日最高价
      R2 = 近 250 日最高价（若与 R1 不同，作为突破后的第二目标）
    输出：
      dist_to_r1   : (R1 - 收盘) / 收盘 * 100，负值表示已站上
      upside_ratio : (R2 - 收盘) / 收盘 * 100，突破后的理论空间
      near_high    : 收盘是否已到近 250 日高点 95% 以上
    """
    close = float(df["close"].iloc[-1])
    hi250 = float(relpos["hi"])
    highs = _swing_highs(df)
    above_highs = sorted([x for x in highs if x > close * 1.005])

    vp = _volume_profile_resistance(df)
    r1_candidates = vp + above_highs
    if r1_candidates:
        r1 = min(r1_candidates)
    elif close < hi250:
        r1 = hi250
    else:
        r1 = hi250  # 已创新高，R1 视为前高

    r2 = hi250 if hi250 > r1 * 1.005 else r1 * 1.04
    dist_r1 = (r1 / close - 1.0) * 100.0 if close > 0 else float("nan")
    upside = (r2 / close - 1.0) * 100.0 if close > 0 else float("nan")

    ma20 = float(df["close"].astype(float).tail(20).mean()) if len(df) >= 20 else float("nan")

    return {
        "r1": float(r1),
        "r2": float(r2),
        "dist_to_r1": float(dist_r1),
        "upside_to_r2": float(upside),
        "near_high": bool(close >= hi250 * 0.95),
        "above_high": bool(close >= hi250),
        "has_volume_peak": bool(vp),
        "off_high_pct": float(relpos.get("off_high_pct", float("nan"))),
        "below_boll_mid": bool(np.isfinite(ma20) and close < ma20),
    }

# --------------------------------------------------------------------------- #
# 量能（放量 / 缩量 / 地量 / 天量 / 量价背离）
# --------------------------------------------------------------------------- #


def compute_volume(df: pd.DataFrame, short: int = 5, long: int = 20, lookback: int = 250) -> dict:
    """成交量因子。

    核心比率都拿「当期 / 之前均值」算，避免当日量把自身均值抬高。
      v_ratio     : 当日量 / 前20日均量
      v_ratio5    : 近5日均量 / 前20日均量（量能趋势）
      v_pct250    : 当日量在近250日中的分位（地量/天量）
      price_5d    : 近5日涨跌幅
      state       : 放量上涨 / 缩量回调 / 放量下跌 / 地量 / 天量 / 平量
    """
    vol = df["volume"].astype(float)
    close = df["close"].astype(float)
    ma20_prev = vol.rolling(long, min_periods=long).mean().shift(1)
    v_ratio = float(vol.iloc[-1] / ma20_prev.iloc[-1]) if ma20_prev.iloc[-1] else float("nan")

    v5 = float(vol.tail(short).mean())
    v20_prev = float(vol.iloc[-long - 1:-1].mean()) if len(vol) > long else float("nan")
    v_ratio5 = v5 / v20_prev if v20_prev and np.isfinite(v20_prev) else float("nan")

    v_win = vol.tail(min(lookback, len(vol))).dropna()
    v_pct250 = float((v_win <= vol.iloc[-1]).mean() * 100.0) if len(v_win) >= 40 else float("nan")

    price_5d = float((close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0) if len(close) > 6 else float("nan")
    chg_today = float((close.iloc[-1] / close.iloc[-2] - 1.0) * 100.0) if len(close) > 2 else float("nan")

    if np.isfinite(v_pct250) and v_pct250 < 10:
        state = "地量"
    elif np.isfinite(v_pct250) and v_pct250 > 95:
        state = "天量"
    elif np.isfinite(price_5d) and np.isfinite(v_ratio5):
        if price_5d > 2 and v_ratio5 > 1.3:
            state = "放量上涨"
        elif price_5d < -2 and v_ratio5 > 1.3:
            state = "放量下跌"
        elif price_5d < 0 and v_ratio5 < 0.85:
            state = "缩量回调"
        else:
            state = "平量"
    else:
        state = "平量"

    # 量价背离：价格 5 日新高但量能低于 20 日均量，或价格新低但放量
    hi5 = bool(len(close) > 6 and close.iloc[-1] >= close.tail(6).max() - 1e-9)
    divergence = bool(
        hi5 and np.isfinite(v_ratio5) and v_ratio5 < 0.8
    )

    return {
        "v_ratio": v_ratio,
        "v_ratio5": v_ratio5,
        "v_pct250": v_pct250,
        "price_5d": price_5d,
        "chg_today": chg_today,
        "state": state,
        "divergence": divergence,
        "avg_vol_20": float(v20_prev) if np.isfinite(v20_prev) else float("nan"),
    }

def compute_zn_limit(df: pd.DataFrame, n: int = 18, m: int = 30, band_n: int = 30) -> dict:
    """指南针「新布林极限」复刻（近似）。

    实测公式（上证指数 7 个连续值，误差 0.02）：
        BIAS18 = (收盘 / MA18 - 1) * 100
        I      = BIAS18 - MA30(BIAS18)
        上/下轨 = ±AVEDEV(I, 30)

    注意：创业板指同结构但幅度约大 10%，可能还有一层按波动率缩放；
    因此标注为「近似」，不保证与指南针逐位一致。
    """
    c = df["close"].astype(float)
    bias = (c / c.rolling(n).mean() - 1.0) * 100.0
    i = bias - bias.rolling(m).mean()
    dev = (i - i.rolling(band_n).mean()).abs().rolling(band_n).mean()
    return {
        "value": float(i.iloc[-1]) if np.isfinite(i.iloc[-1]) else float("nan"),
        "band": float(dev.iloc[-1]) if np.isfinite(dev.iloc[-1]) else float("nan"),
        "bias": float(bias.iloc[-1]) if np.isfinite(bias.iloc[-1]) else float("nan"),
        "series": i,
    }
