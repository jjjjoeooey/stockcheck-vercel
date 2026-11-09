"""七维打分：把指标值翻译成 0~100 分，并合成总分、标签和结论。

权重（可调）
------------
换手 9 / 量能 9 / KDJ 金叉 16 / 相对位置 12 / 新布林极限 14 / 相对压力位 10 / 基本面 30
=> 技术面 70，基本面 30，合计 100。
（依据：267 只股票、29 个横截面、2023-2026 的 IC 检验，详见 research/）
缺数据的分项会被剔除并重新归一，最终报告里会写明「按可用数据覆盖度」。

另：boll_extreme 的说明里会附带「指南针口径」近似值（BIAS18 − MA30，±AVEDEV30），
只作对照，不参与打分。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from numbers import Number
from typing import Any, Optional

import numpy as np
import pandas as pd

from .indicators import (
    _clip01,
    _piecewise,
    compute_boll,
    compute_boll_extreme,
    compute_kdj,
    compute_relpos,
    compute_resistance,
    compute_turnover,
    compute_volume,
    compute_zn_limit,
)

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHTS: dict[str, float] = {
    "turnover": 9.0,
    "volume": 9.0,
    "kdj": 16.0,
    "relpos": 12.0,
    "boll_extreme": 14.0,
    "resistance": 10.0,
    "fundamental": 30.0,
}  # 合计 100（2026-11 实证调整：技术弱、基本面加权）

DIM_LABELS = {
    "turnover": "换手",
    "volume": "量能",
    "kdj": "KDJ金叉",
    "relpos": "相对位置",
    "boll_extreme": "新布林极限",
    "resistance": "相对压力位",
    "fundamental": "基本面",
}


def _jsonable(v: Any) -> Any:
    """只保留标量；Series/数组/NaN/Inf 不进 JSON（避免报告文件巨大且非法）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, Number):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, str):
        return v
    return None


@dataclass
class DimensionScore:
    key: str
    label: str
    score: Optional[float]
    weight: float
    note: str = ""
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metrics"] = {k: _jsonable(v) for k, v in self.metrics.items() if _jsonable(v) is not None}
        return d


@dataclass
class StockResult:
    code: str
    name: str
    close: float
    date: str
    total: Optional[float]
    dims: list[DimensionScore]
    tags: list[str]
    conclusion: str
    coverage: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "close": _jsonable(self.close),
            "date": self.date,
            "total": _jsonable(self.total),
            "coverage": _jsonable(self.coverage),
            "tags": list(self.tags),
            "conclusion": self.conclusion,
            "warnings": list(self.warnings),
            "dims": [d.to_dict() for d in self.dims],
        }

    @property
    def scores(self) -> dict:
        return {d.key: d.score for d in self.dims}


# --------------------------------------------------------------------------- #
# 各维度打分
# --------------------------------------------------------------------------- #


def _score_turnover(t: dict) -> tuple[float, str, dict]:
    used = "量比代理" if t["used_proxy"] else "换手率"
    level, pct, vs5 = t["level"], t["pct_250"], t["vs_5d"]

    if str(used) == "量比代理":
        # 量比：1.0 附近为常态，1.5~3 为放量，>5 为异常放量
        s_level = _piecewise(
            level, [(0.3, 15), (0.6, 45), (1.0, 60), (1.5, 78), (2.5, 88), (4.0, 80), (6.0, 55)]
        )
    else:
        # 换手率%：3%~8% 为「活跃但不失控」
        s_level = _piecewise(
            level, [(0.3, 12), (0.8, 38), (1.5, 55), (3.0, 78), (5.0, 88), (8.0, 82), (12.0, 60), (20.0, 30)]
        )

    s_pct = 50.0 if not np.isfinite(pct) else 25.0 + 0.75 * pct  # 分位 0→25, 100→100

    if np.isfinite(vs5):
        s_rel = _piecewise(vs5, [(0.5, 25), (0.8, 40), (1.0, 55), (1.5, 80), (2.5, 90), (4.0, 70)])
    else:
        s_rel = 50.0

    score = 0.50 * s_level + 0.30 * s_pct + 0.20 * s_rel
    if np.isfinite(level) and str(used) != "量比代理" and level > 15:
        score = min(score, 55.0)  # 极端换手：无论如何不奖励

    note = f"{used} {level:.2f}，近250日分位 {pct:.0f}%，较5日均量 {vs5:.2f}x"
    return float(np.clip(score, 0, 100)), note, t


def _score_volume(v: dict) -> tuple[float, str, dict]:
    """量能维度：放量上涨最好，放量下跌最差；地量/天量都打折。"""
    state = v["state"]
    vr, vr5 = v["v_ratio"], v["v_ratio5"]
    pct = v["v_pct250"]
    chg = v["chg_today"]

    base = {
        "放量上涨": 82.0,
        "缩量回调": 68.0,
        "放量下跌": 22.0,
        "地量": 45.0,
        "天量": 35.0,
        "平量": 52.0,
    }.get(state, 50.0)

    if np.isfinite(vr5):
        if vr5 > 1.2:
            base += 6
        elif vr5 < 0.85:
            base -= 4
    if np.isfinite(vr) and np.isfinite(chg):
        if 1.2 <= vr <= 2.5 and chg > 0:
            base += 5
        if vr > 3 and chg <= 0:
            base -= 12
    if v["divergence"]:
        base -= 8

    score = float(np.clip(base, 0, 100))
    note = (
        f"{state}：当日量 {vr:.2f}x 前20日均量（分位 {pct:.0f}%），"
        f"5日均量 {vr5:.2f}x，近5日 {v['price_5d']:+.1f}%"
    )
    if v["divergence"]:
        note += "，量价背离"
    return score, note, v


def _score_kdj(k: dict) -> tuple[float, str, dict]:
    state = k["state"]
    kv, dv = k["k"], k["d"]
    days = k["days_since_cross"]
    has_cross = np.isfinite(days)

    if state == "below":
        # 死叉/空头排列：分低，但 J 极度超卖给一点「反弹观察」分
        score = 32.0 if k["j"] < 0 else 26.0
        note = f"K {kv:.1f} < D {dv:.1f}，J {k['j']:.1f}，空头排列"
        return score, note, k

    if not has_cross:
        # K 在 D 上方但近期没有金叉记录（高位钝化/持续多头）
        score = 62.0 if kv > 70 else 56.0
        note = f"K {kv:.1f} > D {dv:.1f}，近20日无金叉（持续多头/高位钝化）"
        return score, note, k

    # 有金叉：低位金叉 > 高位金叉；越新鲜越好
    kc = k["k_at_cross"]
    base = 58.0 + 0.45 * np.clip(60.0 - kc, -20.0, 50.0)  # K=60→58, K=20→76
    if kc < 20:
        base += 8
    if k["j_oversold_before"]:
        base += 5
    base += min(8.0, max(0.0, 4.0 * (k["cross_strength"] or 0.0)))

    recency = _piecewise(days, [(0, 1.0), (1, 0.98), (3, 0.92), (7, 0.80), (14, 0.65), (20, 0.55)])
    score = 100.0 - (100.0 - base) * (1.0 / max(recency, 0.55))  # 太远则向中位收敛
    if state == "golden":
        score = max(score, base)
    score = float(np.clip(score, 0, 100))

    tag = "当日金叉" if days == 0 else f"{int(days)}日前金叉"
    note = f"{tag}（金叉位 K {kc:.1f}/D {k['d_at_cross']:.1f}），现 K {kv:.1f}/D {dv:.1f}/J {k['j']:.1f}"
    return score, note, k


def _score_relpos(r: dict) -> tuple[float, str, dict]:
    pos = r["pos"]
    if not np.isfinite(pos):
        return 50.0, "区间无波动，相对位置不可用", r
    # 甜点 0.72：有趋势但不过热；高斯型曲线
    score = 95.0 * np.exp(-((pos - 0.72) ** 2) / (2 * 0.26**2))
    if pos > 0.95:
        score -= (pos - 0.95) * 300  # 极高位额外扣分
    score = max(8.0, score)
    note = f"一年区间 {pos * 100:.1f}%（现价 {r['close']:.2f}，区间 {r['lo']:.2f}~{r['hi']:.2f}，距高点 {r['off_high_pct']:+.1f}%）"
    return float(np.clip(score, 0, 100)), note, r


def _score_boll_extreme(e: dict, boll: dict, close: float) -> tuple[float, str, dict]:
    bo = e["breakout"]
    exp_ = e["width_expansion"]
    pctb = e["pctb"]

    if not np.isfinite(bo):
        return 50.0, "布林数据不足", e

    s_bo = _piecewise(
        bo,
        [
            (-0.8, 8), (-0.4, 22), (-0.1, 35), (0.0, 45), (0.05, 55),
            (0.1, 70), (0.2, 82), (0.35, 88), (0.6, 80), (1.0, 62),
        ],
    )
    if np.isfinite(exp_) and exp_ > 1.0:
        s_exp = _piecewise(exp_, [(0.7, 30), (0.9, 45), (1.0, 50), (1.1, 65), (1.3, 80), (1.6, 88), (2.5, 80)])
    else:
        s_exp = _piecewise(exp_, [(0.7, 30), (0.9, 45), (1.0, 50)]) if np.isfinite(exp_) else 50.0

    score = 0.55 * s_bo + 0.45 * s_exp
    if bo > 0.02 and np.isfinite(exp_) and exp_ < 1.02:
        score = min(score, 62.0)  # 缩口假突破
    if bo < -0.15:
        score = min(score, 40.0)  # 跌破上轨且远离
    if np.isfinite(pctb) and pctb >= 0.95 and bo > 0:
        score = max(score, min(88.0, score + 4))

    # 跌破中轨属于空头背景：此时「带宽扩张」是下跌加速而不是突破，
    # 不能给奖励，整体压到 45 分以下。
    below_mid = bool(np.isfinite(pctb) and pctb < 0.5)
    if below_mid:
        score = min(score, 45.0)
        if bo < -0.5:
            score = min(score, 35.0)
    score = float(np.clip(score, 0, 100))

    flag = "开口突破" if (bo > 0 and np.isfinite(exp_) and exp_ >= 1.05) else (
        "贴上轨" if bo > -0.05 else ("中轨下方" if bo < -0.2 else "轨道内")
    )
    note = (
        f"{flag}：突破度 {bo:+.3f}（%B {pctb:.2f}），带宽 {e['width'] * 100:.2f}%"
        f"（近60日 {exp_:.2f}x）"
    )
    zn = e.get("zn") or {}
    if zn and np.isfinite(zn.get("value", float("nan"))):
        note += f"；指南针口径 {zn['value']:+.2f}（轨道 ±{zn['band']:.2f}，近似）"
        e["zn_value"] = zn["value"]
        e["zn_band"] = zn["band"]
    return score, note, e


def _score_resistance(r: dict) -> tuple[float, str, dict]:
    d = r["dist_to_r1"]
    if not np.isfinite(d):
        return 50.0, "压力位不可用", r

    # 基础分：上方空间越大越好（但已站上压力位/创新高时有独立逻辑）
    s = _piecewise(
        d, [(-8, 18), (-3, 25), (0, 30), (3, 38), (5, 50), (8, 62), (12, 74), (20, 84), (35, 90), (60, 95)]
    )
    up = r["upside_to_r2"]
    if np.isfinite(up) and up > 0:
        s += min(8.0, 0.25 * up)

    # 关键修正：下跌趋势里「上方空间大」往往是深套区域，不是利好。
    # 用「距年内高点回撤」和「是否在 20 日均线之下」做 regime 折扣。
    drawdown = r.get("off_high_pct")
    below_mid = r.get("below_boll_mid", False)
    if np.isfinite(drawdown) and drawdown < -15:
        s *= _piecewise(-drawdown, [(15, 1.0), (25, 0.85), (40, 0.7), (60, 0.55)])
    if below_mid and np.isfinite(drawdown) and drawdown < -8:
        s *= 0.85
    s = float(np.clip(s, 0, 100))

    note = f"R1 {r['r1']:.2f}（距现价 {d:+.1f}%），突破后到 R2 {r['r2']:.2f} 还有 {up:+.1f}%"
    if np.isfinite(drawdown) and drawdown < -15:
        note += f"；但距年内高点 {drawdown:+.1f}%，上方为深套区"
    if r["above_high"]:
        note += "；已创近一年新高"
    return s, note, r


def _score_fundamental(f: Optional[dict]) -> tuple[Optional[float], str, dict]:
    if not f:
        return None, "未提供基本面数据（该维度已剔除并重新归一）", {}

    parts: list[tuple[float, float, str]] = []  # (score, weight, text)

    roe = f.get("roe")
    if roe is not None and np.isfinite(roe):
        s = _piecewise(roe, [(-5, 0), (0, 15), (5, 40), (10, 65), (15, 85), (20, 93), (30, 97)])
        parts.append((s, 0.30, f"ROE {roe:.1f}%"))

    g = f.get("profit_growth")
    if g is not None and np.isfinite(g):
        s = _piecewise(g, [(-50, 5), (-20, 20), (-5, 40), (0, 50), (10, 65), (25, 80), (50, 90), (100, 95)])
        parts.append((s, 0.25, f"净利增速 {g:+.1f}%"))

    rg = f.get("revenue_growth")
    if rg is not None and np.isfinite(rg):
        s = _piecewise(rg, [(-30, 8), (-10, 28), (0, 45), (10, 62), (20, 78), (35, 88), (60, 92)])
        parts.append((s, 0.15, f"营收增速 {rg:+.1f}%"))

    pe = f.get("pe")
    if pe is not None and np.isfinite(pe):
        if pe <= 0:
            s = 20.0
        else:
            s = _piecewise(pe, [(5, 92), (10, 85), (15, 78), (25, 65), (40, 48), (60, 32), (100, 18), (200, 8)])
        parts.append((s, 0.15, f"PE {pe:.1f}"))

    pb = f.get("pb")
    if pb is not None and np.isfinite(pb):
        s = 90.0 if pb <= 0 else _piecewise(pb, [(0.8, 90), (1.5, 80), (3, 65), (5, 45), (8, 28), (15, 12)])
        parts.append((s, 0.05, f"PB {pb:.2f}"))

    debt = f.get("debt_ratio")
    if debt is not None and np.isfinite(debt):
        s = _piecewise(debt, [(20, 90), (35, 82), (50, 70), (65, 50), (75, 32), (90, 12)])
        parts.append((s, 0.10, f"负债率 {debt:.1f}%"))

    if not parts:
        return None, "基本面字段全空（该维度已剔除并重新归一）", f

    total_w = sum(p[1] for p in parts)
    score = sum(p[0] * p[1] for p in parts) / total_w
    note = "，".join(p[2] for p in parts)
    return float(np.clip(score, 0, 100)), note, f


def quick_score(row: dict) -> float:
    """全市场快照的粗筛分（0~100）。

    只用快照字段（换手、量比、涨跌幅、PE、PB），不拉 K 线，所以很快；
    它只用来决定「先深度扫描谁」，不能替代七维总分。
    """
    turn = row.get("turnover")
    vr = row.get("volume_ratio")
    pct = row.get("pct")
    pe = row.get("pe")
    pb = row.get("pb")

    s_turn = _piecewise(turn, [(0.3, 10), (0.8, 30), (1.5, 50), (3, 80), (6, 92), (10, 85), (18, 55), (30, 25)]) if turn is not None else 50.0
    s_vr = _piecewise(vr, [(0.5, 25), (0.8, 40), (1.0, 50), (1.5, 80), (2.5, 92), (4, 75), (6, 55)]) if vr is not None else 50.0
    if pct is None:
        s_pct = 50.0
    elif pct >= 9.5:
        s_pct = 78.0 if pct < 15 else 62.0   # 涨停/大涨：强，但追高折价
    else:
        s_pct = _piecewise(pct, [(-9, 8), (-5, 20), (-2, 35), (0, 48), (2, 65), (5, 82), (8, 88)])
    if pe is None or pe <= 0:
        s_pe = 25.0
    else:
        s_pe = _piecewise(pe, [(5, 92), (15, 82), (30, 68), (60, 45), (120, 22), (300, 8)])
    s_pb = 50.0 if pb is None else (85.0 if pb <= 0 else _piecewise(pb, [(1, 88), (3, 75), (5, 55), (10, 30), (20, 12)]))

    score = 0.35 * s_turn + 0.25 * s_vr + 0.20 * s_pct + 0.10 * s_pe + 0.10 * s_pb
    return float(np.clip(score, 0, 100))


# --------------------------------------------------------------------------- #
# 合成
# --------------------------------------------------------------------------- #


def _make_tags(dims: dict[str, DimensionScore], extra: dict) -> list[str]:
    tags: list[str] = []
    kdj_note = dims["kdj"].note
    if "当日金叉" in kdj_note:
        tags.append("当日KDJ金叉")
    elif "日前金叉" in kdj_note:
        tags.append("近期KDJ金叉")
    if "开口突破" in dims["boll_extreme"].note:
        tags.append("布林开口突破")
    if dims["relpos"].score is not None and dims["relpos"].score >= 80:
        tags.append("强势区间")
    if dims["relpos"].score is not None and dims["relpos"].score <= 25:
        tags.append("弱势区间")
    if extra.get("near_high"):
        tags.append("接近年内高点")
    if dims["turnover"].score is not None and dims["turnover"].score >= 80:
        tags.append("明显放量")
    vol_note = dims["volume"].note
    if "放量上涨" in vol_note:
        tags.append("放量上攻")
    elif "缩量回调" in vol_note:
        tags.append("缩量回踩")
    elif "放量下跌" in vol_note:
        tags.append("放量下跌")
    elif "地量" in vol_note:
        tags.append("地量")
    elif "天量" in vol_note:
        tags.append("天量")
    if "量价背离" in vol_note:
        tags.append("量价背离")
    if extra.get("fundamental_good"):
        tags.append("基本面优良")
    return tags


def _conclusion(total: Optional[float], tags: list[str]) -> str:
    if total is None:
        return "数据不足，无法给出结论。"
    if total >= 85:
        head = "七维共振的强势票"
    elif total >= 75:
        head = "整体偏强，可列入重点观察"
    elif total >= 60:
        head = "中性偏多，趋势未走坏但缺乏共振"
    elif total >= 45:
        head = "偏弱，暂时没有明确买点"
    else:
        head = "弱势，建议回避或等待结构修复"

    if "当日KDJ金叉" in tags and "布林开口突破" in tags:
        return f"{head}：KDJ 金叉与布林开口同日出现，是最值得盯的共振信号，但需次日量价确认。"
    if "布林开口突破" in tags and ("当日KDJ金叉" in tags or "近期KDJ金叉" in tags):
        return f"{head}：布林开口突破叠加 KDJ 金叉，是本次扫描里的标准共振形态，可等回踩上轨不破再介入。"
    if "布林开口突破" in tags:
        return f"{head}：布林已开口突破，等 KDJ 金叉或缩量回踩上轨再介入。"
    if "当日KDJ金叉" in tags or "近期KDJ金叉" in tags:
        return f"{head}：KDJ 已金叉，等布林开口或放量确认后再动手更稳。"
    if "接近年内高点" in tags:
        return f"{head}：已逼近年内高位，压力与获利盘都在，追高性价比低。"
    return f"{head}。"


def build_context(df: pd.DataFrame) -> dict:
    """一次性算出所有指标，便于测试与二次开发。"""
    boll = compute_boll(df)
    return {
        "turnover": compute_turnover(df),
        "volume": compute_volume(df),
        "kdj": compute_kdj(df),
        "relpos": compute_relpos(df),
        "boll": boll,
        "boll_extreme": compute_boll_extreme(df, boll),
        "zn_limit": compute_zn_limit(df),
        "resistance": compute_resistance(df, compute_relpos(df)),
    }


def score_stock(
    df: pd.DataFrame,
    fundamentals: Optional[dict] = None,
    weights: Optional[dict] = None,
    code: str = "",
    name: str = "",
    min_bars: int = 60,
) -> StockResult:
    """对单只股票打分。

    df            : 标准日线（至少 60 根，建议 250+）
    fundamentals  : {'roe':..,'profit_growth':..,'revenue_growth':..,'pe':..,'pb':..,'debt_ratio':..}
    weights       : 覆盖默认权重，键为 turnover/kdj/relpos/boll_extreme/resistance/fundamental
    """
    warnings: list[str] = []
    code = code or df.attrs.get("code", "")
    name = name or df.attrs.get("name", "")

    if df is None or len(df) < min_bars:
        raise ValueError(f"数据不足：需要至少 {min_bars} 根日线，实际 {0 if df is None else len(df)} 根")

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})
    if sum(w.values()) <= 0:
        raise ValueError("权重之和必须为正")

    ctx = build_context(df)
    close = float(df["close"].iloc[-1])
    date = pd.to_datetime(df["date"].iloc[-1]).strftime("%Y-%m-%d")

    if len(df) < 120:
        warnings.append("历史数据不足 120 根，相对位置/压力位/布林分位参考性下降")

    s_turn, n_turn, m_turn = _score_turnover(ctx["turnover"])
    s_vol, n_vol, m_vol = _score_volume(ctx["volume"])
    s_kdj, n_kdj, m_kdj = _score_kdj(ctx["kdj"])
    s_pos, n_pos, m_pos = _score_relpos(ctx["relpos"])
    ctx["boll_extreme"]["zn"] = ctx.get("zn_limit")
    s_boll, n_boll, m_boll = _score_boll_extreme(ctx["boll_extreme"], ctx["boll"], close)
    s_res, n_res, m_res = _score_resistance(ctx["resistance"])
    s_fund, n_fund, m_fund = _score_fundamental(fundamentals)

    dims = {
        "turnover": DimensionScore("turnover", DIM_LABELS["turnover"], s_turn, w["turnover"], n_turn, m_turn),
        "volume": DimensionScore("volume", DIM_LABELS["volume"], s_vol, w["volume"], n_vol, m_vol),
        "kdj": DimensionScore("kdj", DIM_LABELS["kdj"], s_kdj, w["kdj"], n_kdj, m_kdj),
        "relpos": DimensionScore("relpos", DIM_LABELS["relpos"], s_pos, w["relpos"], n_pos, m_pos),
        "boll_extreme": DimensionScore("boll_extreme", DIM_LABELS["boll_extreme"], s_boll, w["boll_extreme"], n_boll, m_boll),
        "resistance": DimensionScore("resistance", DIM_LABELS["resistance"], s_res, w["resistance"], n_res, m_res),
        "fundamental": DimensionScore("fundamental", DIM_LABELS["fundamental"], s_fund, w["fundamental"], n_fund, m_fund),
    }

    available = {k: d for k, d in dims.items() if d.score is not None and d.weight > 0}
    wsum = sum(d.weight for d in available.values())
    total = sum(d.score * d.weight for d in available.values()) / wsum if wsum > 0 else None
    coverage = wsum / sum(w.values())

    if s_fund is None:
        warnings.append("缺少基本面数据：总分已按技术面重新归一，跨股票比较时注意口径")

    extra = {
        "near_high": ctx["resistance"]["near_high"],
        "fundamental_good": bool(s_fund is not None and s_fund >= 75),
    }
    tags = _make_tags(dims, extra)

    return StockResult(
        code=str(code),
        name=str(name),
        close=close,
        date=date,
        total=round(float(total), 2) if total is not None else None,
        dims=list(dims.values()),
        tags=tags,
        conclusion=_conclusion(round(float(total), 2) if total is not None else None, tags),
        coverage=round(coverage, 3),
        warnings=warnings,
    )
