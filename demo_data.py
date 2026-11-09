"""生成演示数据（固定随机种子），用于在没有真实导出文件时验证整条链路。

注意：演示数据是「人造」的，只用来展示模型输出形态，不代表任何真实股票。
每只股票都刻意构造成一种典型形态，方便检验各维度是否按预期给分。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _ohlc(
    closes: np.ndarray,
    volumes: np.ndarray,
    turnovers: np.ndarray,
    seed: int,
    start: str = "",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(closes)
    open_ = np.empty(n)
    open_[0] = closes[0]
    open_[1:] = closes[:-1] * (1 + rng.normal(0, 0.004, n - 1))
    hi_pad = np.abs(rng.normal(0, 0.008, n)) + 0.003
    lo_pad = np.abs(rng.normal(0, 0.008, n)) + 0.003
    high = np.maximum(open_, closes) * (1 + hi_pad)
    low = np.minimum(open_, closes) * (1 - lo_pad)
    end = pd.Timestamp.today().normalize()
    if end.weekday() >= 5:  # 周六/周日 -> 回退到上一个工作日
        end -= pd.Timedelta(days=end.weekday() - 4)
    dates = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame(
        {
            "日期": dates.strftime("%Y-%m-%d"),
            "开盘": np.round(open_, 2),
            "最高": np.round(high, 2),
            "最低": np.round(low, 2),
            "收盘": np.round(closes, 2),
            "成交量": np.round(volumes).astype(int),
            "换手率": np.round(turnovers, 2),
        }
    )


def _noise_path(n: int, start: float, drift: float, vol: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    return start * np.exp(np.cumsum(ret))


def _wiggle(n: int, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, amp, n)) - np.cumsum(rng.normal(0, amp, n)).mean()


def make_breakout(n: int = 320) -> pd.DataFrame:
    """放量突破：长期上行 + 最后 6 日加速放量，收在上轨外。"""
    base = _noise_path(n - 6, 40.0, 0.0008, 0.015, 101)
    last = np.linspace(base[-1] * 0.995, base[-1] * 1.155, 6)
    closes = np.concatenate([base, last])
    vol = np.linspace(1.2e6, 1.4e6, n - 6)
    vol = np.concatenate([vol, np.linspace(1.6e6, 3.2e6, 6)])
    turn = np.concatenate([np.linspace(1.8, 2.6, n - 6), np.linspace(3.0, 5.6, 6)])
    return _ohlc(closes, vol, turn, seed=101)


def make_oversold_turn(n: int = 320) -> pd.DataFrame:
    """超跌反弹：前期阴跌，最后 4 日急速拉起，K/D 低位金叉。"""
    down = _noise_path(n - 4, 60.0, -0.0012, 0.016, 102)
    down = down - np.linspace(0, (down[0] - down[-1]) * 0.25, len(down))  # 让跌幅更实
    last = np.linspace(down[-1] * 1.0, down[-1] * 1.14, 4)
    closes = np.concatenate([down, last])
    vol = np.full(n, 9e5)
    vol[-4:] = np.linspace(1.1e6, 2.0e6, 4)
    turn = np.concatenate([np.full(n - 4, 1.5), np.linspace(2.0, 3.4, 4)])
    return _ohlc(closes, vol, turn, seed=102)


def make_steady_trend(n: int = 320) -> pd.DataFrame:
    """稳健上升通道：20 日均线上方，相对位置 70%~85%，量能温和。"""
    base = _noise_path(n, 100.0, 0.0009, 0.014, 103)
    w = _wiggle(len(base), 0.9, 203)
    closes = base * (1 + w / base.mean() * 0.3)
    # 强制最后 20 日温和上行，确保站上中轨
    closes[-20:] = np.linspace(closes[-21], closes[-21] * 1.07, 20)
    vol = np.full(n, 1.1e6)
    vol[-20:] *= np.linspace(1.0, 1.25, 20)
    turn = np.full(n, 2.4)
    turn[-20:] = np.linspace(2.4, 3.0, 20)
    return _ohlc(closes, vol, turn, seed=103)


def make_range(n: int = 320) -> pd.DataFrame:
    """箱体震荡：上沿附近遇阻回落，布林收口，位置 50%~65%。"""
    base = _noise_path(n, 30.0, 0.0000, 0.012, 104)
    cyc = 0.10 * np.sin(np.arange(n) / 18.0)
    closes = base * (1 + cyc)
    closes[-8:] = np.linspace(closes[-9], closes[-9] * 0.985, 8)
    vol = np.full(n, 8e5)
    turn = np.full(n, 1.6)
    return _ohlc(closes, vol, turn, seed=104)


def make_downtrend(n: int = 320) -> pd.DataFrame:
    """下降趋势：均线空头排列，接近一年低点。"""
    base = _noise_path(n, 50.0, -0.0011, 0.015, 105)
    closes = base
    vol = np.full(n, 7e5)
    turn = np.full(n, 1.2)
    return _ohlc(closes, vol, turn, seed=105)


def make_overheat(n: int = 320) -> pd.DataFrame:
    """高位过热：涨幅巨大、相对位置 >95%、J 值极高，追高风险大。"""
    base = _noise_path(n - 15, 20.0, 0.0018, 0.020, 106)
    closes = np.concatenate([base, np.linspace(base[-1] * 1.02, base[-1] * 1.30, 15)])
    vol = np.linspace(1e6, 1.5e6, n - 15)
    vol = np.concatenate([vol, np.full(15, 2.2e6)])
    turn = np.concatenate([np.full(n - 15, 3.5), np.full(15, 9.0)])
    return _ohlc(closes, vol, turn, seed=106)


DEMO_BUILDERS = {
    "600519": ("贵州茅台", make_steady_trend),
    "300750": ("宁德时代", make_breakout),
    "601318": ("中国平安", make_downtrend),
    "600036": ("招商银行", make_range),
    "000001": ("平安银行", make_oversold_turn),
    "002594": ("比亚迪", make_overheat),
}

FUNDAMENTALS = [
    ("600519", "贵州茅台", 24.0, 7.2, 32.0, 15.0, 12.0, 18.0),
    ("300750", "宁德时代", 21.0, 4.1, 24.0, 32.0, 11.0, 68.0),
    ("601318", "中国平安", 9.0, 1.0, 11.0, -6.0, 2.0, 88.0),
    ("600036", "招商银行", 6.8, 0.9, 16.0, 6.0, -1.0, 92.0),
    ("000001", "平安银行", 5.2, 0.6, 9.0, 2.0, -4.0, 93.0),
    ("002594", "比亚迪", 28.0, 4.8, 19.0, 28.0, 24.0, 78.0),
]


def build_demo(outdir: str = "demo", n: int = 320) -> None:
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for code, (name, builder) in DEMO_BUILDERS.items():
        df = builder(n=n)
        path = os.path.join(outdir, f"{code}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        rows.append({"文件": path, "代码": code, "名称": name})

    # 代码列写成文本（前导零不能丢）
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, "pool.csv"), index=False, encoding="utf-8-sig"
    )
    fund = pd.DataFrame(
        FUNDAMENTALS,
        columns=["代码", "名称", "市盈率", "市净率", "净资产收益率", "净利润增长率", "营收增长率", "资产负债率"],
    )
    fund["代码"] = fund["代码"].astype(str)
    fund.to_csv(os.path.join(outdir, "fundamentals.csv"), index=False, encoding="utf-8-sig")
