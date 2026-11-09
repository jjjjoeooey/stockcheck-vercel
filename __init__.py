"""stockcheck —— 七维股票体检 / 初筛模型。

指标: 换手、KDJ 金叉、相对位置、新布林极限、相对压力位、基本面概况。
"""

__version__ = "0.1.0"

from .indicators import (
    compute_boll,
    compute_boll_extreme,
    compute_kdj,
    compute_relpos,
    compute_resistance,
    compute_turnover,
)
from .data_io import (
    load_fundamentals,
    load_price_file,
    load_universe,
    normalize_price_frame,
)
from .model import build_context, score_stock
from .report import render_text

__all__ = [
    "__version__",
    "compute_boll",
    "compute_boll_extreme",
    "compute_kdj",
    "compute_relpos",
    "compute_resistance",
    "compute_turnover",
    "load_fundamentals",
    "load_price_file",
    "load_universe",
    "normalize_price_frame",
    "build_context",
    "score_stock",
    "render_text",
]
