"""模型入口：对外暴露 build_context / score_stock。

实现放在 score.py（打包时若只想保留一个文件也可直接用 score.py）。
"""

from __future__ import annotations

from .score import DEFAULT_WEIGHTS, DIM_LABELS, StockResult, build_context, score_stock

__all__ = ["build_context", "score_stock", "StockResult", "DEFAULT_WEIGHTS", "DIM_LABELS"]
