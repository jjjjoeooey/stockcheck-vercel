"""终端报告渲染：单票体检 + 批量排行。"""

from __future__ import annotations

import unicodedata

from .score import StockResult


def _width(s: str) -> int:
    """中文字符按 2 列宽计算，便于对齐表格。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int, align: str = "left") -> str:
    gap = max(0, width - _width(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def _bar(score: float | None, width: int = 12) -> str:
    if score is None:
        return "—"
    filled = int(round(max(0.0, min(100.0, score)) / 100.0 * width))
    return "█" * filled + "░" * (width - filled)


def _grade(total: float | None) -> str:
    if total is None:
        return "数据不足"
    if total >= 85:
        return "★★★★★ 重点跟踪"
    if total >= 75:
        return "★★★★ 观察"
    if total >= 60:
        return "★★★ 中性偏多"
    if total >= 45:
        return "★★ 偏弱"
    return "★ 回避"


def render_text(r: StockResult, show_metrics: bool = False) -> str:
    lines: list[str] = []
    title = f" {r.name} {r.code} " if r.name else f" {r.code} "
    lines.append("=" * 68)
    lines.append(_pad(title, 38, "center") + _pad(f"截至于 {r.date}", 30, "right"))
    lines.append("=" * 68)
    lines.append(
        f"收盘 {r.close:>8.2f}   总分 {r.total if r.total is not None else '—':>6}"
        f"   {_grade(r.total)}   覆盖度 {r.coverage * 100:.0f}%"
    )
    lines.append("-" * 68)
    lines.append(f"{'维度':<8}{'得分':>6}  {'权重':>5}  {'强度':<14} 说明")
    for d in r.dims:
        score_s = f"{d.score:.0f}" if d.score is not None else "—"
        lines.append(
            f"{_pad(d.label, 8)}{score_s:>6}  {d.weight:>5.0f}  {_bar(d.score):<14} {d.note}"
        )
        if show_metrics and d.metrics:
            small = {k: v for k, v in d.metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if small:
                lines.append(" " * 22 + "· " + "  ".join(f"{k}={v:.4g}" for k, v in list(small.items())[:8]))
    lines.append("-" * 68)
    if r.tags:
        lines.append("标签：" + "  ".join(f"[{t}]" for t in r.tags))
    lines.append("结论：" + r.conclusion)
    for w in r.warnings:
        lines.append("注意：" + w)
    lines.append("=" * 68)
    return "\n".join(lines)


def render_summary_table(results: list[StockResult]) -> str:
    ranked = sorted(results, key=lambda x: (x.total if x.total is not None else -1), reverse=True)
    lines: list[str] = []
    lines.append("-" * 96)
    header = (
        f"{_pad('排名', 5)}{_pad('代码', 8)}{_pad('名称', 10)}"
        f"{_pad('总分', 7, 'right')}{_pad('换手', 7, 'right')}{_pad('KDJ', 7, 'right')}"
        f"{_pad('位置', 7, 'right')}{_pad('布林', 7, 'right')}{_pad('压力', 7, 'right')}"
        f"{_pad('基本面', 8, 'right')}  评级 / 标签"
    )
    lines.append(header)
    lines.append("-" * 96)
    for i, r in enumerate(ranked, 1):
        s = r.scores
        def f(k):
            v = s.get(k)
            return f"{v:>5.0f}" if v is not None else "   —"
        tag = "/".join(r.tags) if r.tags else ""
        lines.append(
            f"{_pad(str(i), 5)}{_pad(r.code, 8)}{_pad(r.name, 10)}"
            f"{_pad(f'{r.total:.1f}' if r.total is not None else '—', 7, 'right')}"
            f"{_pad(f('turnover'), 7, 'right')}{_pad(f('kdj'), 7, 'right')}"
            f"{_pad(f('relpos'), 7, 'right')}{_pad(f('boll_extreme'), 7, 'right')}"
            f"{_pad(f('resistance'), 7, 'right')}{_pad(f('fundamental'), 8, 'right')}"
            f"  {_grade(r.total)} {tag}"
        )
    lines.append("-" * 96)
    return "\n".join(lines)
