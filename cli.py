"""命令行入口。

用法
----
# 单票体检（指南针导出文件）
python -m stockcheck.cli one --file 600519.csv --code 600519 --name 贵州茅台

# 单票体检（联网取真实前复权行情） + 基本面
python -m stockcheck.cli one --code 300750 --fundamentals fundamentals.csv

# 批量初筛（股票池 csv：文件,代码,名称）
python -m stockcheck.cli screen --universe pool.csv --top 20 --min-score 70

# 批量初筛（所有股票在同一张宽表里，含代码列）
python -m stockcheck.cli screen --wide all.csv --top 20

# 生成演示数据并跑一遍
python -m stockcheck.cli demo
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .data_io import (
    fetch_fundamentals_akshare,
    fetch_quote_em,
    fetch_price_akshare,
    fetch_price_em,
    load_fundamentals,
    load_price_file,
    load_universe,
    normalize_price_frame,
    read_table,
)
from .model import score_stock
from .report import render_summary_table, render_text


class CliError(Exception):
    """可预期的用户错误：打印一句话，返回退出码 2。"""


def _fund_dict_row(row) -> dict:
    keys = ("pe", "pb", "roe", "profit_growth", "revenue_growth", "debt_ratio", "gross_margin", "market_cap")
    out = {}
    for k in keys:
        v = row.get(k)
        if v is not None and pd.notna(v):
            out[k] = float(v)
    return out


def _load_price(args) -> tuple[pd.DataFrame, str, str]:
    if getattr(args, "file", None):
        df = load_price_file(args.file, code=args.code or "", name=args.name or "")
        return df, args.code or df.attrs.get("code", ""), args.name or df.attrs.get("name", "")
    if not args.code:
        raise CliError("请提供 --file（导出的日线文件）或 --code（联网取数）")
    df = fetch_price_em(args.code, days=getattr(args, "days", 600))
    return df, args.code, args.name or str(df.attrs.get("name") or "")


def cmd_one(args) -> int:
    df, code, name = _load_price(args)

    fund = None
    if args.fundamentals:
        table = load_fundamentals(args.fundamentals)
        hit = table[table["code"] == str(code)]
        if len(hit):
            fund = _fund_dict_row(hit.iloc[0])
            name = name or str(hit.iloc[0].get("name", ""))
        else:
            print(f"[warn] 基本面表里没有 {code}，将剔除基本面维度", file=sys.stderr)
    elif args.akshare_fundamentals:
        fund = fetch_fundamentals_akshare(code)
    elif not args.file:
        try:
            q = fetch_quote_em(code)
            fund = {k: q[k] for k in ("pe", "pb", "market_cap") if q.get(k) is not None} or None
        except Exception:
            fund = None

    res = score_stock(df, fundamentals=fund, code=code, name=name)

    payload = res.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_text(res, show_metrics=args.metrics))
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 JSON -> {args.save}")
    return 0


def cmd_screen(args) -> int:
    results = []
    if args.wide:
        df_all = read_table(args.wide)
        df_all = df_all.loc[:, ~df_all.columns.duplicated()]
        name_map = {}
        if "名称" in df_all.columns:
            name_map = dict(zip(df_all["代码"].astype(str), df_all["名称"].astype(str)))
        fund_table = load_fundamentals(args.fundamentals) if args.fundamentals else None

        codes = df_all["代码"].astype(str).str.extract(r"(\d{6})", expand=False).dropna().unique()
        for code in codes:
            try:
                sub = df_all[df_all["代码"].astype(str).str.contains(code, regex=False)]
                price = normalize_price_frame(sub, code=code, name=name_map.get(code, ""))
                fund = None
                if fund_table is not None:
                    hit = fund_table[fund_table["code"] == code]
                    if len(hit):
                        fund = _fund_dict_row(hit.iloc[0])
                results.append(score_stock(price, fundamentals=fund, code=code, name=name_map.get(code, "")))
            except Exception as exc:
                print(f"[skip] {code}: {exc}", file=sys.stderr)
    else:
        pool = load_universe(args.universe)
        fund_table = load_fundamentals(args.fundamentals) if args.fundamentals else None
        for item in pool:
            code = item["code"]
            try:
                if item["file"]:
                    price = load_price_file(item["file"], code=code, name=item["name"])
                else:
                    price = fetch_price_em(code)
                fund = None
                if fund_table is not None:
                    hit = fund_table[fund_table["code"] == code]
                    if len(hit):
                        fund = _fund_dict_row(hit.iloc[0])
                results.append(score_stock(price, fundamentals=fund, code=code, name=item["name"]))
            except Exception as exc:
                print(f"[skip] {code}: {exc}", file=sys.stderr)

    if not results:
        print("没有成功打分的股票。", file=sys.stderr)
        return 1

    filtered = [r for r in results if r.total is not None and r.total >= args.min_score]
    filtered.sort(key=lambda x: x.total, reverse=True)
    if args.top:
        filtered = filtered[: args.top]

    if args.json:
        print(json.dumps([r.to_dict() for r in filtered], ensure_ascii=False, indent=2, default=str))
    else:
        print(render_summary_table(filtered))
        print(f"\n共评估 {len(results)} 只，通过阈值(≥{args.min_score:g}) {len(filtered)} 只。")
    if args.save:
        pd.DataFrame(
            [
                {
                    "代码": r.code, "名称": r.name, "总分": r.total,
                    **{f"{d.label}": d.score for d in r.dims},
                    "标签": "/".join(r.tags), "结论": r.conclusion, "覆盖度": r.coverage,
                }
                for r in filtered
            ]
        ).to_csv(args.save, index=False, encoding="utf-8-sig")
        print(f"已保存 CSV -> {args.save}")
    return 0


def cmd_demo(args) -> int:
    from .demo_data import build_demo

    build_demo(args.outdir)
    print(f"演示数据已生成 -> {args.outdir}/")
    args2 = argparse.Namespace(
        universe=f"{args.outdir}/pool.csv",
        wide=None,
        fundamentals=f"{args.outdir}/fundamentals.csv",
        akshare=False,
        min_score=0.0,
        top=20,
        json=False,
        save=None,
    )
    return cmd_screen(args2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stockcheck", description="七维股票体检/初筛模型")
    sub = p.add_subparsers(dest="cmd", required=True)

    one = sub.add_parser("one", help="单只股票体检")
    one.add_argument("--file", help="指南针导出的 CSV/TXT/XLS 文件")
    one.add_argument("--code", help="6 位股票代码（无 --file 时走 akshare）")
    one.add_argument("--name", default="", help="股票名称（可选）")
    one.add_argument("--start", default="20200101", help="（已废弃）akshare 起始日期")
    one.add_argument("--days", type=int, default=600, help="联网取数的日线根数，默认 600")
    one.add_argument("--fundamentals", help="基本面 CSV")
    one.add_argument("--akshare", action="store_true", help="（已废弃）保留兼容，默认走东方财富")
    one.add_argument("--akshare-fundamentals", action="store_true", help="用 akshare 抓基本面（不稳定）")
    one.add_argument("--metrics", action="store_true", help="打印原始指标值")
    one.add_argument("--json", action="store_true")
    one.add_argument("--save", help="保存 JSON 结果路径")
    one.set_defaults(func=cmd_one)

    scr = sub.add_parser("screen", help="批量初筛")
    scr.add_argument("--universe", help="股票池 CSV（文件,代码,名称）")
    scr.add_argument("--wide", help="宽表 CSV（所有股票在同一张表，含代码列）")
    scr.add_argument("--fundamentals", help="基本面 CSV")
    scr.add_argument("--akshare", action="store_true", help="（已废弃）保留兼容")
    scr.add_argument("--min-score", type=float, default=0.0)
    scr.add_argument("--top", type=int, default=50)
    scr.add_argument("--json", action="store_true")
    scr.add_argument("--save", help="导出 CSV 路径")
    scr.set_defaults(func=cmd_screen)

    demo = sub.add_parser("demo", help="生成演示数据并跑一遍批量初筛")
    demo.add_argument("--outdir", default="demo")
    demo.set_defaults(func=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # 多为 akshare 缺失/网络问题
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
