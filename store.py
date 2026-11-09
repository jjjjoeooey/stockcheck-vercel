"""SQLite 持久化层。

一张 stocks 表存「原始日线 + 打分结果 + 基本面」，所以调权重时可以原地重算，
不用重新上传文件。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    code         TEXT PRIMARY KEY,
    name         TEXT,
    series       TEXT NOT NULL,   -- 原始日线 JSON（date/open/high/low/close/volume/amount/turnover）
    result       TEXT NOT NULL,   -- StockResult.to_dict() JSON
    fundamentals TEXT,             -- 基本面 dict JSON，可为 NULL
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------- settings ----------------
    def get_settings(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_settings(self, **kwargs) -> None:
        with self._conn() as c:
            for k, v in kwargs.items():
                c.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, json.dumps(v, ensure_ascii=False)),
                )

    # ---------------- stocks ----------------
    def upsert(self, code: str, name: str, series: list[dict], result: dict,
               fundamentals: dict | None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO stocks(code, name, series, result, fundamentals, updated_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(code) DO UPDATE SET "
                "name=excluded.name, series=excluded.series, result=excluded.result, "
                "fundamentals=excluded.fundamentals, updated_at=excluded.updated_at",
                (
                    code,
                    name,
                    json.dumps(series, ensure_ascii=False),
                    json.dumps(result, ensure_ascii=False),
                    json.dumps(fundamentals, ensure_ascii=False) if fundamentals else None,
                    time.time(),
                ),
            )

    def set_name(self, code: str, name: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE stocks SET name=? WHERE code=? AND (name IS NULL OR name='')", (name, code))

    def update_result(self, code: str, result: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE stocks SET result=?, updated_at=? WHERE code=?",
                      (json.dumps(result, ensure_ascii=False), time.time(), code))

    def set_fundamentals(self, code: str, fundamentals: dict | None) -> None:
        with self._conn() as c:
            c.execute("UPDATE stocks SET fundamentals=? WHERE code=?",
                      (json.dumps(fundamentals, ensure_ascii=False) if fundamentals else None, code))

    def all(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT code, name, result, fundamentals, updated_at FROM stocks"
            ).fetchall()
        return [
            {
                "code": r["code"],
                "name": r["name"],
                "result": json.loads(r["result"]),
                "fundamentals": json.loads(r["fundamentals"]) if r["fundamentals"] else None,
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def all_raw(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT code, name, series, fundamentals FROM stocks").fetchall()
        return [
            {
                "code": r["code"],
                "name": r["name"],
                "series": json.loads(r["series"]),
                "fundamentals": json.loads(r["fundamentals"]) if r["fundamentals"] else None,
            }
            for r in rows
        ]

    def get(self, code: str) -> dict | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT code, name, series, result, fundamentals, updated_at FROM stocks WHERE code=?",
                (code,),
            ).fetchone()
        if not r:
            return None
        return {
            "code": r["code"],
            "name": r["name"],
            "series": json.loads(r["series"]),
            "result": json.loads(r["result"]),
            "fundamentals": json.loads(r["fundamentals"]) if r["fundamentals"] else None,
            "updated_at": r["updated_at"],
        }

    def delete(self, code: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM stocks WHERE code=?", (code,))
            return cur.rowcount > 0

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM stocks")

    def codes(self) -> set[str]:
        with self._conn() as c:
            return {r["code"] for r in c.execute("SELECT code FROM stocks").fetchall()}

    def count(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM stocks").fetchone()[0])
