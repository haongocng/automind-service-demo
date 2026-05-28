from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
HEART_DIR = ROOT_DIR / "examples" / "heart_disease"
DB_PATH = HEART_DIR / "heart_disease.sqlite"
TABLE_SOURCES = {
    "heart_train": HEART_DIR / "heart_train.csv",
    "heart_test": HEART_DIR / "heart_test.csv",
}


def main() -> None:
    HEART_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        for table_name, csv_path in TABLE_SOURCES.items():
            rows = _read_csv(csv_path)
            if not rows:
                raise ValueError(f"No rows found in {csv_path}")
            columns = list(rows[0].keys())
            column_types = _infer_column_types(rows, columns)
            _recreate_table(conn, table_name, columns, column_types)
            _insert_rows(conn, table_name, rows, columns)
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            print(f"{table_name}: {count} rows")
        conn.commit()
    print(f"SQLite database created: {DB_PATH}")


def _read_csv(path: Path) -> List[Dict[str, str | None]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


def _infer_column_types(rows: List[Dict[str, str | None]], columns: Iterable[str]) -> Dict[str, str]:
    column_types: Dict[str, str] = {}
    for column in columns:
        values = [row.get(column) for row in rows]
        non_empty = [value.strip() for value in values if value is not None and value.strip() != ""]
        if not non_empty:
            column_types[column] = "TEXT"
        elif all(_is_int_like(value) for value in non_empty):
            column_types[column] = "INTEGER"
        elif all(_is_float_like(value) for value in non_empty):
            column_types[column] = "REAL"
        else:
            column_types[column] = "TEXT"
    return column_types


def _recreate_table(
    conn: sqlite3.Connection,
    table_name: str,
    columns: List[str],
    column_types: Dict[str, str],
) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    column_defs = ", ".join(
        f'"{column}" {column_types[column]}' for column in columns
    )
    conn.execute(f'CREATE TABLE "{table_name}" ({column_defs})')


def _insert_rows(
    conn: sqlite3.Connection,
    table_name: str,
    rows: List[Dict[str, str | None]],
    columns: List[str],
) -> None:
    placeholders = ", ".join("?" for _ in columns)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    sql = f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})'
    values = [tuple(_coerce_value(row.get(column)) for column in columns) for row in rows]
    conn.executemany(sql, values)


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return None
    if _is_int_like(stripped):
        return int(stripped)
    if _is_float_like(stripped):
        return float(stripped)
    return stripped


def _is_int_like(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_float_like(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
