from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_parquet(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, target)
    return target


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_blob(obj: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, bytes):
        target.write_bytes(obj)
        return target

    if isinstance(obj, str):
        target.write_text(obj)
        return target

    with target.open("w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def read_blob(path: str | Path) -> Any:
    source = Path(path)
    if source.suffix == ".json":
        with source.open() as handle:
            return json.load(handle)
    return source.read_bytes()
