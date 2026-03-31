from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

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
            append_jsonl_line(handle, row)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True))
    handle.write("\n")
    handle.flush()


def jsonl_prefix(path: str | Path, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    source = Path(path)
    if not source.exists():
        return []

    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(source):
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def jsonl_count(path: str | Path) -> int:
    source = Path(path)
    if not source.exists():
        return 0

    count = 0
    with source.open() as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def jsonl_keys(
    path: str | Path,
    *,
    key_for: Callable[[dict[str, Any]], str | None],
) -> set[str]:
    source = Path(path)
    if not source.exists():
        return set()

    keys: set[str] = set()
    for row in iter_jsonl(source):
        key = key_for(row)
        if key is None:
            continue
        keys.add(key)
    return keys


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
