from __future__ import annotations

from pathlib import Path
from typing import Any

from sdg.commons import store


def load_generated_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    outputs_dir = Path(run_dir) / "outputs"
    rows: list[dict[str, Any]] = []

    memorization_path = outputs_dir / "memorization_rows.jsonl"
    if memorization_path.exists():
        rows.extend(store.read_jsonl(memorization_path))

    grounded_qa_path = outputs_dir / "grounded_qa_rows.jsonl"
    if grounded_qa_path.exists():
        rows.extend(store.read_jsonl(grounded_qa_path))

    return rows


def load_verified_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    outputs_dir = Path(run_dir) / "outputs"
    rows: list[dict[str, Any]] = []

    memorization_path = outputs_dir / "memorization_verified.jsonl"
    if memorization_path.exists():
        rows.extend(store.read_jsonl(memorization_path))

    grounded_qa_path = outputs_dir / "grounded_qa_verified.jsonl"
    if grounded_qa_path.exists():
        rows.extend(store.read_jsonl(grounded_qa_path))

    return rows
