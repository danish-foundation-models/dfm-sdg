from __future__ import annotations

from collections.abc import Callable
from statistics import fmean
from typing import TypedDict

from sdg.commons.model import LLM


class PassedCheck(TypedDict):
    passed: bool


class OkCheck(TypedDict):
    ok: bool


CheckValue = bool | PassedCheck | OkCheck
Row = dict[str, object]


def verify(
    rows: list[Row],
    fn: Callable[[Row], CheckValue],
    *,
    name: str,
) -> list[Row]:
    """Run a deterministic check against each row and attach the result."""

    verified_rows: list[Row] = []
    for row in rows:
        checks = dict(_mapping(row, "checks"))
        checks[name] = fn(row)
        updated = dict(row)
        updated["checks"] = checks
        verified_rows.append(updated)
    return verified_rows


def judge(
    rows: list[Row],
    llm: LLM,
    prompt_fn: Callable[[Row], str | list[dict[str, object]]],
    *,
    name: str,
) -> list[Row]:
    """Attach model-based judgments to each row."""

    judged_rows: list[Row] = []
    for row in rows:
        prompt = prompt_fn(row)
        value = llm.complete(prompt) if isinstance(prompt, str) else llm.chat(prompt)
        scores = dict(_mapping(row, "scores"))
        scores[name] = value
        updated = dict(row)
        updated["scores"] = scores
        judged_rows.append(updated)
    return judged_rows


def aggregate_metrics(rows: list[Row]) -> dict[str, object]:
    check_metrics: dict[str, dict[str, int]] = {}
    numeric_scores: dict[str, list[float]] = {}

    for row in rows:
        for check_name, value in _mapping(row, "checks").items():
            check_summary = check_metrics.setdefault(check_name, {"passed": 0, "failed": 0})
            if _check_passed(value):
                check_summary["passed"] += 1
            else:
                check_summary["failed"] += 1

        for score_name, value in _mapping(row, "scores").items():
            if isinstance(value, (int, float)):
                numeric_scores.setdefault(score_name, []).append(float(value))

    score_metrics: dict[str, dict[str, float | int]] = {}
    for score_name, values in numeric_scores.items():
        score_metrics[score_name] = {
            "mean": fmean(values),
            "count": len(values),
        }

    return {
        "rows": len(rows),
        "checks": check_metrics,
        "scores": score_metrics,
    }


def summarize_failures(rows: list[Row]) -> dict[str, object]:
    rows_with_failures = 0
    check_failures: dict[str, dict[str, int | list[object]]] = {}

    for row in rows:
        row_failed = False
        for check_name, value in _mapping(row, "checks").items():
            if _check_passed(value):
                continue
            row_failed = True
            check_summary = check_failures.setdefault(check_name, {"count": 0, "examples": []})
            check_summary["count"] += 1
            if len(check_summary["examples"]) < 5:
                check_summary["examples"].append(row.get("id"))
        if row_failed:
            rows_with_failures += 1

    return {
        "rows_with_failures": rows_with_failures,
        "checks": check_failures,
    }


def _check_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value

    assert isinstance(value, dict), "check value must be a bool or a status mapping"
    if "passed" in value:
        return bool(value["passed"])
    if "ok" in value:
        return bool(value["ok"])
    raise AssertionError("check status mapping must contain 'passed' or 'ok'")


def _mapping(row: Row, key: str) -> dict[str, object]:
    raw = row.get(key)
    if raw is None:
        return {}

    assert isinstance(raw, dict), f"row {key} must be a mapping"
    return raw
