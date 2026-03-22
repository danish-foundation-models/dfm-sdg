from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sdg.commons import eval as common_eval
from sdg.packs.synth.grounded_qa_filters import (
    row_answer_supported as row_grounded_qa_answer_supported,
    row_citation_supported,
    row_has_citations,
    row_coverage_supported as row_grounded_qa_coverage_supported,
    row_language_quality as row_grounded_qa_language_quality,
    row_reasoning_grounded as row_grounded_qa_reasoning_grounded,
    row_retrieval_grounded as row_grounded_qa_retrieval_grounded,
)
from sdg.packs.synth.memorization_filters import (
    row_answer_supported,
    row_coverage_supported,
    row_language_quality,
    row_reasoning_grounded,
    row_retrieval_grounded,
)
from sdg.packs.synth.memorization_text import normalize_text


def verify_memory_core(
    chunks: list[dict[str, Any]],
    source_table: list[dict[str, Any]],
    index: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_ids = {row["source_id"] for row in source_table}
    indexed_chunk_ids = set(index.get("chunks", {}))
    failures: dict[str, list[str]] = {
        "missing_source_id": [],
        "empty_text": [],
        "missing_index_entry": [],
        "missing_url": [],
        "missing_license": [],
    }

    for chunk in chunks:
        if chunk["source_id"] not in source_ids:
            failures["missing_source_id"].append(chunk["id"])
        if not chunk["text"].strip():
            failures["empty_text"].append(chunk["id"])
        if chunk["id"] not in indexed_chunk_ids:
            failures["missing_index_entry"].append(chunk["id"])
        if not chunk.get("url"):
            failures["missing_url"].append(chunk["id"])
        if not chunk.get("license"):
            failures["missing_license"].append(chunk["id"])

    failure_summary = {
        name: {"count": len(ids), "examples": ids[:5]}
        for name, ids in failures.items()
        if ids
    }

    metrics = aggregate_pack_metrics(chunks, source_table, failure_summary)
    return metrics, failure_summary


def aggregate_pack_metrics(
    chunks: list[dict[str, Any]],
    source_table: list[dict[str, Any]],
    failure_summary: dict[str, Any],
) -> dict[str, Any]:
    avg_chunk_words = 0.0
    if chunks:
        avg_chunk_words = sum(chunk["meta"]["word_count"] for chunk in chunks) / len(chunks)

    wikidata_sources = sum(1 for row in source_table if row.get("meta", {}).get("wikidata_id"))

    return {
        "sources": len(source_table),
        "chunks": len(chunks),
        "avg_chunk_words": avg_chunk_words,
        "sources_with_wikidata": wikidata_sources,
        "sources_with_structured_wikipedia": sum(
            1 for row in source_table if row.get("meta", {}).get("structured_wikipedia")
        ),
        "failed_checks": {name: spec["count"] for name, spec in failure_summary.items()},
    }


def verify_memorization(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verified = common_eval.verify(rows, _has_target, name="has_target")
    verified = common_eval.verify(verified, _has_reasoning, name="has_reasoning")
    verified = common_eval.verify(verified, _retrieval_grounded, name="retrieval_grounded")
    verified = common_eval.verify(verified, _reasoning_grounded, name="reasoning_grounded")
    verified = common_eval.verify(verified, _answer_supported, name="answer_supported")
    verified = common_eval.verify(verified, _coverage_supported, name="coverage_supported")
    verified = common_eval.verify(verified, _answer_not_leaked, name="answer_not_leaked")
    verified = common_eval.verify(verified, _language_quality, name="language_quality")
    verified = common_eval.verify(verified, _judge_pass, name="judge_pass")

    metrics = common_eval.aggregate_metrics(verified)
    metrics["question_types"] = _question_type_counts(verified)
    failure_summary = common_eval.summarize_failures(verified)
    return verified, metrics, failure_summary


def verify_grounded_qa(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    verified = common_eval.verify(rows, _has_target, name="has_target")
    verified = common_eval.verify(verified, _has_reasoning, name="has_reasoning")
    verified = common_eval.verify(verified, _grounded_qa_retrieval_grounded, name="retrieval_grounded")
    verified = common_eval.verify(verified, _grounded_qa_reasoning_grounded, name="reasoning_grounded")
    verified = common_eval.verify(verified, _grounded_qa_answer_supported, name="answer_supported")
    verified = common_eval.verify(verified, _grounded_qa_coverage_supported, name="coverage_supported")
    verified = common_eval.verify(verified, _has_citations, name="has_citations")
    verified = common_eval.verify(verified, _grounded_qa_citation_supported, name="citation_supported")
    verified = common_eval.verify(verified, _answer_not_leaked, name="answer_not_leaked")
    verified = common_eval.verify(verified, _grounded_qa_language_quality, name="language_quality")
    verified = common_eval.verify(verified, _grounded_qa_judge_pass, name="judge_pass")

    metrics = common_eval.aggregate_metrics(verified)
    metrics["question_types"] = _question_type_counts(verified)
    metrics["heuristics"] = {
        "citation_supported": _heuristic_summary(verified, row_citation_supported),
    }
    failure_summary = common_eval.summarize_failures(verified)
    return verified, metrics, failure_summary


def _has_target(row: dict[str, Any]) -> bool:
    return bool(str(row.get("target", "")).strip())


def _has_reasoning(row: dict[str, Any]) -> bool:
    return bool(str(row.get("reasoning", "")).strip())


def _retrieval_grounded(row: dict[str, Any]) -> bool:
    return row_retrieval_grounded(row)


def _reasoning_grounded(row: dict[str, Any]) -> bool:
    return row_reasoning_grounded(row)


def _answer_supported(row: dict[str, Any]) -> bool:
    return row_answer_supported(row)


def _coverage_supported(row: dict[str, Any]) -> bool:
    return row_coverage_supported(row)


def _grounded_qa_retrieval_grounded(row: dict[str, Any]) -> bool:
    return row_grounded_qa_retrieval_grounded(row)


def _grounded_qa_reasoning_grounded(row: dict[str, Any]) -> bool:
    return row_grounded_qa_reasoning_grounded(row)


def _grounded_qa_answer_supported(row: dict[str, Any]) -> bool:
    return row_grounded_qa_answer_supported(row)


def _grounded_qa_coverage_supported(row: dict[str, Any]) -> bool:
    return row_grounded_qa_coverage_supported(row)


def _has_citations(row: dict[str, Any]) -> bool:
    return row_has_citations(row)


def _grounded_qa_citation_supported(row: dict[str, Any]) -> bool:
    judge = row.get("scores", {}).get("judge")
    if not judge:
        return row_citation_supported(row)
    return bool(judge.get("citations", False))



def _answer_not_leaked(row: dict[str, Any]) -> bool:
    prompt = normalize_text(row.get("prompt", ""))
    target = normalize_text(row.get("target", ""))
    return bool(target) and target not in prompt


def _judge_pass(row: dict[str, Any]) -> bool:
    judge = row.get("scores", {}).get("judge")
    if not judge:
        return True
    return bool(judge.get("pass", False))


def _language_quality(row: dict[str, Any]) -> bool:
    return row_language_quality(row)


def _grounded_qa_judge_pass(row: dict[str, Any]) -> bool:
    judge = row.get("scores", {}).get("judge")
    if not judge:
        return True
    return bool(judge.get("pass", False))


def _grounded_qa_language_quality(row: dict[str, Any]) -> bool:
    return row_grounded_qa_language_quality(row)


def _question_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        name = row.get("meta", {}).get("question_type", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _heuristic_summary(
    rows: list[dict[str, Any]],
    fn: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    passed = 0
    failed = 0
    examples: list[str] = []

    for row in rows:
        if fn(row):
            passed += 1
            continue
        failed += 1
        if len(examples) < 5:
            examples.append(str(row.get("id", "")))

    return {
        "passed": passed,
        "failed": failed,
        "examples": examples,
    }
