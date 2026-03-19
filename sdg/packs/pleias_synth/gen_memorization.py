from __future__ import annotations

import asyncio
import json
import re
from random import Random
from typing import Any

from sdg.commons import Artifact, store
from sdg.commons import publish as common_publish
from sdg.commons.model import LLM, load_clients
from sdg.packs.pleias_synth.llm_json import achat_json
from sdg.packs.pleias_synth.memorization_filters import (
    annotate_filter_result,
    row_filter_reasons,
)
from sdg.packs.pleias_synth.memorization_text import (
    as_list,
    clean_recall_list,
    clean_recall_text,
    normalize_text,
    split_sentences,
    title_variants,
    tokenize,
)
from sdg.packs.pleias_synth.personas import build_query_plans


def generate_memorization(
    cfg: dict[str, Any],
    memory: dict[str, Any],
    outputs_dir,
    *,
    seed: int | None,
) -> tuple[dict[str, Artifact], list[dict[str, Any]]]:
    fact_bundles = sample_fact_bundles(memory, cfg, seed=seed)
    query_plans = build_query_plans(fact_bundles, cfg, seed=seed)
    query_plans = attach_planning_evidence(query_plans, memory["index"], memory["chunks"], cfg)
    models = _load_memorization_models(cfg)
    candidate_rows = asyncio.run(_generate_memorization_async(query_plans, memory, cfg, models))

    rows, rejected_rows = filter_rows(candidate_rows)

    path = store.write_jsonl(rows, outputs_dir / "memorization_rows.jsonl")
    candidate_path = store.write_jsonl(candidate_rows, outputs_dir / "memorization_candidates.jsonl")
    rejected_path = store.write_jsonl(rejected_rows, outputs_dir / "memorization_rejected.jsonl")
    common_publish.write_preview(rows, outputs_dir / "memorization_preview.jsonl", n=50)
    common_publish.write_preview(rejected_rows, outputs_dir / "memorization_rejected_preview.jsonl", n=50)

    return (
        {
            "memorization_rows": Artifact(
                name="memorization_rows",
                path=str(path),
                kind="jsonl",
                meta={"rows": len(rows), "family": "memorization"},
            ),
            "memorization_candidates": Artifact(
                name="memorization_candidates",
                path=str(candidate_path),
                kind="jsonl",
                meta={"rows": len(candidate_rows), "family": "memorization"},
            ),
            "memorization_rejected": Artifact(
                name="memorization_rejected",
                path=str(rejected_path),
                kind="jsonl",
                meta={"rows": len(rejected_rows), "family": "memorization"},
            ),
        },
        rows,
    )


async def _generate_memorization_async(
    query_plans: list[dict[str, Any]],
    memory: dict[str, Any],
    cfg: dict[str, Any],
    models: dict[str, LLM],
) -> list[dict[str, Any]]:
    rows = await backtranslate_queries_async(
        query_plans,
        llm=models["query_teacher"],
        planner=models["task_planner"],
    )
    rows = retrieve_support(rows, memory["index"], memory["chunks"], cfg)
    rows = await generate_backreasoning_async(rows, llm=models["reasoning_teacher"])
    rows = await generate_answers_async(rows, llm=models["answer_teacher"])
    rows = await judge_rows_async(rows, llm=models["judge"])
    return rows


def filter_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    filtered_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in rows:
        reasons = row_filter_reasons(row)
        annotated = annotate_filter_result(row, reasons)
        if reasons:
            rejected_rows.append(annotated)
            continue
        filtered_rows.append(annotated)
    return filtered_rows, rejected_rows


def sample_fact_bundles(
    memory: dict[str, Any],
    cfg: dict[str, Any],
    *,
    seed: int | None,
) -> list[dict[str, Any]]:
    generation_cfg = cfg.get("generation", {})
    memorization_cfg = generation_cfg.get("memorization", {})
    max_rows = int(memorization_cfg.get("max_rows", generation_cfg.get("max_rows_per_family", 200)))
    lead_sentences = int(memorization_cfg.get("lead_sentences", 8))
    max_sentences_per_doc = int(memorization_cfg.get("max_sentences_per_doc", 4))
    min_words = int(memorization_cfg.get("min_sentence_words", 5))
    max_words = int(memorization_cfg.get("max_sentence_words", 90))
    support_sentences = int(memorization_cfg.get("support_sentences", 2))
    structured_facts = int(memorization_cfg.get("structured_facts", 4))

    docs = list(memory["docs"])
    rng = Random(seed if seed is not None else 0)
    rng.shuffle(docs)

    fact_bundles: list[dict[str, Any]] = []
    for doc in docs:
        candidates = _candidate_sentences(
            doc["text"],
            lead_sentences=lead_sentences,
            min_words=min_words,
            max_words=max_words,
        )
        if not candidates:
            continue

        structured_context = _structured_facts(doc, limit=structured_facts)
        for candidate_index, candidate in enumerate(candidates[:max_sentences_per_doc]):
            supporting_claims = [
                row["text"]
                for other_index, row in enumerate(candidates)
                if other_index != candidate_index
            ][:support_sentences]

            fact_bundles.append(
                {
                    "doc": doc,
                    "primary_sentence": candidate["text"],
                    "sentence_index": candidate["index"],
                    "support_sentences": supporting_claims,
                    "structured_facts": structured_context,
                }
            )

            if len(fact_bundles) >= max_rows:
                return fact_bundles

    return fact_bundles


def attach_planning_evidence(
    query_plans: list[dict[str, Any]],
    index: dict[str, Any],
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    memorization_cfg = cfg.get("generation", {}).get("memorization", {})
    top_k = int(memorization_cfg.get("planning_retrieve_top_k", memorization_cfg.get("retrieve_top_k", 5)))
    max_claims = int(memorization_cfg.get("planning_claims", 6))
    chunk_lookup = {chunk["id"]: chunk for chunk in chunks}

    enriched_plans: list[dict[str, Any]] = []
    for plan in query_plans:
        bundle = plan["bundle"]
        seed_parts = [
            bundle["doc"]["title"],
            bundle["primary_sentence"],
            *bundle.get("support_sentences", []),
            *bundle.get("structured_facts", []),
        ]
        query_tokens = set(tokenize(" ".join(seed_parts)))
        scored_chunks: list[tuple[int, dict[str, Any]]] = []
        for chunk_id, entry in index["chunks"].items():
            overlap = len(query_tokens.intersection(entry["tokens"]))
            if overlap == 0:
                continue

            chunk = chunk_lookup[chunk_id]
            scored_chunks.append((overlap, chunk))

        scored_chunks.sort(
            key=lambda item: (
                item[0],
                item[1]["source_id"] == bundle["doc"]["id"],
                item[1]["meta"]["word_count"],
            ),
            reverse=True,
        )

        planning_sources = [
            {
                "chunk_id": chunk["id"],
                "source_id": chunk["source_id"],
                "title": chunk["title"],
                "url": chunk.get("url"),
                "score": score,
                "snippet": chunk["text"],
            }
            for score, chunk in scored_chunks[:top_k]
        ]

        updated = dict(plan)
        updated["planning_sources"] = planning_sources
        updated["planning_claims"] = _planning_claims(bundle, planning_sources, limit=max_claims)
        enriched_plans.append(updated)

    return enriched_plans


async def backtranslate_queries_async(
    query_plans: list[dict[str, Any]],
    *,
    llm: LLM,
    planner: LLM,
) -> list[dict[str, Any]]:
    tasks = [
        _make_row_async(plan, index, llm=llm, planner=planner)
        for index, plan in enumerate(query_plans)
    ]
    built_rows = await asyncio.gather(*tasks)
    return [row for row in built_rows if row is not None]


def retrieve_support(
    rows: list[dict[str, Any]],
    index: dict[str, Any],
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    memorization_cfg = cfg.get("generation", {}).get("memorization", {})
    top_k = int(memorization_cfg.get("retrieve_top_k", 5))
    chunk_lookup = {chunk["id"]: chunk for chunk in chunks}

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        query_tokens = set(tokenize(row["prompt"]))
        scored_chunks: list[tuple[int, dict[str, Any]]] = []
        for chunk_id, entry in index["chunks"].items():
            overlap = len(query_tokens.intersection(entry["tokens"]))
            if overlap == 0:
                continue

            chunk = chunk_lookup[chunk_id]
            scored_chunks.append((overlap, chunk))

        scored_chunks.sort(
            key=lambda item: (
                item[0],
                item[1]["doc_id"] == row["hidden"]["source_id"],
                item[1]["meta"]["word_count"],
            ),
            reverse=True,
        )

        sources = [
            {
                "chunk_id": chunk["id"],
                "source_id": chunk["source_id"],
                "title": chunk["title"],
                "url": chunk.get("url"),
                "score": score,
                "snippet": chunk["text"],
            }
            for score, chunk in scored_chunks[:top_k]
        ]

        updated = dict(row)
        updated["sources"] = sources
        enriched_rows.append(updated)

    return enriched_rows


async def generate_backreasoning_async(rows: list[dict[str, Any]], *, llm: LLM) -> list[dict[str, Any]]:
    traces = await asyncio.gather(*[_llm_backreasoning_async(row, llm) for row in rows])
    enriched_rows: list[dict[str, Any]] = []
    for row, trace in zip(rows, traces, strict=True):
        hidden = dict(row["hidden"])
        hidden["teacher_backreasoning"] = trace

        updated = dict(row)
        updated["hidden"] = hidden
        updated["reasoning"] = _render_reasoning(trace)

        proposed_target = str(trace.get("proposed_target", "")).strip()
        if proposed_target and _target_needs_upgrade(updated):
            updated["target"] = proposed_target

        enriched_rows.append(updated)
    return enriched_rows


async def generate_answers_async(rows: list[dict[str, Any]], *, llm: LLM) -> list[dict[str, Any]]:
    targets = await asyncio.gather(*[_generate_answer_async(row, llm) for row in rows])
    enriched_rows: list[dict[str, Any]] = []
    for row, target in zip(rows, targets, strict=True):
        updated = dict(row)
        updated["target"] = target
        enriched_rows.append(updated)
    return enriched_rows


async def judge_rows_async(rows: list[dict[str, Any]], *, llm: LLM) -> list[dict[str, Any]]:
    judgments = await asyncio.gather(*[_judge_row_async(row, llm) for row in rows])
    judged_rows: list[dict[str, Any]] = []
    for row, judge in zip(rows, judgments, strict=True):
        updated = dict(row)
        scores = dict(row.get("scores") or {})
        scores["judge"] = judge
        updated["scores"] = scores
        judged_rows.append(updated)
    return judged_rows


def format_teacher_bundle(teacher_bundle: dict[str, Any]) -> str:
    parts = [
        f"Article title: {teacher_bundle['article_title']}",
        f"Primary claim: {teacher_bundle['primary_claim']}",
    ]

    supporting_claims = teacher_bundle.get("supporting_claims") or []
    if supporting_claims:
        parts.append(
            "Supporting claims:\n" + "\n".join(f"- {claim}" for claim in supporting_claims)
        )

    structured_context = teacher_bundle.get("structured_context") or []
    if structured_context:
        parts.append(
            "Structured context:\n" + "\n".join(f"- {item}" for item in structured_context)
        )

    retrieved_context = teacher_bundle.get("retrieved_context") or []
    if retrieved_context:
        parts.append(
            "Retrieved context:\n"
            + "\n".join(f"- {item['title']}: {item['snippet']}" for item in retrieved_context[:5])
        )

    retrieved_claims = teacher_bundle.get("retrieved_claims") or []
    if retrieved_claims:
        parts.append(
            "Retrieved claims:\n" + "\n".join(f"- {item}" for item in retrieved_claims[:6])
        )

    return "\n".join(parts)


def _planning_claims(
    bundle: dict[str, Any],
    planning_sources: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    seen = {
        normalize_text(bundle["primary_sentence"]),
        *(normalize_text(sentence) for sentence in bundle.get("support_sentences", [])),
    }
    claims: list[str] = []
    for source in planning_sources:
        for sentence in split_sentences(source.get("snippet", "")):
            normalized = normalize_text(sentence)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            claims.append(sentence)
            if len(claims) >= limit:
                return claims
    return claims


async def _make_row_async(
    plan: dict[str, Any],
    index: int,
    *,
    llm: LLM,
    planner: LLM,
) -> dict[str, Any] | None:
    bundle = plan["bundle"]
    persona = plan["persona"]
    query_angle = plan["query_angle"]
    query_profile = plan["query_profile"]
    assistant_style = plan["assistant_style"]
    doc = bundle["doc"]
    task_plan = await _llm_task_plan_async(plan, planner)

    question = await _llm_question_async(
        doc,
        bundle,
        persona,
        query_angle,
        query_profile,
        task_plan,
        plan.get("planning_sources") or [],
        llm,
    )
    if question is None:
        return None

    teacher_bundle = _build_teacher_bundle(doc, bundle)
    teacher_bundle["retrieved_context"] = list(plan.get("planning_sources") or [])
    teacher_bundle["retrieved_claims"] = list(plan.get("planning_claims") or [])

    target_seed = _default_target(doc, bundle, task_plan)

    return {
        "id": f"memorization-{index:06d}",
        "prompt": question["prompt"],
        "target": target_seed,
        "reasoning": "",
        "hidden": {
            "source_id": doc["id"],
            "source_title": doc["title"],
            "source_url": doc.get("url"),
            "sentence": bundle["primary_sentence"],
            "sentence_index": bundle["sentence_index"],
            "question_type": question["question_type"],
            "generation_mode": question.get("generation_mode", "llm"),
            "persona": persona,
            "query_angle": query_angle,
            "query_profile": query_profile,
            "assistant_style": assistant_style,
            "task_plan": task_plan,
            "teacher_bundle": teacher_bundle,
            "target_seed": target_seed,
        },
        "sources": [],
        "checks": {},
        "scores": {},
        "meta": {
            "family": "memorization",
            "question_type": question["question_type"],
            "dataset": doc.get("meta", {}).get("dataset"),
            "language": doc.get("meta", {}).get("language"),
            "vital_level": doc.get("meta", {}).get("vital_level"),
            "persona_id": persona["persona_id"],
            "persona_source": persona["source"],
            "persona_tags": persona["tags"],
            "query_angle": query_angle,
            "task_type": task_plan["task_type"],
            "user_goal": task_plan["user_goal"],
            "query_profile_id": query_profile["profile_id"],
            "query_profile_source": query_profile["source"],
            "query_profile_tags": query_profile["tags"],
            "assistant_style_id": assistant_style["style_id"],
            "assistant_style_source": assistant_style["source"],
            "assistant_style_tags": assistant_style["tags"],
            "reasoning_style": "teacher_backreasoning_v1",
        },
    }


def _build_teacher_bundle(doc: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_title": doc["title"],
        "primary_claim": bundle["primary_sentence"],
        "supporting_claims": list(bundle.get("support_sentences") or []),
        "structured_context": list(bundle.get("structured_facts") or []),
        "retrieved_context": [],
        "retrieved_claims": [],
    }


def _default_target(doc: dict[str, Any], bundle: dict[str, Any], task_plan: dict[str, Any]) -> str:
    if task_plan.get("task_type") in {"reverse_definition", "source_clue", "identification"}:
        return doc["title"]

    coverage_points = as_list(task_plan.get("coverage_points"))
    if len(coverage_points) >= 2:
        return " ".join(coverage_points[:2])

    definition_target = _definition_target(doc["title"], bundle["primary_sentence"])
    if definition_target:
        return definition_target

    return bundle["primary_sentence"]


def _definition_target(title: str, sentence: str) -> str | None:
    for variant in title_variants(title):
        pattern = re.compile(
            rf"^{re.escape(variant)}(?:\s*\([^)]*\))?\s+(is|was|are|were)\s+(.+?)[.!?]?$",
            flags=re.IGNORECASE,
        )
        match = pattern.match(sentence)
        if not match:
            continue

        predicate = match.group(2).strip(" ,;:")
        if predicate:
            return predicate
    return None


async def _llm_task_plan_async(plan: dict[str, Any], llm: LLM) -> dict[str, Any]:
    messages = _task_plan_messages(plan)
    parsed = await achat_json(llm, messages, temperature=0.4)
    return _parse_task_plan(plan, parsed)


def _task_plan_messages(plan: dict[str, Any]) -> list[dict[str, str]]:
    bundle = plan["bundle"]
    persona = plan["persona"]
    query_profile = plan["query_profile"]
    planning_sources = plan.get("planning_sources") or []

    evidence_block = "\n".join(
        f"- {source['title']}: {source['snippet']}"
        for source in planning_sources[:6]
    )

    return [
        {
            "role": "system",
            "content": (
                "You plan realistic memorization tasks from hidden evidence. "
                "Return strict JSON with keys task_type, user_goal, answer_shape, coverage_points, and query_brief. "
                "task_type should describe the kind of user need, such as overview, planning, comparison, explanation, "
                "clarification, timeline, identification, lookup, or another realistic need supported by the evidence. "
                "Prefer broader multi-point tasks when the hidden evidence can support them. "
                "Do not default to lookup just because one concrete fact is present. "
                "If you have multiple distinct evidence points, the task should usually be overview, explanation, timeline, clarification, or comparison. "
                "Only fall back to narrow single-fact lookup when the evidence is genuinely thin. "
                "coverage_points must be a list of the main things the final answer should cover."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Persona id: {persona['persona_id']}\n"
                f"Persona intent: {persona['intent']}\n"
                f"Persona knowledge level: {persona['knowledge_level']}\n"
                f"Persona tone: {persona['tone']}\n"
                f"Persona answer granularity: {persona['answer_granularity']}\n"
                f"Query profile channel: {query_profile['channel']}\n"
                f"Query profile register: {query_profile['register']}\n"
                f"Query profile shape: {query_profile['query_shape']}\n"
                f"Query profile instructions: {query_profile['instructions']}\n"
                f"Legacy sampling hint: {plan['query_angle']}\n"
                "Treat that hint as weak; ignore it if the evidence supports a broader task.\n\n"
                f"Article title: {bundle['doc']['title']}\n"
                f"Primary claim: {bundle['primary_sentence']}\n"
                "Supporting claims:\n"
                + "\n".join(f"- {claim}" for claim in bundle.get("support_sentences", []))
                + "\n\nStructured context:\n"
                + "\n".join(f"- {item}" for item in bundle.get("structured_facts", []))
                + "\n\nRetrieved evidence:\n"
                + evidence_block
            ),
        },
    ]


def _parse_task_plan(plan: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    task_type = str(parsed.get("task_type", "")).strip() or str(parsed.get("question_type", "")).strip()
    if not task_type:
        task_type = str(plan["query_angle"]).strip() or "lookup"

    user_goal = str(parsed.get("user_goal", "")).strip()
    if not user_goal:
        user_goal = str(plan["persona"]["intent"]).strip() or "understand the topic"

    answer_shape = str(parsed.get("answer_shape", "")).strip() or "direct answer"
    query_brief = str(parsed.get("query_brief", "")).strip() or bundle_query_brief(plan)
    coverage_points = as_list(parsed.get("coverage_points"))
    if not coverage_points:
        coverage_points = _default_coverage_points(plan)

    return {
        "task_type": task_type,
        "user_goal": user_goal,
        "answer_shape": answer_shape,
        "coverage_points": coverage_points,
        "query_brief": query_brief,
    }


def bundle_query_brief(plan: dict[str, Any]) -> str:
    bundle = plan["bundle"]
    doc = bundle["doc"]
    return (
        f"Ask about {doc['title']} in a way that matches the persona and profile while staying grounded "
        "in the primary claim, supporting claims, and retrieved context."
    )


def _default_coverage_points(plan: dict[str, Any]) -> list[str]:
    bundle = plan["bundle"]
    coverage_points = [bundle["primary_sentence"]]
    coverage_points.extend(bundle.get("support_sentences", [])[:2])
    coverage_points.extend(bundle.get("structured_facts", [])[:2])
    coverage_points.extend(plan.get("planning_claims", [])[:2])
    return coverage_points[:4]


async def _llm_question_async(
    doc: dict[str, Any],
    bundle: dict[str, Any],
    persona: dict[str, Any],
    query_angle: str,
    query_profile: dict[str, Any],
    task_plan: dict[str, Any],
    planning_sources: list[dict[str, Any]],
    llm: LLM,
) -> dict[str, str] | None:
    messages = _question_messages(doc, bundle, persona, query_angle, query_profile, task_plan, planning_sources)
    parsed = await achat_json(llm, messages, temperature=0.4)
    return _parse_question(parsed, task_plan)


def _question_messages(
    doc: dict[str, Any],
    bundle: dict[str, Any],
    persona: dict[str, Any],
    query_angle: str,
    query_profile: dict[str, Any],
    task_plan: dict[str, Any],
    planning_sources: list[dict[str, Any]],
) -> list[dict[str, str]]:
    teacher_bundle = _build_teacher_bundle(doc, bundle)
    teacher_bundle["retrieved_context"] = list(planning_sources)
    exemplars = query_profile.get("exemplars") or []
    exemplar_block = ""
    if exemplars:
        exemplar_block = "Query profile exemplars:\n" + "\n".join(f"- {item}" for item in exemplars) + "\n\n"

    return [
        {
            "role": "system",
            "content": (
                "You create memorization questions from teacher-side evidence. "
                "Return strict JSON with keys prompt and question_type. "
                "question_type should match the hidden task plan rather than defaulting to narrow fact lookup. "
                "The prompt must be user-facing and must not reveal the answer verbatim. "
                "The prompt may ask for an overview, explanation, planning help, comparison, timeline, clarification, or a narrower lookup, "
                "depending on what the hidden task plan supports. "
                "Realize the query from the structured profile you are given rather than defaulting to the same polished style every time. "
                "If the profile implies imperfect language, keep it light and realistic."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Bias hint: {query_angle}\n"
                f"Persona id: {persona['persona_id']}\n"
                f"Persona name: {persona['name']}\n"
                f"Intent: {persona['intent']}\n"
                f"Knowledge level: {persona['knowledge_level']}\n"
                f"Tone: {persona['tone']}\n"
                f"Question style: {persona['question_style']}\n"
                f"Answer granularity: {persona['answer_granularity']}\n"
                f"Constraints: {json.dumps(persona['constraints'])}\n\n"
                f"Query profile id: {query_profile['profile_id']}\n"
                f"Query profile name: {query_profile['name']}\n"
                f"Channel: {query_profile['channel']}\n"
                f"Fluency: {query_profile['fluency']}\n"
                f"Register: {query_profile['register']}\n"
                f"Urgency: {query_profile['urgency']}\n"
                f"Query shape: {query_profile['query_shape']}\n"
                f"Noise level: {query_profile['noise_level']}\n"
                f"Instructions: {query_profile['instructions']}\n\n"
                f"Task type: {task_plan['task_type']}\n"
                f"User goal: {task_plan['user_goal']}\n"
                f"Answer shape: {task_plan['answer_shape']}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan["coverage_points"])
                + "\n\n"
                f"Query brief: {task_plan['query_brief']}\n\n"
                f"{exemplar_block}"
                f"{format_teacher_bundle(teacher_bundle)}\n\n"
                "Write one realistic user-facing query that matches the persona, task plan, and query profile."
            ),
        },
    ]


def _parse_question(parsed: dict[str, Any], task_plan: dict[str, Any]) -> dict[str, str] | None:
    prompt = str(parsed.get("prompt", "")).strip()
    question_type = str(parsed.get("question_type", task_plan["task_type"])).strip() or task_plan["task_type"]
    if not prompt:
        return None

    return {
        "prompt": prompt,
        "question_type": question_type,
        "generation_mode": "llm",
    }


async def _llm_backreasoning_async(row: dict[str, Any], llm: LLM) -> dict[str, Any]:
    messages = _backreasoning_messages(row)
    parsed = await achat_json(llm, messages, temperature=0.2)
    return _parse_backreasoning(row, parsed)


def _backreasoning_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    teacher_bundle = row["hidden"]["teacher_bundle"]
    task_plan = row["hidden"].get("task_plan") or {}
    source_snippets = "\n\n".join(
        f"- {source['title']}: {source['snippet']}"
        for source in row.get("sources", [])[:3]
    )
    return [
        {
            "role": "system",
            "content": (
                "You write strongly opinionated recall-style reasoning for synthetic memorization data. "
                "The visible reasoning must read like recalled knowledge, not document analysis. "
                "Never mention texts, passages, snippets, sources, evidence, retrieved support, or teacher bundles. "
                "Return strict JSON with keys key_question, assumption_check, known_facts, reasoning_steps, "
                "caveats, synthesis, and proposed_target. "
                "known_facts, reasoning_steps, and caveats must be arrays of short strings. "
                "Ground every step in the hidden facts you were given, but phrase it as memory recall. "
                "If the query carries an implicit assumption, say it directly in assumption_check."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt: {row['prompt']}\n"
                f"Question type: {row['meta'].get('question_type')}\n"
                f"Persona id: {row['meta'].get('persona_id')}\n"
                f"Query angle: {row['meta'].get('query_angle')}\n\n"
                f"Task type: {task_plan.get('task_type', '')}\n"
                f"User goal: {task_plan.get('user_goal', '')}\n"
                f"Answer shape: {task_plan.get('answer_shape', '')}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan.get("coverage_points", []))
                + "\n\n"
                f"{format_teacher_bundle(teacher_bundle)}\n\n"
                f"Hidden grounding facts:\n{source_snippets}\n\n"
                "Write compact but forceful recall notes that isolate the key question, "
                "surface any needed clarification, work from remembered facts, and end with a direct synthesis."
            ),
        },
    ]


def _parse_backreasoning(row: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "key_question": clean_recall_text(str(parsed.get("key_question", "")).strip()) or _default_key_question(row),
        "assumption_check": clean_recall_text(str(parsed.get("assumption_check", "")).strip()),
        "known_facts": clean_recall_list(parsed.get("known_facts") or parsed.get("evidence_points")),
        "reasoning_steps": clean_recall_list(parsed.get("reasoning_steps")),
        "caveats": clean_recall_list(parsed.get("caveats")),
        "synthesis": clean_recall_text(str(parsed.get("synthesis", "")).strip())
        or _default_synthesis(row, row.get("target", "")),
        "proposed_target": str(parsed.get("proposed_target", "")).strip(),
    }


def _render_reasoning(trace: dict[str, Any]) -> str:
    parts = [f"Key question: {clean_recall_text(trace['key_question'])}"]

    assumption_check = clean_recall_text(str(trace.get("assumption_check", "")).strip())
    if assumption_check:
        parts.append(f"Assumption check: {assumption_check}")

    known_facts = clean_recall_list(trace.get("known_facts") or trace.get("evidence_points"))
    if known_facts:
        parts.append("### 1. Known facts\n" + "\n".join(f"- {item}" for item in known_facts))

    reasoning_steps = clean_recall_list(trace.get("reasoning_steps"))
    if reasoning_steps:
        parts.append("### 2. Resolution\n" + "\n".join(f"- {item}" for item in reasoning_steps))

    caveats = clean_recall_list(trace.get("caveats"))
    if caveats:
        parts.append("### 3. Caveats\n" + "\n".join(f"- {item}" for item in caveats))

    synthesis = clean_recall_text(str(trace.get("synthesis", "")).strip())
    if synthesis:
        parts.append("### 4. Synthesis\n" + synthesis)

    return "\n\n".join(parts)


async def _generate_answer_async(row: dict[str, Any], llm: LLM) -> str:
    messages = _answer_messages(row)
    parsed = await achat_json(llm, messages, temperature=0.0)
    return _parse_target(parsed, row)


def _answer_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    teacher_bundle = row["hidden"]["teacher_bundle"]
    persona = row["hidden"]["persona"]
    query_profile = row["hidden"]["query_profile"]
    assistant_style = row["hidden"]["assistant_style"]
    task_plan = row["hidden"].get("task_plan") or {}
    target_seed = row["hidden"].get("target_seed", "")
    source_snippets = "\n\n".join(
        f"- {source['title']}: {source['snippet']}"
        for source in row.get("sources", [])[:5]
    )

    return [
        {
            "role": "system",
            "content": (
                "You write the final assistant response for synthetic memorization data. "
                "Return strict JSON with the single key target. "
                "The target is the final user-visible answer, not an extractive span. "
                "Write a complete natural-language response that directly answers the prompt. "
                "Use the assistant style specification consistently across rows. "
                "Adapt to the user's goal and knowledge level, but do not change assistant identity from one row to another. "
                "Follow the assistant style's formatting and punctuation constraints. "
                "Use as much hidden information as needed to answer well. "
                "If the question is broad or explanatory, give a brief explanation in full sentences. "
                "If the question is narrow, answer directly in one clear sentence. "
                "Every target must contain at least one complete sentence. "
                "A bare title, name, date, noun phrase, or clipped span is invalid. "
                "If the core answer is a title, person, year, role, or attribute, place it inside a natural sentence. "
                "Do not mention hidden facts, sources, snippets, or retrieval. "
                "Do not imitate the user's errors or slang. Keep the assistant voice steady, clear, and grounded."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt: {row['prompt']}\n\n"
                f"Question type: {row['meta'].get('question_type')}\n"
                f"Task type: {task_plan.get('task_type', '')}\n"
                f"User goal: {task_plan.get('user_goal', '')}\n"
                f"Answer shape: {task_plan.get('answer_shape', '')}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan.get("coverage_points", []))
                + "\n\n"
                f"User persona name: {persona['name']}\n"
                f"User intent: {persona['intent']}\n"
                f"User knowledge level: {persona['knowledge_level']}\n"
                f"User constraints: {json.dumps(persona['constraints'])}\n"
                f"Query profile channel: {query_profile['channel']}\n"
                f"Query profile fluency: {query_profile['fluency']}\n"
                f"Query profile register: {query_profile['register']}\n"
                f"Query profile shape: {query_profile['query_shape']}\n"
                f"Query profile urgency: {query_profile['urgency']}\n"
                f"Query profile noise level: {query_profile['noise_level']}\n\n"
                f"Assistant style id: {assistant_style['style_id']}\n"
                f"Assistant style name: {assistant_style['name']}\n"
                f"Assistant tone: {assistant_style['tone']}\n"
                f"Assistant detail level: {assistant_style['detail_level']}\n"
                f"Assistant structure: {assistant_style['structure']}\n"
                f"Assistant voice: {assistant_style['voice']}\n"
                f"Assistant formatting style: {assistant_style['formatting_style']}\n"
                f"Assistant punctuation style: {assistant_style['punctuation_style']}\n"
                f"Assistant instructions: {assistant_style['instructions']}\n"
                "Assistant exemplars:\n"
                + "\n".join(f"- {item}" for item in assistant_style["exemplars"])
                + "\n\n"
                f"Initial answer seed: {target_seed}\n\n"
                f"Hidden facts:\n{format_teacher_bundle(teacher_bundle)}\n\n"
                f"Recall notes:\n{row['reasoning']}\n\n"
                f"Hidden grounding details:\n{source_snippets}\n\n"
                "Write the final answer exactly as it should be shown to the user. "
                "Return a complete sentence even if the answer could be written as just a title or year."
            ),
        },
    ]


def _parse_target(parsed: dict[str, Any], row: dict[str, Any]) -> str:
    target = str(parsed.get("target", "")).strip()
    if target:
        return " ".join(target.split())
    return _default_answer_from_trace(row)


def _default_answer_from_trace(row: dict[str, Any]) -> str:
    trace = row["hidden"].get("teacher_backreasoning") or {}
    proposed_target = str(trace.get("proposed_target", "")).strip()
    if proposed_target:
        return " ".join(proposed_target.split())
    return " ".join(str(row["target"]).split())


async def _judge_row_async(row: dict[str, Any], llm: LLM) -> dict[str, Any]:
    messages = _judge_messages(row)
    parsed = await achat_json(llm, messages, temperature=0.0)
    return _parse_judge(parsed)


def _judge_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    teacher_bundle = row["hidden"]["teacher_bundle"]
    task_plan = row["hidden"].get("task_plan") or {}
    source_snippets = "\n\n".join(
        f"- {source['title']}: {source['snippet']}"
        for source in row.get("sources", [])[:3]
    )
    return [
        {
            "role": "system",
            "content": (
                "You judge synthetic memorization examples. "
                "Return strict JSON with keys pass, support, leakage, style_distinct, reasoning_quality, and reason. "
                "pass must be true only if the answer is supported, the prompt does not trivially leak it, "
                "the reasoning is grounded, the query is not just a bland default definition question, "
                "and the target is a complete natural-language answer rather than a bare span."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt: {row['prompt']}\n"
                f"Answer: {row['target']}\n"
                f"Reasoning: {row.get('reasoning', '')}\n"
                f"Persona id: {row['meta'].get('persona_id')}\n"
                f"Query angle: {row['meta'].get('query_angle')}\n\n"
                f"Query profile id: {row['meta'].get('query_profile_id')}\n\n"
                f"Task type: {task_plan.get('task_type', '')}\n"
                f"User goal: {task_plan.get('user_goal', '')}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan.get("coverage_points", []))
                + "\n\n"
                f"{format_teacher_bundle(teacher_bundle)}\n\n"
                f"Retrieved support:\n{source_snippets}"
            ),
        },
    ]


def _parse_judge(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": bool(parsed.get("pass", False)),
        "support": bool(parsed.get("support", False)),
        "leakage": bool(parsed.get("leakage", False)),
        "style_distinct": bool(parsed.get("style_distinct", parsed.get("diversity", False))),
        "reasoning_quality": bool(parsed.get("reasoning_quality", False)),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def _default_key_question(row: dict[str, Any]) -> str:
    task_plan = row["hidden"].get("task_plan") or {}
    task_type = task_plan.get("task_type", row["meta"].get("question_type"))
    if task_type in {"reverse_definition", "source_clue", "identification"}:
        return "Which article title is uniquely identified by the clue?"
    if task_type in {"overview", "planning", "comparison", "timeline", "explanation"}:
        return "Which remembered points are necessary to satisfy the user goal?"
    return "Which remembered fact pattern best resolves the prompt?"


def _default_synthesis(row: dict[str, Any], target: str) -> str:
    if target:
        return f"The answer is {target} because that is the remembered fact that resolves the prompt."
    return "The answer has to come from the remembered fact pattern, not from free paraphrase."


def _target_needs_upgrade(row: dict[str, Any]) -> bool:
    current_target = str(row.get("target", "")).strip()
    if not current_target:
        return True
    return current_target == row["hidden"]["sentence"]


def _candidate_sentences(
    text: str,
    *,
    lead_sentences: int,
    min_words: int,
    max_words: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, sentence in enumerate(split_sentences(text)[:lead_sentences]):
        words = len(sentence.split())
        if words < min_words or words > max_words:
            continue
        candidates.append({"index": index, "text": sentence})
    return candidates


def _structured_facts(doc: dict[str, Any], *, limit: int) -> list[str]:
    meta = doc.get("meta") or {}
    wikidata = meta.get("wikidata") or {}
    structured = meta.get("structured_wikipedia") or {}

    facts: list[str] = []

    description = str(wikidata.get("description") or meta.get("wikibase_shortdesc") or "").strip()
    if description:
        facts.append(f"Wikidata description: {description}")

    aliases = [str(alias).strip() for alias in wikidata.get("aliases", []) if str(alias).strip()]
    if aliases:
        facts.append("Aliases: " + ", ".join(aliases[:3]))

    categories = [str(item).strip() for item in structured.get("categories", []) if str(item).strip()]
    if categories:
        facts.append("Categories: " + ", ".join(categories[:3]))

    infobox_templates = [str(item).strip() for item in structured.get("infobox_templates", []) if str(item).strip()]
    if infobox_templates:
        facts.append("Infobox templates: " + ", ".join(infobox_templates[:2]))

    outgoing_links = [str(item).strip() for item in structured.get("outgoing_links", []) if str(item).strip()]
    if outgoing_links:
        facts.append("Related links: " + ", ".join(outgoing_links[:3]))

    return facts[:limit]


def _load_memorization_models(cfg: dict[str, Any]) -> dict[str, LLM]:
    memorization_cfg = cfg.get("generation", {}).get("memorization", {})
    if memorization_cfg.get("use_llm") is False:
        raise ValueError("PleIAs memorization generation requires LLM-backed models")

    model_refs = cfg.get("models", {})
    required_roles = ["query_teacher", "answer_teacher", "judge"]
    missing = [role for role in required_roles if role not in model_refs]
    if missing:
        raise ValueError(f"Missing model roles for memorization LLM path: {', '.join(missing)}")

    requested_roles = dict(model_refs)
    if "reasoning_teacher" not in requested_roles:
        requested_roles["reasoning_teacher"] = requested_roles["answer_teacher"]
    if "task_planner" not in requested_roles:
        requested_roles["task_planner"] = requested_roles["query_teacher"]

    roles = ["task_planner", "query_teacher", "reasoning_teacher", "answer_teacher", "judge"]
    clients = load_clients({role: requested_roles[role] for role in roles})
    return {role: clients[role] for role in roles}
