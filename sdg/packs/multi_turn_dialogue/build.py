from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import concurrency as common_concurrency
from sdg.commons import eval as common_eval
from sdg.commons import publish as common_publish
from sdg.commons.model import LLM, load_clients
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json
from sdg.commons.work_queue import map_async_unordered

METHOD_SOURCES = (
    {
        "kind": "method",
        "name": "APIGen-MT",
        "role": "blueprint-first hidden-state design before simulated dialogue",
        "url": "https://arxiv.org/abs/2504.03601",
    },
    {
        "kind": "method",
        "name": "ConsistentChat",
        "role": "intent trajectory and skeleton-guided dialogue structure",
        "url": "https://arxiv.org/abs/2506.03558",
    },
    {
        "kind": "method",
        "name": "Action-Based Contrastive Self-Training",
        "role": "chosen/rejected action contrast pairs",
        "url": "https://research.google/blog/learning-to-clarify-multi-turn-conversations-with-action-based-contrastive-self-training/",
    },
    {
        "kind": "method",
        "name": "MDS",
        "role": "dialogue-level coverage and structure selection",
        "url": "https://arxiv.org/abs/2604.07892",
    },
)

CONSISTENTCHAT_INTENTS = (
    "Problem Solving Interaction: user works through a practical problem from diagnosis or need to solution",
    "Educational Interaction: user learns a concept through explanation, correction, examples, and checks",
    "Health Consultation Interaction: user seeks health-adjacent guidance with safety boundaries and next steps",
    "Exploratory Interaction: user explores an idea, phenomenon, cause, implication, or cross-domain connection",
    "Entertainment Interaction: user engages in playful, creative, game-like, or recreational conversation",
    "Simulation Interaction: user asks to simulate a scenario, process, role, system, or decision environment",
    "Emotional Support Interaction: user needs supportive reflection, decision support, or coping-oriented guidance",
    "Information Retrieval Interaction: user seeks facts, explanations, evidence extraction, or source-grounded synthesis",
    "Transaction Interaction: user completes a bounded task, form, revision, checklist, message, or decision",
)

CONSISTENTCHAT_INTENT_NAMES = {item.split(":", maxsplit=1)[0] for item in CONSISTENTCHAT_INTENTS}

CONSISTENTCHAT_SUBTASKS = {
    "Problem Solving Interaction": [
        "planning",
        "troubleshooting",
        "practical constraint solving",
        "optimization under tradeoffs",
        "late constraint recovery",
    ],
    "Educational Interaction": [
        "concept tutoring",
        "misconception correction",
        "worked example",
        "intuition building",
        "knowledge check",
    ],
    "Health Consultation Interaction": [
        "general wellness planning",
        "symptom triage boundary",
        "nutrition guidance",
        "medication-safety next steps",
        "clinician escalation",
    ],
    "Exploratory Interaction": [
        "causal exploration",
        "timeline expansion",
        "cross-domain comparison",
        "hypothesis discussion",
        "implication mapping",
    ],
    "Entertainment Interaction": [
        "game setup",
        "creative play",
        "interactive quiz",
        "story continuation",
        "light roleplay",
    ],
    "Simulation Interaction": [
        "role simulation",
        "policy scenario walkthrough",
        "resource-allocation scenario",
        "decision rehearsal",
        "process simulation",
    ],
    "Emotional Support Interaction": [
        "decision support",
        "confidence building",
        "stress-aware planning",
        "perspective taking",
        "low-pressure next step",
    ],
    "Information Retrieval Interaction": [
        "source-grounded explanation",
        "document extraction",
        "comparison rubric",
        "policy interpretation",
        "evidence summary",
    ],
    "Transaction Interaction": [
        "form assistance",
        "artifact revision",
        "checklist creation",
        "message drafting",
        "bounded workflow completion",
    ],
}

WORK_SESSION_GUIDANCE = {
    "Problem Solving Interaction": {
        "work_session_type": "practical_solution_session",
        "deliverable_family": "plan, checklist, diagnosis, recommendation, or constraint-aware next-step sequence",
        "visible_input_payload": "requirements, constraints, numbers, prior attempt, symptoms of the problem, or messy notes",
        "progression": "move from visible inputs to a first useful solution, then revise after a concrete constraint or correction",
        "completion_bar": "the final assistant turn should leave the user with a usable action plan or solved subproblem",
    },
    "Educational Interaction": {
        "work_session_type": "learning_work_session",
        "deliverable_family": "corrected explanation, worked example, rewritten answer, study plan, or short practice item",
        "visible_input_payload": "the user's attempted answer, confusion point, assignment text, rubric, or example problem",
        "progression": "diagnose the misunderstanding, teach from the user's material, then produce a corrected version or check",
        "completion_bar": "the final assistant turn should improve the user's answer or understanding in a concrete way",
    },
    "Health Consultation Interaction": {
        "work_session_type": "safety_bounded_planning_session",
        "deliverable_family": "general plan, symptom timeline, escalation checklist, questions for a clinician, or safer routine",
        "visible_input_payload": "visible symptoms, timeline, goals, limitations, medication context, or current routine",
        "progression": "separate general guidance from medical uncertainty, refine after risk details, and produce safe next steps",
        "completion_bar": "the final assistant turn should give bounded, safety-aware next steps without diagnosis overreach",
    },
    "Exploratory Interaction": {
        "work_session_type": "structured_exploration_session",
        "deliverable_family": "structured memo, hypothesis map, pros/cons frame, timeline, or implications summary",
        "visible_input_payload": "question, source excerpt, observed pattern, thesis, or competing interpretations",
        "progression": "turn broad curiosity into a structured artifact, then refine scope or uncertainty after follow-up",
        "completion_bar": "the final assistant turn should organize the exploration into a usable frame, not just continue discussion",
    },
    "Entertainment Interaction": {
        "work_session_type": "creative_output_session",
        "deliverable_family": "quiz, story beat, game rules, prompt, playful script, or creative revision",
        "visible_input_payload": "theme, audience, draft, constraints, examples, or preferred tone",
        "progression": "create a playable or usable draft, adapt it after preference feedback, and finish with the revised artifact",
        "completion_bar": "the final assistant turn should contain something the user can play, read, use, or continue from",
    },
    "Simulation Interaction": {
        "work_session_type": "rehearsal_and_feedback_session",
        "deliverable_family": "roleplay exchange, decision rehearsal, feedback notes, script, or scenario walkthrough",
        "visible_input_payload": "role, situation, goals, constraints, user's attempted response, or evaluation criteria",
        "progression": "simulate a realistic step, evaluate or adapt based on the user's response, then produce feedback or a script",
        "completion_bar": "the final assistant turn should give concrete rehearsal value: feedback, next line, or improved approach",
    },
    "Emotional Support Interaction": {
        "work_session_type": "supportive_next_step_session",
        "deliverable_family": "decision frame, grounding plan, message draft, low-pressure checklist, or next-step script",
        "visible_input_payload": "situation, emotional constraint, decision pressure, relationship context, or attempted coping step",
        "progression": "validate briefly, turn the situation into one manageable artifact, then refine based on the user's boundary",
        "completion_bar": "the final assistant turn should be supportive but also leave a concrete next step or wording the user can use",
    },
    "Information Retrieval Interaction": {
        "work_session_type": "grounded_synthesis_session",
        "deliverable_family": "summary, extraction table, comparison, evidence-grounded explanation, or source-bounded answer",
        "visible_input_payload": "source excerpt, options, criteria, document text, public question, or facts the user has supplied",
        "progression": "extract or compare from visible material, mark uncertainty, then revise after source or criteria changes",
        "completion_bar": "the final assistant turn should synthesize the visible information into a directly usable answer",
    },
    "Transaction Interaction": {
        "work_session_type": "bounded_artifact_completion_session",
        "deliverable_family": "email, form checklist, revised text, message, table, checklist, or completed workflow draft",
        "visible_input_payload": "draft, form fields, policy excerpt, notes, recipient, tone, deadline, or requirements",
        "progression": "produce or transform the artifact early, then revise after specific user feedback or missing details",
        "completion_bar": "the final assistant turn should contain the completed artifact or a clearly bounded final version",
    },
}

LANGUAGE_NAMES = {
    "en": "English",
    "da": "Danish",
}

SALIENT_TERM_STOPWORDS = {
    "about",
    "after",
    "because",
    "before",
    "could",
    "first",
    "help",
    "there",
    "these",
    "think",
    "under",
    "which",
    "would",
    "again",
    "eller",
    "efter",
    "fordi",
    "gerne",
    "hjælp",
    "hvordan",
    "ikke",
    "noget",
    "omkring",
    "selvom",
    "skulle",
    "ville",
}

DEFAULT_SOURCE_METHODS = [
    "APIGen-MT",
    "ConsistentChat",
    "Action-Based Contrastive Self-Training",
    "MDS",
]

SEED_EXPANSION_VARIANTS = (
    {
        "id": "sparse_opening",
        "target_user_turns": 3,
        "first_turn_shape": "brief opening with the main goal plus at least one concrete user-owned fact, artifact pointer, constraint, role context, prior attempt, named option, or visible source snippet",
        "opener_payload_rule": "may be one sentence, but must include one concrete payload item beyond the broad topic",
        "user_context": "ordinary user context with a small but useful amount of task material",
        "information_flow": "assistant uses the visible payload for bounded partial work, user supplies additional core facts, user later adds one new constraint",
        "work_session_focus": "do not use a zero-information 'can you help with this' opening; even the shortest opening should let the assistant start useful work",
    },
    {
        "id": "moderately_informative_opening",
        "target_user_turns": 4,
        "first_turn_shape": "opening includes the goal plus two or three user-owned facts",
        "opener_payload_rule": "should be at least two sentences or a sentence plus a compact list of user-owned facts",
        "user_context": "the user has already thought about the task but still needs help structuring it",
        "information_flow": "assistant checks a remaining ambiguity, user narrows the request, user later asks for a format or priority change",
        "work_session_focus": "assistant should create a useful first pass after one targeted clarification or immediately if enough is visible",
    },
    {
        "id": "concrete_numbers_opening",
        "target_user_turns": 5,
        "first_turn_shape": "opening includes concrete quantities, dates, deadlines, amounts, or constraints when the domain supports them",
        "opener_payload_rule": "must include at least two visible numeric/date/quantity/budget/limit details when the domain supports numbers",
        "user_context": "the user wants practical help with visible numbers or constraints",
        "information_flow": "assistant works from visible values, user later reveals an exception, missing item, or changed assumption",
        "work_session_focus": "conversation should make visible calculations, tradeoffs, schedule details, or constraint satisfaction useful",
    },
    {
        "id": "visible_artifact_opening",
        "target_user_turns": 4,
        "first_turn_shape": "opening includes a visible artifact such as an excerpt, draft, error message, form text, policy snippet, or notes when the archetype supports it",
        "opener_payload_rule": "should include a short pasted artifact, draft, notes list, error text, source excerpt, or quoted snippet in USER 1",
        "user_context": "the user brings their own source material rather than asking from scratch",
        "information_flow": "assistant grounds in the visible artifact, user later adds omitted context or asks for a targeted revision",
        "work_session_focus": "assistant should transform, critique, extract from, or revise the visible artifact rather than discuss the topic generally",
    },
    {
        "id": "prior_attempt_opening",
        "target_user_turns": 3,
        "first_turn_shape": "opening includes the user's attempted solution, partial plan, or current understanding",
        "opener_payload_rule": "must describe what the user tried and what failed, confused them, or felt unsatisfactory",
        "user_context": "the user has tried something and wants correction without losing ownership of the task",
        "information_flow": "assistant diagnoses the attempt, user clarifies what they meant, user later asks for a final corrected version",
        "work_session_focus": "assistant should diagnose the attempt and produce a corrected or improved version by the final turn",
    },
    {
        "id": "comparison_or_choice_opening",
        "target_user_turns": 5,
        "first_turn_shape": "opening frames a choice between options, approaches, interpretations, or next steps",
        "opener_payload_rule": "must name or describe at least two options, or state one option plus explicit decision criteria and missing details",
        "user_context": "the user needs help weighing tradeoffs under incomplete information",
        "information_flow": "assistant asks for decision criteria, user supplies priorities, user later changes or adds one deciding criterion",
        "work_session_focus": "assistant should build a comparison or decision artifact that changes after the new criterion appears",
    },
    {
        "id": "stakeholder_context_opening",
        "target_user_turns": 4,
        "first_turn_shape": "opening includes audience, stakeholder, tone, risk, or situational context",
        "opener_payload_rule": "must identify the audience/recipient/stakeholder and at least one tone, risk, or relationship constraint",
        "user_context": "the user is adapting the answer for someone else or for a specific setting",
        "information_flow": "assistant aligns to the context, user later introduces a stakeholder constraint or tone adjustment",
        "work_session_focus": "assistant should adapt the work product for the audience, tone, or stakeholder boundary",
    },
    {
        "id": "careful_boundary_opening",
        "target_user_turns": 5,
        "first_turn_shape": "opening asks for help while acknowledging that some details may be missing, current, source-specific, or uncertain",
        "opener_payload_rule": "must include at least one known fact and one explicit uncertainty, missing source, or assumption boundary",
        "user_context": "the user knows the answer may depend on information they have not supplied yet",
        "information_flow": "assistant separates known facts from assumptions, user supplies missing evidence, user later asks for a bounded final answer",
        "work_session_focus": "assistant should produce a bounded artifact that clearly separates visible facts, assumptions, and unknowns",
    },
)

SCENARIO_INSTANCE_VARIANTS = (
    {
        "id": "artifact_payload",
        "variation_axis": "put concrete task material near the front of the conversation",
        "first_turn_bias": "include a short draft, notes list, quote, traceback, excerpt, or rough artifact when the domain supports it",
    },
    {
        "id": "constraint_payload",
        "variation_axis": "change the visible constraints and priorities while preserving the same archetype",
        "first_turn_bias": "include explicit constraints such as time, budget, audience, risk tolerance, dietary need, deadline, or success bar",
    },
    {
        "id": "prior_attempt_payload",
        "variation_axis": "make the user arrive with an attempted answer, plan, fix, draft, or interpretation",
        "first_turn_bias": "include what the user tried and the concrete symptom, weakness, or uncertainty that remains",
    },
    {
        "id": "named_options_payload",
        "variation_axis": "vary names, options, entities, or alternative paths instead of reusing the seed's default example",
        "first_turn_bias": "include named options, candidate choices, roles, or entities even if some details remain missing",
    },
)

PACK_DIR = Path(__file__).resolve().parent

def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="multi_turn_dialogue",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    verified_rows = common_eval.verify(rows, _messages_alternate, name="messages_alternate")
    verified_rows = common_eval.verify(verified_rows, _no_role_label_leak, name="no_role_label_leak")
    verified_rows = common_eval.verify(verified_rows, _skeleton_matches_messages, name="skeleton_matches_messages")
    verified_rows = common_eval.verify(verified_rows, _evidence_boundary_respected, name="evidence_boundary_respected")
    verified_rows = common_eval.verify(verified_rows, _visible_artifact_claims_grounded, name="visible_artifact_claims_grounded")
    verified_rows = common_eval.verify(verified_rows, _reaction_references_grounded, name="reaction_references_grounded")
    verified_rows = common_eval.verify(verified_rows, _assistant_format_restrained, name="assistant_format_restrained")
    verified_rows = common_eval.verify(verified_rows, _selection_accepted, name="selection_accepted")
    failures = [row for row in verified_rows if _row_failed(row)]

    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verified_rows, outputs_dir / "verified.jsonl")
    store.write_jsonl(failures, outputs_dir / "failures.jsonl")

    metrics = common_eval.aggregate_metrics(verified_rows)
    failure_summary = common_eval.summarize_failures(verified_rows)
    dataset_checks = _dataset_checks(verified_rows)
    write_json(metrics, outputs_dir / "metrics.json")
    write_json(failure_summary, outputs_dir / "failure_summary.json")
    write_json(dataset_checks, outputs_dir / "dataset_checks.json")
    common_publish.write_preview(verified_rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "run_id": result.run_id,
        "verified_rows": len(verified_rows),
        "failed_rows": len(failures),
        "metrics": metrics,
        "failure_summary": failure_summary,
        "dataset_checks": dataset_checks,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    selection_report_path = outputs_dir / "selection_report.json"
    preference_pairs_path = outputs_dir / "preference_pairs.jsonl"

    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)
    selection_report = read_json(selection_report_path)
    preference_pairs = store.read_jsonl(preference_pairs_path) if preference_pairs_path.exists() else []

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "preference_pairs": len(preference_pairs),
        "artifacts": sorted(result.artifacts),
        "families": dict(Counter(str(row["meta"]["family"]) for row in rows)),
        "intent_trajectories": dict(Counter(str(row["meta"]["intent_trajectory"]) for row in rows)),
        "subtask_trajectories": dict(Counter(str(row["meta"].get("subtask_trajectory", "")) for row in rows)),
        "work_session_types": dict(Counter(str(row["meta"].get("work_session_type", "")) for row in rows)),
        "mds_bins": dict(Counter(str(_hidden(row)["selection"]["mds"]["global_bin"]) for row in rows)),
        "selection_report": selection_report,
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    export_rows = [_strip_hidden(row) for row in rows]
    failures = [row for row in export_rows if _row_failed(row)]
    generation = cfg["generation"]
    train_rows, eval_rows = _split_rows(export_rows, generation["train_fraction"])

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(export_rows, target_dir / "sample_preview.jsonl", n=20)

    published_artifacts = [
        "train.parquet",
        "eval.parquet",
        "failures.parquet",
        "sample_preview.jsonl",
        "manifest.json",
        "metrics.json",
        "failure_summary.json",
        "dataset_checks.json",
        "selection_report.json",
        "report.json",
    ]

    preference_pairs = _load_preference_pairs(result)
    if preference_pairs:
        store.write_parquet([_strip_hidden(row) for row in preference_pairs], target_dir / "preference_pairs.parquet")
        published_artifacts.append("preference_pairs.parquet")

    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = _load_or_compute(outputs_dir / "metrics.json", common_eval.aggregate_metrics(rows))
    failure_summary = _load_or_compute(outputs_dir / "failure_summary.json", common_eval.summarize_failures(rows))
    dataset_checks = _load_or_compute(outputs_dir / "dataset_checks.json", _dataset_checks(rows))
    selection_report = read_json(outputs_dir / "selection_report.json")

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
    write_json(dataset_checks, target_dir / "dataset_checks.json")
    write_json(selection_report, target_dir / "selection_report.json")
    common_publish.write_report(metrics, failure_summary, target_dir / "report.json")
    common_publish.write_manifest(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "source_run_dir": result.run_dir,
            "source_artifacts": sorted(result.artifacts),
            "published_artifacts": published_artifacts,
        },
        target_dir / "manifest.json",
    )

    return {
        "run_id": result.run_id,
        "out_dir": str(target_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "failure_rows": len(failures),
        "preference_pairs": len(preference_pairs),
    }


def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    model_roles = _load_model_roles(cfg)
    rows, candidates, rejected_rows, preference_pairs, selection_report = asyncio.run(
        _make_rows(
            cfg,
            seed,
            model_roles,
            outputs_dir=outputs_dir,
        )
    )
    blueprints = [_blueprint_artifact(row) for row in rows]
    intent_models = [_intent_model_artifact(row) for row in rows]
    skeletons = [_skeleton_artifact(row) for row in rows]

    blueprints_path = store.write_jsonl(blueprints, outputs_dir / "blueprints.jsonl")
    intent_models_path = store.write_jsonl(intent_models, outputs_dir / "intent_models.jsonl")
    skeletons_path = store.write_jsonl(skeletons, outputs_dir / "skeletons.jsonl")
    candidates_path = outputs_dir / "candidates.jsonl"
    if not candidates_path.exists():
        candidates_path = store.write_jsonl(candidates, candidates_path)
    rejected_path = store.write_jsonl(rejected_rows, outputs_dir / "rejected_candidates.jsonl")
    dataset_path = store.write_jsonl(rows, outputs_dir / "dataset.jsonl")
    preference_pairs_path = store.write_jsonl(preference_pairs, outputs_dir / "preference_pairs.jsonl")
    selection_report_path = write_json(selection_report, outputs_dir / "selection_report.json")
    common_publish.write_preview(rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "blueprints": Artifact(
            name="blueprints",
            path=str(blueprints_path),
            kind="jsonl",
            meta={"rows": len(blueprints), "family": "multi_turn_dialogue"},
        ),
        "skeletons": Artifact(
            name="skeletons",
            path=str(skeletons_path),
            kind="jsonl",
            meta={"rows": len(skeletons), "family": "multi_turn_dialogue"},
        ),
        "intent_models": Artifact(
            name="intent_models",
            path=str(intent_models_path),
            kind="jsonl",
            meta={"rows": len(intent_models), "family": "multi_turn_dialogue"},
        ),
        "candidates": Artifact(
            name="candidates",
            path=str(candidates_path),
            kind="jsonl",
            meta={"rows": len(candidates), "family": "multi_turn_dialogue"},
        ),
        "rejected_candidates": Artifact(
            name="rejected_candidates",
            path=str(rejected_path),
            kind="jsonl",
            meta={"rows": len(rejected_rows), "family": "multi_turn_dialogue"},
        ),
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={"rows": len(rows), "family": "multi_turn_dialogue"},
        ),
        "preference_pairs": Artifact(
            name="preference_pairs",
            path=str(preference_pairs_path),
            kind="jsonl",
            meta={"rows": len(preference_pairs), "family": "multi_turn_dialogue"},
        ),
        "selection_report": Artifact(
            name="selection_report",
            path=str(selection_report_path),
            kind="json",
            meta={"accepted": len(rows), "preference_pairs": len(preference_pairs)},
        ),
    }


async def _make_rows(
    cfg: dict[str, Any],
    seed: int | None,
    model_roles: dict[str, LLM],
    *,
    outputs_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    generation = cfg["generation"]
    count = int(generation["count"])
    candidate_multiplier = int(generation.get("candidate_multiplier", 1))
    candidate_count = max(count * candidate_multiplier, count)
    families = set(generation.get("families") or [])
    seed_specs = [seed_spec for seed_spec in _seed_specs(generation) if not families or seed_spec["family"] in families]
    assert seed_specs, "multi_turn_dialogue requires at least one enabled seed family"

    if generation.get("shuffle_seeds", False):
        rng = Random(seed if seed is not None else 0)
        rng.shuffle(seed_specs)

    candidates = await _generate_candidates_to_disk(
        candidate_count=candidate_count,
        seed_specs=seed_specs,
        generation=generation,
        model_roles=model_roles,
        outputs_dir=outputs_dir,
    )

    rows, rejected_rows, selection_report = _select_rows_mds(candidates, target_count=count)

    preference_pairs: list[dict[str, Any]] = []
    if generation.get("include_preference_pairs", True):
        for row in rows:
            preference_pairs.append(_generate_preference_pair(model_roles["preference_writer"], row, generation))

    selection_report["preference_pairs"] = len(preference_pairs)
    return rows, candidates, rejected_rows, preference_pairs, selection_report


async def _generate_candidates_to_disk(
    *,
    candidate_count: int,
    seed_specs: list[dict[str, Any]],
    generation: dict[str, Any],
    model_roles: dict[str, LLM],
    outputs_dir: Path,
) -> list[dict[str, Any]]:
    candidate_path = outputs_dir / "candidates.jsonl"
    existing_ids = store.jsonl_keys(candidate_path, key_for=_row_id)
    pending = [
        {
            "index": index,
            "row_id": _candidate_id(index),
            "seed_spec": _candidate_seed_spec(seed_specs[index % len(seed_specs)], candidate_index=index),
        }
        for index in range(candidate_count)
        if _candidate_id(index) not in existing_ids
    ]

    concurrency = _candidate_concurrency(generation, model_roles)
    mode = "a" if candidate_path.exists() else "w"
    with candidate_path.open(mode) as handle:
        async for row in map_async_unordered(
            pending,
            lambda _index, item: _generate_candidate_async(item, generation, model_roles),
            concurrency=concurrency,
            total=len(pending),
        ):
            store.append_jsonl_line(handle, row)

    candidates = store.read_jsonl(candidate_path)
    candidates.sort(key=lambda row: str(row["id"]))
    return candidates


async def _generate_candidate_async(
    item: dict[str, Any],
    generation: dict[str, Any],
    model_roles: dict[str, LLM],
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _generate_candidate,
        str(item["row_id"]),
        item["seed_spec"],
        generation,
        model_roles,
    )


def _candidate_concurrency(generation: dict[str, Any], model_roles: dict[str, LLM]) -> int:
    configured = generation.get("candidate_concurrency")
    if configured is not None:
        assert isinstance(configured, int) and configured > 0, "candidate_concurrency must be a positive integer"
        return configured
    return common_concurrency.effective_concurrency(model_roles.values())


def _candidate_id(index: int) -> str:
    return f"multi-turn-dialogue-candidate-{index:05d}"


def _candidate_seed_spec(seed_spec: dict[str, Any], *, candidate_index: int) -> dict[str, Any]:
    candidate = dict(seed_spec)
    candidate["scenario_instance"] = _scenario_instance(seed_spec, candidate_index=candidate_index)
    return candidate


def _scenario_instance(seed_spec: dict[str, Any], *, candidate_index: int) -> dict[str, Any]:
    instance = dict(SCENARIO_INSTANCE_VARIANTS[candidate_index % len(SCENARIO_INSTANCE_VARIANTS)])
    instance["candidate_index"] = candidate_index
    instance["seed_id"] = seed_spec["seed_id"]
    instance["instruction"] = (
        "Instantiate this archetype as a fresh scenario. Preserve the domain, evidence boundary, and success criteria, "
        "but vary visible payload details, names, quantities, constraints, examples, and user wording where possible. "
        "Do not simply restate the handwritten seed or previous common example for this archetype."
    )
    return instance


def _row_id(row: dict[str, Any]) -> str | None:
    row_id = row.get("id")
    if row_id is None:
        return None
    return str(row_id)


def _load_model_roles(cfg: dict[str, Any]) -> dict[str, LLM]:
    model_refs = cfg.get("models")
    assert isinstance(model_refs, dict), "multi_turn_dialogue requires a models mapping"

    required_roles = {
        "blueprint_writer",
        "intent_modeler",
        "skeleton_writer",
        "user_simulator",
        "assistant_teacher",
        "reviewer",
        "preference_writer",
    }
    missing = sorted(required_roles - set(model_refs))
    assert not missing, f"multi_turn_dialogue missing model roles: {missing}"

    clients = load_clients(model_refs)
    roles: dict[str, LLM] = {}
    for role in required_roles:
        client = clients[role]
        assert isinstance(client, LLM), f"models.{role} must resolve to an LLM"
        roles[role] = client
    return roles


def _generate_candidate(
    row_id: str,
    seed_spec: dict[str, Any],
    generation: dict[str, Any],
    model_roles: dict[str, LLM],
) -> dict[str, Any]:
    seed_spec = _generate_scenario_instance(model_roles["blueprint_writer"], seed_spec, generation)
    blueprint = _generate_blueprint(model_roles["blueprint_writer"], seed_spec, generation)
    intent_model = _generate_intent_model(model_roles["intent_modeler"], seed_spec, blueprint, generation)
    skeleton = _generate_skeleton(model_roles["skeleton_writer"], seed_spec, blueprint, intent_model, generation)
    user_turns = _generate_user_turns(model_roles["user_simulator"], seed_spec, blueprint, intent_model, skeleton, generation)
    assistant_turns = _generate_assistant_turns(
        model_roles["assistant_teacher"],
        seed_spec,
        skeleton,
        user_turns,
        generation,
    )
    messages = _interleave_messages(skeleton, user_turns, assistant_turns)
    messages = _polish_messages(model_roles["reviewer"], seed_spec, messages, generation)
    review = _generate_review(model_roles["reviewer"], seed_spec, blueprint, intent_model, skeleton, messages, generation)

    for _ in range(_repair_attempts(generation)):
        if not _needs_repair(review, messages, seed_spec):
            break
        messages = _repair_messages(
            model_roles["assistant_teacher"],
            seed_spec,
            skeleton,
            messages,
            review,
            generation,
        )
        messages = _polish_messages(model_roles["reviewer"], seed_spec, messages, generation)
        review = _generate_review(model_roles["reviewer"], seed_spec, blueprint, intent_model, skeleton, messages, generation)

    return _row_from_generated_parts(
        row_id=row_id,
        seed_spec=seed_spec,
        blueprint=blueprint,
        intent_model=intent_model,
        skeleton=skeleton,
        messages=messages,
        review=review,
    )


def _generate_scenario_instance(
    llm: LLM,
    seed_spec: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    if generation.get("generate_scenario_instances", True) is False:
        return seed_spec

    artifact = _chat_text(
        llm,
        _scenario_instance_messages(seed_spec),
        temperature=float(generation.get("scenario_temperature", 0.9)),
    )
    assert artifact, "scenario instance generation returned an empty artifact"

    candidate = dict(seed_spec)
    instance = dict(seed_spec.get("scenario_instance", {}))
    instance["artifact"] = artifact
    candidate["scenario_instance"] = instance
    return candidate


def _repair_attempts(generation: dict[str, Any]) -> int:
    if not generation.get("repair_failed_reviews", True):
        return 0
    return max(int(generation.get("repair_attempts", 2)), 0)


def _generate_blueprint(
    llm: LLM,
    seed_spec: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    artifact = _chat_text(
        llm,
        _blueprint_messages(seed_spec),
        temperature=float(generation.get("blueprint_temperature", 0.5)),
    )
    assert artifact, "blueprint_writer returned an empty blueprint"
    return {
        "domain": seed_spec["domain"],
        "family": seed_spec["family"],
        "seed_intent": seed_spec["seed_intent"],
        "artifact": artifact,
        "success_criteria": list(seed_spec["success_criteria"]),
        "work_session": dict(seed_spec["work_session"]),
        "dialogue_language": seed_spec["dialogue_language"],
        "latent_language": seed_spec["latent_language"],
        "evidence_boundary": _evidence_boundary(seed_spec),
        "method_pattern": "blueprint_first_hidden_state",
    }


def _generate_intent_model(
    llm: LLM,
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    text = _chat_text(
        llm,
        _intent_model_messages(seed_spec, blueprint),
        temperature=float(generation.get("intent_temperature", 0.3)),
    )
    return _parse_intent_model(text, seed_spec)


def _generate_skeleton(
    llm: LLM,
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    generation: dict[str, Any],
) -> list[dict[str, str]]:
    text = _chat_text(
        llm,
        _skeleton_messages(seed_spec, blueprint, intent_model),
        temperature=float(generation.get("skeleton_temperature", 0.4)),
    )
    skeleton = _parse_skeleton(text)
    expected_roles = _expected_skeleton_roles(seed_spec)
    assert len(skeleton) == len(expected_roles), f"skeleton_writer must produce exactly {len(expected_roles)} skeleton steps"
    assert [step["role"] for step in skeleton] == expected_roles
    return skeleton


def _generate_user_turns(
    llm: LLM,
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
    generation: dict[str, Any],
) -> list[str]:
    text = _chat_text(
        llm,
        _user_simulator_messages(seed_spec, blueprint, intent_model, skeleton),
        temperature=float(generation.get("user_temperature", 0.7)),
    )
    turns = _parse_numbered_blocks(text, label="USER")
    assert len(turns) == _role_count(skeleton, "user"), "user_simulator returned the wrong number of user turns"
    return turns


def _generate_assistant_turns(
    llm: LLM,
    seed_spec: dict[str, Any],
    skeleton: list[dict[str, str]],
    user_turns: list[str],
    generation: dict[str, Any],
    *,
    review: dict[str, Any] | None = None,
) -> list[str]:
    expected_turns = _role_count(skeleton, "assistant")
    assistant_turns: list[str] = []
    for assistant_index in range(expected_turns):
        text = _chat_text(
            llm,
            _assistant_turn_messages(
                seed_spec,
                skeleton,
                user_turns,
                assistant_turns,
                assistant_index=assistant_index,
                review=review,
            ),
            temperature=float(generation.get("assistant_temperature", 0.5)),
        )
        turns = _parse_numbered_blocks(text, label="ASSISTANT")
        if len(turns) != 1:
            text = _chat_text(
                llm,
                _reformat_numbered_blocks_messages(text, label="ASSISTANT", count=1),
                temperature=0.0,
            )
            turns = _parse_numbered_blocks(text, label="ASSISTANT")
        assert len(turns) == 1, (
            f"assistant_teacher returned {len(turns)} turns for assistant {assistant_index + 1}; "
            f"expected 1. Raw response: {text}"
        )
        assistant_turns.append(turns[0])
    return assistant_turns


def _polish_messages(
    llm: LLM,
    seed_spec: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> list[dict[str, str]]:
    if not _should_surface_polish(seed_spec, generation):
        return messages

    text = _chat_text(
        llm,
        _surface_polish_messages(seed_spec, messages),
        temperature=float(generation.get("surface_polish_temperature", 0.0)),
    )
    polished = _parse_polished_messages(text, messages)
    if polished is None:
        reformatted = _chat_text(
            llm,
            _reformat_surface_polish_messages(seed_spec, text, messages),
            temperature=0.0,
        )
        polished = _parse_polished_messages(reformatted, messages)
    if polished is None:
        return messages
    return polished


def _should_surface_polish(seed_spec: dict[str, Any], generation: dict[str, Any]) -> bool:
    if generation.get("surface_polish") is False:
        return False
    return str(seed_spec["dialogue_language"]) != "en"


def _interleave_messages(
    skeleton: list[dict[str, str]],
    user_turns: list[str],
    assistant_turns: list[str],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    user_index = 0
    assistant_index = 0
    for step in skeleton:
        role = step["role"]
        if role == "user":
            messages.append(_message(role, user_turns[user_index]))
            user_index += 1
            continue
        messages.append(_message(role, assistant_turns[assistant_index]))
        assistant_index += 1
    return messages


def _generate_review(
    llm: LLM,
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> dict[str, Any]:
    text = _chat_text(
        llm,
        _review_messages(seed_spec, blueprint, intent_model, skeleton, messages),
        temperature=float(generation.get("review_temperature", 0.0)),
    )
    review = _parse_review(
        text,
        seed_spec,
        threshold=float(generation.get("review_accept_threshold", 0.8)),
        expected_assistant_turns=_role_count(skeleton, "assistant"),
    )
    if generation.get("strict_review", True):
        strict_text = _chat_text(
            llm,
            _strict_review_messages(seed_spec, blueprint, intent_model, skeleton, messages),
            temperature=float(generation.get("strict_review_temperature", generation.get("review_temperature", 0.0))),
        )
        strict_review = _parse_strict_review(strict_text, threshold=float(generation.get("strict_review_accept_threshold", 0.9)))
        review = _merge_strict_review(review, strict_review)
    return _apply_local_review_gates(review, messages)


def _repair_messages(
    llm: LLM,
    seed_spec: dict[str, Any],
    skeleton: list[dict[str, str]],
    messages: list[dict[str, str]],
    review: dict[str, Any],
    generation: dict[str, Any],
) -> list[dict[str, str]]:
    user_turns = [message["content"] for message in messages if message["role"] == "user"]
    assistant_turns = _generate_assistant_turns(
        llm,
        seed_spec,
        skeleton,
        user_turns,
        generation,
        review=review,
    )
    return _interleave_messages(skeleton, user_turns, assistant_turns)


def _needs_repair(
    review: dict[str, Any],
    messages: list[dict[str, str]],
    seed_spec: dict[str, Any],
) -> bool:
    if not bool(review["selection"]["accepted"]):
        return True
    return not _dialogue_respects_evidence_boundary(messages, _evidence_boundary(seed_spec))


def _row_from_generated_parts(
    *,
    row_id: str,
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
    messages: list[dict[str, str]],
    review: dict[str, Any],
) -> dict[str, Any]:
    required_acts = [str(step["dialogue_act"]) for step in skeleton]
    source_methods = list(seed_spec["source_methods"])

    return {
        "id": row_id,
        "messages": messages,
        "sources": [
            source
            for source in METHOD_SOURCES
            if source["name"] in source_methods
        ],
        "hidden": {
            "blueprint": blueprint,
            "intent_model": intent_model,
            "skeleton": skeleton,
            "review": review["review"],
            "verification": {
                "required_dialogue_acts": required_acts,
                "success_criteria": list(seed_spec["success_criteria"]),
                "work_session": dict(seed_spec["work_session"]),
            },
            "selection": review["selection"],
        },
        "meta": {
            "seed_id": seed_spec["seed_id"],
            "archetype_id": seed_spec["archetype_id"],
            "scenario_variant": seed_spec["scenario_variant"],
            "scenario_instance": seed_spec.get("scenario_instance", {}),
            "scenario_variant_index": seed_spec["scenario_variant_index"],
            "family": seed_spec["family"],
            "domain": blueprint["domain"],
            "intent_trajectory": intent_model["trajectory"],
            "subtask_trajectory": intent_model["subtask_trajectory"],
            "work_session_type": seed_spec["work_session"]["work_session_type"],
            "work_session_fit": seed_spec["work_session"]["fit"],
            "deliverable_family": seed_spec["work_session"]["deliverable_family"],
            "consistentchat_query_plan": intent_model["user_query_plan_steps"],
            "turn_count": len(messages),
            "source_methods": source_methods,
            "dialogue_acts": required_acts,
            "language": seed_spec["dialogue_language"],
            "source_language": seed_spec["source_language"],
            "prompt_language": seed_spec["dialogue_language"],
            "reasoning_language": seed_spec["latent_language"],
            "target_language": seed_spec["dialogue_language"],
            "language_mode": _language_mode(
                source=seed_spec["source_language"],
                prompt=seed_spec["dialogue_language"],
                reasoning=seed_spec["latent_language"],
                target=seed_spec["dialogue_language"],
            ),
        },
    }


def _generate_preference_pair(
    llm: LLM,
    row: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    text = _chat_text(
        llm,
        _preference_pair_messages(row),
        temperature=float(generation.get("preference_temperature", 0.5)),
    )
    parsed = _parse_preference_pair(text)

    return {
        "id": f"{row['id']}-pref-000",
        "source_row_id": row["id"],
        "prompt_messages": row["messages"][:-1],
        "chosen": row["messages"][-1],
        "rejected": {
            "role": "assistant",
            "content": parsed["rejected_answer"],
        },
        "hidden": {
            "failure_mode": parsed["failure_mode"],
            "chosen_action": parsed["chosen_action"],
            "rejected_action": parsed["rejected_action"],
            "blueprint_domain": row["meta"]["domain"],
        },
        "meta": {
            "family": row["meta"]["family"],
            "domain": row["meta"]["domain"],
            "intent_trajectory": row["meta"]["intent_trajectory"],
            "subtask_trajectory": row["meta"].get("subtask_trajectory", ""),
            "source_methods": ["Action-Based Contrastive Self-Training", "MDS"],
        },
    }


def _blueprint_artifact(row: dict[str, Any]) -> dict[str, Any]:
    hidden = _hidden(row)
    return {
        "id": f"{row['id']}-blueprint",
        "source_row_id": row["id"],
        "blueprint": hidden["blueprint"],
        "source_methods": row["meta"]["source_methods"],
    }


def _skeleton_artifact(row: dict[str, Any]) -> dict[str, Any]:
    hidden = _hidden(row)
    return {
        "id": f"{row['id']}-skeleton",
        "source_row_id": row["id"],
        "intent_trajectory": row["meta"]["intent_trajectory"],
        "subtask_trajectory": row["meta"].get("subtask_trajectory", ""),
        "skeleton": hidden["skeleton"],
    }


def _intent_model_artifact(row: dict[str, Any]) -> dict[str, Any]:
    hidden = _hidden(row)
    return {
        "id": f"{row['id']}-intent-model",
        "source_row_id": row["id"],
        "intent_model": hidden["intent_model"],
    }


def _select_rows_mds(
    candidates: list[dict[str, Any]],
    *,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scored = [_with_mds_scores(row) for row in candidates]
    accepted_pool = [
        row
        for row in scored
        if _mds_eligible(row)
    ]

    by_bin: dict[str, list[dict[str, Any]]] = {}
    for row in accepted_pool:
        mds = _hidden(row)["selection"]["mds"]
        by_bin.setdefault(str(mds["global_bin"]), []).append(row)

    for rows in by_bin.values():
        rows.sort(key=_mds_rank_key, reverse=True)

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    family_cursors: dict[str, int] = {}
    by_family = _group_mds_bins_by_family(by_bin)
    while len(selected) < target_count and by_family:
        for family in _ordered_mds_families(by_family):
            row = _pop_next_mds_row(by_family[family], family_cursors, family)
            if row is None:
                continue
            selected.append(_mark_mds_selected(row, accepted=True))
            selected_ids.add(str(row["id"]))
            if len(selected) >= target_count:
                break
        by_family = {
            family: bins
            for family, bins in by_family.items()
            if any(bucket for bucket in bins.values())
        }

    rejected = [
        _mark_mds_selected(row, accepted=False)
        for row in scored
        if row["id"] not in selected_ids
    ]

    report = _mds_selection_report(scored, selected, rejected)
    report["target_rows"] = target_count
    report["eligible_rows"] = len(accepted_pool)
    report["selection_complete"] = len(selected) >= target_count
    if len(selected) < target_count:
        report["shortfall"] = target_count - len(selected)
        report["status"] = "partial"
        report["message"] = (
            f"MDS selection needed {target_count} rows, "
            f"but only {len(accepted_pool)} candidates passed review and hard checks"
        )
    else:
        report["shortfall"] = 0
        report["status"] = "completed"
    return selected, rejected, report


def _group_mds_bins_by_family(by_bin: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for bin_name, bucket in by_bin.items():
        if not bucket:
            continue
        family = str(bucket[0]["meta"]["family"])
        by_family.setdefault(family, {})[bin_name] = bucket
    return by_family


def _ordered_mds_families(by_family: dict[str, dict[str, list[dict[str, Any]]]]) -> list[str]:
    return sorted(
        by_family,
        key=lambda family: (
            sum(len(bucket) for bucket in by_family[family].values()),
            family,
        ),
    )


def _pop_next_mds_row(
    bins: dict[str, list[dict[str, Any]]],
    cursors: dict[str, int],
    family: str,
) -> dict[str, Any] | None:
    bin_names = sorted(name for name, bucket in bins.items() if bucket)
    if not bin_names:
        return None

    start = cursors.get(family, 0) % len(bin_names)
    for offset in range(len(bin_names)):
        index = (start + offset) % len(bin_names)
        bucket = bins[bin_names[index]]
        if not bucket:
            continue
        cursors[family] = (index + 1) % len(bin_names)
        return bucket.pop(0)
    return None


def _with_mds_scores(row: dict[str, Any]) -> dict[str, Any]:
    hidden = _hidden(row)
    selection = dict(hidden["selection"])
    mds = _mds_scores(row)
    selection["mds"] = mds

    updated_hidden = dict(hidden)
    updated_hidden["selection"] = selection
    updated = dict(row)
    updated["hidden"] = updated_hidden
    return updated


def _mark_mds_selected(row: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
    hidden = _hidden(row)
    selection = dict(hidden["selection"])
    selection["accepted"] = bool(selection.get("accepted")) and accepted
    selection["mds"] = {
        **selection["mds"],
        "selected": accepted,
    }

    updated_hidden = dict(hidden)
    updated_hidden["selection"] = selection
    updated = dict(row)
    updated["hidden"] = updated_hidden
    return updated


def _mds_scores(row: dict[str, Any]) -> dict[str, Any]:
    hidden = _hidden(row)
    user_messages = [message["content"] for message in row["messages"] if message["role"] == "user"]
    assistant_messages = [message["content"] for message in row["messages"] if message["role"] == "assistant"]
    skeleton = hidden["skeleton"]

    entity_grounding = _visible_entity_grounding_score(user_messages, assistant_messages)

    user_acts = [step["dialogue_act"] for step in skeleton if step["role"] == "user"]
    information_progress = len(set(user_acts)) / max(len(user_acts), 1)
    if any("late" in act or "constraint" in act or "correction" in act for act in user_acts):
        information_progress = min(information_progress + 0.15, 1.0)

    nonempty_assistant = sum(1 for message in assistant_messages if str(message).strip())
    qa_form_consistency = nonempty_assistant / max(len(assistant_messages), 1)
    if len(user_messages) == len(assistant_messages):
        qa_form_consistency = min(qa_form_consistency + 0.1, 1.0)

    evidence_boundary = 1.0 if _dialogue_respects_evidence_boundary(row["messages"], _evidence_boundary_for_row(row)) else 0.0
    artifact_grounding = 1.0 if _visible_artifact_claims_grounded(row)["passed"] else 0.0
    reaction_grounding = 1.0 if _reaction_references_grounded(row)["passed"] else 0.0
    task_completion = _task_completion_score(row)
    first_turn_substance = _first_turn_substance_score(user_messages[0] if user_messages else "")
    local_score = round(
        (
            entity_grounding
            + information_progress
            + qa_form_consistency
            + evidence_boundary
            + artifact_grounding
            + reaction_grounding
            + task_completion
            + first_turn_substance
        )
        / 8,
        4,
    )
    global_bin = "|".join(
        [
            str(row["meta"]["family"]),
            str(row["meta"]["domain"]),
            str(row["meta"]["intent_trajectory"]),
            str(row["meta"].get("subtask_trajectory", "")),
            str(row["meta"].get("work_session_type", "")),
            _consistentchat_query_shape(row),
            _trajectory_shape(row),
        ]
    )

    return {
        "global_bin": global_bin,
        "user_query_trajectory": user_messages,
        "consistentchat_query_plan": _consistentchat_query_plan(row),
        "consistentchat_query_shape": _consistentchat_query_shape(row),
        "trajectory_shape": _trajectory_shape(row),
        "local_scores": {
            "entity_grounding": round(entity_grounding, 4),
            "information_progress": round(information_progress, 4),
            "query_answer_form_consistency": round(qa_form_consistency, 4),
            "evidence_boundary": evidence_boundary,
            "artifact_grounding": artifact_grounding,
            "reaction_grounding": reaction_grounding,
            "task_completion": task_completion,
            "first_turn_substance": first_turn_substance,
        },
        "local_score": local_score,
        "selected": False,
    }


def _first_turn_substance_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0

    words = re.findall(r"[A-Za-zÆØÅæøå0-9]+", stripped)
    has_payload_marker = bool(
        re.search(r"\d|[$€£]|[`\"]|“|”|:|;", stripped)
        or "\n" in stripped
        or re.search(r"\b(?:kr|dkk|usd|gb|ram|ssd|python|error|fejl|frist|budget|dato)\b", stripped, flags=re.IGNORECASE)
    )
    salient_terms = _salient_visible_terms(stripped, limit=8)

    score = 0.2
    if len(words) >= 10:
        score += 0.2
    if len(words) >= 18:
        score += 0.2
    if has_payload_marker:
        score += 0.3
    if len(salient_terms) >= 2:
        score += 0.1
    if _looks_like_zero_information_help_request(stripped):
        score -= 0.35
    return round(min(max(score, 0.0), 1.0), 4)


def _looks_like_zero_information_help_request(text: str) -> bool:
    lower = text.lower()
    help_markers = [
        "kan du hjælpe",
        "kan du hjælpe mig",
        "vil du hjælpe",
        "hjælp mig",
        "can you help",
        "could you help",
    ]
    if not any(marker in lower for marker in help_markers):
        return False
    words = re.findall(r"[A-Za-zÆØÅæøå0-9]+", text)
    has_specific_payload = bool(re.search(r"\d|[$€£]|[`\"]|“|”|:", text))
    return len(words) < 14 and not has_specific_payload


def _task_completion_score(row: dict[str, Any]) -> float:
    selection = _hidden(row).get("selection", {})
    assert isinstance(selection, dict), "row selection must be a mapping"
    review_score = _bounded_float(selection.get("score", 0.0))
    outcome_score = _reviewer_score(row, "outcome")
    return round((review_score + outcome_score) / 2, 4)


def _reviewer_score(row: dict[str, Any], name: str) -> float:
    review = _hidden(row).get("review", {})
    assert isinstance(review, dict), "row review must be a mapping"
    reviewers = review.get("reviewers", [])
    assert isinstance(reviewers, list), "row review.reviewers must be a list"
    for reviewer in reviewers:
        if not isinstance(reviewer, dict):
            continue
        if reviewer.get("name") == name:
            return 1.0 if bool(reviewer.get("passed")) else 0.0
    return 0.0


def _bounded_float(value: object) -> float:
    number = float(value)
    return min(max(number, 0.0), 1.0)


def _mds_eligible(row: dict[str, Any]) -> bool:
    hidden = _hidden(row)
    selection = hidden["selection"]
    mds = selection["mds"]
    if not bool(selection.get("accepted")):
        return False
    if float(mds["local_scores"]["evidence_boundary"]) < 1.0:
        return False
    if float(mds["local_scores"]["artifact_grounding"]) < 1.0:
        return False
    if float(mds["local_scores"]["reaction_grounding"]) < 1.0:
        return False
    return bool(
        _messages_alternate(row)["passed"]
        and _no_role_label_leak(row)["passed"]
        and _skeleton_matches_messages(row)["passed"]
        and _evidence_boundary_respected(row)["passed"]
        and _visible_artifact_claims_grounded(row)["passed"]
        and _reaction_references_grounded(row)["passed"]
        and _assistant_format_restrained(row)["passed"]
    )


def _trajectory_shape(row: dict[str, Any]) -> str:
    hidden = _hidden(row)
    acts = [step["dialogue_act"] for step in hidden["skeleton"] if step["role"] == "user"]
    return ">".join(str(act) for act in acts)


def _visible_entity_grounding_score(user_messages: list[str], assistant_messages: list[str]) -> float:
    terms = _salient_visible_terms("\n".join(user_messages))
    if not terms:
        return 1.0

    assistant_text = "\n".join(assistant_messages).lower()
    hits = sum(1 for term in terms if term.lower() in assistant_text)
    return round(hits / len(terms), 4)


def _salient_visible_terms(text: str, *, limit: int = 16) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"[$€£]?\d[\d.,]*(?:\s*(?:%|kr\.?|dollars?|minutes?|minutter|days?|dage))?",
        r"\b[A-ZÆØÅ]{2,}(?:-[A-Z0-9]+)?\b",
        r"\b[A-Za-zÆØÅæøå]+-[A-Za-z0-9]+\b",
    ]
    for pattern in patterns:
        candidates.extend(match.group(0).strip() for match in re.finditer(pattern, text))

    for token in re.findall(r"[A-Za-zÆØÅæøå]{5,}", text.lower()):
        if token not in SALIENT_TERM_STOPWORDS:
            candidates.append(token)

    return _dedupe([term for term in candidates if term])[:limit]


def _consistentchat_query_plan(row: dict[str, Any]) -> list[str]:
    hidden = _hidden(row)
    intent_model = hidden["intent_model"]
    steps = intent_model.get("user_query_plan_steps", [])
    if isinstance(steps, list) and steps:
        return [str(step) for step in steps]
    plan = str(intent_model.get("user_query_plan", ""))
    parsed = _split_query_plan(plan)
    if parsed:
        return parsed
    return [step["dialogue_act"] for step in hidden["skeleton"] if step["role"] == "user"]


def _consistentchat_query_shape(row: dict[str, Any]) -> str:
    return ">".join(_normalize_shape_part(step) for step in _consistentchat_query_plan(row))


def _normalize_shape_part(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unknown"


def _mds_rank_key(row: dict[str, Any]) -> tuple[float, float]:
    selection = _hidden(row)["selection"]
    mds = selection["mds"]
    return float(mds["local_score"]), float(selection.get("score", 0.0))


def _mds_selection_report(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_bins = Counter(str(_hidden(row)["selection"]["mds"]["global_bin"]) for row in selected)
    candidate_bins = Counter(str(_hidden(row)["selection"]["mds"]["global_bin"]) for row in candidates)
    selected_scores = [float(_hidden(row)["selection"]["mds"]["local_score"]) for row in selected]

    return {
        "candidate_rows": len(candidates),
        "accepted_rows": len(selected),
        "rejected_rows": len(rejected),
        "families": dict(Counter(str(row["meta"]["family"]) for row in selected)),
        "intent_trajectories": dict(Counter(str(row["meta"]["intent_trajectory"]) for row in selected)),
        "subtask_trajectories": dict(Counter(str(row["meta"].get("subtask_trajectory", "")) for row in selected)),
        "work_session_types": dict(Counter(str(row["meta"].get("work_session_type", "")) for row in selected)),
        "consistentchat_query_shapes": dict(Counter(str(_consistentchat_query_shape(row)) for row in selected)),
        "domains": dict(Counter(str(row["meta"]["domain"]) for row in selected)),
        "candidate_bins": dict(candidate_bins),
        "selected_bins": dict(selected_bins),
        "mean_selected_local_score": _average_float(selected_scores),
        "selection_policy": {
            "method": "MDS-style dialogue-level selection",
            "global_coverage": "family-stratified round-robin over user-query trajectory bins",
            "local_structure": [
                "entity-grounded topic grounding",
                "information progress",
                "query-answer form consistency",
                "evidence-boundary compliance",
                "reviewed task completion",
            ],
        },
    }


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)
    rows = _load_rows(result)
    return common_eval.verify(rows, _messages_alternate, name="messages_alternate")


def _load_preference_pairs(result: BuildResult) -> list[dict[str, Any]]:
    artifact = result.artifacts.get("preference_pairs")
    if artifact is None:
        return []
    return store.read_jsonl(Path(artifact.path))


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _messages_alternate(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}
    if len(messages) < 4 or len(messages) % 2 != 0:
        return {"passed": False}

    expected_role = "user"
    for message in messages:
        if not isinstance(message, dict):
            return {"passed": False}
        if message.get("role") != expected_role:
            return {"passed": False}
        if not str(message.get("content", "")).strip():
            return {"passed": False}
        expected_role = "assistant" if expected_role == "user" else "user"
    return {"passed": True}


def _no_role_label_leak(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}
    for message in messages:
        if not isinstance(message, dict):
            return {"passed": False}
        if _role_label_leaked(str(message.get("content", ""))):
            return {"passed": False}
    return {"passed": True}


def _skeleton_matches_messages(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    hidden = _hidden(row)
    skeleton = hidden.get("skeleton")
    if not isinstance(messages, list) or not isinstance(skeleton, list):
        return {"passed": False}
    if len(messages) != len(skeleton):
        return {"passed": False}

    for index, step in enumerate(skeleton):
        if not isinstance(step, dict):
            return {"passed": False}
        if step.get("role") != messages[index].get("role"):
            return {"passed": False}
        if not str(step.get("dialogue_act", "")).strip():
            return {"passed": False}
    return {"passed": True}


def _evidence_boundary_respected(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}
    return {"passed": _dialogue_respects_evidence_boundary(messages, _evidence_boundary_for_row(row))}


def _visible_artifact_claims_grounded(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}

    for message in messages:
        if not isinstance(message, dict):
            return {"passed": False}
        if message.get("role") != "user":
            continue

        content = str(message.get("content", ""))
        if _claims_visible_artifact(content) and not _has_visible_artifact_payload(content):
            return {"passed": False}

    return {"passed": True}


def _reaction_references_grounded(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}

    visible_prefix: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            return {"passed": False}

        content = str(message.get("content", ""))
        if message.get("role") == "user" and not _user_reaction_references_are_visible(content, visible_prefix):
            return {"passed": False}

        visible_prefix.append(content)

    return {"passed": True}


def _assistant_format_restrained(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}
    for message in messages:
        if not isinstance(message, dict):
            return {"passed": False}
        if message.get("role") != "assistant":
            continue
        if not _assistant_text_has_restrained_format(str(message.get("content", ""))):
            return {"passed": False}
    return {"passed": True}


def _assistant_text_has_restrained_format(text: str) -> bool:
    if re.search(r"[\u2705\u274c\u26a0\ufe0f\U0001f300-\U0001faff]", text):
        return False
    if re.search(r"(?m)^\s*[-*_]{3,}\s*$", text):
        return False
    if re.search(r"\*\*[^*\n][\s\S]*?\*\*|\*[^*\n]+\*", text):
        return False
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", text):
        return False
    if re.search(r"(?m)^\s*\|.+\|\s*$", text):
        return False
    return True


def _claims_visible_artifact(text: str) -> bool:
    lower = text.lower()
    has_claim_marker = any(marker in lower for marker in _ARTIFACT_CLAIM_MARKERS)
    has_artifact_term = any(_contains_artifact_term(lower, term) for term in _ARTIFACT_TERMS)
    return (has_claim_marker and has_artifact_term) or bool(_artifact_label_pattern().search(text))


def _contains_artifact_term(lower_text: str, term: str) -> bool:
    escaped = re.escape(term)
    if " " in term or "-" in term:
        return bool(re.search(rf"(?<![a-zæøå]){escaped}(?![a-zæøå])", lower_text))
    suffix = r"(?:en|et|er|erne|ene|s)?"
    return bool(re.search(rf"(?<![a-zæøå]){escaped}{suffix}(?![a-zæøå])", lower_text))


def _has_visible_artifact_payload(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if "```" in stripped:
        return True
    if re.search(r"(?m)^\s*>.+", stripped):
        return True
    if _has_long_quoted_span(stripped):
        return True
    if _has_labeled_artifact_payload(stripped):
        return True

    nonempty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return len(nonempty_lines) >= 3 and len(stripped) >= 180


def _has_long_quoted_span(text: str) -> bool:
    for match in _quoted_span_pattern().finditer(text):
        if len(match.group("phrase").strip()) >= 25:
            return True
    return False


def _has_labeled_artifact_payload(text: str) -> bool:
    matches = list(_artifact_label_pattern().finditer(text))
    if not matches:
        return False

    if len(matches) >= 2:
        return True

    match = matches[0]
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    same_line = text[match.end() : line_end]
    if len(same_line.strip()) >= 20:
        return True

    following_lines = [line.strip() for line in text[match.end() :].splitlines() if line.strip()]
    return bool(following_lines and len(following_lines[0]) >= 20)


def _user_reaction_references_are_visible(content: str, visible_prefix: list[str]) -> bool:
    for phrase, start, end in _quoted_spans(content):
        if len(phrase.strip()) < 4:
            continue
        if not _quoted_phrase_requires_prior_visibility(content, start, end):
            continue

        current_without_quote = f"{content[:start]} {content[end:]}"
        visible_text = "\n".join([*visible_prefix, current_without_quote])
        if _normal_reference_text(phrase) not in _normal_reference_text(visible_text):
            return False

    return True


def _quoted_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group("phrase"), match.start("phrase"), match.end("phrase")) for match in _quoted_span_pattern().finditer(text)]


def _quoted_phrase_requires_prior_visibility(content: str, start: int, end: int) -> bool:
    lower = content.lower()
    before = lower[max(0, start - 80) : start]
    after = lower[end : min(len(lower), end + 80)]
    if re.search(r"(?:\btil\b|\bto\b|\bmed\b|\bwith\b|\bsom\b)\s*$", before):
        return False

    context = f"{before} {after}"
    return any(marker in context for marker in _EXISTING_CONTENT_REACTION_MARKERS)


def _normal_reference_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _artifact_label_pattern() -> re.Pattern[str]:
    label_terms = "|".join(re.escape(term) for term in _ARTIFACT_LABEL_TERMS)
    return re.compile(rf"(?im)^\s*(?:{label_terms})\s*:", flags=re.IGNORECASE)


def _quoted_span_pattern() -> re.Pattern[str]:
    return re.compile(r"[\"'“”„](?P<phrase>[^\"'“”„]{4,})[\"'“”„]")


_ARTIFACT_CLAIM_MARKERS = (
    "her er",
    "her kommer",
    "jeg har indsat",
    "jeg har sat ind",
    "jeg har kopieret",
    "jeg har vedhæftet",
    "jeg vedhæfter",
    "nedenfor",
    "følgende",
    "here is",
    "here are",
    "i pasted",
    "i copied",
    "i attached",
    "attached",
    "pasted",
    "below",
    "following",
)

_ARTIFACT_TERMS = (
    "abstract",
    "aftale",
    "annonce",
    "ansøgning",
    "artikel",
    "brev",
    "cv",
    "dokument",
    "draft",
    "email",
    "e-mail",
    "excerpt",
    "fejlbesked",
    "formular",
    "jobopslag",
    "kilde",
    "kladde",
    "kode",
    "kontrakt",
    "kvittering",
    "lejeaftale",
    "mail",
    "noter",
    "opslag",
    "policy",
    "rapport",
    "regning",
    "source",
    "stack trace",
    "stillingsopslag",
    "tekst",
    "traceback",
    "uddrag",
    "udkast",
    "vilkår",
)

_ARTIFACT_LABEL_TERMS = (
    "abstract",
    "aftale",
    "ansøgning",
    "brev",
    "cv",
    "draft",
    "email",
    "e-mail",
    "excerpt",
    "fejl",
    "fejlbesked",
    "formular",
    "jobopslag",
    "kilde",
    "kladde",
    "kode",
    "kvittering",
    "mail",
    "noter",
    "opslag",
    "policy",
    "source",
    "stillingsopslag",
    "tekst",
    "traceback",
    "uddrag",
    "udkast",
    "vilkår",
)

_EXISTING_CONTENT_REACTION_MARKERS = (
    "delete",
    "drop",
    "erstat",
    "fjern",
    "fjerne",
    "korriger",
    "lav om",
    "remove",
    "replace",
    "ret",
    "rette",
    "rewrite",
    "slet",
    "tag ud",
    "ændr",
    "ændre",
)


def _evidence_boundary_for_row(row: dict[str, Any]) -> dict[str, Any]:
    hidden = _hidden(row)
    blueprint = hidden.get("blueprint", {})
    assert isinstance(blueprint, dict), "row blueprint must be a mapping"
    boundary = blueprint.get("evidence_boundary")
    assert isinstance(boundary, dict), "row blueprint evidence_boundary must be a mapping"
    return boundary


def _dialogue_respects_evidence_boundary(messages: list[dict[str, Any]], boundary: dict[str, Any]) -> bool:
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "user":
            continue
        if role != "assistant":
            return False
        if not _answer_respects_evidence_boundary(content, boundary):
            return False
    return True


def _answer_respects_evidence_boundary(text: str, boundary: dict[str, Any]) -> bool:
    return True


def _selection_accepted(row: dict[str, Any]) -> dict[str, bool]:
    hidden = _hidden(row)
    selection = hidden.get("selection", {})
    if not isinstance(selection, dict):
        return {"passed": False}
    return {"passed": bool(selection.get("accepted"))}


def _dataset_checks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mds_scores = [float(_hidden(row)["selection"]["mds"]["local_score"]) for row in rows]
    return {
        "rows": len(rows),
        "families": dict(Counter(str(row["meta"]["family"]) for row in rows)),
        "domains": dict(Counter(str(row["meta"]["domain"]) for row in rows)),
        "intent_trajectories": dict(Counter(str(row["meta"]["intent_trajectory"]) for row in rows)),
        "subtask_trajectories": dict(Counter(str(row["meta"].get("subtask_trajectory", "")) for row in rows)),
        "work_session_types": dict(Counter(str(row["meta"].get("work_session_type", "")) for row in rows)),
        "consistentchat_query_shapes": dict(Counter(str(_consistentchat_query_shape(row)) for row in rows)),
        "mds_bins": dict(Counter(str(_hidden(row)["selection"]["mds"]["global_bin"]) for row in rows)),
        "mean_mds_local_score": _average_float(mds_scores),
        "avg_turn_count": _average([int(row["meta"]["turn_count"]) for row in rows]),
    }


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _average_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(value)
    return deduped


def _hidden(row: dict[str, Any]) -> dict[str, Any]:
    hidden = row.get("hidden")
    assert isinstance(hidden, dict), "row hidden state must be a mapping"
    return hidden


def _row_failed(row: dict[str, Any]) -> bool:
    checks = row.get("checks", {})
    assert isinstance(checks, dict), "row checks must be a mapping"
    return any(not _check_passed(value) for value in checks.values())


def _check_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    assert isinstance(value, dict), "check value must be a bool or mapping"
    if "passed" in value:
        return bool(value["passed"])
    if "ok" in value:
        return bool(value["ok"])
    return False


def _strip_hidden(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "hidden"}


def _split_rows(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return reports_root() / result.pack / result.run_id


def _load_or_compute(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return fallback


def _chat_text(llm: LLM, messages: list[dict[str, str]], *, temperature: float) -> str:
    response = llm.chat(messages, temperature=temperature)
    text = str(response).strip()
    assert text, "LLM response must be non-empty"
    return text


def _work_session_contract(seed_spec: dict[str, Any]) -> dict[str, Any]:
    top_level_intent = _consistentchat_top_level_intent(
        str(seed_spec.get("intent_trajectory_hint", "")),
        domain=str(seed_spec.get("domain", "")),
    )
    guidance = WORK_SESSION_GUIDANCE[top_level_intent]
    subtask = _subtask_trajectory_hint(seed_spec)
    variant = seed_spec.get("scenario_variant", {})
    assert isinstance(variant, dict), "scenario_variant must be a mapping"

    return {
        "top_level_intent": top_level_intent,
        "subtask_trajectory": subtask,
        "work_session_type": guidance["work_session_type"],
        "deliverable_family": guidance["deliverable_family"],
        "visible_input_payload": guidance["visible_input_payload"],
        "progression": guidance["progression"],
        "completion_bar": guidance["completion_bar"],
        "scenario_focus": str(variant.get("work_session_focus", "")),
        "first_turn_shape": str(variant.get("first_turn_shape", "")),
        "fit": f"{top_level_intent} > {subtask}",
    }


def _format_work_session_contract(seed_spec: dict[str, Any]) -> str:
    contract = seed_spec.get("work_session")
    if not isinstance(contract, dict):
        contract = _work_session_contract(seed_spec)
    return json.dumps(contract, indent=2, sort_keys=True)


def _scenario_instance_messages(seed_spec: dict[str, Any]) -> list[dict[str, str]]:
    latent_language = _language_name(str(seed_spec["latent_language"]))
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    return [
        {
            "role": "system",
            "content": (
                "You expand a handwritten multi-turn dialogue archetype into one concrete scenario instance. "
                "This is the ConsistentChat diversity fanout layer before APIGen-MT-style blueprint generation: "
                "the seed is an archetype, not the final scenario. "
                "Do not write the dialogue. Do not output JSON-only. "
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one fresh scenario instance for this candidate. "
                "Preserve the top-level domain, intent category, evidence boundary, and success criteria, but vary "
                "the concrete task material wherever possible: names, dates, quantities, artifacts, excerpts, user "
                "background, constraints, stakeholder context, options, prior attempts, and disclosure timing. "
                "If the seed success criteria require a specific entity, keep that entity; otherwise avoid reusing "
                "the seed's default examples. Treat the seed query trajectory as a pattern, not a fixed script.\n\n"
                f"Hidden artifact language: {latent_language}\n"
                f"Later visible dialogue language: {dialogue_language}\n"
                f"Seed family: {seed_spec['family']}\n"
                f"Domain: {seed_spec['domain']}\n"
                f"Intent hint: {seed_spec['intent_trajectory_hint']}\n"
                f"Subtask hint: {_subtask_trajectory_hint(seed_spec)}\n"
                f"Difficulty axis: {seed_spec['difficulty_axis']}\n"
                f"Seed success criteria:\n{_format_success_criteria(seed_spec['success_criteria'])}\n\n"
                "Seed facts and constraints:\n"
                f"{json.dumps(seed_spec['seed_blueprint'], indent=2, sort_keys=True)}\n\n"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate variation axis:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                "Write a compact scenario instance with headings for concrete user situation, visible first-turn "
                "payload, hidden-but-unrevealed details, likely follow-up turns, deliverable target, and evidence "
                "boundary notes. Keep it concise but specific enough that two candidates from the same seed can "
                "become visibly different conversations."
            ),
        },
    ]


def _blueprint_messages(seed_spec: dict[str, Any]) -> list[dict[str, str]]:
    latent_language = _language_name(str(seed_spec["latent_language"]))
    target_user_turns = _target_user_turns(seed_spec)
    return [
        {
            "role": "system",
            "content": (
                "You generate the hidden world state for synthetic multi-turn training data. "
                "Borrow APIGen-MT's blueprint-first pattern, but keep this as non-tool dialogue data: "
                "first define the latent task configuration, user intent, "
                "generator-only facts, assistant-visible context requirements, unrevealed constraints, "
                "success criteria, and verification hooks. "
                "Represent useful LLM work as a work-session contract under the ConsistentChat top-level intent; "
                "do not replace the top-level intent taxonomy. "
                "This blueprint is hidden from the visible assistant transcript, so hidden facts are not assistant evidence. "
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one hidden conversation blueprint from this seed. "
                "Make it realistic and keep enough latent state to support "
                f"{target_user_turns} user turns and {target_user_turns} assistant turns.\n\n"
                f"Seed family: {seed_spec['family']}\n"
                f"Domain: {seed_spec['domain']}\n"
                f"Hidden artifact language: {latent_language}\n"
                f"Seed source language: {_language_name(str(seed_spec['source_language']))}\n"
                f"Visible dialogue language later in the pipeline: {_language_name(str(seed_spec['dialogue_language']))}\n"
                f"Intent hint: {seed_spec['seed_intent']}\n"
                f"Difficulty axis: {seed_spec['difficulty_axis']}\n"
                f"Seed success criteria:\n{_format_success_criteria(seed_spec['success_criteria'])}\n\n"
                "Seed facts and constraints:\n"
                f"{json.dumps(seed_spec['seed_blueprint'], indent=2, sort_keys=True)}\n\n"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate scenario instance:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                "Use the candidate scenario instance as the concrete row-specific scenario. Use the seed as an "
                "archetype and constraint source, not as a fixed transcript plan. Preserve the archetype's domain, "
                "evidence boundary, and success criteria, but prefer the candidate instance for user-owned facts, "
                "named entities, quantities, constraints, first-turn informativeness, and disclosure timing where "
                "the archetype permits it. "
                "Do not fabricate current/live, official, private, or source-specific facts unless the variant also "
                "plans for them to appear in visible user/source text before assistant use.\n\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                "Write a compact blueprint with headings for hidden goal, known constraints, "
                "unrevealed constraints, generator-only facts, assistant-visible context requirements, "
                "work-session contract, visible input payload plan, expected intermediate artifact, final deliverable, "
                "conversation flow, and success criteria. "
                "Also include an evidence boundary section that states what the assistant can know, what it cannot know, "
                "which private facts must be supplied in visible user/source text, "
                "and which details must be hedged as assumptions or estimates. "
                "For most seeds, make the dialogue look like real LLM work: transforming notes, drafting, revising, "
                "explaining a pasted item, comparing visible options, checking calculations, planning from constraints, "
                "or producing a bounded next-step artifact. For softer seeds, the deliverable may be a decision frame, "
                "message draft, grounding plan, rehearsal feedback, or structured explanation rather than a literal file. "
                f"Write this hidden blueprint in {latent_language}; do not translate the visible dialogue yet. "
                "Do not introduce tool calls, APIs, executable actions, tool observations, or agent environment state."
            ),
        },
    ]


def _intent_model_messages(seed_spec: dict[str, Any], blueprint: dict[str, Any]) -> list[dict[str, str]]:
    intents = "\n".join(f"- {item}" for item in CONSISTENTCHAT_INTENTS)
    subtasks = _format_consistentchat_subtasks()
    latent_language = _language_name(str(seed_spec["latent_language"]))
    return [
        {
            "role": "system",
            "content": (
                "You perform ConsistentChat-style intent modeling. "
                "Choose a global conversation intent trajectory before any query skeleton is generated. "
                "The intent must constrain information flow, role interaction, and topic progression across turns."
            ),
        },
        {
            "role": "user",
            "content": (
                "Assign the best intent trajectory and information-flow plan for this hidden blueprint. "
                "Use a compact artifact with these lines: INTENT_TRAJECTORY, SUBTASK_TRAJECTORY, INFORMATION_FLOW, "
                "ROLE_INTERACTION, TOPIC_GUARDRAILS, WORK_SESSION_TYPE, DELIVERABLE_PROGRESS, USER_QUERY_PLAN.\n\n"
                "INTENT_TRAJECTORY must exactly match one of the nine ConsistentChat top-level categories below. "
                "SUBTASK_TRAJECTORY should be a more specific domain/task pattern under that top-level category. "
                "WORK_SESSION_TYPE must fit underneath INTENT_TRAJECTORY rather than adding a competing taxonomy.\n\n"
                f"Write the intent artifact in {latent_language}. The future visible dialogue language is "
                f"{_language_name(str(seed_spec['dialogue_language']))}.\n\n"
                "ConsistentChat top-level interaction categories:\n"
                f"{intents}\n\n"
                "Subtask expansion space under those categories:\n"
                f"{subtasks}\n\n"
                f"Seed top-level ConsistentChat hint: {seed_spec['intent_trajectory_hint']}\n"
                f"Seed subtask hint: {_subtask_trajectory_hint(seed_spec)}\n\n"
                "Seed user-query trajectory hint:\n"
                f"{_format_query_trajectory(seed_spec.get('query_trajectory_hint', []))}\n\n"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate scenario instance:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}"
            ),
        },
    ]


def _skeleton_messages(
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
) -> list[dict[str, str]]:
    latent_language = _language_name(str(seed_spec["latent_language"]))
    target_user_turns = _target_user_turns(seed_spec)
    target_steps = target_user_turns * 2
    role_order = ", ".join(_expected_skeleton_roles(seed_spec))
    return [
        {
            "role": "system",
            "content": (
                "You design ConsistentChat-style dialogue skeletons. "
                "The skeleton should specify query progression before any dialogue text is written. "
                "Keep the top-level intent fixed, and express useful LLM work through the subtask and work-session "
                "contract underneath it. "
                "If a user step rejects, corrects, or compares against assistant content, the immediately prior "
                "assistant step must have visibly introduced the content being reacted to; otherwise plan the user "
                "step as a new constraint or clarification. "
                "For private policy, organizational, document, or data-source facts, plan a user turn that reveals "
                "the relevant source text before the assistant uses those facts. "
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Build a {target_steps}-step skeleton from the hidden blueprint. "
                f"Use the roles in this exact order: {role_order}. "
                "Each user step should be a structurally grounded query move aligned to the intent model; "
                "each assistant step should state the conversational response purpose needed at that point. "
                "State deltas must track both visible information progress and work-product progress: what input became "
                "visible, what draft/analysis/checklist/plan was produced, and what changed after user feedback.\n\n"
                f"Preserve the ConsistentChat user-query trajectory. The {target_user_turns} user skeleton steps "
                "should map to the USER_QUERY_PLAN in order, while still using compact dialogue_act labels.\n\n"
                f"Write the skeleton artifact in {latent_language}. The future visible dialogue language is "
                f"{_language_name(str(seed_spec['dialogue_language']))}.\n\n"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate scenario instance:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                "Prefer skeletons where the assistant does useful work early when the visible input is sufficient. "
                "The final assistant step should usually produce the completed deliverable, revised artifact, bounded "
                "decision frame, or concrete next-step output required by the work-session contract.\n\n"
                "Use this line form so the artifact can be replayed:\n"
                "STEP 1: user | initial_user_request | what changes in the visible state\n"
                "STEP 2: assistant | clarify_missing_slots | what changes in the visible state"
            ),
        },
    ]


def _user_simulator_messages(
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
) -> list[dict[str, str]]:
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    user_turns = _role_count(skeleton, "user")
    user_labels = ", ".join(f"USER {index}:" for index in range(1, user_turns + 1))
    return [
        {
            "role": "system",
            "content": (
                "You are a user simulator for multi-turn data. "
                "Act like a realistic human user: choose the first turn's level of detail from the skeleton and seed "
                "trajectory. The first turn may be vague, partial, or reasonably informative; do not make every "
                "conversation start with underspecification. Reveal hidden constraints gradually, refer back to previous "
                "entities, and do not expose the full blueprint at once. "
                "Make the dialogue look like a real language-model work session when the seed supports it: the user "
                "often brings messy notes, a draft, numbers, requirements, a source excerpt, an attempted answer, or "
                "specific feedback on a previous assistant artifact. "
                "The first user turn should normally contain at least one useful work payload: a concrete fact, number, "
                "deadline, draft, excerpt, named option, role/context, prior attempt, error message, or constraint. "
                "Avoid empty openings such as merely asking whether the assistant can help with a topic. "
                "If a user turn says it provides, pastes, attaches, or shows an artifact such as a draft, source, "
                "job posting, policy, abstract, receipt, error message, or code, include a short visible payload in "
                "that same turn; do not imply invisible attachments or omitted source text. "
                "When private policy, organizational, or document facts are needed, reveal them as user-visible "
                "source text before expecting the assistant to rely on them. Reveal concrete condition status, such as "
                "receipts, documentation, dates, exclusions, and missing fields, before expecting a final eligibility "
                "or calculation answer. Each visible user turn should realize the skeleton step and seed query plan, "
                "including requested stance, format, source boundary, or factual-only constraints. "
                "Because assistant turns are generated later, phrase late constraints as new information unless the "
                "skeleton explicitly says the previous assistant turn introduced the thing being rejected or corrected. "
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
                f"Write the visible user turns in {dialogue_language}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write only the user turns for this skeleton. "
                "The assistant turns will be generated later from the same blueprint. "
                f"Use one labeled block per user turn with these labels: {user_labels}. "
                "A user block may contain multiple sentences, a compact list, or a short pasted excerpt when the "
                "scenario variant calls for visible payload. "
                "Only sparse_opening may be a very short opening; all other opener variants should make USER 1 "
                "substantive enough for the assistant to start useful work.\n\n"
                f"Visible dialogue language: {dialogue_language}. Keep the USER labels exactly as requested, "
                f"but write the turn content in {dialogue_language}.\n"
                f"Domain: {seed_spec['domain']}\n"
                f"Seed query trajectory hint:\n{_format_query_trajectory(seed_spec.get('query_trajectory_hint'))}\n\n"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate scenario instance:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                f"Skeleton:\n{_format_skeleton(skeleton)}"
            ),
        },
    ]


def _assistant_turn_messages(
    seed_spec: dict[str, Any],
    skeleton: list[dict[str, str]],
    user_turns: list[str],
    assistant_turns: list[str],
    *,
    assistant_index: int,
    review: dict[str, Any] | None,
) -> list[dict[str, str]]:
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    review_text = ""
    if review is not None:
        review_text = (
            "This is a repair pass. The previous candidate failed review. Improve coherence, grounding, recovery, "
            "and format while using only the visible transcript prefix as assistant evidence.\n"
            f"Previous review findings to fix:\n{_format_repair_context(review)}\n\n"
        )
    turn_number = assistant_index + 1
    assistant_steps = [step for step in skeleton if step["role"] == "assistant"]
    current_step = assistant_steps[assistant_index]
    transcript_prefix = _assistant_transcript_prefix(user_turns, assistant_turns, assistant_index=assistant_index)
    return [
        {
            "role": "system",
            "content": (
                "You realize one assistant turn for ConsistentChat-style multi-turn data. "
                "Use the ConsistentChat role/action outline for dialogue structure, but use only the visible transcript "
                "prefix as evidence. You cannot see future user turns, hidden blueprints, or future skeleton state deltas. "
                "Treat the work-session contract as generic task-shape guidance under the ConsistentChat intent, not "
                "as evidence for user-specific facts. "
                f"{_evidence_boundary_instruction(seed_spec)}"
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
                f"Write the visible assistant turns in {dialogue_language}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Write ASSISTANT {turn_number} only. "
                f"Output exactly one labeled block: ASSISTANT {turn_number}:.\n\n"
                f"Visible dialogue language: {dialogue_language}. Keep the ASSISTANT labels exactly as requested, "
                f"but write the assistant content in {dialogue_language}. Do not translate the ASSISTANT labels.\n"
                f"Domain: {seed_spec['domain']}\n"
                "If this is the final assistant turn, resolve the visible user request as far as the visible conversation allows. "
                "Carry forward concrete user-visible constraints, deadlines, quantities, entity names, requested stance, "
                "and requested output format when they remain relevant. "
                "If enough visible information is already present, do useful work now: draft, revise, calculate, compare, "
                "explain, diagnose, or create the requested bounded artifact instead of defaulting to a broad clarification loop. "
                "If required information is private, live/current, source-specific, jurisdiction-specific, or numeric and "
                "has not been supplied by the user, say that it is missing or use explicit illustrative placeholders. "
                "Do not treat a user phrase like here is the draft, below is the source, or I attached the document as "
                "evidence unless the actual draft, source excerpt, document text, error message, or code is visible in "
                "the transcript. If a needed artifact is missing, ask for it instead of drafting from it. If the user "
                "asks to remove, rewrite, or react to a quoted phrase that does not appear in the visible transcript, "
                "acknowledge the mismatch and treat it as new wording or ask a concise clarification. "
                "For any task, not only product comparisons, treat exact source-dependent details as unknown unless "
                "they are visible in the conversation. Do not fill in specs, prices, policy thresholds, schedules, "
                "legal/procedural rules, compatibility outcomes, provider rules, stock or delivery details, named "
                "institution details, or other precise source facts. It is fine to use broad public knowledge, but hedge "
                "it as general background, mention that it may need current or source-specific verification, and if "
                "lookup is not available, preserve missing details as unknown fields and state how to verify them. "
                "For appeal deadlines, eligibility windows, legal/procedural rules, or policy thresholds, do not invent "
                "a date or precise rule; give a source-bounded checklist and ask for the controlling letter, policy, "
                "jurisdiction, or receipt date when needed. "
                "For calculations, show exclusions and totals from the visible numbers instead of jumping to a result.\n\n"
                "Use restrained plain-text formatting. Prefer short paragraphs and simple bullets or numbering only when "
                "they make the answer easier to scan. Avoid markdown bold, italics, horizontal rules, decorative "
                "separators, and large heading stacks unless the user explicitly asks for a formatted artifact.\n\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"{review_text}"
                "Visible ConsistentChat role/action outline:\n"
                f"{_assistant_visible_dialogue_act_outline(skeleton, assistant_index=assistant_index)}\n\n"
                "Current assistant purpose:\n"
                f"ASSISTANT {turn_number}: {current_step['dialogue_act']}\n\n"
                "Visible transcript prefix:\n"
                f"{transcript_prefix}"
            ),
        },
    ]


def _format_repair_context(review: dict[str, Any]) -> str:
    review_state = review.get("review", {})
    assert isinstance(review_state, dict), "review.review must be a mapping"
    lines: list[str] = []
    for reviewer in review_state.get("reviewers", []):
        if not isinstance(reviewer, dict) or bool(reviewer.get("passed")):
            continue
        lines.append(f"- {reviewer.get('name', 'reviewer')}: {reviewer.get('finding', 'failed')}")
    for action in review_state.get("repair_actions", []):
        lines.append(f"- repair_action: {action}")
    if not lines:
        return "- Previous review failed; regenerate cautiously with stricter grounding, naturalness, and format."
    return "\n".join(lines[:10])


def _surface_polish_messages(seed_spec: dict[str, Any], messages: list[dict[str, str]]) -> list[dict[str, str]]:
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    label_list = ", ".join(
        f"{str(message['role']).upper()} {_role_label_index(messages, position)}:"
        for position, message in enumerate(messages)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a surface editor for synthetic dialogue data. "
                "Fix only visible-language quality issues: typos, malformed words, awkward calques, invented compounds, "
                "unnatural phrasing, and over-precise localized UI/legal/policy labels. Preserve roles, order, meaning, "
                "numbers, dates, code, commands, quoted source text, product names, and evidence boundaries. Do not add "
                "new facts, remove constraints, make the assistant smarter, or rewrite the task. If an official localized "
                "label is uncertain, use a generic description rather than a precise-sounding invented label. "
                "For Danish, replace accidental English words in prose with ordinary Danish unless they are code, product "
                "names, quoted source text, or established loanwords. Fix misspellings and malformed Danish compounds. "
                "Use established Danish technical terms or keep standard English terms when natural; do not create ad hoc "
                "literal compounds, hyphenated participles, or word-for-word translations of English technical phrases. "
                "Check Danish agreement in adjective and noun phrases. "
                "Remove emojis and decorative emoji bullets from assistant turns, replacing them with plain text bullets "
                "or headings. Remove excessive markdown bold/italic emphasis, horizontal rules, decorative separators, "
                "and heading stacks unless the user explicitly requested a formatted artifact. No turn content may "
                "contain USER n: or ASSISTANT n: labels; labels only mark turn boundaries. "
                f"Write polished turns in {dialogue_language}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Surface-edit this dialogue. Return exactly the same number of labeled turns, using these labels in "
                f"this order: {label_list}\n\n"
                f"Dialogue:\n{_format_messages(messages)}"
            ),
        },
    ]


def _reformat_surface_polish_messages(
    seed_spec: dict[str, Any],
    text: str,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    label_list = ", ".join(
        f"{str(message['role']).upper()} {_role_label_index(messages, position)}:"
        for position, message in enumerate(messages)
    )
    return [
        {
            "role": "system",
            "content": (
                "Restore labels for a surface-edited synthetic dialogue. "
                "Do not change the dialogue substance. Do not combine turns. "
                "Each turn content must be in the visible dialogue language and must not contain other turn labels. "
                f"Visible dialogue language: {dialogue_language}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly the same number of labeled turns, using these labels in this order: "
                f"{label_list}\n\n"
                f"Surface-edited text:\n{text}"
            ),
        },
    ]


def _role_label_index(messages: list[dict[str, str]], position: int) -> int:
    role = messages[position]["role"]
    return sum(1 for message in messages[: position + 1] if message["role"] == role)


def _reformat_numbered_blocks_messages(text: str, *, label: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Reformat the supplied text without adding, removing, or changing substantive content. "
                "Only restore the requested numbered labels."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rewrite this as exactly {count} labeled blocks: {label} 1: through {label} {count}:. "
                f"Keep the {label} labels in English and ASCII. Do not add commentary.\n\n"
                f"Text to reformat:\n{text}"
            ),
        },
    ]


def _review_messages(
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    latent_language = _language_name(str(seed_spec["latent_language"]))
    assistant_turns = _role_count(skeleton, "assistant")
    audit_labels = ", ".join(f"ASSISTANT {index}" for index in range(1, assistant_turns + 1))
    return [
        {
            "role": "system",
            "content": (
                "You run an evidence review and repair gate for a synthetic multi-turn dialogue. "
                "Use separate checks for coherence, naturalness, grounding, source boundaries, recovery, outcome, and format. "
                "Accept only if the conversation preserves hidden state, avoids future leakage, "
                "recovers after late constraints, satisfies the success criteria, and accomplishes useful work for the user. "
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
                f"Write the review artifact in {latent_language}, even if the dialogue is in another language."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this dialogue and produce a compact review artifact. "
                "Use SELECTION only for the review decision: ACCEPT or REPAIR. "
                "Use SCORE for a numeric 0-1 quality score. "
                "Use these lines: SELECTION, SCORE, INTENT_TRAJECTORY, SUCCESS_CRITERIA, "
                "SUBTASK_TRAJECTORY, TURN_EVIDENCE_AUDIT, REVIEWER coherence, REVIEWER naturalness, REVIEWER grounding, "
                "REVIEWER source_boundary, REVIEWER language_quality, REVIEWER recovery, "
                "REVIEWER outcome, REVIEWER format, REPAIR_ACTIONS.\n\n"
                f"TURN_EVIDENCE_AUDIT must contain exactly {assistant_turns} semicolon-separated clauses, one for each "
                f"assistant turn: {audit_labels}. Each clause must start with ASSISTANT n: PASS/FAIL for the matching "
                "turn number. "
                "The hidden blueprint and skeleton are references for expected dialogue state, not evidence available "
                "to the assistant. Do not let an assistant claim pass because the fact appears later in the dialogue, "
                "blueprint, or skeleton. "
                "For each assistant turn, judge whether every user-specific, source-specific, current/live, "
                "legal/jurisdictional, numeric, and private/policy claim is supported by prior visible user text, stable "
                "general knowledge, or explicit illustrative framing. If any claim lacks that support, mark that assistant "
                "turn FAIL and make the grounding or source_boundary reviewer fail.\n\n"
                "Grounding review must be turn-causal: for each assistant turn, reject if it introduces user-specific, "
                "private/internal, current/live, source-applied, or numeric facts without the support required by the "
                "epistemic policy below. Stable public knowledge is allowed, but reject it if the assistant presents "
                "current, official, source-dependent, provider-specific, product-model, named-institution, legal/procedural, safety, "
                "medical, financial, compatibility, benchmark, price, schedule, stock, delivery, warranty, certification, "
                "form, policy, local-market, or availability details as verified rather than general/check-current guidance. Reject precise appeal "
                "windows, complaint bodies, fees, official URLs, rights/entitlements, success likelihood, and likely "
                "outcomes unless prior visible source text supports them or the assistant clearly frames them as "
                "general and check-current. For source-, policy-, form-, or eligibility-grounded "
                "dialogues, reject conclusions that contradict the visible source text or invent private/internal policy "
                "exceptions. Reject foreshadowing when an assistant turn anticipates a later user constraint, preference, "
                "document detail, named option, or source fact before it appears in prior visible user text. For numeric dialogues, independently "
                "check thresholds, exclusions, subtractions, and totals using only visible numbers; reject incorrect arithmetic. "
                "For deadlines, appeal windows, eligibility rules, legal/procedural requirements, and policy-specific "
                "thresholds, reject precise dates or rules unless the controlling visible source supports them; a checklist "
                "or formula using missing-source placeholders is acceptable. For source-dependent tasks in any domain, "
                "reject assistant-filled exact fields such as specs, prices, battery Wh, benchmark hours, schedules, "
                "deadlines, rules, fees, form requirements, weights, ports, certifications, compatibility outcomes, "
                "delivery dates, stock claims, named-provider details, or institution policies when the user has not "
                "supplied them; a blank template, verification checklist, hedged background generalization, or clearly "
                "illustrative placeholder is acceptable.\n\n"
                "Language quality review must reject corrupted text, wrong-language fragments, malformed commands, broken "
                "words, incoherent punctuation artifacts, and unnatural translations that would make the dialogue unsuitable "
                "for training. For Danish output, reject non-Danish code-like fragments inside prose unless the user supplied "
                "them or they are valid commands. Reject assistant turns that use emojis or decorative emoji bullets. "
                "Format review must reject markdown-heavy assistant turns with unnecessary **bold**, *italics*, horizontal "
                "rules, decorative separators, or large heading stacks unless the user explicitly requested that artifact "
                "format. Prefer restrained plain text.\n\n"
                "Naturalness review must also reject semantic oddities in either user or assistant turns: contradictory "
                "or impossible premises, ambiguous wording that materially changes the task, literal translation artifacts, "
                "invented words, and unnatural wording that a native speaker would likely not use. Review conversational "
                "references turn-causally: if a user rejects, corrects, or compares against assistant content that was not "
                "visible earlier, the next assistant should acknowledge the mismatch and treat it as a new constraint or "
                "ask a clarification. Reject the dialogue if the assistant silently goes along with the unsupported reaction. "
                "Reject missing visible artifacts: if a user says they provided, pasted, attached, or showed a draft, "
                "source, job posting, policy, abstract, receipt, error message, code, or similar artifact, but no actual "
                "payload is visible in that turn, the assistant must not rely on it. Reject unsupported quoted edits: if "
                "a user asks to remove, rewrite, or react to a quoted phrase that never appeared in the visible transcript "
                "or a same-turn source payload, the assistant must not act as if that phrase was present.\n\n"
                "Outcome review must reject vague conversations that merely discuss a topic without accomplishing a useful "
                "LLM work session. The dialogue should transform visible material, produce or revise an artifact, run a "
                "visible calculation, explain a supplied item, compare visible options, provide rehearsal feedback, or "
                "create a bounded plan/decision frame/next-step output. For softer support or exploratory seeds, do not "
                "force a formal artifact, but require a concrete output that a real user could use after the conversation. "
                "Reject zero-information first turns when they only ask whether the assistant can help with a topic and "
                "provide no concrete fact, source, draft, number, role/context, named option, prior attempt, or constraint. "
                "Reject generic advice, repeated broad clarification, or final turns that do not satisfy the work-session "
                "contract when the visible information was sufficient to do more. Reject assistant turns that only ask "
                "broad slot-filling questions when they could have also produced a bounded partial draft, immediate safe "
                "step, worked structure, or assumption-labeled first pass.\n\n"
                f"Expected intent trajectory: {seed_spec['intent_trajectory_hint']}\n"
                f"Expected subtask trajectory: {_subtask_trajectory_hint(seed_spec)}\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"Success criteria:\n{_format_success_criteria(seed_spec['success_criteria'])}\n\n"
                f"{_evidence_boundary_instruction(seed_spec)}"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                f"Skeleton:\n{_format_skeleton(skeleton)}\n\n"
                f"Dialogue:\n{_format_messages(messages)}"
            ),
        },
    ]


def _strict_review_messages(
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    latent_language = _language_name(str(seed_spec["latent_language"]))
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    return [
        {
            "role": "system",
            "content": (
                "You are a strict second-pass audit for synthetic multi-turn dialogue data. "
                "The first reviewer may have accepted the row; your job is to look only for reject-worthy failures. "
                "Be stricter than a helpful assistant. Do not reward plausibility when the visible transcript lacks "
                "evidence. Stable public knowledge is allowed, but exact source-dependent details must be hedged, "
                "framed as general memory, or left as fields to verify. "
                "For Danish dialogue, reject malformed Danish, odd compounds, calques, wrong agreement, and invented "
                "words that would be bad training data. "
                f"Write the audit artifact in {latent_language}. The visible dialogue language is {dialogue_language}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Run a strict audit of this dialogue. Use exactly these lines: STRICT_SELECTION, STRICT_SCORE, "
                "STRICT_CHECK source_boundary, STRICT_CHECK language_quality, STRICT_CHECK factuality, "
                "STRICT_CHECK contradiction, STRICT_CHECK format, STRICT_REPAIR_ACTIONS.\n\n"
                "STRICT_SELECTION must be PASS only if every strict check passes. STRICT_SCORE is 0-1. "
                "Each STRICT_CHECK line must begin PASS or FAIL, followed by a brief finding.\n\n"
                "Fail source_boundary if the assistant states exact source-dependent facts without visible source "
                "support or clear hedging. This includes product specs, prices, policies, legal/procedural rules, "
                "health/nutrition/sleep claims, local planning claims, schedules, official labels, forms, fees, "
                "institution/provider details, compatibility outcomes, stock, delivery, benchmarks, or current facts. "
                "The acceptable alternative is a hedged general explanation, a verification checklist, or unknown fields.\n\n"
                "Fail language_quality for unnatural Danish, malformed words, calques, broken grammar, invented "
                "compounds, wrong idioms, tense disagreement, or odd phrasing that a native speaker would not accept "
                "as clean training data.\n\n"
                "Fail factuality for suspicious trivia, made-up explanations, incorrect general claims, or numbers "
                "presented as facts when the row is not source-grounded. Entertainment rows are not exempt: playful "
                "tone cannot license fabricated facts.\n\n"
                "Fail contradiction if later assistant turns contradict earlier advice, user constraints, or their own "
                "uncertainty framing.\n\n"
                "Fail format for markdown emphasis, tables, horizontal rules, decorative separators, emojis, heading "
                "stacks, or overly formatted assistant output unless the user explicitly asked for that exact artifact.\n\n"
                f"Expected intent trajectory: {seed_spec['intent_trajectory_hint']}\n"
                f"Expected subtask trajectory: {_subtask_trajectory_hint(seed_spec)}\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"{_evidence_boundary_instruction(seed_spec)}"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                f"Skeleton:\n{_format_skeleton(skeleton)}\n\n"
                f"Dialogue:\n{_format_messages(messages)}"
            ),
        },
    ]


def _preference_pair_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    hidden = _hidden(row)
    target_language = _language_name(str(row["meta"].get("target_language", row["meta"].get("language", "en"))))
    return [
        {
            "role": "system",
            "content": (
                "You create ACT-style contrastive preference data for multi-turn training. "
                "The rejected answer should be realistic but flawed: missed clarification, stale context, ignored late constraint, "
                "unsupported fact use, contradiction, generic advice instead of the requested work product, missed revision, "
                "or wrong final action. Do not use tool-call failure modes for this pack. "
                f"Write the rejected assistant answer in {target_language}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one rejected final assistant answer for this accepted conversation. "
                "Keep the chosen answer unchanged. "
                "Use these lines: FAILURE_MODE, CHOSEN_ACTION, REJECTED_ACTION, REJECTED_ANSWER.\n\n"
                f"Rejected answer language: {target_language}.\n\n"
                "Work-session contract:\n"
                f"{json.dumps(hidden['blueprint'].get('work_session', {}), indent=2, sort_keys=True)}\n\n"
                f"Hidden blueprint:\n{hidden['blueprint']['artifact']}\n\n"
                f"Dialogue:\n{_format_messages(row['messages'])}"
            ),
        },
    ]


def _parse_skeleton(text: str) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("STEP "):
            continue
        _, _, payload = stripped.partition(":")
        parts = [part.strip() for part in payload.split("|")]
        assert len(parts) >= 3, f"skeleton step must use role | act | state delta: {line}"
        role = parts[0].lower()
        assert role in {"user", "assistant"}, f"unsupported skeleton role: {role}"
        steps.append(
            {
                "role": role,
                "dialogue_act": parts[1],
                "state_delta": " | ".join(parts[2:]),
            }
        )
    return steps


def _parse_intent_model(text: str, seed_spec: dict[str, Any]) -> dict[str, Any]:
    fields = _parse_keyed_lines(text)
    raw_trajectory = fields.get("intent_trajectory", seed_spec["intent_trajectory_hint"]).strip()
    trajectory = _consistentchat_top_level_intent(raw_trajectory, domain=str(seed_spec["domain"]))
    assert trajectory, "intent model must include an intent trajectory"
    subtask_trajectory = fields.get("subtask_trajectory", _subtask_trajectory_hint(seed_spec)).strip()
    contract = _work_session_contract({**seed_spec, "intent_trajectory_hint": trajectory})
    query_plan = fields.get("user_query_plan", "").strip()
    query_plan_steps = _split_query_plan(query_plan)
    if not query_plan_steps:
        query_plan_steps = list(seed_spec.get("query_trajectory_hint", []))
    return {
        "trajectory": trajectory,
        "subtask_trajectory": subtask_trajectory,
        "information_flow": fields.get("information_flow", "").strip(),
        "role_interaction": fields.get("role_interaction", "").strip(),
        "topic_guardrails": fields.get("topic_guardrails", "").strip(),
        "work_session_type": fields.get("work_session_type", contract["work_session_type"]).strip(),
        "deliverable_progress": fields.get("deliverable_progress", "").strip(),
        "user_query_plan": query_plan,
        "user_query_plan_steps": query_plan_steps,
        "artifact": text,
        "method_pattern": "consistentchat_intent_modeling",
    }


def _parse_numbered_blocks(text: str, *, label: str) -> list[str]:
    blocks: list[str] = []
    label_pattern = re.escape(label)
    patterns = [
        re.compile(
            rf"^-?\s*{label_pattern}\s+(?:turn\s+|response\s+)?\d+(?:\s*\([^)]*\))?\s*[:.\-\u2013\u2014]\s*(.*)$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"^-?\s*\d+\s*[\).:\-\u2013\u2014]\s*{label_pattern}(?:\s+turn|\s+response)?\s*[:.\-\u2013\u2014]\s*(.*)$",
            flags=re.IGNORECASE,
        ),
    ]
    for line in text.splitlines():
        stripped = line.strip()
        match = next((candidate for pattern in patterns if (candidate := pattern.match(stripped))), None)
        if match:
            blocks.append(match.group(1).strip())
            continue
        if blocks and stripped:
            blocks[-1] = f"{blocks[-1]}\n{stripped}".strip()
    if blocks:
        return [_clean_numbered_block(block) for block in blocks]

    generic_pattern = re.compile(r"^-?\s*\d+\s*[\).:-]\s*(.*)$")
    for line in text.splitlines():
        stripped = line.strip()
        match = generic_pattern.match(stripped)
        if match:
            blocks.append(match.group(1).strip())
            continue
        if blocks and stripped:
            blocks[-1] = f"{blocks[-1]}\n{stripped}".strip()
    return [_clean_numbered_block(block) for block in blocks]


def _parse_polished_messages(text: str, original: list[dict[str, str]]) -> list[dict[str, str]] | None:
    parsed = _parse_mixed_labeled_messages(text)
    if len(parsed) != len(original):
        return None

    original_roles = [message["role"] for message in original]
    parsed_roles = [message["role"] for message in parsed]
    if parsed_roles != original_roles:
        return None
    if any(_role_label_leaked(message["content"]) for message in parsed):
        return None
    return parsed


def _parse_mixed_labeled_messages(text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"^-?\s*(USER|ASSISTANT)\s+(?:turn\s+|response\s+)?\d+(?:\s*\([^)]*\))?\s*[:.\-\u2013\u2014]\s*(.*)$",
        flags=re.IGNORECASE,
    )
    messages: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        match = pattern.match(stripped)
        if match:
            role = match.group(1).lower()
            current = _message(role, match.group(2).strip())
            messages.append(current)
            continue
        if current is not None and stripped:
            current["content"] = f"{current['content']}\n{stripped}".strip()
    return [
        _message(message["role"], _clean_numbered_block(message["content"]))
        for message in messages
    ]


def _role_label_leaked(text: str) -> bool:
    return re.search(r"\b(?:USER|ASSISTANT)\s+\d+\s*:", text, flags=re.IGNORECASE) is not None


def _clean_numbered_block(text: str) -> str:
    text = re.sub(r"^\s*\.\s+(?=\S)", "", text.strip())
    lines = text.splitlines()
    while lines and lines[0].strip() in {".", "-", "\u2013", "\u2014"}:
        lines.pop(0)
    return "\n".join(lines).strip()


def _parse_review(
    text: str,
    seed_spec: dict[str, Any],
    *,
    threshold: float,
    expected_assistant_turns: int = 3,
) -> dict[str, Any]:
    fields = _parse_keyed_lines(text)
    selection_text = fields.get("selection", "ACCEPT")
    score_text = fields.get("score", "0.0")
    score = _parse_score(score_text)
    success_criteria = _split_success_criteria(fields.get("success_criteria", ""))
    if not success_criteria:
        success_criteria = list(seed_spec["success_criteria"])

    reviewers = []
    for name in [
        "coherence",
        "naturalness",
        "grounding",
        "source_boundary",
        "language_quality",
        "recovery",
        "outcome",
        "format",
    ]:
        raw = fields.get(f"reviewer_{name}", "PASS - no issue found")
        reviewers.append(
            {
                "name": name,
                "passed": _reviewer_passed(raw),
                "finding": _reviewer_finding(raw),
            }
        )

    repair_actions = fields.get("repair_actions", "none").strip()
    turn_evidence_audit = fields.get("turn_evidence_audit", "").strip()
    reviewer_accepted = all(reviewer["passed"] for reviewer in reviewers)
    audit_accepted = _turn_evidence_audit_passed(turn_evidence_audit, expected_turns=expected_assistant_turns)
    parsed_repair_actions = _parse_repair_actions(repair_actions)
    accepted = (
        _review_selection_accepted(selection_text, score=score, threshold=threshold)
        and reviewer_accepted
        and audit_accepted
        and not parsed_repair_actions
    )
    return {
        "review": {
            "reviewers": reviewers,
            "turn_evidence_audit": turn_evidence_audit,
            "repair_actions": parsed_repair_actions,
            "review_decision": "accept" if accepted else "repair",
            "difficulty_axis": seed_spec["difficulty_axis"],
            "artifact": text,
        },
        "verification": {
            "success_criteria": success_criteria,
        },
        "selection": {
            "accepted": accepted,
            "score": score,
            "intent_trajectory": _consistentchat_top_level_intent(
                fields.get("intent_trajectory", seed_spec["intent_trajectory_hint"]).strip(),
                domain=str(seed_spec.get("domain", "")),
            ),
            "subtask_trajectory": fields.get("subtask_trajectory", _subtask_trajectory_hint(seed_spec)).strip(),
            "reasons": [
                reviewer["finding"]
                for reviewer in reviewers
                if reviewer["passed"]
            ],
        },
    }


def _parse_strict_review(text: str, *, threshold: float) -> dict[str, Any]:
    fields = _parse_keyed_lines(text)
    score = _parse_score(fields.get("strict_score", "0.0"))
    checks = []
    for name in ["source_boundary", "language_quality", "factuality", "contradiction", "format"]:
        raw = fields.get(f"strict_check_{name}", "FAIL - missing strict check")
        checks.append(
            {
                "name": name,
                "passed": _reviewer_passed(raw),
                "finding": _reviewer_finding(raw),
            }
        )
    repair_actions = _parse_repair_actions(fields.get("strict_repair_actions", "none").strip())
    accepted = (
        _review_selection_accepted(fields.get("strict_selection", "FAIL"), score=score, threshold=threshold)
        and all(check["passed"] for check in checks)
        and not repair_actions
    )
    return {
        "accepted": accepted,
        "score": score,
        "checks": checks,
        "repair_actions": repair_actions,
        "artifact": text,
    }


def _merge_strict_review(review: dict[str, Any], strict_review: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "review": dict(review["review"]),
        "verification": dict(review["verification"]),
        "selection": dict(review["selection"]),
    }
    reviewers = list(merged["review"].get("reviewers", []))
    reviewers.extend(
        {
            "name": f"strict_{check['name']}",
            "passed": bool(check["passed"]),
            "finding": str(check["finding"]),
        }
        for check in strict_review["checks"]
    )
    merged["review"]["reviewers"] = reviewers
    merged["review"]["strict_audit"] = strict_review
    merged["review"]["repair_actions"] = [
        *list(merged["review"].get("repair_actions", [])),
        *list(strict_review.get("repair_actions", [])),
    ]
    merged["selection"]["score"] = min(float(merged["selection"].get("score", 0.0)), float(strict_review["score"]))
    if not strict_review["accepted"]:
        merged["selection"]["accepted"] = False
        merged["review"]["review_decision"] = "repair"
    return merged


def _apply_local_review_gates(review: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    gates = [
        ("assistant_format_restrained", _assistant_format_restrained({"messages": messages})),
        ("visible_artifact_claims_grounded", _visible_artifact_claims_grounded({"messages": messages})),
        ("reaction_references_grounded", _reaction_references_grounded({"messages": messages})),
    ]
    failures = [name for name, result in gates if not bool(result["passed"])]
    if not failures:
        return review

    merged = {
        "review": dict(review["review"]),
        "verification": dict(review["verification"]),
        "selection": dict(review["selection"]),
    }
    reviewers = list(merged["review"].get("reviewers", []))
    reviewers.extend(
        {
            "name": f"local_{name}",
            "passed": False,
            "finding": f"Local gate failed: {name}",
        }
        for name in failures
    )
    merged["review"]["reviewers"] = reviewers
    merged["review"]["review_decision"] = "repair"
    merged["review"]["repair_actions"] = [
        *list(merged["review"].get("repair_actions", [])),
        *(f"Fix local gate failure: {name}" for name in failures),
    ]
    merged["selection"]["accepted"] = False
    merged["selection"]["score"] = min(float(merged["selection"].get("score", 0.0)), 0.0)
    return merged


def _parse_preference_pair(text: str) -> dict[str, str]:
    fields = _parse_keyed_lines(text)
    rejected = _field_tail(text, "REJECTED_ANSWER").strip()
    if not rejected:
        rejected = fields.get("rejected_answer", "").strip()
    assert rejected, "preference pair must include a rejected answer"
    return {
        "failure_mode": fields.get("failure_mode", "ignored_late_constraint").strip(),
        "chosen_action": fields.get("chosen_action", "revise_with_updated_state").strip(),
        "rejected_action": fields.get("rejected_action", "continue_from_stale_state").strip(),
        "rejected_answer": rejected,
    }


def _parse_score(text: str) -> float:
    lower = text.lower().strip()
    if lower in {"pass", "passed", "accept", "accepted"}:
        return 1.0
    if lower in {"fail", "failed", "reject", "rejected", "repair"}:
        return 0.0

    fraction = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if fraction is not None:
        numerator = float(fraction.group(1))
        denominator = float(fraction.group(2))
        if denominator <= 0:
            return 0.0
        return round(numerator / denominator, 4)

    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return 0.0

    score = float(match.group(0))
    if score > 1.0:
        return round(score / 10.0, 4)
    return score


def _reviewer_passed(text: str) -> bool:
    lower = text.lower()
    stripped = lower.lstrip()
    if stripped.startswith(("fail", "reject")):
        return False
    if stripped.startswith("pass"):
        return True
    lower = lower.replace("no contradictions", "")
    lower = lower.replace("no contradiction", "")
    lower = lower.replace("without contradiction", "")
    lower = lower.replace("does not assert", "")
    lower = lower.replace("does not introduce", "")
    lower = lower.replace("does not imply", "")
    lower = lower.replace("does not claim", "")
    lower = lower.replace("avoids hallucination", "")
    lower = lower.replace("no hallucination", "")
    lower = lower.replace("no unsupported", "")
    lower = lower.replace("without unsupported", "")
    lower = lower.replace("not unsupported", "")
    lower = lower.replace("no invented source", "")
    lower = lower.replace("no source invention", "")
    lower = lower.replace("without invented source", "")
    negative_markers = [
        "fail",
        "reject",
        "missing",
        "does not",
        "doesn't",
        "lacks",
        "contradict",
        "unsupported",
        "hallucinat",
        "future leakage",
        "invented source",
        "source invention",
    ]
    return not any(marker in lower for marker in negative_markers)


def _turn_evidence_audit_passed(text: str, *, expected_turns: int) -> bool:
    if not text.strip():
        return False
    for index in range(1, expected_turns + 1):
        pattern = re.compile(rf"\bASSISTANT\s+{index}\s*:\s*PASS\b", flags=re.IGNORECASE)
        if pattern.search(text) is None:
            return False
    unexpected = re.search(rf"\bASSISTANT\s+{expected_turns + 1}\s*:", text, flags=re.IGNORECASE)
    if unexpected is not None:
        return False
    return "FAIL" not in text.upper()


def _reviewer_finding(text: str) -> str:
    match = re.match(r"^\s*(?:pass|fail|reject)\b\s*[-:\u2013\u2014]?\s*(.*)$", text, flags=re.IGNORECASE)
    if match is None:
        return text.strip()
    return match.group(1).strip() or text.strip()


def _parse_repair_actions(text: str) -> list[str]:
    lower = text.lower().strip(" .")
    if lower in {"", "none", "n/a"}:
        return []
    no_action_markers = [
        "none required",
        "none needed",
        "no repair",
        "no action",
        "not needed",
        "not required",
    ]
    if any(lower.startswith(marker) for marker in no_action_markers):
        return []
    return [text.strip()]


def _review_selection_accepted(text: str, *, score: float, threshold: float) -> bool:
    lower = text.lower()
    if "reject" in lower or "fail" in lower:
        return False
    if "accept" in lower or "accepted" in lower:
        return True
    if "repair" in lower:
        return False
    return score >= threshold


def _parse_keyed_lines(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        normalized = key.strip().lower().replace(" ", "_")
        fields[normalized] = value.strip()
    return fields


def _field_tail(text: str, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*)$", flags=re.IGNORECASE)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = pattern.match(line.strip())
        if not match:
            continue
        first = match.group(1).strip()
        rest = [candidate.strip() for candidate in lines[index + 1 :] if candidate.strip()]
        if rest:
            return "\n".join([first, *rest]).strip()
        return first
    return ""


def _split_success_criteria(text: str) -> list[str]:
    return [
        criterion.strip(" -\t")
        for criterion in re.split(r"\s*(?:;|\n)\s*", text)
        if criterion.strip(" -\t")
    ]


def _split_query_plan(text: str) -> list[str]:
    return [
        item.strip(" -\t")
        for item in re.split(r"\s*(?:->|=>|;|\n)\s*", text)
        if item.strip(" -\t")
    ]


def _role_count(skeleton: list[dict[str, str]], role: str) -> int:
    return sum(1 for step in skeleton if step["role"] == role)


def _target_user_turns(seed_spec: dict[str, Any]) -> int:
    variant = seed_spec.get("scenario_variant", {})
    assert isinstance(variant, dict), "seed scenario_variant must be a mapping"
    target = variant.get("target_user_turns", 3)
    assert isinstance(target, int) and 3 <= target <= 5, "scenario_variant.target_user_turns must be between 3 and 5"
    return target


def _expected_skeleton_roles(seed_spec: dict[str, Any]) -> list[str]:
    roles: list[str] = []
    for _ in range(_target_user_turns(seed_spec)):
        roles.extend(["user", "assistant"])
    return roles


def _format_skeleton(skeleton: list[dict[str, str]]) -> str:
    lines = []
    for index, step in enumerate(skeleton, start=1):
        lines.append(f"STEP {index}: {step['role']} | {step['dialogue_act']} | {step['state_delta']}")
    return "\n".join(lines)


def _format_numbered(items: list[str], *, label: str) -> str:
    return "\n".join(f"{label} {index}: {item}" for index, item in enumerate(items, start=1))


def _assistant_transcript_prefix(
    user_turns: list[str],
    assistant_turns: list[str],
    *,
    assistant_index: int,
) -> str:
    lines = []
    for index in range(assistant_index):
        lines.append(f"USER {index + 1}: {user_turns[index]}")
        lines.append(f"ASSISTANT {index + 1}: {assistant_turns[index]}")
    lines.append(f"USER {assistant_index + 1}: {user_turns[assistant_index]}")
    return "\n".join(lines)


def _dialogue_act_outline(skeleton: list[dict[str, str]]) -> str:
    lines = []
    user_index = 0
    assistant_index = 0
    for step in skeleton:
        if step["role"] == "user":
            user_index += 1
            lines.append(f"USER {user_index}: {step['dialogue_act']}")
            continue
        assistant_index += 1
        lines.append(f"ASSISTANT {assistant_index}: {step['dialogue_act']}")
    return "\n".join(lines)


def _assistant_visible_dialogue_act_outline(
    skeleton: list[dict[str, str]],
    *,
    assistant_index: int,
) -> str:
    lines = []
    user_index = 0
    assistant_count = 0
    current_assistant_number = assistant_index + 1

    for step in skeleton:
        if step["role"] == "user":
            user_index += 1
            lines.append(f"USER {user_index}: {step['dialogue_act']}")
            continue

        assistant_count += 1
        lines.append(f"ASSISTANT {assistant_count}: {step['dialogue_act']}")
        if assistant_count == current_assistant_number:
            break

    return "\n".join(lines)


def _format_query_trajectory(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "- infer from hidden blueprint"
    return "\n".join(f"- {item}" for item in items)


def _format_consistentchat_subtasks() -> str:
    lines = []
    for intent, subtasks in CONSISTENTCHAT_SUBTASKS.items():
        lines.append(f"- {intent}: {', '.join(subtasks)}")
    return "\n".join(lines)


def _subtask_trajectory_hint(seed_spec: dict[str, Any]) -> str:
    value = seed_spec.get("subtask_trajectory_hint")
    if value:
        return str(value)
    return str(seed_spec.get("intent_trajectory_hint", "domain_specific_progression"))


def _format_success_criteria(criteria: object) -> str:
    if not isinstance(criteria, list) or not criteria:
        return "- preserve coherent task progress and avoid unsupported assumptions"
    return "\n".join(f"- {criterion}" for criterion in criteria)


def _format_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    user_index = 0
    assistant_index = 0
    for message in messages:
        role = message["role"]
        if role == "user":
            user_index += 1
            lines.append(f"USER {user_index}: {message['content']}")
            continue
        assistant_index += 1
        lines.append(f"ASSISTANT {assistant_index}: {message['content']}")
    return "\n".join(lines)


def _seed_specs(generation: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    languages = _generation_languages(generation)
    latent_language = _language_code(generation.get("latent_language", "en"), label="generation.latent_language")
    templates = _templates(generation)
    variant_count = _seed_expansion_count(generation)
    for variant_index in range(variant_count):
        scenario_variant = _scenario_variant(variant_index, variant_count=variant_count)
        for template in templates:
            for language in languages:
                seed_blueprint = template["blueprint"]
                archetype_id = _template_id(template)
                seed_id = f"{archetype_id}::v{variant_index:02d}"
                spec = {
                    "seed_id": seed_id,
                    "archetype_id": archetype_id,
                    "scenario_variant": scenario_variant,
                    "scenario_variant_index": variant_index,
                    "scenario_variant_count": variant_count,
                    "family": template["family"],
                    "domain": seed_blueprint["domain"],
                    "seed_intent": seed_blueprint["hidden_goal"],
                    "seed_blueprint": seed_blueprint,
                    "success_criteria": list(template["success_criteria"]),
                    "intent_trajectory_hint": template["intent_trajectory"],
                    "subtask_trajectory_hint": template["subtask_trajectory"],
                    "query_trajectory_hint": list(template.get("query_trajectory_hint", [])),
                    "source_methods": template["source_methods"],
                    "source_language": _language_code(
                        template.get("source_language", language),
                        label="seed source_language",
                    ),
                    "difficulty_axis": template["difficulty_axis"],
                    "requires_epistemic_hedging": bool(template.get("requires_epistemic_hedging")),
                    "volatile_items": list(template.get("volatile_items", [])),
                    "dialogue_language": language,
                    "latent_language": latent_language,
                }
                spec["work_session"] = _work_session_contract(spec)
                specs.append(spec)
    return specs


def _seed_expansion_count(generation: dict[str, Any]) -> int:
    config = generation.get("seed_expansion", {})
    if config is None:
        return 1
    assert isinstance(config, dict), "generation.seed_expansion must be a mapping"
    count = config.get("variants_per_seed", 1)
    assert isinstance(count, int) and count > 0, "generation.seed_expansion.variants_per_seed must be a positive integer"
    return count


def _scenario_variant(index: int, *, variant_count: int) -> dict[str, Any]:
    variant = dict(SEED_EXPANSION_VARIANTS[index % len(SEED_EXPANSION_VARIANTS)])
    variant["index"] = index
    variant["variant_count"] = variant_count
    return variant


def _template_id(template: dict[str, Any]) -> str:
    value = template.get("id")
    if value:
        return str(value)
    blueprint = template["blueprint"]
    return str(blueprint["domain"])


def _consistentchat_top_level_intent(subtask_trajectory: str, *, domain: str) -> str:
    if subtask_trajectory in CONSISTENTCHAT_INTENT_NAMES:
        return subtask_trajectory

    domain_mapping = {
        "budget_narrative_revision": "Transaction Interaction",
        "cooking_advice": "Problem Solving Interaction",
        "debt_payoff_cashflow_plan": "Problem Solving Interaction",
        "essay_revision": "Transaction Interaction",
        "event_planning": "Problem Solving Interaction",
        "home_network_troubleshooting": "Problem Solving Interaction",
        "hr_policy_parental_leave": "Information Retrieval Interaction",
        "immigration_form_checklist": "Transaction Interaction",
        "insurance_claim_explanation": "Information Retrieval Interaction",
        "language_learning": "Educational Interaction",
        "legal_form_assistance": "Transaction Interaction",
        "meal_planning": "Health Consultation Interaction",
        "medication_advice": "Health Consultation Interaction",
        "meeting_notes_summarization": "Transaction Interaction",
        "moving_planning": "Problem Solving Interaction",
        "personal_finance_budgeting": "Problem Solving Interaction",
        "policy_grounded_reimbursement": "Information Retrieval Interaction",
        "product_comparison": "Information Retrieval Interaction",
        "proposal_revision": "Transaction Interaction",
        "resume_revision": "Transaction Interaction",
        "shopping_return_policy": "Transaction Interaction",
        "software_debugging": "Problem Solving Interaction",
        "statistics_tutoring": "Educational Interaction",
        "subscription_cancellation": "Transaction Interaction",
        "tax_planning": "Information Retrieval Interaction",
        "tenant_email_revision": "Transaction Interaction",
        "travel_planning": "Problem Solving Interaction",
    }
    if domain in domain_mapping:
        return domain_mapping[domain]

    subtask_mapping = {
        "advisory_interaction": "Problem Solving Interaction",
        "comparison_interaction": "Information Retrieval Interaction",
        "critique_interaction": "Information Retrieval Interaction",
        "educational_interaction": "Educational Interaction",
        "planning_interaction": "Problem Solving Interaction",
        "problem_solving_interaction": "Problem Solving Interaction",
        "refinement_interaction": "Transaction Interaction",
        "transactional_interaction": "Transaction Interaction",
        "troubleshooting_interaction": "Problem Solving Interaction",
    }
    return subtask_mapping.get(subtask_trajectory, "Problem Solving Interaction")


def _format_scenario_variant(seed_spec: dict[str, Any]) -> str:
    variant = seed_spec.get("scenario_variant", _scenario_variant(0, variant_count=1))
    return json.dumps(variant, indent=2, sort_keys=True)


def _format_scenario_instance(seed_spec: dict[str, Any]) -> str:
    instance = seed_spec.get(
        "scenario_instance",
        {
            "id": "template_instance",
            "variation_axis": "use the handwritten seed as the only scenario source",
            "first_turn_bias": "follow the scenario expansion variant",
        },
    )
    return json.dumps(instance, indent=2, sort_keys=True)


def _generation_languages(generation: dict[str, Any]) -> list[str]:
    values = generation.get("languages", ["en"])
    assert isinstance(values, list) and values, "generation.languages must be a non-empty list"
    return [_language_code(value, label="generation.languages item") for value in values]


def _language_code(value: object, *, label: str) -> str:
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    assert value in LANGUAGE_NAMES, f"unsupported {label}: {value}"
    return value


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES[_language_code(code, label="language code")]


def _language_mode(
    *,
    source: str,
    prompt: str,
    reasoning: str,
    target: str,
) -> str:
    if source == prompt == reasoning == target:
        return "same_language"
    return "cross_language"


def _evidence_boundary(seed_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "available_information": "visible conversation, visible source excerpts in the conversation, and stable general background knowledge",
        "generation_hidden_state": "hidden blueprint facts are for data generation only and are not assistant-visible evidence",
        "unavailable_information": [
            "hidden blueprint facts not revealed in the visible conversation",
            "live web data",
            "current prices",
            "real schedules",
            "inventory",
            "booking status",
            "private data not shown in the conversation",
        ],
        "requires_hedging": bool(seed_spec.get("requires_epistemic_hedging")),
        "volatile_items": list(seed_spec.get("volatile_items", [])),
        "dialogue_language": seed_spec["dialogue_language"],
    }


def _epistemic_policy_instruction() -> str:
    return (
        "Epistemic policy: stable public and general knowledge, including broadly known public procedures, may be "
        "used as background knowledge, but prefer careful epistemics throughout. Keep public knowledge high-level "
        "unless the visible conversation supplies the controlling source. Do not sound as if background knowledge was "
        "freshly verified, complete, current, official, or guaranteed; use qualifiers such as generally, often, may, "
        "or check the current official/source-specific details when precision matters. Do not present details whose "
        "correctness depends on current or live data, a specific provider, a named institution, an official current "
        "source, a jurisdiction, a private organization, the user's personal situation, or a pasted/private source as "
        "verified unless that evidence is visible in the conversation. For those source-dependent details, use general "
        "framing, say that details may vary, ask for the relevant source when needed, or label numbers and examples as "
        "illustrative placeholders. Private, internal, user-specific, organization-specific, company-policy, personal, "
        "and pasted-source facts require visible user or source evidence before being applied. When the user provides "
        "a source, policy, form, or excerpt, treat that visible text as controlling evidence for the task; do not "
        "contradict it or invent exceptions. Calculations must use visible numbers, state assumptions, and separate "
        "estimates from known values. Across domains, exact facts that normally come from a current or controlling "
        "source are source-dependent: product or software specs, benchmark measurements, compatibility guarantees, "
        "policy thresholds, fees, prices, schedules, delivery dates, stock, warranty terms, legal/procedural details, "
        "named-organization rules, official labels, form requirements, local market ranges, and provider-specific "
        "availability. The assistant may use broad public knowledge, but should present it as background memory rather "
        "than verified lookup, hedge uncertainty, and say what should be checked in the current source when precision "
        "matters or lookup would normally be needed. For deadlines, appeal windows, fees, complaint bodies, eligibility rules, "
        "legal/procedural requirements, rights or entitlements, success likelihood, likely outcomes, official URLs, "
        "or policy-specific thresholds, prefer a source-bounded checklist or formula using the visible dates and source "
        "text; do not invent receipt dates, official time limits, free/paid status, institution names, "
        "jurisdiction-specific rules, or predictions of success as if verified. If the user names an institution, procedure, product, form, or "
        "office in a way that sounds odd or possibly wrong, do not silently normalize it into a precise official fact; "
        "treat the name as user-provided wording and ask for or suggest checking the official source. "
        "Do not hedge purely visible facts or arithmetic so much that the answer becomes "
        "unhelpful.\n\n"
    )


def _discourse_quality_instruction() -> str:
    return (
        "Discourse-quality policy: keep the visible dialogue semantically coherent and idiomatic. User turns should "
        "sound like plausible human requests, not literal translations of the hidden blueprint. If a visible user turn "
        "contains odd wording, a likely typo or translation artifact, a contradictory premise, or an ambiguity that "
        "materially changes the task, the assistant should recover by asking a concise clarification, stating a "
        "reasonable assumption, or gently reframing the request instead of silently going along with the oddity. If the "
        "user appears to reject, correct, or compare against assistant content that is not present in the prior visible "
        "transcript, the assistant should not pretend that content was present; it should acknowledge the mismatch and "
        "treat the turn as a new constraint or ask for clarification. If a user says they have pasted, attached, or "
        "provided a draft, source, excerpt, policy, job posting, error message, or other artifact, the relevant text "
        "must be visible in the conversation before the assistant relies on it; otherwise the assistant should ask for "
        "the missing artifact. Danish "
        "text should use natural, ordinary Danish phrasing and avoid invented words, awkward compounds, and calques. "
        "Do not mix English words into Danish prose unless they are code, product names, quoted source text, or common "
        "established loanwords. For technical Danish, use established terms or keep the standard English term when that "
        "is what Danish speakers would normally use; do not invent literal compounds, hyphenated participles, or "
        "word-for-word translations of English technical phrases. Danish adjectives and noun phrases should agree "
        "naturally in gender and number. "
        "For localized dialogues, do not invent official product, operating-system, legal, form, policy, or UI labels. "
        "If an official localized term is uncertain, use a generic description or say the label may vary instead of "
        "writing a precise-sounding translation. Assistant turns should not use emojis or decorative emoji bullets; "
        "use plain text headings, bullets, or numbering instead. Assistant turns should use restrained plain-text "
        "formatting: short paragraphs, simple bullets, or simple numbering when useful. Avoid markdown emphasis such "
        "as **bold** or *italics*, horizontal rules, decorative separators, and long stacked heading structures unless "
        "the user explicitly asks for a formatted artifact that requires them.\n\n"
    )


def _work_session_quality_instruction() -> str:
    return (
        "Work-session policy: optimize for conversations that look like real language-model usage, not generic "
        "back-and-forth about a topic. When enough visible information exists, the assistant should produce useful "
        "work immediately: a draft, revision, calculation, comparison, diagnosis, explanation, checklist, plan, "
        "decision frame, or next-step script. When information is missing, ask only the next necessary question and "
        "still provide bounded partial value when possible, such as a reusable structure, assumption-labeled first "
        "pass, safe immediate step, or explanation of what can be done with the missing input. Avoid repeated broad "
        "clarification turns. Later user turns should usually add concrete material, correction, constraint, or "
        "preference that causes a meaningful revision. First user turns should normally include enough concrete "
        "payload to start useful work; avoid zero-information openings that only ask whether help is possible. "
        "The final assistant turn should normally leave a usable "
        "artifact, answer, plan, or bounded next step.\n\n"
    )


def _evidence_boundary_instruction(seed_spec: dict[str, Any]) -> str:
    _language_code(seed_spec["dialogue_language"], label="seed dialogue_language")
    return (
        "Evidence-boundary requirement: the hidden blueprint is generator state, not assistant evidence. "
        f"{_epistemic_policy_instruction()}"
    )


def _templates(generation: dict[str, Any]) -> list[dict[str, Any]]:
    sources = generation.get("seed_sources")
    if not sources:
        return _builtin_templates()

    templates: list[dict[str, Any]] = []
    for source in sources:
        assert isinstance(source, dict), "seed source entries must be mappings"
        source_type = source.get("type")
        assert source_type == "handwritten", f"unsupported multi_turn_dialogue seed source: {source_type}"
        templates.extend(_load_handwritten_templates(source))
    assert templates, "handwritten seed sources must contain at least one seed"
    return templates


def _load_handwritten_templates(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_seed_source_path(str(source["path"]))
    document = read_yaml(path)
    assert isinstance(document, dict), "handwritten seed source must be a mapping"
    seeds = document.get("seeds")
    assert isinstance(seeds, list), "handwritten seed source must contain a seeds list"
    return [_normalize_handwritten_seed(seed) for seed in seeds]


def _resolve_seed_source_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    pack_relative = PACK_DIR / candidate
    if pack_relative.exists():
        return pack_relative
    return candidate.resolve()


def _normalize_handwritten_seed(seed: object) -> dict[str, Any]:
    assert isinstance(seed, dict), "handwritten seed must be a mapping"
    blueprint = seed.get("blueprint")
    assert isinstance(blueprint, dict), "handwritten seed must include a blueprint mapping"
    success_criteria = blueprint.get("success_criteria", [])
    assert isinstance(success_criteria, list) and success_criteria, "handwritten seed blueprint must include success_criteria"
    intent_trajectory = str(seed["intent_trajectory"])
    subtask_trajectory = str(seed.get("subtask_trajectory", intent_trajectory))
    return {
        "id": str(seed["id"]),
        "family": str(seed["family"]),
        "intent_trajectory": _consistentchat_top_level_intent(intent_trajectory, domain=str(seed["domain"])),
        "subtask_trajectory": subtask_trajectory,
        "query_trajectory_hint": [str(item) for item in seed.get("query_trajectory_hint", [])],
        "blueprint": blueprint,
        "success_criteria": [str(criterion) for criterion in success_criteria],
        "source_methods": [str(method) for method in seed.get("source_methods", DEFAULT_SOURCE_METHODS)],
        "source_language": str(seed.get("source_language", "en")),
        "difficulty_axis": str(seed["difficulty_axis"]),
        "requires_epistemic_hedging": bool(seed.get("requires_epistemic_hedging", False)),
        "volatile_items": [str(item) for item in seed.get("volatile_items", [])],
    }


def _builtin_templates() -> list[dict[str, Any]]:
    return [
        _travel_template(),
        _network_template(),
        _policy_template(),
        _proposal_template(),
    ]


def _travel_template() -> dict[str, Any]:
    blueprint = {
        "domain": "travel_planning",
        "user_persona": "busy parent planning a short trip with two children",
        "hidden_goal": "choose a 4-day low-cost Lisbon itinerary",
        "known_constraints": {
            "budget": "$1800 total",
            "children_ages": [6, 10],
        },
        "unrevealed_constraints": [
            "one child has a peanut allergy",
            "avoid late-night flights",
        ],
        "available_facts": [
            "Lisbon has walkable central neighborhoods and family-friendly museums.",
            "Many traditional desserts and sauces can contain nuts or cross-contact risk.",
        ],
        "conversation_flow": [
            "vague initial request",
            "assistant asks clarifying question",
            "user reveals budget and children ages",
            "assistant proposes a first plan",
            "user adds allergy and flight-time constraints",
            "assistant revises safely",
        ],
        "success_criteria": [
            "fits the $1800 budget",
            "mentions peanut allergy precautions",
            "avoids late-night flights",
            "frames costs and flight times as estimates or assumptions, not verified live data",
            "ends with a concise itinerary",
        ],
    }
    skeleton = [
        _step("user", "vague_initial_request", "introduce trip planning without slots"),
        _step("assistant", "clarify_missing_slots", "ask for budget, travelers, dates, and constraints"),
        _step("user", "reveal_known_constraints", "add budget and children ages"),
        _step("assistant", "draft_candidate_plan", "propose a first feasible plan"),
        _step("user", "late_constraint", "add allergy and flight-time constraints"),
        _step("assistant", "revise_final_answer", "revise plan using updated hidden state"),
    ]
    messages = [
        _message("user", "Can you help me plan a quick Lisbon trip for my family?"),
        _message("assistant", "What budget, trip length, traveler ages, and hard constraints should I plan around?"),
        _message("user", "Four days, two kids aged 6 and 10, and about $1800 total if possible."),
        _message(
            "assistant",
            "A low-cost version could use central lodging, transit passes, free viewpoints, and one paid museum day.",
        ),
        _message("user", "I forgot one important thing: one child has a peanut allergy, and we cannot take late-night flights."),
        _message(
            "assistant",
            "Use Lisbon as the base, keep flights daytime only, and reserve roughly $1800 across airfare, lodging, transit, and meals. For the four days: day 1 arrive before evening and settle near Baixa or Chiado; day 2 do Alfama, the castle exterior, and riverfront time; day 3 choose one kid-friendly museum plus a park; day 4 keep a short morning walk before the daytime return. For food, flag the peanut allergy in writing, avoid bakeries or sauces where nut cross-contact is unclear, and keep backup grocery options close to the hotel.",
        ),
    ]
    return _template(
        family="general_chat",
        intent_trajectory="clarify_reveal_revise_finalize",
        blueprint=blueprint,
        skeleton=skeleton,
        messages=messages,
        source_methods=[
            "APIGen-MT",
            "ConsistentChat",
            "Action-Based Contrastive Self-Training",
            "MDS",
        ],
        difficulty_axis="late safety constraint",
        selection_score=0.96,
        requires_epistemic_hedging=True,
        volatile_items=[
            "current flight prices",
            "real flight schedules",
            "hotel inventory",
            "restaurant availability",
            "booking status",
        ],
    )


def _network_template() -> dict[str, Any]:
    blueprint = {
        "domain": "home_network_troubleshooting",
        "user_persona": "remote worker with a time-sensitive video call",
        "hidden_goal": "restore stable Wi-Fi for a work call in 30 minutes",
        "known_constraints": {
            "device": "laptop",
            "deadline": "30 minutes",
        },
        "unrevealed_constraints": [
            "phone Wi-Fi works",
            "laptop recently joined a VPN",
        ],
        "available_facts": [
            "If one device fails while others work, device configuration is more likely than router failure.",
            "VPN clients and DNS settings can make Wi-Fi appear connected but unusable.",
        ],
        "conversation_flow": [
            "vague problem",
            "assistant triages scope",
            "user reveals device-specific symptoms",
            "assistant gives quick checks",
            "user reveals VPN clue",
            "assistant revises toward device/VPN fix",
        ],
        "success_criteria": [
            "does not recommend router replacement",
            "prioritizes quick checks",
            "uses phone-versus-laptop contrast",
            "mentions VPN or DNS",
        ],
    }
    skeleton = [
        _step("user", "vague_initial_request", "report Wi-Fi failure"),
        _step("assistant", "clarify_scope", "ask which devices fail and what changed"),
        _step("user", "reveal_observation", "phone works but laptop fails"),
        _step("assistant", "propose_diagnostic_steps", "give quick isolation checks"),
        _step("user", "correction_or_new_clue", "add recent VPN change"),
        _step("assistant", "revise_final_answer", "switch from router diagnosis to laptop VPN/DNS fix"),
    ]
    messages = [
        _message("user", "My Wi-Fi is broken and I have a work call soon. What should I do?"),
        _message("assistant", "Is every device offline, or only one device? Also, did anything change today?"),
        _message("user", "My phone works on Wi-Fi, but my laptop says connected and pages do not load."),
        _message(
            "assistant",
            "Since the phone works, first toggle laptop Wi-Fi, forget and rejoin the network, and test one known site.",
        ),
        _message("user", "I did connect to a new VPN client this morning."),
        _message(
            "assistant",
            "Because the phone still works, do not replace the router. For the next 30 minutes, disconnect the VPN, disable any custom DNS it added, renew the laptop network connection, and test a browser page before the call. If that works, join the meeting without VPN; if it does not, use the phone hotspot as the fallback and inspect the VPN client after the call.",
        ),
    ]
    return _template(
        family="general_chat",
        intent_trajectory="triage_observe_correct_finalize",
        blueprint=blueprint,
        skeleton=skeleton,
        messages=messages,
        source_methods=[
            "APIGen-MT",
            "ConsistentChat",
            "Action-Based Contrastive Self-Training",
            "MDS",
        ],
        difficulty_axis="wrong-assumption recovery",
        selection_score=0.94,
    )


def _policy_template() -> dict[str, Any]:
    blueprint = {
        "domain": "policy_grounded_reimbursement",
        "user_persona": "employee submitting an expense report after a short trip",
        "hidden_goal": "know which travel expenses can be reimbursed",
        "known_constraints": {
            "trip": "one-night client visit",
            "receipt_status": "some receipts missing",
        },
        "unrevealed_constraints": [
            "hotel breakfast was included",
            "taxi receipt is missing",
        ],
        "available_facts": [
            "Meals are reimbursable up to $65 per day when itemized receipts are attached.",
            "A taxi can be reimbursed without a receipt only when the amount is under $25 and the route is documented.",
            "Meals included in lodging cannot be claimed again as a separate meal expense.",
        ],
        "assistant_visible_context_requirement": (
            "The reimbursement policy facts are private organizational information. "
            "The user must provide the relevant policy excerpt before the assistant can apply these rules."
        ),
        "conversation_flow": [
            "vague reimbursement question with pasted policy excerpt",
            "assistant asks for expense types and receipts",
            "user reveals meal, taxi, and lodging details",
            "assistant grounds answer in policy facts",
            "user adds included-breakfast detail",
            "assistant revises to avoid double-claiming",
        ],
        "success_criteria": [
            "uses only the provided policy facts",
            "mentions $65 daily meal cap",
            "handles missing taxi receipt under $25",
            "excludes hotel breakfast from separate meal claim",
        ],
    }
    skeleton = [
        _step("user", "underspecified_policy_query_with_source", "ask if expenses are reimbursable and provide the policy excerpt"),
        _step("assistant", "clarify_missing_slots", "ask for categories, amounts, and receipt status"),
        _step("user", "reveal_facts", "provide meal, taxi, hotel, and receipt details"),
        _step("assistant", "grounded_answer", "answer against available policy facts"),
        _step("user", "late_constraint", "add breakfast-included detail"),
        _step("assistant", "revise_final_answer", "revise grounded answer with no double claim"),
    ]
    messages = [
        _message(
            "user",
            (
                "Can I expense my client-trip costs? I have this policy excerpt: meals are reimbursable up to "
                "$65 per day with itemized receipts; taxis without receipts are allowed only under $25 if the route "
                "is documented; meals included in lodging cannot be claimed separately."
            ),
        ),
        _message("assistant", "Which costs, amounts, and receipts do you have?"),
        _message("user", "One dinner was $48 with a receipt, the taxi was $18 but I lost the receipt, and the hotel receipt is itemized."),
        _message(
            "assistant",
            "Based on the policy facts, the $48 dinner can fit under the $65 daily meal cap with its itemized receipt, and the $18 taxi can be reimbursed if you document the route.",
        ),
        _message("user", "The hotel also included breakfast, so I was going to add that as a meal too."),
        _message(
            "assistant",
            "Do not claim the hotel breakfast as a separate meal because meals included in lodging cannot be claimed again. The $48 dinner remains eligible under the $65 per-day meal cap with the itemized receipt. The $18 taxi can be reimbursed without a receipt because it is under $25, but document the route. Keep the itemized hotel receipt with the report.",
        ),
    ]
    return _template(
        family="grounded_dialogue",
        intent_trajectory="clarify_ground_revise_finalize",
        blueprint=blueprint,
        skeleton=skeleton,
        messages=messages,
        source_methods=[
            "APIGen-MT",
            "ConsistentChat",
            "MDS",
        ],
        difficulty_axis="document-grounded late constraint",
        selection_score=0.97,
    )


def _proposal_template() -> dict[str, Any]:
    blueprint = {
        "domain": "proposal_revision",
        "user_persona": "nonprofit program lead preparing a short grant update",
        "hidden_goal": "turn a rough update into a funder-ready paragraph",
        "known_constraints": {
            "tone": "plain and credible",
            "length": "short paragraph",
        },
        "unrevealed_constraints": [
            "avoid promising outcomes not yet measured",
            "mention the rural clinics first",
        ],
        "available_facts": [
            "The pilot trained 18 clinic staff.",
            "Three rural clinics started using the intake checklist.",
            "Outcome data will be reviewed next quarter.",
        ],
        "conversation_flow": [
            "rough editing request",
            "assistant asks audience and proof level",
            "user reveals funder audience and facts",
            "assistant drafts a paragraph",
            "user adds caution and ordering constraints",
            "assistant revises with grounded claims",
        ],
        "success_criteria": [
            "mentions three rural clinics before staff count",
            "says outcome data is pending",
            "does not overclaim impact",
            "keeps the response concise",
        ],
    }
    skeleton = [
        _step("user", "vague_edit_request", "ask to improve update"),
        _step("assistant", "clarify_audience", "ask audience, tone, and evidence"),
        _step("user", "reveal_source_facts", "provide facts and target audience"),
        _step("assistant", "draft_candidate_answer", "write first paragraph"),
        _step("user", "late_constraint", "add order and no-overclaim constraint"),
        _step("assistant", "revise_final_answer", "revise with fact ordering and uncertainty"),
    ]
    messages = [
        _message("user", "Can you make this program update sound better?"),
        _message("assistant", "Who is the audience, and which facts are already safe to claim?"),
        _message(
            "user",
            "It is for a funder. We trained 18 clinic staff, three rural clinics started using the intake checklist, and outcomes will be reviewed next quarter.",
        ),
        _message(
            "assistant",
            "Our pilot trained 18 clinic staff and introduced a new intake checklist across participating clinics, with outcomes to be reviewed next quarter.",
        ),
        _message("user", "Put the rural clinics first and do not make it sound like we already proved impact."),
        _message(
            "assistant",
            "Three rural clinics have started using the new intake checklist, and 18 clinic staff have been trained to support the pilot. Outcome data will be reviewed next quarter, so this update focuses on implementation progress rather than claiming measured impact.",
        ),
    ]
    return _template(
        family="grounded_dialogue",
        intent_trajectory="edit_clarify_draft_revise",
        blueprint=blueprint,
        skeleton=skeleton,
        messages=messages,
        source_methods=[
            "APIGen-MT",
            "ConsistentChat",
            "MDS",
        ],
        difficulty_axis="grounded revision without overclaiming",
        selection_score=0.95,
    )


def _template(
    *,
    family: str,
    intent_trajectory: str,
    blueprint: dict[str, Any],
    skeleton: list[dict[str, str]],
    messages: list[dict[str, str]],
    source_methods: list[str],
    difficulty_axis: str,
    selection_score: float,
    requires_epistemic_hedging: bool = False,
    volatile_items: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "intent_trajectory": _consistentchat_top_level_intent(intent_trajectory, domain=str(blueprint["domain"])),
        "subtask_trajectory": intent_trajectory,
        "blueprint": blueprint,
        "skeleton": skeleton,
        "messages": messages,
        "source_methods": source_methods,
        "difficulty_axis": difficulty_axis,
        "selection_score": selection_score,
        "requires_epistemic_hedging": requires_epistemic_hedging,
        "volatile_items": volatile_items or [],
        "selection_reasons": [
            "persistent hidden state",
            "incremental user reveal",
            "late-constraint recovery",
            "dialogue-level coherence",
        ],
    }


def _step(role: str, dialogue_act: str, state_delta: str) -> dict[str, str]:
    return {
        "role": role,
        "dialogue_act": dialogue_act,
        "state_delta": state_delta,
    }


def _message(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content,
    }
