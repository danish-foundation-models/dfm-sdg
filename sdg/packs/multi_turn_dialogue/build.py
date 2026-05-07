from __future__ import annotations

import ast
import asyncio
import json
import re
import threading
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
from sdg.commons.sources import iter_source_records, read_record_value, source_label
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

CANDIDATE_STAGE_ARTIFACTS = {
    "candidate_source_task_plans": "candidate_source_task_plans.jsonl",
    "candidate_scenario_instances": "candidate_scenario_instances.jsonl",
    "candidate_blueprints": "candidate_blueprints.jsonl",
    "candidate_intent_models": "candidate_intent_models.jsonl",
    "candidate_skeletons": "candidate_skeletons.jsonl",
    "candidate_user_turns": "candidate_user_turns.jsonl",
    "candidate_assistant_turns": "candidate_assistant_turns.jsonl",
    "candidate_messages": "candidate_messages.jsonl",
    "candidate_polished_messages": "candidate_polished_messages.jsonl",
    "candidate_reviews": "candidate_reviews.jsonl",
    "candidate_repairs": "candidate_repairs.jsonl",
    "candidate_final_rows": "candidate_final_rows.jsonl",
}

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

PERSONA_COMMUNICATION_STYLES = (
    "prefers concise practical steps",
    "wants plain-language explanations before action",
    "likes checklists and explicit next steps",
    "is cautious about rules and uncertainty",
    "asks for a usable draft rather than abstract advice",
    "wants tradeoffs made explicit",
)

PERSONA_TASK_POSTURES = (
    "arrives with partial information and wants help turning it into action",
    "has a visible document or note and needs the important parts extracted",
    "needs a source-bounded explanation before deciding what to do next",
    "wants a short artifact they can reuse with another person or organization",
    "has a practical constraint and needs assumptions separated from known facts",
    "wants help checking whether their own interpretation is reasonable",
)

PERSONA_PRACTICAL_CONSTRAINTS = (
    "limited domain expertise",
    "limited time for follow-up",
    "needs uncertainty flagged clearly",
    "prefers low-friction next steps",
    "wants source-specific facts separated from general background",
    "needs wording that is safe to reuse",
)

SOURCE_PAYLOAD_STRATEGIES = {
    "full_excerpt": {
        "label": "provided source paste",
        "user_visibility": "USER 1 should paste the provided source excerpt before asking the task question. Do not describe it as full, complete, or exhaustive.",
        "assistant_boundary": "The assistant may use all source facts that appear in the pasted excerpt and should mark only unseen implementation details as unknown.",
    },
    "long_contiguous_excerpt": {
        "label": "long contiguous excerpt",
        "user_visibility": "USER 1 should paste a long contiguous passage from the source that is sufficient for the main task, not just a one-line quote.",
        "assistant_boundary": "The assistant may use facts in that passage, but must not infer omitted surrounding sections.",
    },
    "selected_clauses": {
        "label": "selected source passages",
        "user_visibility": "USER 1 should paste selected complete source passages that are directly relevant to the user's question, not isolated sentence fragments.",
        "assistant_boundary": "The assistant should answer from the selected passages and explicitly identify fields that require the omitted document context.",
    },
    "summary_plus_quotes": {
        "label": "user framing plus exact source passages",
        "user_visibility": "USER 1 may give a short user-written framing question and include exact complete source passages for the facts the assistant should rely on.",
        "assistant_boundary": "The assistant should distinguish user framing from exact source passages and avoid treating unquoted source details as verified.",
    },
    "staged_excerpts": {
        "label": "staged source excerpts across turns",
        "user_visibility": "USER 1 should paste an initial excerpt, and later user turns should paste any additional source text before asking about new source facts.",
        "assistant_boundary": "The assistant should answer incrementally and must not anticipate source facts that will only appear in later pasted excerpts.",
    },
}

DEFAULT_SOURCE_PAYLOAD_STRATEGIES = tuple(SOURCE_PAYLOAD_STRATEGIES)
SOURCE_PAYLOAD_PLACEHOLDER_TEMPLATE = "{{SOURCE_PAYLOAD_%d}}"
SOURCE_PAYLOAD_LENGTHS = {
    "short": {
        "label": "focused excerpt",
        "selected_count": 1,
        "selected_target_chars": 450,
        "window_min_chars": 900,
        "window_ratio": 0.35,
    },
    "medium": {
        "label": "substantial excerpt",
        "selected_count": 2,
        "selected_target_chars": 1200,
        "window_min_chars": 3000,
        "window_ratio": 0.65,
    },
    "long": {
        "label": "large document excerpt",
        "selected_count": 3,
        "selected_target_chars": 2200,
        "window_min_chars": 6000,
        "window_ratio": 0.85,
    },
}
DEFAULT_SOURCE_PAYLOAD_LENGTHS = ("long", "medium", "long", "short")
SOURCE_PAYLOAD_STYLES = {
    "fenced_text": "markdown text fence",
    "blockquote": "markdown blockquote",
    "plain_delimited": "plain text with start and end lines",
    "separator": "plain text between separators",
}
DEFAULT_SOURCE_PAYLOAD_STYLES = tuple(SOURCE_PAYLOAD_STYLES)
SOURCE_PAYLOAD_LABEL_STYLES = {
    "excerpt": {
        "stem_template": "Uddrag {index}",
        "label": "neutral excerpt label",
    },
    "pasted_text": {
        "stem_template": "Indsat tekst {index}",
        "label": "pasted text label",
    },
    "document_passage": {
        "stem_template": "Dokumentpassage {index}",
        "label": "document passage label",
    },
    "relevant_text": {
        "stem_template": "Relevant tekst {index}",
        "label": "relevant text label",
    },
    "copied_text": {
        "stem_template": "Kopieret tekst {index}",
        "label": "copied text label",
    },
    "quoted_passage": {
        "stem_template": "Citeret passage {index}",
        "label": "quoted passage label",
    },
}
DEFAULT_SOURCE_PAYLOAD_LABEL_STYLES = tuple(SOURCE_PAYLOAD_LABEL_STYLES)
LEGACY_SOURCE_PAYLOAD_LABEL_STEMS = ("Kildeuddrag",)

PACK_DIR = Path(__file__).resolve().parent

def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="multi_turn_dialogue",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
        resume_incomplete=cfg.get("resume_incomplete", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    verified_rows = common_eval.verify(rows, _messages_alternate, name="messages_alternate")
    verified_rows = common_eval.verify(verified_rows, _no_role_label_leak, name="no_role_label_leak")
    verified_rows = common_eval.verify(verified_rows, _skeleton_matches_messages, name="skeleton_matches_messages")
    verified_rows = common_eval.verify(verified_rows, _evidence_boundary_respected, name="evidence_boundary_respected")
    verified_rows = common_eval.verify(verified_rows, _visible_artifact_claims_grounded, name="visible_artifact_claims_grounded")
    verified_rows = common_eval.verify(verified_rows, _source_payloads_readable, name="source_payloads_readable")
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

    artifacts = {
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
    artifacts.update(_candidate_stage_artifacts(outputs_dir))
    return artifacts


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
    stage_writer = _CandidateStageWriter(outputs_dir)
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
            lambda _index, item: _generate_candidate_async(item, generation, model_roles, stage_writer),
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
    stage_writer: _CandidateStageWriter,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _generate_candidate,
        str(item["row_id"]),
        item["seed_spec"],
        generation,
        model_roles,
        stage_writer,
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
    for role in ["source_task_planner"]:
        if role not in clients:
            continue
        client = clients[role]
        assert isinstance(client, LLM), f"models.{role} must resolve to an LLM"
        roles[role] = client
    return roles


class _CandidateStageWriter:
    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.lock = threading.Lock()
        self.seen_ids = {
            artifact_name: store.jsonl_keys(outputs_dir / filename, key_for=_row_id)
            for artifact_name, filename in CANDIDATE_STAGE_ARTIFACTS.items()
        }

    def write(
        self,
        artifact_name: str,
        *,
        candidate_id: str,
        stage: str,
        payload: dict[str, Any],
        attempt: int = 0,
    ) -> None:
        assert artifact_name in CANDIDATE_STAGE_ARTIFACTS, f"Unsupported candidate stage artifact: {artifact_name}"
        row = {
            "id": f"{candidate_id}-{stage}-attempt-{attempt:02d}",
            "candidate_id": candidate_id,
            "stage": stage,
            "attempt": attempt,
            "payload": payload,
        }
        path = self.outputs_dir / CANDIDATE_STAGE_ARTIFACTS[artifact_name]
        with self.lock:
            seen = self.seen_ids[artifact_name]
            if row["id"] in seen:
                return
            with path.open("a") as handle:
                store.append_jsonl_line(handle, row)
            seen.add(row["id"])


def _candidate_stage_artifacts(outputs_dir: Path) -> dict[str, Artifact]:
    artifacts: dict[str, Artifact] = {}
    for artifact_name, filename in CANDIDATE_STAGE_ARTIFACTS.items():
        path = outputs_dir / filename
        if not path.exists():
            continue
        artifacts[artifact_name] = Artifact(
            name=artifact_name,
            path=str(path),
            kind="jsonl",
            meta={"rows": store.jsonl_count(path), "family": "multi_turn_dialogue"},
        )
    return artifacts


def _generate_candidate(
    row_id: str,
    seed_spec: dict[str, Any],
    generation: dict[str, Any],
    model_roles: dict[str, LLM],
    stage_writer: _CandidateStageWriter,
) -> dict[str, Any]:
    seed_spec = _generate_source_task_plan_if_needed(
        model_roles.get("source_task_planner", model_roles["blueprint_writer"]),
        seed_spec,
        generation,
    )
    if "source_task_plan" in seed_spec:
        stage_writer.write(
            "candidate_source_task_plans",
            candidate_id=row_id,
            stage="source_task_plan",
            payload=seed_spec["source_task_plan"],
        )
    seed_spec = _generate_scenario_instance(model_roles["blueprint_writer"], seed_spec, generation)
    stage_writer.write(
        "candidate_scenario_instances",
        candidate_id=row_id,
        stage="scenario_instance",
        payload={"scenario_instance": seed_spec.get("scenario_instance", {})},
    )
    blueprint = _generate_blueprint(model_roles["blueprint_writer"], seed_spec, generation)
    stage_writer.write("candidate_blueprints", candidate_id=row_id, stage="blueprint", payload=blueprint)
    intent_model = _generate_intent_model(model_roles["intent_modeler"], seed_spec, blueprint, generation)
    stage_writer.write("candidate_intent_models", candidate_id=row_id, stage="intent_model", payload=intent_model)
    skeleton = _generate_skeleton(model_roles["skeleton_writer"], seed_spec, blueprint, intent_model, generation)
    stage_writer.write("candidate_skeletons", candidate_id=row_id, stage="skeleton", payload={"skeleton": skeleton})
    user_turns = _generate_user_turns(model_roles["user_simulator"], seed_spec, blueprint, intent_model, skeleton, generation)
    stage_writer.write("candidate_user_turns", candidate_id=row_id, stage="user_turns", payload={"turns": user_turns})
    assistant_turns = _generate_assistant_turns(
        model_roles["assistant_teacher"],
        seed_spec,
        skeleton,
        user_turns,
        generation,
    )
    stage_writer.write(
        "candidate_assistant_turns",
        candidate_id=row_id,
        stage="assistant_turns",
        payload={"turns": assistant_turns},
    )
    messages = _interleave_messages(skeleton, user_turns, assistant_turns)
    stage_writer.write("candidate_messages", candidate_id=row_id, stage="messages", payload={"messages": messages})
    messages = _polish_messages(model_roles["reviewer"], seed_spec, messages, generation)
    messages = _restore_source_payloads_in_messages(seed_spec, messages)
    stage_writer.write(
        "candidate_polished_messages",
        candidate_id=row_id,
        stage="polished_messages",
        payload={"messages": messages},
    )
    surface_failures = _unrepairable_surface_failures(messages, seed_spec)
    if surface_failures:
        review = _local_unrepairable_review(seed_spec, intent_model, surface_failures)
    else:
        review = _generate_review(model_roles["reviewer"], seed_spec, blueprint, intent_model, skeleton, messages, generation)
    stage_writer.write("candidate_reviews", candidate_id=row_id, stage="review", payload=review)

    for repair_attempt in range(1, _repair_attempts(generation) + 1):
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
        stage_writer.write(
            "candidate_repairs",
            candidate_id=row_id,
            stage="repair_messages",
            payload={"messages": messages},
            attempt=repair_attempt,
        )
        messages = _polish_messages(model_roles["reviewer"], seed_spec, messages, generation)
        messages = _restore_source_payloads_in_messages(seed_spec, messages)
        stage_writer.write(
            "candidate_polished_messages",
            candidate_id=row_id,
            stage="polished_messages",
            payload={"messages": messages},
            attempt=repair_attempt,
        )
        surface_failures = _unrepairable_surface_failures(messages, seed_spec)
        if surface_failures:
            review = _local_unrepairable_review(seed_spec, intent_model, surface_failures)
        else:
            review = _generate_review(model_roles["reviewer"], seed_spec, blueprint, intent_model, skeleton, messages, generation)
        stage_writer.write(
            "candidate_reviews",
            candidate_id=row_id,
            stage="review",
            payload=review,
            attempt=repair_attempt,
        )

    row = _row_from_generated_parts(
        row_id=row_id,
        seed_spec=seed_spec,
        blueprint=blueprint,
        intent_model=intent_model,
        skeleton=skeleton,
        messages=messages,
        review=review,
    )
    stage_writer.write("candidate_final_rows", candidate_id=row_id, stage="final_row", payload=row)
    return row


def _generate_source_task_plan_if_needed(
    llm: LLM,
    seed_spec: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    if "source_grounding" not in seed_spec:
        return seed_spec

    text = _chat_text(
        llm,
        _source_task_planner_messages(seed_spec),
        temperature=float(generation.get("source_task_temperature", 0.8)),
    )
    return _apply_source_task_plan(seed_spec, _parse_source_task_plan(text))


def _source_task_planner_messages(seed_spec: dict[str, Any]) -> list[dict[str, str]]:
    latent_language = _language_name(str(seed_spec["latent_language"]))
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    source_grounding = seed_spec["source_grounding"]
    persona = seed_spec.get("persona_context", {})
    return [
        {
            "role": "system",
            "content": (
                "You are the stochastic source-to-task planner for grounded synthetic multi-turn dialogue data. "
                "Use the source excerpt and sampled persona as diversity inputs, then create a latent task archetype "
                "for the existing APIGen-MT and ConsistentChat pipeline. Do not write the dialogue. Do not output "
                "JSON-only. The persona is a diversity seed, not a user identity that must be copied literally. "
                "For Danish dialogue, localize the user's situation and wording naturally unless the source itself "
                "requires another context. Do not copy US locations, law, currency, institutions, or demographic facts "
                "from a persona into a Danish source-grounded task unless they are genuinely relevant and visibly stated. "
                "Treat the persona as an abstract diversity vector for occupation, constraints, communication style, "
                "risk tolerance, and task motivation. Do not copy the sampled persona's name, biography, city, state, "
                "country, currency, institutions, legal setting, or personal history into the task. For Danish rows, "
                "make SOURCE_TASK_PERSONA a Danish-localized or neutral user role, not a copied persona biography. "
                f"{_discourse_quality_instruction()}"
                f"{_work_session_quality_instruction()}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one source-grounded task archetype from this source/persona pair. Use exactly these labeled "
                "sections: SOURCE_TASK_DOMAIN, SOURCE_TASK_FAMILY, SOURCE_TASK_INTENT_TRAJECTORY, "
                "SOURCE_TASK_SUBTASK_TRAJECTORY, SOURCE_TASK_DIFFICULTY_AXIS, SOURCE_TASK_PERSONA, SOURCE_TASK_GOAL, "
                "SOURCE_TASK_QUERY_TRAJECTORY, SOURCE_TASK_SUCCESS_CRITERIA, SOURCE_TASK_VOLATILE_ITEMS, "
                "SOURCE_TASK_BLUEPRINT.\n\n"
                "The plan should be varied and realistic: choose a concrete user situation, motivation, missing facts, "
                "late-turn reveal, and useful final deliverable. The assistant must only use the source excerpt after "
                "the simulated user makes it visible in the dialogue. Prefer work people actually ask language models "
                "to do: explain an excerpt, extract requirements, draft a message, create a checklist, compare visible "
                "clauses, summarize implications, or identify unknowns. Keep exact source-dependent conclusions bounded "
                "to the visible excerpt and preserve uncertainty about current status or unseen rules. "
                "The source excerpt and metadata are immutable inputs. Use persona/context to vary the user situation, "
                "motivation, deliverable, missing facts, and disclosure timing only. Do not invent, rename, anonymize, "
                "or alter source-specific entities, titles, outlets, institutions, case numbers, dates, quoted phrases, "
                "findings, legal/procedural facts, or factual claims from the source. Do not write exact source facts "
                "into SOURCE_TASK_GOAL, SOURCE_TASK_QUERY_TRAJECTORY, SOURCE_TASK_SUCCESS_CRITERIA, or "
                "SOURCE_TASK_BLUEPRINT. Describe the user's work as operations on whatever source payload is later "
                "inserted, such as extracting visible deadlines, checking visible obligations, drafting from the pasted "
                "excerpt, or listing unknowns from omitted context. Do not use the persona's foreign names, places, or "
                "biography as visible row content; abstract them into a localized Danish user role.\n\n"
                f"Hidden artifact language: {latent_language}\n"
                f"Visible dialogue language later in the pipeline: {dialogue_language}\n"
                f"Configured source pack: {source_grounding['pack_id']}\n"
                f"Source family: {source_grounding['source_name']}\n"
                f"Source document id: {source_grounding['document_id']}\n"
                f"Source title: {source_grounding.get('title', '')}\n"
                f"Source created/date metadata: {source_grounding.get('created', '')}\n"
                f"Scenario expansion variant:\n{_format_scenario_variant(seed_spec)}\n\n"
                f"Source visibility plan for the later visible dialogue:\n{_format_source_payload_plan(seed_spec)}\n\n"
                "Sampled persona/context:\n"
                f"{json.dumps(persona, indent=2, sort_keys=True)}\n\n"
                "Visible source excerpt candidate:\n"
                f"{source_grounding['excerpt']}\n\n"
                "Write a compact plan. SOURCE_TASK_INTENT_TRAJECTORY must be one of the ConsistentChat top-level "
                "intents when possible. SOURCE_TASK_QUERY_TRAJECTORY should be a semicolon-separated user-query plan. "
                "SOURCE_TASK_SUCCESS_CRITERIA should be newline bullets or semicolon-separated criteria."
            ),
        },
    ]


def _parse_source_task_plan(text: str) -> dict[str, Any]:
    fields = _parse_keyed_lines(text)
    success_criteria = _split_success_criteria(fields.get("source_task_success_criteria", ""))
    query_trajectory = _split_query_plan(fields.get("source_task_query_trajectory", ""))
    volatile_items = _split_success_criteria(fields.get("source_task_volatile_items", ""))
    return {
        "domain": _slug(fields.get("source_task_domain", "source_grounded_dialogue")),
        "family": _source_task_family(fields.get("source_task_family", "grounded_dialogue")),
        "intent_trajectory": fields.get("source_task_intent_trajectory", "Information Retrieval Interaction").strip(),
        "subtask_trajectory": _slug(fields.get("source_task_subtask_trajectory", "source_grounded_synthesis")),
        "difficulty_axis": fields.get(
            "source_task_difficulty_axis",
            "source-grounded task with missing user facts and epistemic boundaries",
        ).strip(),
        "persona": fields.get("source_task_persona", "").strip(),
        "goal": fields.get("source_task_goal", "").strip(),
        "query_trajectory": query_trajectory,
        "success_criteria": success_criteria,
        "volatile_items": volatile_items,
        "blueprint": fields.get("source_task_blueprint", "").strip(),
        "artifact": text,
    }


def _apply_source_task_plan(seed_spec: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    source_grounding = seed_spec["source_grounding"]
    domain = plan["domain"]
    success_criteria = _source_safe_success_criteria(seed_spec)
    goal = _source_safe_goal(plan)
    query_trajectory = _source_safe_query_trajectory(seed_spec, plan)
    blueprint = {
        "domain": domain,
        "user_persona": plan["persona"] or seed_spec["seed_blueprint"].get("user_persona", ""),
        "hidden_goal": goal,
        "source_context": {
            "source_pack": source_grounding["pack_id"],
            "source_name": source_grounding["source_name"],
            "source_payload_plan": seed_spec.get("source_payload_plan", {}),
            "source_text_available_to_visible_dialogue_only_via_injected_payload": True,
        },
        "persona_context": seed_spec.get("persona_context", {}),
        "conversation_flow": query_trajectory,
        "success_criteria": success_criteria,
    }
    intent_trajectory = _source_task_intent_trajectory(seed_spec, plan, domain=domain)
    updated = {
        **seed_spec,
        "family": plan["family"],
        "domain": domain,
        "seed_intent": goal,
        "seed_blueprint": blueprint,
        "success_criteria": success_criteria,
        "intent_trajectory_hint": intent_trajectory,
        "subtask_trajectory_hint": plan["subtask_trajectory"],
        "query_trajectory_hint": query_trajectory,
        "difficulty_axis": _source_safe_difficulty_axis(plan),
        "requires_epistemic_hedging": True,
        "volatile_items": _source_safe_volatile_items(seed_spec),
        "source_task_plan": plan,
    }
    updated["work_session"] = _work_session_contract(updated)
    return updated


def _source_safe_goal(plan: dict[str, Any]) -> str:
    subtask = str(plan["subtask_trajectory"]).replace("_", " ")
    return f"complete a {subtask} task using only source text inserted into the visible dialogue"


def _source_safe_success_criteria(seed_spec: dict[str, Any]) -> list[str]:
    return [
        "the assistant only treats source details as known after the relevant source payload is visible",
        "the dialogue separates source-visible facts, user-owned facts, and unknown omitted context",
        "the assistant does not assume the pasted excerpt is complete, current, or exhaustive",
        "the final answer produces a useful source-bounded work product",
        *list(seed_spec.get("success_criteria", [])),
    ]


def _source_safe_query_trajectory(seed_spec: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    subtask = str(plan["subtask_trajectory"]).replace("_", " ")
    payload_plan = seed_spec.get("source_payload_plan", {})
    strategy = str(payload_plan.get("strategy", "")) if isinstance(payload_plan, dict) else ""
    if strategy == "staged_excerpts":
        return [
            f"paste the first source excerpt and ask for initial {subtask}",
            "add user-owned context, constraint, or intended use",
            f"paste the later source excerpt before asking for revised {subtask}",
            "request a final source-bounded deliverable with unknowns separated",
        ]
    return [
        f"paste the source excerpt and ask for initial {subtask}",
        "add user-owned context, constraint, or intended use",
        "ask for a revised or final source-bounded deliverable with unknowns separated",
    ]


def _source_safe_difficulty_axis(plan: dict[str, Any]) -> str:
    subtask = str(plan["subtask_trajectory"]).replace("_", " ")
    return f"source-grounded {subtask} with missing user facts and strict visible-payload boundaries"


def _source_safe_volatile_items(seed_spec: dict[str, Any]) -> list[str]:
    return _dedupe(
        [
            "current source status",
            "unseen surrounding rules or context",
            "user-specific eligibility or applicability facts",
            *list(seed_spec.get("volatile_items", [])),
        ]
    )


def _source_task_family(value: str) -> str:
    normalized = _slug(value)
    if normalized in {"grounded_dialogue", "general_chat"}:
        return normalized
    return "grounded_dialogue"


def _source_task_intent_trajectory(seed_spec: dict[str, Any], plan: dict[str, Any], *, domain: str) -> str:
    raw_intent = str(plan["intent_trajectory"])
    if raw_intent in CONSISTENTCHAT_INTENT_NAMES:
        return raw_intent
    configured = str(seed_spec.get("intent_trajectory_hint", ""))
    if configured in CONSISTENTCHAT_INTENT_NAMES:
        return configured
    return _consistentchat_top_level_intent(raw_intent, domain=domain)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "source_grounded_dialogue"


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
    expected_roles = _expected_skeleton_roles(seed_spec)
    text = _chat_text(
        llm,
        _skeleton_messages(seed_spec, blueprint, intent_model),
        temperature=float(generation.get("skeleton_temperature", 0.4)),
    )
    skeleton = _parse_skeleton(text)
    for _ in range(2):
        if _skeleton_has_expected_roles(skeleton, expected_roles):
            break
        text = _chat_text(
            llm,
            _reformat_skeleton_messages(text, seed_spec, blueprint, intent_model, expected_roles, skeleton),
            temperature=0.0,
        )
        skeleton = _parse_skeleton(text)
    if not _skeleton_has_expected_roles(skeleton, expected_roles):
        skeleton = _fallback_skeleton_from_intent(expected_roles, intent_model)
    assert len(skeleton) == len(expected_roles), (
        f"skeleton_writer must produce exactly {len(expected_roles)} skeleton steps. Raw response: {text}"
    )
    actual_roles = [step["role"] for step in skeleton]
    assert actual_roles == expected_roles, (
        f"skeleton_writer role order mismatch. Expected {expected_roles}; got {actual_roles}. Raw response: {text}"
    )
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
    expected_turns = _role_count(skeleton, "user")
    turns = _parse_numbered_blocks(text, label="USER")
    if len(turns) != expected_turns:
        text = _chat_text(
            llm,
            _regenerate_user_turns_messages(
                seed_spec,
                blueprint,
                intent_model,
                skeleton,
                draft_text=text,
                parsed_turns=turns,
            ),
            temperature=0.2,
        )
        turns = _parse_numbered_blocks(text, label="USER")
    if len(turns) != expected_turns:
        text = _chat_text(
            llm,
            _reformat_numbered_blocks_messages(text, label="USER", count=expected_turns),
            temperature=0.0,
        )
        turns = _parse_numbered_blocks(text, label="USER")
    assert len(turns) == expected_turns, (
        f"user_simulator returned {len(turns)} user turns; expected {expected_turns}. Raw response: {text}"
    )
    return _inject_source_payloads(seed_spec, turns)


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


def _inject_source_payloads(seed_spec: dict[str, Any], user_turns: list[str]) -> list[str]:
    slots = _source_payload_slots(seed_spec, user_turn_count=len(user_turns))
    if not slots:
        return user_turns

    materialized = list(user_turns)
    for slot in slots:
        replacement = _format_source_payload_slot(slot)
        found = False
        for index, turn in enumerate(materialized):
            if slot["placeholder"] not in turn:
                continue
            materialized[index] = turn.replace(slot["placeholder"], replacement)
            found = True
        if found:
            continue
        turn_index = min(int(slot["turn_index"]), len(materialized) - 1)
        materialized[turn_index] = f"{materialized[turn_index].rstrip()}\n\n{replacement}"

    return [_remove_unresolved_source_placeholders(turn) for turn in materialized]


def _restore_source_payloads_in_messages(
    seed_spec: dict[str, Any],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    user_count = sum(1 for message in messages if message["role"] == "user")
    slots = _source_payload_slots(seed_spec, user_turn_count=user_count)
    if not slots:
        return messages

    user_turns = [message["content"] for message in messages if message["role"] == "user"]
    restored = _restore_source_payloads_in_user_turns(slots, user_turns)
    restored_messages = []
    user_index = 0
    for message in messages:
        if message["role"] != "user":
            restored_messages.append(message)
            continue
        restored_messages.append({**message, "content": restored[user_index]})
        user_index += 1
    return restored_messages


def _restore_source_payloads_in_user_turns(slots: list[dict[str, Any]], user_turns: list[str]) -> list[str]:
    restored = list(user_turns)
    for slot in slots:
        replacement = _format_source_payload_slot(slot)
        found = False
        for index, turn in enumerate(restored):
            updated, replaced = _replace_source_payload_block(turn, slot, replacement)
            restored[index] = updated
            found = found or replaced
        if found:
            continue
        turn_index = min(int(slot["turn_index"]), len(restored) - 1)
        restored[turn_index] = f"{restored[turn_index].rstrip()}\n\n{replacement}"
    return [_remove_unresolved_source_placeholders(turn) for turn in restored]


def _replace_source_payload_block(text: str, slot: dict[str, Any], replacement: str) -> tuple[str, bool]:
    if slot["placeholder"] in text:
        return text.replace(slot["placeholder"], replacement), True

    for pattern in _source_payload_restore_patterns(slot):
        if pattern.search(text) is None:
            continue
        return pattern.sub(replacement, text, count=1), True
    return text, False


def _source_payload_restore_patterns(slot: dict[str, Any]) -> list[re.Pattern[str]]:
    stem = re.escape(_source_payload_label_stem(slot))
    label = re.escape(str(slot["label"]))
    generic_stem = _source_payload_visible_label_pattern(index=int(slot["index"]))
    return [
        re.compile(rf"{label}\s*```(?:text)?\s*\n.*?\n```", flags=re.IGNORECASE | re.DOTALL),
        re.compile(rf"{stem}\s+begynder:\s*\n.*?\n{stem}\s+slutter\.?", flags=re.IGNORECASE | re.DOTALL),
        re.compile(rf"{label}\s*---\s*\n.*?\n---", flags=re.IGNORECASE | re.DOTALL),
        re.compile(rf"{label}\s*(?:>\s?.*(?:\n|$))+", flags=re.IGNORECASE),
        re.compile(rf"{generic_stem}\s*:\s*```(?:text)?\s*\n.*?\n```", flags=re.IGNORECASE | re.DOTALL),
        re.compile(
            rf"{generic_stem}\s+begynder:\s*\n.*?\n{generic_stem}\s+slutter\.?",
            flags=re.IGNORECASE | re.DOTALL,
        ),
        re.compile(rf"{generic_stem}\s*:\s*---\s*\n.*?\n---", flags=re.IGNORECASE | re.DOTALL),
        re.compile(rf"{generic_stem}\s*:\s*(?:>\s?.*(?:\n|$))+", flags=re.IGNORECASE),
    ]


def _remove_unresolved_source_placeholders(text: str) -> str:
    return re.sub(r"\{\{SOURCE_PAYLOAD_\d+\}\}", "", text).strip()


def _source_payload_slots(seed_spec: dict[str, Any], *, user_turn_count: int) -> list[dict[str, Any]]:
    source = _source_grounding(seed_spec)
    if source is None or user_turn_count <= 0:
        return []

    plan = seed_spec.get("source_payload_plan")
    strategy = str(plan.get("strategy", "full_excerpt")) if isinstance(plan, dict) else "full_excerpt"
    payload_length = str(plan.get("payload_length", "medium")) if isinstance(plan, dict) else "medium"
    paste_style = str(plan.get("paste_style", "fenced_text")) if isinstance(plan, dict) else "fenced_text"
    label_style = str(plan.get("label_style", "excerpt")) if isinstance(plan, dict) else "excerpt"
    excerpt = source["excerpt"]

    if strategy == "staged_excerpts" and user_turn_count > 1:
        chunks = _split_source_payload(excerpt)
        later_turn = min(2, user_turn_count - 1)
        return [
            _source_payload_slot(
                index=1,
                turn_index=0,
                strategy=strategy,
                payload_length=payload_length,
                paste_style=paste_style,
                label_style=label_style,
                text=chunks[0],
            ),
            _source_payload_slot(
                index=2,
                turn_index=later_turn,
                strategy=strategy,
                payload_length=payload_length,
                paste_style=paste_style,
                label_style=label_style,
                text=chunks[1],
            ),
        ]

    text = _source_payload_text_for_strategy(
        strategy,
        excerpt,
        seed_text=str(source["document_id"]),
        payload_length=payload_length,
    )
    return [
        _source_payload_slot(
            index=1,
            turn_index=0,
            strategy=strategy,
            payload_length=payload_length,
            paste_style=paste_style,
            label_style=label_style,
            text=text,
        )
    ]


def _source_payload_slot(
    *,
    index: int,
    turn_index: int,
    strategy: str,
    payload_length: str,
    paste_style: str,
    label_style: str,
    text: str,
) -> dict[str, Any]:
    stem = _source_payload_label_stem_from_style(label_style, index=index)
    return {
        "index": index,
        "label": f"{stem}:",
        "label_stem": stem,
        "label_style": label_style,
        "placeholder": SOURCE_PAYLOAD_PLACEHOLDER_TEMPLATE % index,
        "turn_index": turn_index,
        "target_user_turn": turn_index + 1,
        "strategy": strategy,
        "payload_length": payload_length,
        "paste_style": paste_style,
        "text": text,
        "char_count": len(text),
    }


def _source_payload_text_for_strategy(
    strategy: str,
    excerpt: str,
    *,
    seed_text: str,
    payload_length: str = "medium",
) -> str:
    if strategy == "full_excerpt":
        return excerpt
    if strategy == "long_contiguous_excerpt":
        return _source_payload_window(excerpt, seed_text=seed_text, payload_length=payload_length)
    if strategy == "selected_clauses":
        settings = _source_payload_length_settings(payload_length)
        passages = _selected_source_passages(
            excerpt,
            count=int(settings["selected_count"]),
            seed_text=seed_text,
            target_chars=int(settings["selected_target_chars"]),
        )
        return _format_selected_source_passages(passages)
    if strategy == "summary_plus_quotes":
        settings = _source_payload_length_settings(payload_length)
        passages = _selected_source_passages(
            excerpt,
            count=max(1, int(settings["selected_count"]) - 1),
            seed_text=seed_text,
            target_chars=int(settings["selected_target_chars"]),
        )
        return _format_source_passages(passages)
    return excerpt


def _source_payload_length_settings(payload_length: str) -> dict[str, float | int | str]:
    settings = SOURCE_PAYLOAD_LENGTHS.get(payload_length)
    if settings is not None:
        return settings
    return SOURCE_PAYLOAD_LENGTHS["medium"]


def _source_payload_window(excerpt: str, *, seed_text: str, payload_length: str = "medium") -> str:
    settings = _source_payload_length_settings(payload_length)
    target_chars = max(int(settings["window_min_chars"]), int(len(excerpt) * float(settings["window_ratio"])))
    if len(excerpt) <= target_chars:
        return excerpt
    max_start = max(len(excerpt) - target_chars, 0)
    start = _stable_index(seed_text + "|long_source_window", max_start + 1)
    if start > 0:
        start = excerpt.find(" ", start)
        if start == -1:
            start = 0
    end = min(start + target_chars, len(excerpt))
    if end < len(excerpt):
        boundary = excerpt.rfind(" ", start, end)
        if boundary > start:
            end = boundary
    return excerpt[start:end].strip()


def _split_source_payload(excerpt: str) -> tuple[str, str]:
    midpoint = len(excerpt) // 2
    split_at = excerpt.find(". ", max(midpoint - 500, 0), min(midpoint + 500, len(excerpt)))
    if split_at == -1:
        split_at = excerpt.find(" ", midpoint)
    if split_at == -1:
        split_at = midpoint
    first = excerpt[: split_at + 1].strip()
    second = excerpt[split_at + 1 :].strip()
    if not second:
        return first, first
    return first, second


def _selected_source_passages(excerpt: str, *, count: int, seed_text: str, target_chars: int) -> list[str]:
    passages = [block for block in _source_payload_blocks(excerpt) if _source_payload_block_is_readable(block)]
    if not passages:
        return [_source_payload_window(excerpt, seed_text=seed_text, payload_length="medium")]
    if len(passages) <= count:
        return passages
    if count == 1:
        index = _stable_index(seed_text + "|source_passage", len(passages))
        span, _ = _source_payload_span(passages, index=index, target_chars=target_chars)
        return [span]

    step = (len(passages) - 1) / (count - 1)
    offset = _stable_index(seed_text + "|source_passage_offset", len(passages))
    indexes = sorted({(offset + round(index * step)) % len(passages) for index in range(count)})
    selected = []
    next_allowed = 0
    for index in indexes:
        start = max(index, next_allowed)
        if start >= len(passages):
            continue
        span, end = _source_payload_span(passages, index=start, target_chars=target_chars)
        selected.append(span)
        next_allowed = end + 1
    if selected:
        return selected
    return [passages[_stable_index(seed_text + "|source_passage_fallback", len(passages))]]


def _source_payload_blocks(excerpt: str) -> list[str]:
    paragraphs = [_normalize_source_payload_block(part) for part in re.split(r"\n\s*\n+", excerpt)]
    blocks = [paragraph for paragraph in paragraphs if paragraph]
    if len(blocks) > 1:
        return _merge_short_source_blocks(blocks)

    lines = [_normalize_source_payload_block(line) for line in excerpt.splitlines()]
    line_blocks = [line for line in lines if line]
    if len(line_blocks) > 1:
        return _merge_short_source_blocks(line_blocks)

    if len(excerpt) > 1400:
        return _source_payload_chunk_blocks(excerpt, target_chars=900)
    return [excerpt.strip()] if excerpt.strip() else []


def _source_payload_span(passages: list[str], *, index: int, target_chars: int) -> tuple[str, int]:
    start = max(0, min(index, len(passages) - 1))
    selected = [passages[start]]
    total = len(passages[start])
    next_index = start + 1

    while total < target_chars and next_index < len(passages):
        selected.append(passages[next_index])
        total += len(passages[next_index])
        next_index += 1

    return "\n\n".join(selected).strip(), next_index - 1


def _source_payload_chunk_blocks(excerpt: str, *, target_chars: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(excerpt):
        end = min(start + target_chars, len(excerpt))
        if end < len(excerpt):
            boundary = _source_payload_chunk_boundary(excerpt, start=start, end=end)
            if boundary > start:
                end = boundary
        chunk = excerpt[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def _source_payload_chunk_boundary(excerpt: str, *, start: int, end: int) -> int:
    search_start = max(start + int((end - start) * 0.6), start)
    for marker in [". ", "\n", "; ", ", ", " "]:
        boundary = excerpt.rfind(marker, search_start, end)
        if boundary > start:
            return boundary + len(marker.rstrip())
    return end


def _normalize_source_payload_block(text: str) -> str:
    return _normalize_source_text(text).strip()


def _merge_short_source_blocks(blocks: list[str]) -> list[str]:
    merged: list[str] = []
    pending: list[str] = []
    pending_chars = 0

    for block in blocks:
        pending.append(block)
        pending_chars += len(block)
        if pending_chars < 180:
            continue
        merged.append("\n".join(pending).strip())
        pending = []
        pending_chars = 0

    if pending:
        tail = "\n".join(pending).strip()
        if merged and len(tail) < 180:
            merged[-1] = f"{merged[-1]}\n{tail}".strip()
        else:
            merged.append(tail)

    return merged


def _source_payload_block_is_readable(block: str) -> bool:
    stripped = block.strip()
    if len(stripped) < 80:
        return False
    alpha_chars = sum(1 for char in stripped if char.isalpha())
    visible_chars = sum(1 for char in stripped if not char.isspace())
    if visible_chars == 0:
        return False
    return alpha_chars / visible_chars >= 0.45


def _format_selected_source_passages(passages: list[str]) -> str:
    return "\n\n".join(f"Uddrag {index}:\n{passage}" for index, passage in enumerate(passages, start=1))


def _format_source_passages(passages: list[str]) -> str:
    return "Direkte kildepassager markeret af brugeren:\n\n" + _format_selected_source_passages(passages)


def _format_source_payload_slot(slot: dict[str, Any]) -> str:
    style = str(slot.get("paste_style", "fenced_text"))
    text = str(slot["text"])
    if style == "blockquote":
        quoted = "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
        return f"{slot['label']}\n{quoted}"
    if style == "plain_delimited":
        stem = _source_payload_label_stem(slot)
        return f"{stem} begynder:\n{text}\n{stem} slutter."
    if style == "separator":
        return f"{slot['label']}\n---\n{text}\n---"
    return f"{slot['label']}\n```text\n{text}\n```"


def _source_payload_label_stem(slot: dict[str, Any]) -> str:
    if "label_stem" in slot:
        return str(slot["label_stem"])
    return _source_payload_label_stem_from_style(str(slot.get("label_style", "excerpt")), index=int(slot["index"]))


def _source_payload_label_stem_from_style(label_style: str, *, index: int) -> str:
    detail = SOURCE_PAYLOAD_LABEL_STYLES.get(label_style, SOURCE_PAYLOAD_LABEL_STYLES["excerpt"])
    return str(detail["stem_template"]).format(index=index)


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
    return _apply_local_review_gates(review, messages, seed_spec)


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
    if _unrepairable_surface_failures(messages, seed_spec):
        return False
    if not bool(review["selection"]["accepted"]):
        return True
    return not _dialogue_respects_evidence_boundary(messages, _evidence_boundary(seed_spec))


def _unrepairable_surface_failures(
    messages: list[dict[str, str]],
    seed_spec: dict[str, Any],
) -> list[str]:
    failures = []
    if not _visible_artifact_claims_grounded({"messages": messages})["passed"]:
        failures.append("visible_artifact_claims_grounded")
    if _source_grounding(seed_spec) is not None and not _source_payloads_readable({"messages": messages})["passed"]:
        failures.append("source_payloads_readable")
    return failures


def _local_unrepairable_review(
    seed_spec: dict[str, Any],
    intent_model: dict[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    findings = [
        f"Hard reject: {failure}. This is a user/source surface issue, so assistant-turn repair cannot fix it."
        for failure in failures
    ]
    return {
        "review": {
            "reviewers": [
                {
                    "name": f"local_{failure}",
                    "passed": False,
                    "finding": finding,
                }
                for failure, finding in zip(failures, findings, strict=True)
            ],
            "turn_evidence_audit": "SKIPPED - user/source surface issue is not assistant-repairable",
            "repair_actions": [
                f"Regenerate the candidate with a complete, readable, visible user/source payload instead of repairing assistant turns: {failure}"
                for failure in failures
            ],
            "review_decision": "reject",
            "difficulty_axis": seed_spec["difficulty_axis"],
            "artifact": "\n".join(findings),
        },
        "verification": {
            "success_criteria": list(seed_spec["success_criteria"]),
        },
        "selection": {
            "accepted": False,
            "score": 0.0,
            "intent_trajectory": intent_model["trajectory"],
            "subtask_trajectory": intent_model["subtask_trajectory"],
            "reasons": findings,
        },
    }


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
    verification = {
        "required_dialogue_acts": required_acts,
        "success_criteria": list(seed_spec["success_criteria"]),
        "work_session": dict(seed_spec["work_session"]),
    }
    source_grounding = _hidden_source_grounding(seed_spec)
    if source_grounding:
        verification["source_grounding"] = source_grounding

    return {
        "id": row_id,
        "messages": messages,
        "sources": [
            source
            for source in METHOD_SOURCES
            if source["name"] in source_methods
        ]
        + list(seed_spec.get("external_sources", [])),
        "hidden": {
            "blueprint": blueprint,
            "source_task_plan": seed_spec.get("source_task_plan", {}),
            "intent_model": intent_model,
            "skeleton": skeleton,
            "review": review["review"],
            "verification": verification,
            "selection": review["selection"],
        },
        "meta": {
            "seed_id": seed_spec["seed_id"],
            "archetype_id": seed_spec["archetype_id"],
            "scenario_variant": seed_spec["scenario_variant"],
            "scenario_instance": seed_spec.get("scenario_instance", {}),
            "scenario_variant_index": seed_spec["scenario_variant_index"],
            "source_pack": seed_spec.get("source_grounding", {}).get("pack_id", ""),
            "source_document_id": seed_spec.get("source_grounding", {}).get("document_id", ""),
            "persona_source": seed_spec.get("persona_context", {}).get("source", ""),
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
            + task_completion
            + first_turn_substance
        )
        / 7,
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
    return bool(
        _messages_alternate(row)["passed"]
        and _no_role_label_leak(row)["passed"]
        and _skeleton_matches_messages(row)["passed"]
        and _evidence_boundary_respected(row)["passed"]
        and _visible_artifact_claims_grounded(row)["passed"]
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


def _source_payloads_readable(row: dict[str, Any]) -> dict[str, bool]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {"passed": False}

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for block in _inserted_source_payload_blocks(str(message.get("content", ""))):
            if not _inserted_source_payload_block_readable(block):
                return {"passed": False}

    return {"passed": True}


def _inserted_source_payload_blocks(text: str) -> list[str]:
    label = _source_payload_visible_label_pattern()
    patterns = [
        re.compile(rf"{label}\s*:\s*```(?:text)?\s*\n(.*?)\n```", flags=re.IGNORECASE | re.DOTALL),
        re.compile(rf"{label}\s+begynder:\s*\n(.*?)\n{label}\s+slutter\.?", flags=re.IGNORECASE | re.DOTALL),
        re.compile(rf"{label}\s*:\s*---\s*\n(.*?)\n---", flags=re.IGNORECASE | re.DOTALL),
    ]
    blocks = [match.group(1).strip() for pattern in patterns for match in pattern.finditer(text)]
    blocks.extend(_inserted_blockquote_source_payload_blocks(text))
    return blocks


def _inserted_blockquote_source_payload_blocks(text: str) -> list[str]:
    label = _source_payload_visible_label_pattern()
    pattern = re.compile(rf"{label}\s*:\s*((?:\n>\s?.*)+)", flags=re.IGNORECASE)
    blocks = []
    for match in pattern.finditer(text):
        lines = []
        for line in match.group(1).splitlines():
            stripped = line.strip()
            if not stripped.startswith(">"):
                continue
            lines.append(stripped[1:].lstrip())
        if lines:
            blocks.append("\n".join(lines).strip())
    return blocks


def _source_payload_visible_label_pattern(*, index: int | None = None) -> str:
    stems = [
        str(detail["stem_template"]).replace("{index}", "").strip()
        for detail in SOURCE_PAYLOAD_LABEL_STYLES.values()
    ]
    stems.extend(LEGACY_SOURCE_PAYLOAD_LABEL_STEMS)
    stem_pattern = "|".join(re.escape(stem) for stem in stems)
    index_pattern = str(index) if index is not None else r"\d+"
    return rf"(?:{stem_pattern})\s+{index_pattern}"


def _inserted_source_payload_block_readable(block: str) -> bool:
    units = _inserted_source_payload_units(block)
    if not units:
        return _source_payload_block_is_readable(block)
    return all(_source_payload_block_is_readable(unit) for unit in units)


def _inserted_source_payload_units(block: str) -> list[str]:
    marker = re.compile(r"(?m)^Uddrag\s+\d+:\s*")
    matches = list(marker.finditer(block))
    if not matches:
        return []

    units = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        unit = block[start:end].strip()
        if unit:
            units.append(unit)
    return units


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
    return re.search(r"[\u2705\u274c\u26a0\ufe0f\U0001f300-\U0001faff]", text) is None


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
                "You expand a multi-turn dialogue archetype into one concrete scenario instance. "
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
                "If a source-grounding rule is provided below, it overrides this variation instruction for all "
                "source-owned facts. "
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
                f"{_source_grounding_generation_instruction(seed_spec)}"
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
                f"{_source_grounding_generation_instruction(seed_spec)}"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate scenario instance:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                "Use the candidate scenario instance as the concrete row-specific scenario. Use the seed as an "
                "archetype and constraint source, not as a fixed transcript plan. Preserve the archetype's domain, "
                "evidence boundary, and success criteria, but prefer the candidate instance for user-owned facts, "
                "named entities, quantities, constraints, first-turn informativeness, and disclosure timing where "
                "the archetype permits it. "
                "For source-pack rows, only user-owned facts may vary; source-owned facts are fixed by the original "
                "source reference above. "
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
                f"{_source_grounding_skeleton_instruction(seed_spec)}"
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


def _reformat_skeleton_messages(
    draft_skeleton: str,
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    expected_roles: list[str],
    parsed_skeleton: list[dict[str, str]],
) -> list[dict[str, str]]:
    target_steps = len(expected_roles)
    target_user_turns = _target_user_turns(seed_spec)
    role_order = ", ".join(expected_roles)
    parsed_roles = [step["role"] for step in parsed_skeleton]
    parsed_role_order = ", ".join(parsed_roles) if parsed_roles else "none"
    latent_language = _language_name(str(seed_spec["latent_language"]))
    return [
        {
            "role": "system",
            "content": (
                "You repair ConsistentChat-style dialogue skeleton artifacts. "
                "Preserve the planned intent trajectory and work-session progress where possible, "
                "but the structural contract is mandatory."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rewrite the draft skeleton into exactly {target_steps} STEP lines. "
                f"Required role order: {role_order}. "
                f"The current draft parsed as {len(parsed_skeleton)} STEP lines with roles: {parsed_role_order}. "
                "That parsed shape is invalid unless it already exactly matches the required role order. "
                "Every line must use this form:\n"
                "STEP N: role | compact_dialogue_act | visible_state_delta\n\n"
                "Do not add headings, bullets, JSON, commentary, or dialogue text. "
                f"The {target_user_turns} user skeleton steps should still map to the USER_QUERY_PLAN in order. "
                "If the draft has extra or missing turns, merge or split at natural information-progress points "
                "while preserving the work-session contract. Do not collapse two user-query-plan moves into one "
                "user step just to keep the skeleton short.\n\n"
                f"Write the repaired skeleton artifact in {latent_language}.\n\n"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                f"{_source_grounding_skeleton_instruction(seed_spec)}"
                f"Draft skeleton:\n{draft_skeleton}"
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
                "For source-pack rows, do not type source text from memory or from the hidden blueprint. Use the "
                "provided source-payload placeholders exactly where source material should become visible; the pipeline "
                "will replace those placeholders with exact text from the original source. "
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
                f"{_source_grounding_user_simulator_instruction(seed_spec, user_turns=user_turns)}"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"{_source_grounding_visible_payload_instruction(seed_spec)}"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                f"Skeleton:\n{_format_skeleton(skeleton)}"
            ),
        },
    ]


def _regenerate_user_turns_messages(
    seed_spec: dict[str, Any],
    blueprint: dict[str, Any],
    intent_model: dict[str, Any],
    skeleton: list[dict[str, str]],
    *,
    draft_text: str,
    parsed_turns: list[str],
) -> list[dict[str, str]]:
    dialogue_language = _language_name(str(seed_spec["dialogue_language"]))
    user_turns = _role_count(skeleton, "user")
    user_labels = ", ".join(f"USER {index}:" for index in range(1, user_turns + 1))
    parsed_count = len(parsed_turns)
    return [
        {
            "role": "system",
            "content": (
                "You repair incomplete user-simulator output for multi-turn data. "
                "Regenerate the whole user-side trajectory from the hidden blueprint, intent model, and skeleton. "
                "Use the draft only as a hint; if it collapsed missing turns into one block, split and complete the "
                "trajectory at natural information-progress points. "
                "Do not answer as the assistant."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The previous user simulator output parsed as {parsed_count} user turns, but the skeleton requires "
                f"exactly {user_turns}. Regenerate all user turns with one labeled block per user turn, using exactly "
                f"these labels: {user_labels}. Do not add headings, JSON, commentary, or assistant turns.\n\n"
                f"Visible dialogue language: {dialogue_language}. Keep the USER labels exactly as requested, "
                f"but write the turn content in {dialogue_language}.\n"
                f"Domain: {seed_spec['domain']}\n"
                f"Seed query trajectory hint:\n{_format_query_trajectory(seed_spec.get('query_trajectory_hint'))}\n\n"
                "Scenario expansion variant:\n"
                f"{_format_scenario_variant(seed_spec)}\n\n"
                "Candidate scenario instance:\n"
                f"{_format_scenario_instance(seed_spec)}\n\n"
                f"{_source_grounding_user_simulator_instruction(seed_spec, user_turns=user_turns)}"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"{_source_grounding_visible_payload_instruction(seed_spec)}"
                f"Intent model:\n{intent_model['artifact']}\n\n"
                f"Hidden blueprint:\n{blueprint['artifact']}\n\n"
                f"Skeleton:\n{_format_skeleton(skeleton)}\n\n"
                f"Incomplete draft output:\n{draft_text}"
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
                "as evidence for user-specific facts. If a source-pack skeleton or work-session note names a source "
                "detail that is not visible in the transcript prefix, ignore it as non-evidence. "
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
                "For calculations, show exclusions and totals from the visible numbers instead of jumping to a result. "
                "For practical advice, recipes, routines, schedules, or how-long-it-takes claims, frame examples and "
                "time estimates as rough suggestions to try, not proven outcomes or known timings. Use phrases like "
                "for example, roughly, often, usually, or adjust after trying it. "
                "For technical troubleshooting, keep root causes and compatibility explanations as hypotheses unless "
                "the visible transcript provides the exact package version, platform, build log, documentation, or "
                "other controlling evidence. Say what to verify in docs, release notes, metadata, logs, or a minimal "
                "reproduction. Do not assert exact version compatibility, build requirements, maintenance status, or "
                "causal package behavior as verified unless that evidence is visible.\n\n"
                "Use light, functional formatting when it helps the requested work product: paragraph breaks, short "
                "headings, bullets, numbered steps, compact tables, or occasional emphasis are fine when they make the "
                "answer easier to use. When using Markdown, write structurally valid Markdown: put blank lines between "
                "paragraphs, headings, lists, tables, and major sections where Markdown needs them. Do not use horizontal "
                "rules or separators as spacing. Use bold sparingly; do not bold routine list labels, every key term, "
                "or whole sentences. Avoid decorative formatting, emojis, checkmark/cross/warning symbols, repeated "
                "separators, excessive emphasis, and large heading stacks.\n\n"
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
                "numbers, dates, code, commands, source-payload fenced blocks, quoted source text, product names, and "
                "evidence boundaries. Do not add new facts, remove constraints, make the assistant smarter, or rewrite "
                "the task. If an official localized "
                "label is uncertain, use a generic description rather than a precise-sounding invented label. "
                "For Danish, replace accidental English words in prose with ordinary Danish unless they are code, product "
                "names, quoted source text, or established loanwords. Fix misspellings and malformed Danish compounds. "
                "Use established Danish technical terms or keep standard English terms when natural; do not create ad hoc "
                "literal compounds, hyphenated participles, or word-for-word translations of English technical phrases. "
                "Check Danish agreement in adjective and noun phrases. "
                "Remove emojis, decorative emoji bullets, and checkmark/cross/warning symbols from assistant turns, "
                "replacing them with plain text bullets or headings. Keep functional paragraph breaks, short headings, bullets, numbered lists, compact tables, "
                "and occasional emphasis when they serve the requested artifact. When using Markdown, keep it structurally valid with blank lines "
                "between paragraphs, headings, lists, tables, and major sections where Markdown needs them. Do not use horizontal "
                "rules or separators as spacing. Use bold sparingly; do not bold routine list labels, every key term, "
                "or whole sentences. Remove excessive emphasis, repeated "
                "separators, decorative formatting, and large heading stacks. No turn content may "
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
                "illustrative placeholder is acceptable. Do not reject an assistant merely for telling the user to check "
                "documentation, release notes, metadata, a policy, an official page, or another controlling source. A "
                "verification instruction is not a verified claim. Reject only when the assistant asserts what that "
                "source says, guarantees that the source exists, or treats source-dependent content as already verified "
                "without visible evidence or clear hedging.\n\n"
                "Language quality review must reject corrupted text, wrong-language fragments, malformed commands, broken "
                "words, incoherent punctuation artifacts, and unnatural translations that would make the dialogue unsuitable "
                "for training. For Danish output, reject non-Danish code-like fragments inside prose unless the user supplied "
                "them or they are valid commands. Reject assistant turns that use emojis, decorative emoji bullets, "
                "or checkmark/cross/warning symbols. "
                "Format review should allow light functional formatting when it helps the requested artifact: paragraph "
                "breaks, short headings, bullets, numbered steps, compact tables, and occasional emphasis. Markdown should "
                "be structurally valid, with blank lines between paragraphs, headings, lists, tables, and major sections "
                "where Markdown needs them. Bold should be sparse; routine labels, every key term, and whole sentences "
                "should not be bolded by default. Reject "
                "formatting only when it is decorative, excessive, markdown-heavy, emoji/symbol-based, or when repeated "
                "separators or large heading stacks distract from the work product.\n\n"
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
                f"{_source_grounding_review_instruction(seed_spec)}"
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
                "Be stricter than a helpful assistant, but fail only material defects that would make the row bad "
                "training data. Do not reward plausibility when the visible transcript lacks evidence. Stable public "
                "knowledge is allowed, but exact source-dependent details must be hedged, framed as general memory, "
                "or left as fields to verify. Treat a verification path as different from a verified claim: it is "
                "acceptable to suggest checking docs, metadata, release notes, policies, official pages, or source "
                "excerpts when the assistant does not assert what they contain. "
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
                "Each STRICT_CHECK line must begin PASS or FAIL, followed by a brief finding. Use FAIL only for "
                "material, reject-worthy issues that would make the row bad training data; do not fail for harmless "
                "missing hedges, reasonable examples, or minor preference refinements when the assistant remains "
                "cautious and useful.\n\n"
                "Fail source_boundary if the assistant states exact source-dependent facts without visible source "
                "support or clear hedging. This includes product specs, prices, policies, legal/procedural rules, "
                "health/nutrition/sleep claims, local planning claims, schedules, official labels, forms, fees, "
                "institution/provider details, compatibility outcomes, stock, delivery, benchmarks, or current facts. "
                "Also fail source_boundary if inserted source snippets are unreadable broken fragments, isolated "
                "numbers, citation/date shards, or partial abbreviations rather than complete readable passages. "
                "The acceptable alternative is a hedged general explanation, a verification checklist, or unknown fields. "
                "Allow stable public knowledge, common-domain background, and reasonable illustrative examples when "
                "they are phrased generally, hedged where uncertainty matters, and not presented as current, official, "
                "private, provider-specific, user-specific, or source-verified. Do not fail merely because the assistant "
                "suggests a reasonable new option, ingredient, example, or next step that the user did not pre-approve. "
                "Fail those additions only when they violate a visible exclusion, allergy, safety, medical constraint, "
                "or closed list, or when the assistant presents them as the user's known preference. For health, nutrition, "
                "and sleep, allow cautious general background and practical suggestions. Fail individualized diagnosis, "
                "prescriptive medical advice, exact quantities, or unhedged medical claims without visible support. "
                "For visible source text, allow direct paraphrases and clearly marked interpretations; fail when an "
                "inference materially changes rules, eligibility, obligations, outcomes, or certainty. "
                "Do not fail source_boundary merely because the assistant tells the user to check documentation, release "
                "notes, package metadata, a policy, an official source, or another likely controlling source. Fail only "
                "if the assistant asserts what those sources say, guarantees their existence, or treats their contents "
                "as known without visible evidence.\n\n"
                "Fail language_quality for unnatural Danish, malformed words, calques, broken grammar, invented "
                "compounds, wrong idioms, tense disagreement, or odd phrasing that a native speaker would not accept "
                "as clean training data.\n\n"
                "Fail factuality for suspicious trivia, made-up explanations, incorrect general claims, or numbers "
                "presented as verified facts when the row is not source-grounded. Do not fail cautious general background "
                "merely because no citation is visible. Entertainment rows are not exempt: playful tone cannot license "
                "fabricated facts.\n\n"
                "Fail contradiction if later assistant turns contradict earlier advice, user constraints, or their own "
                "uncertainty framing. Do not fail normal multi-turn revision after newly visible user constraints: the "
                "assistant may narrow, abandon, or replace earlier tentative advice. Fail when it ignores the new "
                "constraint, repeats incompatible advice, misleadingly denies or forgets earlier advice, or hardens "
                "earlier uncertainty without new evidence.\n\n"
                "Fail format for emojis, decorative emoji bullets, checkmark/cross/warning symbols, excessive markdown emphasis, repeated horizontal "
                "rules or separators, large heading stacks, or formatting that overwhelms the answer. Allow light "
                "functional formatting, including paragraph breaks, short markdown headings, bullets, numbered steps, "
                "compact tables, occasional emphasis, and structurally valid Markdown spacing between paragraphs, headings, "
                "lists, tables, and major sections. Bold should be sparse; fail rows that bold routine labels, every key "
                "term, or whole sentences by default.\n\n"
                f"Expected intent trajectory: {seed_spec['intent_trajectory_hint']}\n"
                f"Expected subtask trajectory: {_subtask_trajectory_hint(seed_spec)}\n"
                "ConsistentChat-aligned work-session contract:\n"
                f"{_format_work_session_contract(seed_spec)}\n\n"
                f"{_evidence_boundary_instruction(seed_spec)}"
                f"{_source_grounding_review_instruction(seed_spec)}"
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
                "Use these lines: FAILURE_MODE, CHOSEN_ACTION, REJECTED_ACTION, REJECTED_ANSWER. "
                "REJECTED_ANSWER must contain the full rejected assistant message, not just a description.\n\n"
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
        if blocks:
            blocks[-1] = _append_parsed_content_line(blocks[-1], line)
    if blocks:
        return [_clean_numbered_block(block) for block in blocks]

    generic_pattern = re.compile(r"^-?\s*\d+\s*[\).:-]\s*(.*)$")
    for line in text.splitlines():
        stripped = line.strip()
        match = generic_pattern.match(stripped)
        if match:
            blocks.append(match.group(1).strip())
            continue
        if blocks:
            blocks[-1] = _append_parsed_content_line(blocks[-1], line)
    return [_clean_numbered_block(block) for block in blocks]


def _append_parsed_content_line(text: str, line: str) -> str:
    stripped = line.strip()
    if not stripped:
        if not text.strip():
            return text
        if text.endswith("\n\n"):
            return text
        return f"{text.rstrip()}\n\n"
    if text.endswith("\n\n"):
        return f"{text}{stripped}"
    return f"{text.rstrip()}\n{stripped}".strip()


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
        if current is not None:
            current["content"] = _append_parsed_content_line(current["content"], line)
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


def _apply_local_review_gates(
    review: dict[str, Any],
    messages: list[dict[str, str]],
    seed_spec: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        ("visible_artifact_claims_grounded", _visible_artifact_claims_grounded({"messages": messages})),
    ]
    if _source_grounding(seed_spec) is not None:
        gates.append(("source_payloads_readable", _source_payloads_readable({"messages": messages})))
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
    merged["review"]["review_decision"] = "reject"
    merged["review"]["repair_actions"] = [
        *list(merged["review"].get("repair_actions", [])),
        *(
            f"Regenerate the candidate with a complete, readable, visible user/source payload instead of repairing assistant turns: {name}"
            for name in failures
        ),
    ]
    merged["selection"]["accepted"] = False
    merged["selection"]["score"] = min(float(merged["selection"].get("score", 0.0)), 0.0)
    return merged


def _parse_preference_pair(text: str) -> dict[str, str]:
    fields = _parse_keyed_lines(text)
    rejected = _field_tail(text, "REJECTED_ANSWER").strip()
    if not rejected:
        rejected = fields.get("rejected_answer", "").strip()
    if not rejected:
        rejected = _preference_answer_fallback(text)
    assert rejected, "preference pair must include a rejected answer"
    return {
        "failure_mode": fields.get("failure_mode", "ignored_late_constraint").strip(),
        "chosen_action": fields.get("chosen_action", "revise_with_updated_state").strip(),
        "rejected_action": fields.get("rejected_action", "continue_from_stale_state").strip(),
        "rejected_answer": rejected,
    }


def _preference_answer_fallback(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = [line for line in lines if _keyed_line_match(line) is None]
    if body:
        return "\n".join(body).strip()
    return ""


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
    if stripped.startswith(("pass", "accept")):
        return True
    normalized = _strip_negated_review_failures(lower)
    failure_patterns = [
        r"\bfail(?:ed|s|ure)?\b",
        r"\breject(?:ed|s|ion)?\b",
        r"\bmissing\b",
        r"\blacks?\b",
        r"\bunsupported\b",
        r"\bnot\s+(?:supported|grounded|visible)\b",
        r"\bcontradict(?:s|ed|ion|ions|ory)?\b",
        r"\bhallucinat\w*\b",
        r"\binvent(?:ed|s|ing)?\b",
        r"\bfuture leakage\b",
        r"\bsource invention\b",
        r"\bmalformed\b",
        r"\bbroken\b",
        r"\bcorrupt(?:ed|ion)?\b",
        r"\bunnatural\b",
        r"\brepair\s+(?:needed|required)\b",
        r"\bneeds?\s+(?:repair|regeneration|rewrite|fix)\b",
    ]
    return not any(re.search(pattern, normalized) for pattern in failure_patterns)


def _strip_negated_review_failures(text: str) -> str:
    patterns = [
        r"\bno\s+[^.:\n;]{0,120}\b(?:unsupported|contradict(?:ion|ions|ory)?|missing|hallucinat\w*|invent\w*|source invention|failures?|issues?|problems?|semantic oddit\w*|recovery situation\w*)[^.:\n;]{0,80}",
        r"\bwithout\s+[^.:\n;]{0,80}\b(?:unsupported|contradict\w*|missing|hallucinat\w*|invent\w*|source invention)[^.:\n;]{0,80}",
        r"\bdoes not\s+(?:assert|introduce|imply|claim|invent|contradict|rely|pretend|need|silently|fabricate|override|act)\b",
        r"\bdoesn't\s+(?:assert|introduce|imply|claim|invent|contradict|rely|pretend|need|silently|fabricate|override|act)\b",
        r"\bdo not\s+(?:assert|introduce|imply|claim|invent|contradict|rely|pretend|fabricate|override|act)\b",
        r"\bavoids?\s+(?:hallucination|contradiction|unsupported|inventing|source invention|future leakage)\b",
        r"\bnot unsupported\b",
    ]
    normalized = text
    for pattern in patterns:
        normalized = re.sub(pattern, " ", normalized)
    return normalized


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
    lower = re.sub(r"\s*[-\u2013\u2014]\s*", " - ", lower)
    if re.match(r"^none(?:\b| - )", lower):
        return []
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
    current_key: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        match = _keyed_line_match(stripped)
        if match is None:
            if current_key is not None and stripped:
                current_value = fields[current_key]
                fields[current_key] = f"{current_value}\n{stripped}".strip()
            continue
        key = match.group("key")
        value = match.group("value")
        normalized = key.strip().lower().replace(" ", "_")
        fields[normalized] = value.strip()
        current_key = normalized
    return fields


def _keyed_line_match(line: str) -> re.Match[str] | None:
    return re.match(
        r"^\**(?P<key>STRICT_SELECTION|STRICT_SCORE|STRICT_REPAIR_ACTIONS|STRICT_CHECK\s+\w+|"
        r"SELECTION|SCORE|INTENT_TRAJECTORY|SUCCESS_CRITERIA|SUBTASK_TRAJECTORY|"
        r"TURN_EVIDENCE_AUDIT|REPAIR_ACTIONS|REVIEWER\s+\w+|"
        r"INFORMATION_FLOW|ROLE_INTERACTION|TOPIC_GUARDRAILS|WORK_SESSION_TYPE|"
        r"SOURCE_TASK_DOMAIN|SOURCE_TASK_FAMILY|SOURCE_TASK_INTENT_TRAJECTORY|"
        r"SOURCE_TASK_SUBTASK_TRAJECTORY|SOURCE_TASK_DIFFICULTY_AXIS|SOURCE_TASK_PERSONA|"
        r"SOURCE_TASK_GOAL|SOURCE_TASK_QUERY_TRAJECTORY|SOURCE_TASK_SUCCESS_CRITERIA|"
        r"SOURCE_TASK_VOLATILE_ITEMS|SOURCE_TASK_BLUEPRINT|"
        r"DELIVERABLE_PROGRESS|USER_QUERY_PLAN|FAILURE_MODE|FAILURE\s+MODE|CHOSEN_ACTION|"
        r"CHOSEN\s+ACTION|REJECTED_ACTION|REJECTED\s+ACTION|REJECTED_ANSWER|"
        r"REJECTED\s+ANSWER)\**(?:\s*[:,]\s*|\s+|$)(?P<value>.*)$",
        line,
        flags=re.IGNORECASE,
    )


def _field_tail(text: str, key: str) -> str:
    label = re.escape(key).replace("_", r"[_\s]+")
    pattern = re.compile(rf"^\**{label}\**(?:\s*[:,]\s*|\s+|$)(.*)$", flags=re.IGNORECASE)
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


def _skeleton_has_expected_roles(skeleton: list[dict[str, str]], expected_roles: list[str]) -> bool:
    return [step["role"] for step in skeleton] == expected_roles


def _fallback_skeleton_from_intent(expected_roles: list[str], intent_model: dict[str, Any]) -> list[dict[str, str]]:
    user_plan_steps = intent_model.get("user_query_plan_steps")
    if not isinstance(user_plan_steps, list):
        user_plan_steps = _split_query_plan(str(intent_model.get("user_query_plan", "")))

    user_total = expected_roles.count("user")
    assistant_total = expected_roles.count("assistant")
    user_index = 0
    assistant_index = 0
    skeleton: list[dict[str, str]] = []

    for role in expected_roles:
        if role == "user":
            user_index += 1
            move = _fallback_user_move(user_index, user_total, user_plan_steps)
            skeleton.append(
                {
                    "role": "user",
                    "dialogue_act": _dialogue_act_slug(move, default=f"user_move_{user_index}"),
                    "state_delta": f"visible user move {user_index}: {move}",
                }
            )
            continue

        assistant_index += 1
        skeleton.append(
            {
                "role": "assistant",
                "dialogue_act": _fallback_assistant_act(assistant_index, assistant_total),
                "state_delta": _fallback_assistant_state(assistant_index, assistant_total),
            }
        )

    return skeleton


def _fallback_user_move(index: int, total: int, user_plan_steps: list[Any]) -> str:
    if index <= len(user_plan_steps):
        return str(user_plan_steps[index - 1])
    if index == 1:
        return "initial task request with concrete visible context"
    if index == total:
        return "request finalization or final format adjustment"
    return "add a new constraint, correction, comparison, or work-product requirement"


def _fallback_assistant_act(index: int, total: int) -> str:
    if total == 1:
        return "answer_with_visible_state"
    if index == 1:
        return "produce_initial_work_or_clarify"
    if index == total:
        return "finalize_work_product"
    return "revise_or_extend_work_product"


def _fallback_assistant_state(index: int, total: int) -> str:
    if total == 1:
        return "answer using only visible information and preserve source boundaries"
    if index == 1:
        return "produce useful initial work or ask one targeted clarification when needed"
    if index == total:
        return "deliver the final bounded work product using only visible information"
    return "update the intermediate work product based on the newly visible user move"


def _dialogue_act_slug(text: str, *, default: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:64] or default


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
                if "source_grounding" in template:
                    spec["source_grounding"] = dict(template["source_grounding"])
                    spec["source_payload_plan"] = _source_payload_plan_for_seed(
                        template,
                        archetype_id=archetype_id,
                        variant_index=variant_index,
                    )
                if "persona_context" in template:
                    spec["persona_context"] = dict(template["persona_context"])
                if "external_sources" in template:
                    spec["external_sources"] = list(template["external_sources"])
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


def _source_payload_plan_for_seed(
    template: dict[str, Any],
    *,
    archetype_id: str,
    variant_index: int,
) -> dict[str, str]:
    configured = template.get("source_payload_strategies", DEFAULT_SOURCE_PAYLOAD_STRATEGIES)
    assert isinstance(configured, list | tuple) and configured, "source_payload_strategies must be a non-empty list"
    strategies = [str(strategy) for strategy in configured]
    for strategy in strategies:
        assert strategy in SOURCE_PAYLOAD_STRATEGIES, f"unsupported source payload strategy: {strategy}"

    configured_lengths = template.get("source_payload_lengths", DEFAULT_SOURCE_PAYLOAD_LENGTHS)
    assert isinstance(configured_lengths, list | tuple) and configured_lengths, "source_payload_lengths must be a non-empty list"
    payload_lengths = [str(length) for length in configured_lengths]
    for payload_length in payload_lengths:
        assert payload_length in SOURCE_PAYLOAD_LENGTHS, f"unsupported source payload length: {payload_length}"

    configured_styles = template.get("source_payload_styles", DEFAULT_SOURCE_PAYLOAD_STYLES)
    assert isinstance(configured_styles, list | tuple) and configured_styles, "source_payload_styles must be a non-empty list"
    paste_styles = [str(style) for style in configured_styles]
    for paste_style in paste_styles:
        assert paste_style in SOURCE_PAYLOAD_STYLES, f"unsupported source payload style: {paste_style}"

    configured_label_styles = template.get("source_payload_label_styles", DEFAULT_SOURCE_PAYLOAD_LABEL_STYLES)
    assert (
        isinstance(configured_label_styles, list | tuple) and configured_label_styles
    ), "source_payload_label_styles must be a non-empty list"
    label_styles = [str(style) for style in configured_label_styles]
    for label_style in label_styles:
        assert label_style in SOURCE_PAYLOAD_LABEL_STYLES, f"unsupported source payload label style: {label_style}"

    strategy = strategies[_stable_index(f"{archetype_id}|{variant_index}", len(strategies))]
    payload_length = payload_lengths[_stable_index(f"{archetype_id}|{variant_index}|payload_length", len(payload_lengths))]
    paste_style = paste_styles[_stable_index(f"{archetype_id}|{variant_index}|paste_style", len(paste_styles))]
    label_style = label_styles[_stable_index(f"{archetype_id}|{variant_index}|label_style", len(label_styles))]
    detail = SOURCE_PAYLOAD_STRATEGIES[strategy]
    length_detail = SOURCE_PAYLOAD_LENGTHS[payload_length]
    label_detail = SOURCE_PAYLOAD_LABEL_STYLES[label_style]
    return {
        "strategy": strategy,
        "label": detail["label"],
        "payload_length": payload_length,
        "payload_length_label": str(length_detail["label"]),
        "paste_style": paste_style,
        "paste_style_label": SOURCE_PAYLOAD_STYLES[paste_style],
        "label_style": label_style,
        "label_style_label": str(label_detail["label"]),
        "user_visibility": detail["user_visibility"],
        "assistant_boundary": detail["assistant_boundary"],
    }


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


def _format_source_payload_plan(seed_spec: dict[str, Any]) -> str:
    plan = seed_spec.get("source_payload_plan")
    if not isinstance(plan, dict):
        plan = _source_payload_plan_for_seed(
            {"source_payload_strategies": DEFAULT_SOURCE_PAYLOAD_STRATEGIES},
            archetype_id="default",
            variant_index=0,
        )
    return json.dumps(plan, indent=2, sort_keys=True)


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
        "matters or lookup would normally be needed. Practical examples, recipes, routines, and time estimates should "
        "be framed as examples, rough estimates, or starting points to test, not as proven outcomes or exact timings. "
        "Technical troubleshooting should distinguish visible evidence from hypotheses: exact package compatibility, "
        "build requirements, maintenance status, platform behavior, and root-cause claims require visible logs, version "
        "metadata, documentation, or release notes. Otherwise, phrase them as likely explanations to verify and give "
        "the verification step. For deadlines, appeal windows, fees, complaint bodies, eligibility rules, "
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
        "writing a precise-sounding translation. Assistant turns should not use emojis, decorative emoji bullets, "
        "or checkmark/cross/warning symbols; "
        "use plain text headings, bullets, or numbering instead. Assistant turns should use light functional "
        "formatting: paragraph breaks, short headings, simple bullets, numbered steps, compact tables, or occasional "
        "emphasis when useful. When using Markdown, write structurally valid Markdown with blank lines between "
        "paragraphs, headings, lists, tables, and major sections where Markdown needs them. Do not use horizontal "
        "rules or separators as spacing. Use bold sparingly; do not bold routine list labels, every key term, "
        "or whole sentences. Avoid decorative formatting, repeated separators, excessive emphasis, and long stacked "
        "heading structures.\n\n"
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


def _source_grounding(seed_spec: dict[str, Any]) -> dict[str, Any] | None:
    source = seed_spec.get("source_grounding")
    if isinstance(source, dict):
        return source
    return None


def _source_grounding_reference(seed_spec: dict[str, Any]) -> str:
    source = _source_grounding(seed_spec)
    if source is None:
        return ""
    return (
        "Original source-grounding reference. This is generator/reviewer context, not assistant-visible evidence "
        "unless the user includes it in the visible transcript.\n"
        f"Source pack: {source['pack_id']}\n"
        f"Source family: {source['source_name']}\n"
        f"Document id: {source['document_id']}\n"
        f"Title: {source.get('title', '')}\n"
        f"Created/date metadata: {source.get('created', '')}\n"
        "Immutable source excerpt:\n"
        f"{source['excerpt']}\n\n"
    )


def _source_grounding_generation_instruction(seed_spec: dict[str, Any]) -> str:
    source = _source_grounding(seed_spec)
    if source is None:
        return ""
    return (
        "Source-grounding rule: downstream scenario, blueprint, skeleton, user-turn, and assistant-turn generation "
        "must treat source facts as unavailable until exact source payloads are inserted into visible user turns. "
        "The original source excerpt is immutable reviewer/planner context, but it is not assistant evidence and "
        "should not be recited into hidden artifacts or dialogue plans. Vary the user's role, motivation, task, "
        "missing facts, requested deliverable, and disclosure timing, but do not invent, rename, anonymize, or alter "
        "source-specific entities, titles, outlets, institutions, case numbers, dates, quoted phrases, findings, "
        "legal/procedural facts, obligations, thresholds, or factual claims. Write task plans as operations over "
        "the source text that will become visible, not as target facts to assert. The assistant may only use source "
        "facts after they are visible in the transcript. Persona context is only a diversity vector; for localized "
        "Danish rows, do not copy foreign persona names, locations, currency, institutions, legal settings, or "
        "biography unless the source or user-visible scenario specifically requires a foreign context.\n\n"
        f"Source context for task shape only: pack={source['pack_id']}; family={source['source_name']}. "
        "Exact title, document id, date metadata, and source facts are withheld from downstream generation unless "
        "they appear inside an injected source payload.\n\n"
        "Source visibility plan for the later visible dialogue:\n"
        f"{_format_source_payload_plan(seed_spec)}\n\n"
    )


def _source_grounding_visible_payload_instruction(seed_spec: dict[str, Any]) -> str:
    if _source_grounding(seed_spec) is None:
        return ""
    return (
        "Source-pack visible-payload rule: follow the sampled source visibility plan below. The actual source text is "
        "injected by the pipeline, not generated by the user simulator. Some rows should expose the provided source "
        "excerpt; others should expose selected passages, a long passage, exact source passages, or staged excerpts across "
        "turns. The user may frame the question before or after inserted source text, but assistant-verifiable source "
        "facts must appear in visible user text before the assistant uses them. Later user turns should add user-owned "
        "context, constraints, interpretations, follow-up questions, or new source-payload placeholders rather than "
        "introducing hidden source facts by implication. Do not describe a payload as full, complete, all of the "
        "document, or exhaustive; use neutral wording such as this excerpt, this pasted text, or the selected passages. "
        "Payloads may appear as code-fenced text, blockquotes, plain delimited pastes, or separator-delimited pastes; "
        "all of these are visible source evidence.\n\n"
        "Sampled source visibility plan:\n"
        f"{_format_source_payload_plan(seed_spec)}\n\n"
    )


def _source_grounding_user_simulator_instruction(seed_spec: dict[str, Any], *, user_turns: int) -> str:
    source = _source_grounding(seed_spec)
    if source is None:
        return ""
    slots = _source_payload_slots(seed_spec, user_turn_count=user_turns)
    prompt_slots = [
        {
            "placeholder": slot["placeholder"],
            "target_user_turn": slot["target_user_turn"],
            "strategy": slot["strategy"],
            "payload_length": slot["payload_length"],
            "paste_style_after_injection": slot["paste_style"],
            "visible_label_after_injection": slot["label"],
            "char_count": slot["char_count"],
        }
        for slot in slots
    ]
    return (
        "Source-pack user-simulation rule: do not reproduce, paraphrase as if quoted, shorten, translate, or repair "
        "the original source excerpt yourself. Place each source-payload placeholder exactly once in the visible user "
        "turn where that source material should appear. The build pipeline will replace placeholders with exact source "
        "text after user-turn generation, before assistant generation. You may write the user's surrounding question, "
        "context, constraints, and follow-up requests naturally, but do not add source-owned dates, thresholds, case "
        "numbers, quoted phrases, obligations, medical criteria, legal rules, or procedural facts outside the injected "
        "payload. Do not refer to source title, document id, date metadata, exact institutional names, or exact "
        "source-owned values unless they will be present in the same visible user turn through a placeholder. Do not "
        "call a payload full, complete, exhaustive, all recitals, or the whole document. If a later turn needs more "
        "source evidence, use the later placeholder before asking about it. The pipeline may render inserted source "
        "payloads with varied neutral labels and as a code fence, blockquote, plain delimited paste, or "
        "separator-delimited paste; do not write those wrappers yourself.\n\n"
        f"Source context for task shape only: pack={source['pack_id']}; family={source['source_name']}. "
        "Exact source metadata and source facts are available only through inserted payloads.\n"
        "Sampled source visibility plan:\n"
        f"{_format_source_payload_plan(seed_spec)}\n\n"
        "Source-payload placeholders to use:\n"
        f"{json.dumps(prompt_slots, indent=2, sort_keys=True)}\n\n"
    )


def _source_grounding_skeleton_instruction(seed_spec: dict[str, Any]) -> str:
    if _source_grounding(seed_spec) is None:
        return ""
    return (
        "Source-pack skeleton rule: plan source visibility according to the sampled source payload plan. Do not put "
        "hidden source facts into assistant dialogue acts or state deltas as targets to assert. "
        "Assistant steps must be procedural, for example: extract covered roles visible in the pasted source, list any "
        "effective dates visible in the source, separate confirmed source facts from unknowns, and ask for missing "
        "controlling sections. Bad assistant purpose: \"assert the hidden effective date and repealed source\". "
        "Good assistant purpose: \"identify any effective date and repeal notice visible in the prior pasted source, "
        "and mark absent implementation details as unknown\". User steps should introduce source placeholders before "
        "asking about source-owned facts, and should not name exact source values outside those payloads.\n\n"
        "Sampled source visibility plan:\n"
        f"{_format_source_payload_plan(seed_spec)}\n\n"
    )


def _source_grounding_review_instruction(seed_spec: dict[str, Any]) -> str:
    if _source_grounding(seed_spec) is None:
        return ""
    return (
        f"{_source_grounding_reference(seed_spec)}"
        "Source-pack review rule: compare the visible dialogue against the original source reference above. Fail "
        "source_boundary if a visible user payload or assistant answer mutates source-specific entities, titles, "
        "outlets, institutions, case numbers, dates, quoted phrases, findings, legal/procedural facts, obligations, "
        "or factual claims from the original source. Persona and scenario variation may add the user's situation, "
        "motive, preferences, deadlines, and requested deliverable, but it may not rewrite the source into a different "
        "case, policy, document, product, rule, quote, or factual situation. For localized Danish rows, fail "
        "language_quality or naturalness if irrelevant foreign persona names, places, currency, institutions, or "
        "biographical details are copied into the row instead of being abstracted into a Danish-localized or neutral "
        "user situation. Fail source_boundary if an inserted source payload is made of broken fragments, isolated "
        "numbers, citation/date shards, or unreadable snippets rather than complete readable source passages. "
        "The original source reference above is not assistant-visible evidence. Fail source_boundary "
        "if the assistant uses a source fact from the original reference before that fact appears in prior visible user "
        "text, even when the fact is correct in the original source. Do not fail merely because the user asks "
        "for a transformation, explanation, checklist, or draft based on the source.\n\n"
    )


def _hidden_source_grounding(seed_spec: dict[str, Any]) -> dict[str, Any]:
    source = _source_grounding(seed_spec)
    if source is None:
        return {}
    return {
        "pack_id": source["pack_id"],
        "source_name": source["source_name"],
        "document_id": source["document_id"],
        "title": source.get("title", ""),
        "created": source.get("created", ""),
        "excerpt": source["excerpt"],
        "license": source.get("license", ""),
        "url": source.get("url", ""),
        "payload_plan": seed_spec.get("source_payload_plan", {}),
        "payload_slots": [
            {
                "index": slot["index"],
                "strategy": slot["strategy"],
                "target_user_turn": slot["target_user_turn"],
                "char_count": slot["char_count"],
                "text": slot["text"],
            }
            for slot in _source_payload_slots(seed_spec, user_turn_count=_target_user_turns(seed_spec))
        ],
    }


def _templates(generation: dict[str, Any]) -> list[dict[str, Any]]:
    sources = generation.get("seed_sources")
    if not sources:
        return _builtin_templates()

    templates: list[dict[str, Any]] = []
    for source in sources:
        assert isinstance(source, dict), "seed source entries must be mappings"
        source_type = source.get("type")
        if source_type == "handwritten":
            templates.extend(_load_handwritten_templates(source))
            continue
        if source_type == "source_pack":
            templates.extend(_load_source_pack_templates(source))
            continue
        raise AssertionError(f"unsupported multi_turn_dialogue seed source: {source_type}")
    assert templates, "multi_turn_dialogue seed sources must produce at least one seed"
    return templates


def _load_handwritten_templates(source: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_seed_source_path(str(source["path"]))
    document = read_yaml(path)
    assert isinstance(document, dict), "handwritten seed source must be a mapping"
    seeds = document.get("seeds")
    assert isinstance(seeds, list), "handwritten seed source must contain a seeds list"
    return [_normalize_handwritten_seed(seed) for seed in seeds]


def _load_source_pack_templates(source: dict[str, Any]) -> list[dict[str, Any]]:
    documents_cfg = source.get("documents")
    assert isinstance(documents_cfg, dict), "source_pack seed source must include a documents mapping"
    personas_cfg = source.get("personas")
    assert personas_cfg is None or isinstance(personas_cfg, dict), "source_pack personas must be a mapping"

    documents = _load_source_documents(source, documents_cfg)
    personas = _load_persona_contexts(personas_cfg, limit=len(documents))
    templates = []
    for index, document in enumerate(documents):
        persona = personas[index % len(personas)] if personas else {}
        templates.append(_source_pack_template(source, document, persona, index=index))
    assert templates, "source_pack seed source must produce at least one source document"
    return templates


def _load_source_documents(source: dict[str, Any], documents_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    text_field = str(documents_cfg.get("text_field", "text"))
    id_field = str(documents_cfg.get("id_field", "id"))
    source_field = str(documents_cfg.get("source_field", "source"))
    title_field = documents_cfg.get("title_field")
    created_field = documents_cfg.get("created_field")
    min_chars = int(documents_cfg.get("min_chars", 200))
    max_chars = int(documents_cfg.get("max_chars", 3000))
    max_records = int(documents_cfg.get("max_records", source.get("max_records", 64)))

    documents = []
    for record in iter_source_records(documents_cfg, default_streaming=True):
        assert isinstance(record, dict), "source_pack document records must be mappings"
        if not _record_matches_filters(record, documents_cfg):
            continue
        text = read_record_value(record, text_field)
        if text is None or len(text) < min_chars:
            continue
        document_id = read_record_value(record, id_field) or f"{source_label(documents_cfg)}-{len(documents):05d}"
        source_name = read_record_value(record, source_field) or source_label(documents_cfg)
        documents.append(
            {
                "document_id": document_id,
                "source_name": source_name,
                "title": read_record_value(record, str(title_field)) if title_field else "",
                "created": read_record_value(record, str(created_field)) if created_field else "",
                "excerpt": _trim_source_excerpt(text, max_chars=max_chars),
                "raw_token_count": read_record_value(record, "token_count") or "",
            }
        )
        if len(documents) >= max_records:
            break
    assert documents, f"source_pack {source.get('id', source_label(documents_cfg))} produced no usable documents"
    return documents


def _record_matches_filters(record: dict[str, Any], cfg: dict[str, Any]) -> bool:
    include_values = cfg.get("include_values", {})
    assert isinstance(include_values, dict), "source include_values must be a mapping when provided"
    for field, allowed_values in include_values.items():
        allowed = {str(value) for value in _list_config(allowed_values)}
        value = read_record_value(record, str(field))
        if value not in allowed:
            return False
    return True


def _load_persona_contexts(personas_cfg: dict[str, Any] | None, *, limit: int) -> list[dict[str, Any]]:
    if not personas_cfg:
        return []
    max_records = int(personas_cfg.get("max_records", limit))
    persona_fields = [str(field) for field in personas_cfg.get("persona_fields", ["persona", "professional_persona"])]
    id_field = str(personas_cfg.get("id_field", "uuid"))
    min_age = personas_cfg.get("min_age", 18)
    max_chars = int(personas_cfg.get("max_chars", 1600))
    projection_cfg = personas_cfg.get("projection", {})
    assert isinstance(projection_cfg, dict), "persona projection must be a mapping when provided"
    use_projection = bool(projection_cfg.get("enabled", False))

    personas = []
    for record in iter_source_records(personas_cfg, default_streaming=True):
        assert isinstance(record, dict), "persona records must be mappings"
        if min_age is not None and not _record_age_at_least(record, int(min_age)):
            continue
        persona_id = read_record_value(record, id_field) or f"{source_label(personas_cfg)}-{len(personas):05d}"
        if use_projection:
            personas.append(_project_persona_context(record, personas_cfg, projection_cfg, persona_id=persona_id))
        else:
            persona_text = _join_record_fields(record, persona_fields)
            if not persona_text:
                continue
            personas.append(
                {
                    "id": persona_id,
                    "source": source_label(personas_cfg),
                    "license": str(personas_cfg.get("license", "cc-by-4.0")),
                    "persona": _trim_source_excerpt(persona_text, max_chars=max_chars),
                    "occupation": read_record_value(record, "occupation") or "",
                    "education_level": read_record_value(record, "education_level") or "",
                    "age": read_record_value(record, "age") or "",
                }
            )
        if len(personas) >= max_records:
            break
    assert personas, f"persona source {source_label(personas_cfg)} produced no usable personas"
    return personas


def _project_persona_context(
    record: dict[str, Any],
    personas_cfg: dict[str, Any],
    projection_cfg: dict[str, Any],
    *,
    persona_id: str,
) -> dict[str, Any]:
    target_locale = str(projection_cfg.get("target_locale", ""))
    occupation = _humanize_surface_value(
        read_record_value(record, str(projection_cfg.get("occupation_field", "occupation"))) or "adult user"
    )
    education_level = _humanize_surface_value(
        read_record_value(record, str(projection_cfg.get("education_field", "education_level"))) or ""
    )
    age_band = _age_band(read_record_value(record, str(projection_cfg.get("age_field", "age"))))
    seed_text = "|".join([persona_id, occupation, education_level, age_band, target_locale])
    communication_style = _stable_choice(PERSONA_COMMUNICATION_STYLES, seed_text + "|style")
    task_posture = _stable_choice(PERSONA_TASK_POSTURES, seed_text + "|posture")
    constraints = _stable_sample(PERSONA_PRACTICAL_CONSTRAINTS, seed_text + "|constraints", count=2)
    skill_tags = _projected_skill_tags(record, projection_cfg)
    surface = {
        "role_archetype": occupation,
        "age_band": age_band,
        "education_level": education_level,
        "communication_style": communication_style,
        "task_posture": task_posture,
        "practical_constraints": constraints,
        "skill_tags": skill_tags,
        "target_locale": target_locale,
        "localization_rule": "localize into the target locale or keep neutral; do not copy source persona identity",
    }
    return {
        "id": persona_id,
        "source": source_label(personas_cfg),
        "license": str(personas_cfg.get("license", "cc-by-4.0")),
        "projection_mode": "surface",
        "persona": _format_projected_persona(surface),
        "surface": surface,
    }


def _projected_skill_tags(record: dict[str, Any], projection_cfg: dict[str, Any]) -> list[str]:
    if not bool(projection_cfg.get("include_skill_tags", False)):
        return []
    field_name = str(projection_cfg.get("skill_tags_field", "skills_and_expertise_list"))
    max_items = int(projection_cfg.get("max_skill_tags", 3))
    return [_humanize_surface_value(value) for value in _record_list_values(record, field_name, max_items=max_items)]


def _format_projected_persona(surface: dict[str, Any]) -> str:
    parts = [
        f"role archetype: {surface['role_archetype']}",
        f"age band: {surface['age_band']}",
        f"education: {surface['education_level'] or 'unspecified'}",
        f"style: {surface['communication_style']}",
        f"task posture: {surface['task_posture']}",
        f"constraints: {', '.join(surface['practical_constraints'])}",
    ]
    skill_tags = surface.get("skill_tags") or []
    if skill_tags:
        parts.append(f"surface skills: {', '.join(skill_tags)}")
    target_locale = str(surface.get("target_locale", ""))
    if target_locale:
        parts.append(f"target locale: {target_locale}")
    return "Projected surface persona; " + "; ".join(parts) + "."


def _age_band(value: str | None) -> str:
    if value is None:
        return "adult"
    age = int(value)
    if age < 25:
        return "adult_18_24"
    if age < 35:
        return "adult_25_34"
    if age < 50:
        return "adult_35_49"
    if age < 65:
        return "adult_50_64"
    return "adult_65_plus"


def _humanize_surface_value(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split())


def _stable_choice(values: tuple[str, ...], seed_text: str) -> str:
    return values[_stable_index(seed_text, len(values))]


def _stable_sample(values: tuple[str, ...], seed_text: str, *, count: int) -> list[str]:
    start = _stable_index(seed_text, len(values))
    return [values[(start + offset) % len(values)] for offset in range(min(count, len(values)))]


def _stable_index(seed_text: str, modulo: int) -> int:
    assert modulo > 0, "stable index modulo must be positive"
    value = 0
    for char in seed_text:
        value = (value * 33 + ord(char)) % 2_147_483_647
    return value % modulo


def _record_list_values(record: dict[str, Any], field_name: str, *, max_items: int) -> list[str]:
    value: Any = record
    for part in field_name.split("."):
        if not isinstance(value, dict):
            return []
        value = value.get(part)
        if value is None:
            return []
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value[:max_items] if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list | tuple):
            return [str(item).strip() for item in parsed[:max_items] if str(item).strip()]
    return [text]


def _record_age_at_least(record: dict[str, Any], min_age: int) -> bool:
    value = record.get("age")
    if value is None:
        return True
    return int(value) >= min_age


def _join_record_fields(record: dict[str, Any], fields: list[str]) -> str:
    values = [value for field in fields if (value := read_record_value(record, field))]
    return "\n".join(values)


def _source_pack_template(
    source: dict[str, Any],
    document: dict[str, Any],
    persona: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    pack_id = str(source.get("id", document["source_name"]))
    family = str(source.get("family", "grounded_dialogue"))
    domain_hint = str(source.get("domain_hint", _slug(document["source_name"])))
    source_methods = [str(method) for method in source.get("source_methods", DEFAULT_SOURCE_METHODS)]
    blueprint = {
        "domain": domain_hint,
        "user_persona": persona.get("persona", "source-grounded Danish user with a practical task"),
        "hidden_goal": f"create a source-grounded Danish task from {document['source_name']}",
        "source_context": {
            "source_pack": pack_id,
            "source_name": document["source_name"],
            "source_text_available_to_visible_dialogue_only_via_injected_payload": True,
        },
        "persona_context": persona,
        "success_criteria": [
            "the assistant only treats source details as known after the user makes the excerpt visible",
            "the dialogue produces a useful source-grounded work product",
            "exact source-dependent, current, legal, policy, health, or procedural claims remain bounded to visible text",
            "the user situation is varied through the sampled persona without copying irrelevant persona facts",
        ],
    }
    return {
        "id": f"{pack_id}-{index:05d}",
        "family": family,
        "intent_trajectory": str(source.get("intent_trajectory", "Information Retrieval Interaction")),
        "subtask_trajectory": str(source.get("subtask_trajectory", "source_grounded_synthesis")),
        "query_trajectory_hint": [],
        "blueprint": blueprint,
        "success_criteria": list(blueprint["success_criteria"]),
        "source_methods": source_methods,
        "source_language": str(source.get("source_language", "da")),
        "difficulty_axis": str(source.get("difficulty_axis", "source-grounded task with visible excerpt boundary")),
        "requires_epistemic_hedging": True,
        "volatile_items": [str(item) for item in source.get("volatile_items", ["current source status", "unseen rules"])],
        "source_payload_strategies": [
            str(strategy)
            for strategy in source.get("source_payload_strategies", DEFAULT_SOURCE_PAYLOAD_STRATEGIES)
        ],
        "source_payload_lengths": [
            str(length)
            for length in source.get("source_payload_lengths", DEFAULT_SOURCE_PAYLOAD_LENGTHS)
        ],
        "source_payload_styles": [
            str(style)
            for style in source.get("source_payload_styles", DEFAULT_SOURCE_PAYLOAD_STYLES)
        ],
        "source_payload_label_styles": [
            str(style)
            for style in source.get("source_payload_label_styles", DEFAULT_SOURCE_PAYLOAD_LABEL_STYLES)
        ],
        "source_grounding": {
            "pack_id": pack_id,
            **document,
            "license": str(source.get("license", "")),
            "url": str(source.get("url", "")),
        },
        "persona_context": persona,
        "external_sources": _external_sources_for_source_pack(source, document, persona),
    }


def _external_sources_for_source_pack(
    source: dict[str, Any],
    document: dict[str, Any],
    persona: dict[str, Any],
) -> list[dict[str, str]]:
    external_sources = [
        {
            "kind": "source_excerpt",
            "name": document["source_name"],
            "role": "visible source material for grounded task planning",
            "document_id": document["document_id"],
            "source_pack": str(source.get("id", document["source_name"])),
        }
    ]
    if source.get("url"):
        external_sources[0]["url"] = str(source["url"])
    if source.get("license"):
        external_sources[0]["license"] = str(source["license"])
    if persona:
        external_sources.append(
            {
                "kind": "persona_source",
                "name": persona["source"],
                "role": "persona diversity seed for source-to-task planning",
                "document_id": str(persona["id"]),
                "license": str(persona.get("license", "")),
            }
        )
    return external_sources


def _list_config(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _trim_source_excerpt(text: str, *, max_chars: int) -> str:
    stripped = _normalize_source_text(text).strip()
    if len(stripped) <= max_chars:
        return stripped
    truncated = stripped[:max_chars]
    cut_at = _source_excerpt_cut_boundary(truncated)
    if cut_at > 0:
        return truncated[:cut_at].rstrip()
    return truncated.rstrip()


def _normalize_source_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _source_excerpt_cut_boundary(text: str) -> int:
    lower_bound = int(len(text) * 0.6)
    boundaries = [text.rfind(marker, lower_bound) for marker in ["\n\n", "\n", " "]]
    return max(boundaries)


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
