from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from random import Random

from sdg.packs.pleias_synth.assistant_styles import load_assistant_style
from sdg.packs.pleias_synth.query_profiles import load_weighted_query_profiles
from sdg.packs.pleias_synth.record_loader import load_records_source
from sdg.packs.pleias_synth.types import Persona, QueryPlan, Record

DEFAULT_QUERY_ANGLES = [
    "overview",
    "explanation",
    "planning",
    "comparison",
    "timeline",
    "clarification",
    "identification",
    "lookup",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
PRESETS = {"starter": DATA_DIR / "starter_personas.yaml"}


def load_personas(cfg: Record) -> list[Persona]:
    source, records = load_records_source(
        cfg,
        key="personas",
        label="persona",
        presets=PRESETS,
    )
    return _normalize_personas(records, source=source)


def build_query_plans(
    fact_bundles: list[Record],
    cfg: Record,
    *,
    seed: int | None,
) -> list[QueryPlan]:
    return list(iter_query_plans(fact_bundles, cfg, seed=seed))


def iter_query_plans(
    fact_bundles: Iterable[Record],
    cfg: Record,
    *,
    seed: int | None,
) -> Iterator[QueryPlan]:
    personas = load_personas(cfg)
    assert personas, "No personas available for memorization generation"

    rng = Random(0 if seed is None else seed)
    rng.shuffle(personas)

    query_angles = _query_angles(cfg)
    query_profiles = load_weighted_query_profiles(cfg, seed=seed)
    assistant_style = load_assistant_style(cfg)
    persona_count = len(personas)
    for index, bundle in enumerate(fact_bundles):
        persona = personas[index % persona_count]
        preferred_angles = [angle for angle in persona["preferred_angles"] if angle in query_angles] or query_angles
        angle = preferred_angles[(index // persona_count) % len(preferred_angles)]
        yield {
            "bundle": bundle,
            "persona": persona,
            "query_angle": angle,
            "query_profile": query_profiles[index % len(query_profiles)],
            "assistant_style": assistant_style,
        }


def _query_angles(cfg: Record) -> list[str]:
    generation_cfg = cfg.get("generation")
    if generation_cfg is None:
        return list(DEFAULT_QUERY_ANGLES)
    assert isinstance(generation_cfg, dict), "generation config must be a mapping"

    memorization_cfg = generation_cfg.get("memorization")
    if memorization_cfg is None:
        return list(DEFAULT_QUERY_ANGLES)
    assert isinstance(memorization_cfg, dict), "memorization config must be a mapping"

    raw_angles = memorization_cfg.get("query_angles", DEFAULT_QUERY_ANGLES)
    assert isinstance(raw_angles, list), "query_angles must be a list"
    angles = [str(angle) for angle in raw_angles]
    assert angles, "query_angles must not be empty"
    return angles


def _normalize_personas(records: list[Record], *, source: str) -> list[Persona]:
    normalized: list[Persona] = []
    for record in records:
        normalized.append(
            {
                "persona_id": _required_str(record, "persona_id", label="persona persona_id"),
                "source": source,
                "name": _required_str(record, "name", label="persona name"),
                "intent": _required_str(record, "intent", label="persona intent"),
                "knowledge_level": _required_str(record, "knowledge_level", label="persona knowledge_level"),
                "tone": _required_str(record, "tone", label="persona tone"),
                "question_style": _required_str(record, "question_style", label="persona question_style"),
                "answer_granularity": _required_str(record, "answer_granularity", label="persona answer_granularity"),
                "constraints": _string_list(record, "constraints", label="persona constraints"),
                "tags": _string_list(record, "tags", label="persona tags"),
                "preferred_angles": _required_string_list(
                    record,
                    "preferred_angles",
                    label="persona preferred_angles",
                ),
                "meta": _meta(record, label="persona meta"),
            }
        )
    return normalized


def _required_str(record: Record, key: str, *, label: str) -> str:
    value = record.get(key)
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    return value


def _required_string_list(record: Record, key: str, *, label: str) -> list[str]:
    values = _string_list(record, key, label=label)
    assert values, f"{label} must not be empty"
    return values


def _string_list(record: Record, key: str, *, label: str) -> list[str]:
    value = record.get(key, [])
    assert isinstance(value, list), f"{label} must be a list"
    return [str(item) for item in value if str(item).strip()]


def _meta(record: Record, *, label: str) -> Record:
    value = record.get("meta", {})
    assert isinstance(value, dict), f"{label} must be a mapping"
    return dict(value)
