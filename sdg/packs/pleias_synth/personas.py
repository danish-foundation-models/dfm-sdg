from __future__ import annotations

from pathlib import Path
from random import Random

from sdg.packs.pleias_synth.assistant_styles import load_assistant_style
from sdg.packs.pleias_synth.query_profiles import assign_query_profiles
from sdg.packs.pleias_synth.record_loader import load_records, load_source_config
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


def load_personas(cfg: Record) -> list[Persona]:
    personas_config = load_source_config(
        cfg,
        key="personas",
        label="persona",
        default_preset="starter",
    )
    match personas_config["source"]:
        case "preset":
            preset = personas_config["preset"]
            assert preset == "starter", f"Unknown persona preset: {preset}"
            source = f"preset:{preset}"
            records = load_records(DATA_DIR / "starter_personas.yaml", label="persona preset")
        case "inline":
            source = "inline"
            records = personas_config["records"]
        case "path":
            path = personas_config["path"]
            source = str(path)
            records = load_records(path, label="persona file")
        case _:
            raise AssertionError(f"Unknown persona source: {personas_config['source']}")

    return _normalize_personas(records, source=source)


def build_query_plans(
    fact_bundles: list[Record],
    cfg: Record,
    *,
    seed: int | None,
) -> list[QueryPlan]:
    personas = load_personas(cfg)
    assert personas, "No personas available for memorization generation"

    rng = Random(0 if seed is None else seed)
    rng.shuffle(personas)

    query_angles = _query_angles(cfg)
    query_profiles = assign_query_profiles(len(fact_bundles), cfg, seed=seed)
    assistant_style = load_assistant_style(cfg)
    persona_count = len(personas)
    plans: list[QueryPlan] = []
    for index, bundle in enumerate(fact_bundles):
        persona = personas[index % persona_count]
        preferred_angles = [angle for angle in persona["preferred_angles"] if angle in query_angles] or query_angles
        angle = preferred_angles[(index // persona_count) % len(preferred_angles)]
        plans.append(
            {
                "bundle": bundle,
                "persona": persona,
                "query_angle": angle,
                "query_profile": query_profiles[index],
                "assistant_style": assistant_style,
            }
        )
    return plans


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
    for index, record in enumerate(records):
        constraints = record.get("constraints", [])
        tags = record.get("tags", [])
        preferred_angles = record.get("preferred_angles", DEFAULT_QUERY_ANGLES)
        meta = record.get("meta", {})

        assert isinstance(constraints, list), "persona constraints must be a list"
        assert isinstance(tags, list), "persona tags must be a list"
        assert isinstance(preferred_angles, list), "persona preferred_angles must be a list"
        assert isinstance(meta, dict), "persona meta must be a mapping"

        persona_id = str(record.get("persona_id") or f"persona-{index:03d}")

        normalized.append(
            {
                "persona_id": persona_id,
                "source": source,
                "name": str(record.get("name") or persona_id),
                "intent": str(record.get("intent") or "ask a clear factual question"),
                "knowledge_level": str(record.get("knowledge_level") or "general"),
                "tone": str(record.get("tone") or "neutral"),
                "question_style": str(record.get("question_style") or "concise"),
                "answer_granularity": str(record.get("answer_granularity") or "short answer"),
                "constraints": [str(item) for item in constraints],
                "tags": [str(item) for item in tags],
                "preferred_angles": [str(item) for item in preferred_angles],
                "meta": dict(meta),
            }
        )
    return normalized
