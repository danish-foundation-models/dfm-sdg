from __future__ import annotations

from pathlib import Path
from random import Random

from sdg.packs.pleias_synth.record_loader import load_records, load_source_config
from sdg.packs.pleias_synth.types import QueryProfile, Record

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_query_profiles(cfg: Record) -> list[QueryProfile]:
    profiles_config = load_source_config(
        cfg,
        key="query_profiles",
        label="query profile",
        default_preset="starter",
    )
    match profiles_config["source"]:
        case "preset":
            preset = profiles_config["preset"]
            assert preset == "starter", f"Unknown query profile preset: {preset}"
            source = f"preset:{preset}"
            records = load_records(
                DATA_DIR / "starter_query_profiles.yaml",
                label="query profile preset",
            )
        case "inline":
            source = "inline"
            records = profiles_config["records"]
        case "path":
            path = profiles_config["path"]
            source = str(path)
            records = load_records(path, label="query profile file")
        case _:
            raise AssertionError(f"Unknown query profile source: {profiles_config['source']}")

    return _normalize_query_profiles(records, source=source)


def assign_query_profiles(
    count: int,
    cfg: Record,
    *,
    seed: int | None,
) -> list[QueryProfile]:
    profiles = load_query_profiles(cfg)
    assert profiles, "No query profiles available for memorization generation"

    weighted_profiles = [profile for profile in profiles for _ in range(profile["weight"])]

    rng = Random(0 if seed is None else seed)
    rng.shuffle(weighted_profiles)

    return [weighted_profiles[index % len(weighted_profiles)] for index in range(count)]


def _normalize_query_profiles(records: list[Record], *, source: str) -> list[QueryProfile]:
    normalized: list[QueryProfile] = []
    for index, record in enumerate(records):
        tags = record.get("tags", [])
        exemplars = record.get("exemplars", [])
        meta = record.get("meta", {})
        weight = record.get("weight", 1)

        assert isinstance(tags, list), "query profile tags must be a list"
        assert isinstance(exemplars, list), "query profile exemplars must be a list"
        assert isinstance(meta, dict), "query profile meta must be a mapping"
        assert isinstance(weight, int), "query profile weight must be an integer"
        assert weight > 0, "query profile weight must be positive"

        profile_id = str(record.get("profile_id") or f"profile-{index:03d}")

        normalized.append(
            {
                "profile_id": profile_id,
                "source": source,
                "name": str(record.get("name") or profile_id),
                "weight": weight,
                "channel": str(record.get("channel") or "chat"),
                "fluency": str(record.get("fluency") or "native_or_unspecified"),
                "register": str(record.get("register") or "neutral"),
                "urgency": str(record.get("urgency") or "normal"),
                "query_shape": str(record.get("query_shape") or "full_question"),
                "noise_level": str(record.get("noise_level") or "none"),
                "instructions": str(record.get("instructions") or "Write a normal user query."),
                "exemplars": [str(item) for item in exemplars if str(item).strip()],
                "tags": [str(item) for item in tags],
                "meta": dict(meta),
            }
        )
    return normalized
