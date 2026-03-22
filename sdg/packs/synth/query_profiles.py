from __future__ import annotations

from pathlib import Path
from random import Random

from sdg.packs.synth.record_loader import load_records_source
from sdg.packs.synth.types import QueryProfile, Record

DATA_DIR = Path(__file__).resolve().parent / "data"
PRESETS = {"starter": DATA_DIR / "starter_query_profiles.yaml"}


def load_query_profiles(cfg: Record, *, family: str = "memorization") -> list[QueryProfile]:
    source, records = load_records_source(
        cfg,
        family=family,
        key="query_profiles",
        label="query profile",
        presets=PRESETS,
    )
    return _normalize_query_profiles(records, source=source)


def assign_query_profiles(
    count: int,
    cfg: Record,
    *,
    family: str = "memorization",
    seed: int | None,
) -> list[QueryProfile]:
    weighted_profiles = load_weighted_query_profiles(cfg, family=family, seed=seed)
    return [weighted_profiles[index % len(weighted_profiles)] for index in range(count)]


def load_weighted_query_profiles(
    cfg: Record,
    *,
    family: str = "memorization",
    seed: int | None,
) -> list[QueryProfile]:
    profiles = load_query_profiles(cfg, family=family)
    assert profiles, f"No query profiles available for {family} generation"

    weighted_profiles = [profile for profile in profiles for _ in range(profile["weight"])]
    rng = Random(0 if seed is None else seed)
    rng.shuffle(weighted_profiles)
    return weighted_profiles


def _normalize_query_profiles(records: list[Record], *, source: str) -> list[QueryProfile]:
    normalized: list[QueryProfile] = []
    for record in records:
        normalized.append(
            {
                "profile_id": _required_str(record, "profile_id", label="query profile profile_id"),
                "source": source,
                "name": _required_str(record, "name", label="query profile name"),
                "weight": _positive_int(record, "weight", label="query profile weight"),
                "channel": _required_str(record, "channel", label="query profile channel"),
                "fluency": _required_str(record, "fluency", label="query profile fluency"),
                "register": _required_str(record, "register", label="query profile register"),
                "urgency": _required_str(record, "urgency", label="query profile urgency"),
                "query_shape": _required_str(record, "query_shape", label="query profile query_shape"),
                "noise_level": _required_str(record, "noise_level", label="query profile noise_level"),
                "instructions": _required_str(record, "instructions", label="query profile instructions"),
                "exemplars": _string_list(record, "exemplars", label="query profile exemplars"),
                "tags": _string_list(record, "tags", label="query profile tags"),
                "meta": _meta(record, label="query profile meta"),
            }
        )
    return normalized


def _required_str(record: Record, key: str, *, label: str) -> str:
    value = record.get(key)
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    return value


def _positive_int(record: Record, key: str, *, label: str) -> int:
    value = record.get(key)
    assert isinstance(value, int) and value > 0, f"{label} must be a positive integer"
    return value


def _string_list(record: Record, key: str, *, label: str) -> list[str]:
    value = record.get(key, [])
    assert isinstance(value, list), f"{label} must be a list"
    return [str(item) for item in value if str(item).strip()]


def _meta(record: Record, *, label: str) -> Record:
    value = record.get("meta", {})
    assert isinstance(value, dict), f"{label} must be a mapping"
    return dict(value)
