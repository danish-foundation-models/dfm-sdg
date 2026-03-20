from __future__ import annotations

from pathlib import Path

from sdg.packs.pleias_synth.record_loader import load_record_source
from sdg.packs.pleias_synth.types import AssistantStyle, Record

DATA_DIR = Path(__file__).resolve().parent / "data"
PRESETS = {"starter": DATA_DIR / "starter_assistant_styles.yaml"}


def load_assistant_style(cfg: Record) -> AssistantStyle:
    source, record = load_record_source(
        cfg,
        key="assistant_style",
        label="assistant style",
        presets=PRESETS,
        aliases=("assistant_styles",),
    )
    return _normalize_assistant_style(record, source=source)


def _normalize_assistant_style(record: Record, *, source: str) -> AssistantStyle:
    return {
        "style_id": _required_str(record, "style_id", label="assistant style style_id"),
        "source": source,
        "name": _required_str(record, "name", label="assistant style name"),
        "tone": _required_str(record, "tone", label="assistant style tone"),
        "detail_level": _required_str(record, "detail_level", label="assistant style detail_level"),
        "structure": _required_str(record, "structure", label="assistant style structure"),
        "voice": _required_str(record, "voice", label="assistant style voice"),
        "formatting_style": _required_str(record, "formatting_style", label="assistant style formatting_style"),
        "punctuation_style": _required_str(record, "punctuation_style", label="assistant style punctuation_style"),
        "instructions": _required_str(record, "instructions", label="assistant style instructions"),
        "exemplars": _string_list(record, "exemplars", label="assistant style exemplars"),
        "tags": _string_list(record, "tags", label="assistant style tags"),
        "meta": _meta(record, label="assistant style meta"),
    }


def _required_str(record: Record, key: str, *, label: str) -> str:
    value = record.get(key)
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    return value


def _string_list(record: Record, key: str, *, label: str) -> list[str]:
    value = record.get(key, [])
    assert isinstance(value, list), f"{label} must be a list"
    return [str(item) for item in value if str(item).strip()]


def _meta(record: Record, *, label: str) -> Record:
    value = record.get("meta", {})
    assert isinstance(value, dict), f"{label} must be a mapping"
    return dict(value)
