from __future__ import annotations

from pathlib import Path

from sdg.packs.pleias_synth.record_loader import load_record, load_source_config
from sdg.packs.pleias_synth.types import AssistantStyle, Record

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_assistant_style(cfg: Record) -> AssistantStyle:
    style_config = load_source_config(
        cfg,
        key="assistant_style",
        label="assistant style",
        default_preset="starter",
        aliases=("assistant_styles",),
        allow_item=True,
    )
    match style_config["source"]:
        case "preset":
            preset = style_config["preset"]
            assert preset == "starter", f"Unknown assistant style preset: {preset}"
            source = f"preset:{preset}"
            record = load_record(
                DATA_DIR / "starter_assistant_styles.yaml",
                label="assistant style preset",
            )
        case "inline":
            source = "inline"
            records = style_config["records"]
            assert len(records) == 1, "assistant style inline config must contain exactly one record"
            record = records[0]
        case "path":
            path = style_config["path"]
            source = str(path)
            record = load_record(path, label="assistant style file")
        case _:
            raise AssertionError(f"Unknown assistant style source: {style_config['source']}")

    return _normalize_assistant_style(record, source=source)


def _normalize_assistant_style(record: Record, *, source: str) -> AssistantStyle:
    tags = record.get("tags", [])
    exemplars = record.get("exemplars", [])
    meta = record.get("meta", {})

    assert isinstance(tags, list), "assistant style tags must be a list"
    assert isinstance(exemplars, list), "assistant style exemplars must be a list"
    assert isinstance(meta, dict), "assistant style meta must be a mapping"

    style_id = str(record.get("style_id") or "assistant-style")

    return {
        "style_id": style_id,
        "source": source,
        "name": str(record.get("name") or style_id),
        "tone": str(record.get("tone") or "neutral"),
        "detail_level": str(record.get("detail_level") or "balanced"),
        "structure": str(record.get("structure") or "plain paragraphs"),
        "voice": str(record.get("voice") or "helpful and factual"),
        "formatting_style": str(
            record.get("formatting_style")
            or "plain text with minimal formatting"
        ),
        "punctuation_style": str(
            record.get("punctuation_style")
            or "standard punctuation with minimal flourish"
        ),
        "instructions": str(record.get("instructions") or "Answer clearly and directly."),
        "exemplars": [str(item) for item in exemplars if str(item).strip()],
        "tags": [str(item) for item in tags],
        "meta": dict(meta),
    }
