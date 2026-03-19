from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from sdg.commons import store
from sdg.commons.utils import read_json, read_yaml
from sdg.packs.pleias_synth.types import Record


class PresetSourceConfig(TypedDict):
    source: Literal["preset"]
    preset: str


class InlineSourceConfig(TypedDict):
    source: Literal["inline"]
    records: list[Record]


class PathSourceConfig(TypedDict):
    source: Literal["path"]
    path: Path


SourceConfig = PresetSourceConfig | InlineSourceConfig | PathSourceConfig


def load_record(path: Path, *, label: str) -> Record:
    return one_record(_load_path_data(path), label=f"{label} at {path}")


def load_records(path: Path, *, label: str) -> list[Record]:
    return as_records(_load_path_data(path), label=f"{label} at {path}")


def load_source_config(
    cfg: Record,
    *,
    key: str,
    label: str,
    default_preset: str,
    aliases: tuple[str, ...] = (),
    allow_item: bool = False,
) -> SourceConfig:
    raw_config = _raw_source_config(cfg, key=key, label=label, aliases=aliases)
    if raw_config is None:
        return {"source": "preset", "preset": default_preset}

    source = raw_config.get("source", "preset")
    assert isinstance(source, str), f"{label} source must be a string"

    match source:
        case "preset":
            preset = raw_config.get("preset", default_preset)
            assert isinstance(preset, str) and preset, f"{label} preset must be a non-empty string"
            return {"source": "preset", "preset": preset}
        case "inline":
            return {
                "source": "inline",
                "records": _inline_records(raw_config, label=label, allow_item=allow_item),
            }
        case "path":
            raw_path = raw_config.get("path")
            assert isinstance(raw_path, str) and raw_path, f"{label} path must be a non-empty string"
            return {"source": "path", "path": Path(raw_path).expanduser().resolve()}
        case _:
            raise ValueError(f"Unsupported {label} source: {source}")


def one_record(value: object, *, label: str) -> Record:
    if isinstance(value, dict):
        return value

    records = as_records(value, label=label)
    assert len(records) == 1, f"{label} must contain exactly one record"
    return records[0]


def as_records(value: object, *, label: str) -> list[Record]:
    assert isinstance(value, list), f"{label} must contain a list"
    for record in value:
        assert isinstance(record, dict), f"{label} must contain mappings"
    return value


def _inline_records(
    raw_config: Record,
    *,
    label: str,
    allow_item: bool,
) -> list[Record]:
    raw_records = raw_config.get("items")
    if raw_records is not None:
        return as_records(raw_records, label=label)

    assert allow_item, f"{label} inline config requires items"
    raw_record = raw_config.get("item")
    assert raw_record is not None, f"{label} inline config requires item"
    return [one_record(raw_record, label=label)]


def _raw_source_config(
    cfg: Record,
    *,
    key: str,
    label: str,
    aliases: tuple[str, ...],
) -> Record | None:
    generation_cfg = cfg.get("generation")
    if generation_cfg is None:
        return None
    assert isinstance(generation_cfg, dict), "generation config must be a mapping"

    memorization_cfg = generation_cfg.get("memorization")
    if memorization_cfg is None:
        return None
    assert isinstance(memorization_cfg, dict), "memorization config must be a mapping"

    raw_config = memorization_cfg.get(key)
    for alias in aliases:
        if raw_config is not None:
            break
        raw_config = memorization_cfg.get(alias)

    if raw_config is None:
        return None

    assert isinstance(raw_config, dict), f"{label} config must be a mapping"
    return raw_config


def _load_path_data(path: Path) -> object:
    match path.suffix:
        case ".jsonl":
            return store.read_jsonl(path)
        case ".json":
            return read_json(path)
        case ".yaml" | ".yml":
            return read_yaml(path)
        case ".parquet":
            return store.read_parquet(path)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")
