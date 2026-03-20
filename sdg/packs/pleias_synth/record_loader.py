from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from sdg.commons import store
from sdg.commons.utils import read_json, read_yaml
from sdg.packs.pleias_synth.types import Record


class PresetSourceConfig(TypedDict):
    source: Literal["preset"]
    preset: str


class InlineRecordSourceConfig(TypedDict):
    source: Literal["inline"]
    item: Record


class InlineRecordsSourceConfig(TypedDict):
    source: Literal["inline"]
    items: list[Record]


class PathSourceConfig(TypedDict):
    source: Literal["path"]
    path: Path


SingleRecordSourceConfig = PresetSourceConfig | InlineRecordSourceConfig | PathSourceConfig
RecordListSourceConfig = PresetSourceConfig | InlineRecordsSourceConfig | PathSourceConfig


def load_record(path: Path, *, label: str) -> Record:
    return one_record(_load_path_data(path), label=f"{label} at {path}")


def load_records(path: Path, *, label: str) -> list[Record]:
    return as_records(_load_path_data(path), label=f"{label} at {path}")


def load_record_source(
    cfg: Record,
    *,
    key: str,
    label: str,
    presets: dict[str, Path],
    aliases: tuple[str, ...] = (),
) -> tuple[str, Record]:
    source_config = load_single_source_config(
        cfg,
        key=key,
        label=label,
        default_preset=_default_preset(presets),
        aliases=aliases,
    )
    return _resolve_record_source(source_config, label=label, presets=presets)


def load_records_source(
    cfg: Record,
    *,
    key: str,
    label: str,
    presets: dict[str, Path],
    aliases: tuple[str, ...] = (),
) -> tuple[str, list[Record]]:
    source_config = load_record_list_source_config(
        cfg,
        key=key,
        label=label,
        default_preset=_default_preset(presets),
        aliases=aliases,
    )
    return _resolve_records_source(source_config, label=label, presets=presets)


def load_single_source_config(
    cfg: Record,
    *,
    key: str,
    label: str,
    default_preset: str,
    aliases: tuple[str, ...] = (),
) -> SingleRecordSourceConfig:
    raw_config = _raw_source_config(cfg, key=key, label=label, aliases=aliases)
    if raw_config is None:
        return {"source": "preset", "preset": default_preset}

    source = _required_str(raw_config, "source", label=f"{label} source")

    match source:
        case "preset":
            return {
                "source": "preset",
                "preset": _required_str(raw_config, "preset", label=f"{label} preset"),
            }
        case "inline":
            return {
                "source": "inline",
                "item": one_record(_required_value(raw_config, "item", label=f"{label} inline item"), label=label),
            }
        case "path":
            return {"source": "path", "path": _resolve_path(raw_config, label=label)}
        case _:
            raise AssertionError(f"Unsupported {label} source: {source}")


def load_record_list_source_config(
    cfg: Record,
    *,
    key: str,
    label: str,
    default_preset: str,
    aliases: tuple[str, ...] = (),
) -> RecordListSourceConfig:
    raw_config = _raw_source_config(cfg, key=key, label=label, aliases=aliases)
    if raw_config is None:
        return {"source": "preset", "preset": default_preset}

    source = _required_str(raw_config, "source", label=f"{label} source")

    match source:
        case "preset":
            return {
                "source": "preset",
                "preset": _required_str(raw_config, "preset", label=f"{label} preset"),
            }
        case "inline":
            return {
                "source": "inline",
                "items": as_records(_required_value(raw_config, "items", label=f"{label} inline items"), label=label),
            }
        case "path":
            return {"source": "path", "path": _resolve_path(raw_config, label=label)}
        case _:
            raise AssertionError(f"Unsupported {label} source: {source}")


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


def _resolve_record_source(
    source_config: SingleRecordSourceConfig,
    *,
    label: str,
    presets: dict[str, Path],
) -> tuple[str, Record]:
    match source_config["source"]:
        case "preset":
            preset = source_config["preset"]
            path = _preset_path(presets, preset, label=label)
            return f"preset:{preset}", load_record(path, label=f"{label} preset")
        case "inline":
            return "inline", source_config["item"]
        case "path":
            path = source_config["path"]
            return str(path), load_record(path, label=f"{label} file")
        case _:
            raise AssertionError(f"Unsupported {label} source: {source_config['source']}")


def _resolve_records_source(
    source_config: RecordListSourceConfig,
    *,
    label: str,
    presets: dict[str, Path],
) -> tuple[str, list[Record]]:
    match source_config["source"]:
        case "preset":
            preset = source_config["preset"]
            path = _preset_path(presets, preset, label=label)
            return f"preset:{preset}", load_records(path, label=f"{label} preset")
        case "inline":
            return "inline", source_config["items"]
        case "path":
            path = source_config["path"]
            return str(path), load_records(path, label=f"{label} file")
        case _:
            raise AssertionError(f"Unsupported {label} source: {source_config['source']}")


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


def _resolve_path(raw_config: Record, *, label: str) -> Path:
    raw_path = _required_str(raw_config, "path", label=f"{label} path")
    return Path(raw_path).expanduser().resolve()


def _preset_path(presets: dict[str, Path], preset: str, *, label: str) -> Path:
    assert preset in presets, f"Unknown {label} preset: {preset}"
    return presets[preset]


def _default_preset(presets: dict[str, Path]) -> str:
    assert presets, "presets must not be empty"
    return next(iter(presets))


def _required_value(record: Record, key: str, *, label: str) -> object:
    value = record.get(key)
    assert value is not None, f"{label} is required"
    return value


def _required_str(record: Record, key: str, *, label: str) -> str:
    value = _required_value(record, key, label=label)
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    return value
