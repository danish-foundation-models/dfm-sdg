from __future__ import annotations

from typing import Literal, TypedDict, cast

from sdg.packs.pleias_synth.types import Record

LanguageCode = Literal["en", "da"]

LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "en": "English",
    "da": "Danish",
}


class SameLanguagePlan(TypedDict):
    kind: Literal["same_language"]
    source: LanguageCode
    prompt: LanguageCode
    reasoning: LanguageCode
    target: LanguageCode


class CrossLanguagePlan(TypedDict):
    kind: Literal["cross_language"]
    source: LanguageCode
    prompt: LanguageCode
    reasoning: LanguageCode
    target: LanguageCode


LanguagePlan = SameLanguagePlan | CrossLanguagePlan


def language_name(code: LanguageCode) -> str:
    return LANGUAGE_NAMES[code]


def source_language_from_memory_cfg(memory_cfg: Record) -> LanguageCode:
    source_language = memory_cfg.get("source_language")
    legacy_language = memory_cfg.get("language")
    if source_language is not None and legacy_language is not None:
        assert source_language == legacy_language, "memory_core source_language and language must match"

    value = source_language if source_language is not None else legacy_language
    if value is None:
        return "en"
    return _language_code(value, label="memory_core source language")


def load_language_plan(cfg: Record) -> LanguagePlan:
    memory_cfg = cfg.get("memory_core", {})
    assert isinstance(memory_cfg, dict), "memory_core config must be a mapping"
    source = source_language_from_memory_cfg(memory_cfg)

    generation_cfg = cfg.get("generation", {})
    assert isinstance(generation_cfg, dict), "generation config must be a mapping"

    memorization_cfg = generation_cfg.get("memorization", {})
    assert isinstance(memorization_cfg, dict), "memorization config must be a mapping"

    raw_plan = memorization_cfg.get("language_plan")
    if raw_plan is None:
        return {
            "kind": "same_language",
            "source": source,
            "prompt": source,
            "reasoning": source,
            "target": source,
        }

    assert isinstance(raw_plan, dict), "memorization language_plan must be a mapping"
    prompt = _required_language(raw_plan, "prompt", label="memorization language_plan prompt")
    reasoning = _required_language(raw_plan, "reasoning", label="memorization language_plan reasoning")
    target = _required_language(raw_plan, "target", label="memorization language_plan target")

    kind: Literal["same_language", "cross_language"]
    if prompt == source and reasoning == source and target == source:
        kind = "same_language"
    else:
        kind = "cross_language"

    return {
        "kind": kind,
        "source": source,
        "prompt": prompt,
        "reasoning": reasoning,
        "target": target,
    }


def _required_language(record: Record, key: str, *, label: str) -> LanguageCode:
    value = record.get(key)
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    return _language_code(value, label=label)


def _language_code(value: object, *, label: str) -> LanguageCode:
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    assert value in LANGUAGE_NAMES, f"Unsupported {label}: {value}"
    return cast(LanguageCode, value)
