from __future__ import annotations

from pathlib import Path

from sdg.packs.synth.assistant_styles import load_assistant_style
from sdg.packs.synth.personas import build_query_plans, load_personas
from sdg.packs.synth.query_profiles import load_query_profiles


def test_starter_personas_load_by_default() -> None:
    personas = load_personas({"generation": {"memorization": {}}})
    assert len(personas) >= 5
    assert personas[0]["persona_id"]
    assert personas[0]["source"].startswith("preset:")
    assert personas[0]["preferred_angles"]


def test_personas_can_load_from_external_jsonl(tmp_path) -> None:
    persona_path = tmp_path / "personas.jsonl"
    persona_path.write_text(
        '{"persona_id":"archivist","name":"Archivist","intent":"recover a precise fact","knowledge_level":"expert","tone":"direct","question_style":"targeted","answer_granularity":"short factual answer","preferred_angles":["attribute_lookup"]}\n'
    )

    cfg = {
        "generation": {
            "memorization": {
                "personas": {
                    "source": "path",
                    "path": str(persona_path),
                }
            }
        }
    }
    personas = load_personas(cfg)

    assert len(personas) == 1
    assert personas[0]["persona_id"] == "archivist"
    assert personas[0]["source"] == str(persona_path)


def test_query_plans_carry_persona_and_angle() -> None:
    cfg = {"generation": {"memorization": {}}}
    fact_bundles = [
        {
            "doc": {"id": "a", "title": "Alpha", "text": "Alpha is first."},
            "primary_sentence": "Alpha is first.",
            "sentence_index": 0,
            "support_sentences": [],
            "structured_facts": [],
        },
        {
            "doc": {"id": "b", "title": "Beta", "text": "Beta is second."},
            "primary_sentence": "Beta is second.",
            "sentence_index": 0,
            "support_sentences": [],
            "structured_facts": [],
        },
    ]

    plans = build_query_plans(fact_bundles, cfg, seed=3)

    assert len(plans) == 2
    assert plans[0]["persona"]["persona_id"]
    assert plans[0]["query_angle"]
    assert plans[0]["query_profile"]["profile_id"]
    assert plans[0]["assistant_style"]["style_id"]
    assert plans[0]["assistant_style"]["style_id"] == plans[1]["assistant_style"]["style_id"]


def test_starter_query_profiles_load_by_default() -> None:
    profiles = load_query_profiles({"generation": {"memorization": {}}})
    assert len(profiles) >= 4
    assert profiles[0]["profile_id"]
    assert profiles[0]["source"].startswith("preset:")
    assert profiles[0]["weight"] >= 1


def test_starter_assistant_style_loads_by_default() -> None:
    style = load_assistant_style({"generation": {"memorization": {}}})
    assert style["style_id"] == "synth_assistant"
    assert style["source"].startswith("preset:")
    assert style["instructions"]
    assert "minimal formatting" in style["formatting_style"]
    assert "em dashes" in style["punctuation_style"]


def test_assistant_style_can_load_from_external_yaml(tmp_path: Path) -> None:
    style_path = tmp_path / "assistant_style.yaml"
    style_path.write_text(
        "\n".join(
            [
                "style_id: custom_assistant",
                "name: Custom Assistant",
                "tone: precise",
                "detail_level: concise",
                "structure: lead with the answer",
                "voice: factual",
                "formatting_style: minimal formatting",
                "punctuation_style: standard punctuation",
                "instructions: Answer directly.",
            ]
        )
    )

    cfg = {
        "generation": {
            "memorization": {
                "assistant_style": {
                    "source": "path",
                    "path": str(style_path),
                }
            }
        }
    }

    style = load_assistant_style(cfg)
    assert style["style_id"] == "custom_assistant"
    assert style["source"] == str(style_path)


def test_assistant_style_inline_item_loads_single_record() -> None:
    cfg = {
        "generation": {
            "memorization": {
                "assistant_style": {
                    "source": "inline",
                    "item": {
                        "style_id": "inline_assistant",
                        "name": "Inline Assistant",
                        "tone": "direct",
                        "detail_level": "concise",
                        "structure": "lead with the answer",
                        "voice": "factual",
                        "formatting_style": "minimal formatting",
                        "punctuation_style": "standard punctuation",
                        "instructions": "Answer directly.",
                    },
                }
            }
        }
    }

    style = load_assistant_style(cfg)

    assert style["style_id"] == "inline_assistant"
    assert style["source"] == "inline"
