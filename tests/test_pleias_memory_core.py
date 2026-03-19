from __future__ import annotations

import json
from pathlib import Path

from sdg.commons import store
from sdg.packs.pleias_synth import gen_memorization
from sdg.packs.pleias_synth.build import build, publish, summarize, verify
from sdg.packs.pleias_synth.verify import _answer_supported


class FakeLLM:
    def chat(self, messages, temperature=0.0):
        return self._respond(messages)

    async def achat(self, messages, temperature=0.0):
        return self._respond(messages)

    def _respond(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]

        if "You plan realistic memorization tasks" in system:
            primary_claim = _line_value(user, "Primary claim")
            return json.dumps(
                {
                    "task_type": "overview",
                    "user_goal": "understand the topic",
                    "answer_shape": "brief explanation",
                    "coverage_points": [primary_claim],
                    "query_brief": "Ask for a brief grounded explanation.",
                }
            )

        if "You create memorization questions" in system:
            title = _line_value(user, "Article title")
            return json.dumps(
                {
                    "prompt": f"What is {title} and why is it notable?",
                    "question_type": "overview",
                }
            )

        if "You write strongly opinionated recall-style reasoning" in system:
            primary_claim = _line_value(user, "Primary claim")
            return json.dumps(
                {
                    "key_question": "What core remembered fact answers the user's question?",
                    "assumption_check": "",
                    "known_facts": [primary_claim],
                    "reasoning_steps": ["Use the strongest remembered claim and answer directly."],
                    "caveats": [],
                    "synthesis": "The answer should come straight from the remembered core fact.",
                    "proposed_target": primary_claim,
                }
            )

        if "You write the final assistant response" in system:
            primary_claim = _line_value(user, "Primary claim")
            return json.dumps({"target": primary_claim})

        if "You judge synthetic memorization examples" in system:
            return json.dumps(
                {
                    "pass": True,
                    "support": True,
                    "leakage": False,
                    "style_distinct": True,
                    "reasoning_quality": True,
                    "reason": "",
                }
            )

        raise AssertionError(f"Unexpected prompt family: {system}")


def _line_value(text: str, label: str) -> str:
    prefix = f"{label}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"Missing {label} in prompt:\n{text}")


def test_pleias_memory_core_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    fake_llm = FakeLLM()
    monkeypatch.setattr(
        gen_memorization,
        "_load_memorization_models",
        lambda cfg: {
            "task_planner": fake_llm,
            "query_teacher": fake_llm,
            "reasoning_teacher": fake_llm,
            "answer_teacher": fake_llm,
            "judge": fake_llm,
        },
    )

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "pleias_sources.jsonl"
    cfg = {
        "pack": "pleias_synth",
        "reuse_completed": True,
        "memory_core": {
            "source_path": str(fixture_path),
            "chunk_size": 20,
            "chunk_overlap": 5,
        },
        "generation": {
            "families": ["memorization"],
            "max_rows_per_family": 4,
            "train_fraction": 0.75,
            "memorization": {
                "use_llm": True,
                "lead_sentences": 2,
                "max_sentences_per_doc": 1,
                "retrieve_top_k": 3,
            },
        },
    }

    result = build(cfg)
    assert "memorization_rows" in result.artifacts

    verification = verify(result.run_id)
    assert verification["metrics"]["memory_core"]["sources"] == 3
    assert verification["metrics"]["memory_core"]["chunks"] >= 3
    assert verification["metrics"]["memorization"]["rows"] >= 2
    assert verification["metrics"]["memorization"]["checks"]["has_reasoning"]["failed"] == 0
    assert verification["metrics"]["memorization"]["checks"]["retrieval_grounded"]["failed"] == 0
    assert verification["metrics"]["memorization"]["checks"]["reasoning_grounded"]["failed"] == 0
    assert verification["metrics"]["memorization"]["checks"]["coverage_supported"]["failed"] == 0
    assert verification["failure_summary"]["memory_core"] == {}

    summary = summarize(result.run_id)
    assert summary["sources"] == 3
    assert summary["chunks"] >= 3
    assert summary["generated_rows"] >= 2

    published = publish(result.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "memory_chunks.jsonl").exists()
    assert (out_dir / "retrieval_index.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "memorization_candidates.jsonl").exists()
    assert (out_dir / "memorization_rejected.jsonl").exists()

    train_rows = store.read_parquet(out_dir / "train.parquet")
    assert train_rows
    assert "hidden" not in train_rows[0]
    assert "reasoning" in train_rows[0]
    assert train_rows[0]["reasoning"]
    assert "teacher bundle" not in train_rows[0]["reasoning"].lower()
    assert "provided text" not in train_rows[0]["reasoning"].lower()
    assert "retrieved support" not in train_rows[0]["reasoning"].lower()
    assert "evidence frame" not in train_rows[0]["reasoning"].lower()
    assert "persona_id" in train_rows[0]["meta"]
    assert "query_angle" in train_rows[0]["meta"]
    assert "query_profile_id" in train_rows[0]["meta"]
    assert "assistant_style_id" in train_rows[0]["meta"]
    assert "task_type" in train_rows[0]["meta"]
    assert "user_goal" in train_rows[0]["meta"]
    assert train_rows[0]["meta"]["reasoning_style"] == "teacher_backreasoning_v1"
    assert len({row["meta"]["assistant_style_id"] for row in train_rows}) == 1


def test_answer_supported_accepts_full_response() -> None:
    row = {
        "target": "The film follows a voyage to Jupiter to investigate an alien monolith.",
        "meta": {"question_type": "definition"},
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
        },
        "sources": [],
    }

    assert _answer_supported(row)


def test_answer_supported_rejects_unsupported_response() -> None:
    row = {
        "target": "The film is mainly a courtroom drama about corporate fraud on Earth.",
        "meta": {"question_type": "definition"},
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
        },
        "sources": [],
    }

    assert not _answer_supported(row)
