from __future__ import annotations

from pathlib import Path

from sdg.commons.model import LLM
from sdg.commons.run import load
from sdg.commons.store import read_jsonl, write_jsonl
from sdg.commons.utils import read_json, read_yaml
from sdg.packs.multi_turn_dialogue import build as multi_turn_build
from sdg.packs.multi_turn_dialogue.build import build, publish, summarize, verify


def source_seed_spec_for_tests():
    seed_spec = {
        "seed_id": "source-seed::v00",
        "archetype_id": "source-seed",
        "scenario_variant": multi_turn_build._scenario_variant(0, variant_count=1),
        "scenario_variant_index": 0,
        "scenario_variant_count": 1,
        "family": "grounded_dialogue",
        "domain": "source_case",
        "seed_intent": "explain the source excerpt",
        "seed_blueprint": {"domain": "source_case", "user_persona": "generic"},
        "success_criteria": ["use visible source"],
        "intent_trajectory_hint": "Information Retrieval Interaction",
        "subtask_trajectory_hint": "source_grounded_synthesis",
        "query_trajectory_hint": [],
        "source_methods": list(multi_turn_build.DEFAULT_SOURCE_METHODS),
        "source_language": "da",
        "difficulty_axis": "source boundary",
        "requires_epistemic_hedging": True,
        "volatile_items": ["current status"],
        "dialogue_language": "da",
        "latent_language": "en",
        "source_grounding": {
            "pack_id": "dynaword_public_rules_da",
            "source_name": "retsinformationdk",
            "document_id": "AA014830",
            "title": "Pressenævnets kendelse i sag nr. 2024-10933",
            "created": "2025",
            "excerpt": (
                "Pressenævnets kendelse i sag nr. 2024-10933. BPLN.dk får kritik for artiklen "
                "\"Hvorfor bliver han ved?\"."
            ),
            "license": "varies",
            "url": "https://example.invalid",
        },
        "persona_context": {
            "source": "nvidia/Nemotron-Personas-USA",
            "id": "persona-1",
            "persona": "A practical administrative worker who wants concise checklists.",
        },
        "external_sources": [],
    }
    seed_spec["work_session"] = multi_turn_build._work_session_contract(seed_spec)
    return seed_spec


class FakeDialogueLLM(LLM):
    def __init__(self):
        super().__init__(model="fake-dialogue", base_url="https://example.invalid", api_key_env=None)

    def chat(self, messages, **gen):
        prompt = messages[-1]["content"]
        if "Create one fresh scenario instance" in prompt:
            return (
                "Concrete user situation: a parent wants a bounded Lisbon planning draft for a family trip.\n"
                "Visible first-turn payload: four days, two children, and uncertainty about current prices.\n"
                "Hidden-but-unrevealed details: peanut allergy and daytime flight preference.\n"
                "Likely follow-up turns: budget details, then allergy and flight timing.\n"
                "Deliverable target: concise itinerary assumptions and booking-check checklist.\n"
                "Evidence boundary notes: current prices and availability are unknown unless supplied visibly."
            )
        if "Create one hidden conversation blueprint" in prompt:
            return (
                "Hidden goal: plan a four-day Lisbon family trip under $1800.\n"
                "Known constraints: two children, low cost.\n"
                "Unrevealed constraints: peanut allergy; daytime flights.\n"
                "Work-session contract: practical_solution_session with a bounded family trip plan as the deliverable.\n"
                "Conversation flow: vague request, clarification, reveal, draft, late constraint, revision.\n"
                "Success criteria: mention Lisbon, $1800, peanut, and daytime."
            )
        if "Assign the best intent trajectory" in prompt:
            return "\n".join(
                [
                    "INTENT_TRAJECTORY: Problem Solving Interaction",
                    "SUBTASK_TRAJECTORY: planning_interaction",
                    "INFORMATION_FLOW: vague request -> clarify -> reveal constraints -> draft -> late constraint -> final revision",
                    "ROLE_INTERACTION: user reveals slots gradually; assistant asks and revises",
                    "TOPIC_GUARDRAILS: keep all turns about the Lisbon family trip",
                    "WORK_SESSION_TYPE: practical_solution_session",
                    "DELIVERABLE_PROGRESS: clarify constraints -> draft plan -> bounded final itinerary guidance",
                    "USER_QUERY_PLAN: initial vague trip request; budget and children; allergy and daytime flights",
                ]
            )
        if "Build a 6-step skeleton" in prompt:
            return "\n".join(
                [
                    "STEP 1: user | vague_initial_request | introduce the trip without details",
                    "STEP 2: assistant | clarify_missing_slots | ask for budget, travelers, dates, and constraints",
                    "STEP 3: user | reveal_known_constraints | reveal budget and children",
                    "STEP 4: assistant | draft_candidate_plan | provide an initial plan",
                    "STEP 5: user | late_constraint | add allergy and flight-time constraints",
                    "STEP 6: assistant | revise_final_answer | revise with the full revealed state",
                ]
            )
        if "Write only the user turns" in prompt:
            return "\n".join(
                [
                    "USER 1: Can you help me plan a quick Lisbon trip for my family?",
                    "USER 2: Four days, two kids aged 6 and 10, and about $1800 total if possible.",
                    "USER 3: I forgot one important thing: one child has a peanut allergy, and we cannot take late-night flights.",
                ]
            )
        if "Write ASSISTANT 1 only" in prompt:
            return "ASSISTANT 1: What budget, trip length, traveler ages, and hard constraints should I plan around?"
        if "Write ASSISTANT 2 only" in prompt:
            return (
                "ASSISTANT 2: A low-cost Lisbon version could use central lodging, transit passes, "
                "free viewpoints, and one paid museum day."
            )
        if "Write ASSISTANT 3 only" in prompt:
            return (
                "ASSISTANT 3: Use Lisbon as the base, keep flights daytime only as an assumption, and treat the "
                "$1800 split as an estimate rather than live booking data. For food, flag the peanut allergy in "
                "writing, keep backup grocery options close to the hotel, and verify current fares and availability "
                "before booking."
            )
        if "Review this dialogue" in prompt:
            return "\n".join(
                [
                    "SELECTION: ACCEPT",
                    "SCORE: 0.96",
                    "INTENT_TRAJECTORY: Problem Solving Interaction",
                    "SUBTASK_TRAJECTORY: planning_interaction",
                    "SUCCESS_CRITERIA: coherent incremental reveal; grounded final revision",
                    "TURN_EVIDENCE_AUDIT: ASSISTANT 1: PASS - supported by USER 1; ASSISTANT 2: PASS - supported by USER 1-2; ASSISTANT 3: PASS - supported by USER 1-3",
                    "REVIEWER coherence: PASS - the dialogue stays on one latent task",
                    "REVIEWER naturalness: PASS - constraints are revealed incrementally",
                    "REVIEWER grounding: PASS - final answer uses available facts",
                    "REVIEWER source_boundary: PASS - no private or source-specific facts are invented",
                    "REVIEWER language_quality: PASS - language is clean",
                    "REVIEWER recovery: PASS - final answer revises after the late constraint",
                    "REVIEWER outcome: PASS - final answer gives a bounded usable trip-planning output",
                    "REVIEWER format: PASS - roles alternate",
                    "REPAIR_ACTIONS: none",
                ]
            )
        if "Run a strict audit of this dialogue" in prompt:
            return "\n".join(
                [
                    "STRICT_SELECTION: PASS",
                    "STRICT_SCORE: 0.95",
                    "STRICT_CHECK source_boundary: PASS - no exact source-dependent claims",
                    "STRICT_CHECK language_quality: PASS - clean language",
                    "STRICT_CHECK factuality: PASS - no suspicious facts",
                    "STRICT_CHECK contradiction: PASS - no contradiction",
                    "STRICT_CHECK format: PASS - restrained format",
                    "STRICT_REPAIR_ACTIONS: none",
                ]
            )
        if "Create one rejected final assistant answer" in prompt:
            return "\n".join(
                [
                    "FAILURE_MODE: ignores_late_constraint",
                    "CHOSEN_ACTION: revise_with_updated_state",
                    "REJECTED_ACTION: continue_from_stale_state",
                    "REJECTED_ANSWER: Book the cheapest late-night flights and pick any restaurants near the hotel.",
                ]
            )
        raise AssertionError(f"unexpected prompt: {prompt}")


def test_numbered_block_parser_removes_leading_punctuation_artifact() -> None:
    turns = multi_turn_build._parse_numbered_blocks("ASSISTANT 1: .\nDet her er svaret.", label="ASSISTANT")
    same_line_turns = multi_turn_build._parse_numbered_blocks("ASSISTANT 1: . Det her er svaret.", label="ASSISTANT")

    assert turns == ["Det her er svaret."]
    assert same_line_turns == ["Det her er svaret."]


def test_numbered_block_parser_preserves_paragraph_breaks() -> None:
    turns = multi_turn_build._parse_numbered_blocks(
        "\n".join(
            [
                "ASSISTANT 1: Første afsnit.",
                "",
                "Andet afsnit.",
                "",
                "- Punkt",
            ]
        ),
        label="ASSISTANT",
    )

    assert turns == ["Første afsnit.\n\nAndet afsnit.\n\n- Punkt"]


def test_handwritten_seed_source_loads_richer_seed_cards() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "sdg" / "packs" / "multi_turn_dialogue" / "configs" / "base.yaml"
    cfg = read_yaml(cfg_path)

    specs = multi_turn_build._seed_specs(cfg["generation"])
    domains = {spec["domain"] for spec in specs}
    archetypes = {spec["archetype_id"] for spec in specs}
    variants_per_seed = cfg["generation"]["seed_expansion"]["variants_per_seed"]
    top_level_intents = {spec["intent_trajectory_hint"] for spec in specs}
    subtask_trajectories = {spec["subtask_trajectory_hint"] for spec in specs}

    assert len(specs) == len(archetypes) * variants_per_seed
    assert len(archetypes) >= 24
    assert top_level_intents <= multi_turn_build.CONSISTENTCHAT_INTENT_NAMES
    assert "Problem Solving Interaction" in top_level_intents
    assert "Information Retrieval Interaction" in top_level_intents
    assert "planning_interaction" in subtask_trajectories
    assert "refinement_interaction" in subtask_trajectories
    assert "statistics_tutoring" in domains
    assert "product_comparison" in domains
    assert "tenant_email_revision" in domains
    assert "tax_planning" in domains
    assert "legal_form_assistance" in domains
    assert "medication_advice" in domains
    assert "insurance_claim_explanation" in domains
    assert "software_debugging" in domains
    assert all(spec["query_trajectory_hint"] for spec in specs)
    assert all("private_fact_markers" not in spec for spec in specs)
    assert all("forbidden_assistant_claim_markers" not in spec for spec in specs)
    assert all(spec["scenario_variant"]["first_turn_shape"] for spec in specs)
    assert all(spec["work_session"]["top_level_intent"] in multi_turn_build.CONSISTENTCHAT_INTENT_NAMES for spec in specs)
    assert all(spec["work_session"]["fit"].startswith(spec["intent_trajectory_hint"]) for spec in specs)
    assert all(spec["work_session"]["work_session_type"] for spec in specs)
    assert all(spec["scenario_variant"]["work_session_focus"] for spec in specs)
    assert all(spec["scenario_variant"]["opener_payload_rule"] for spec in specs)
    assert any(spec["requires_epistemic_hedging"] for spec in specs)


def test_danish_seed_specs_keep_latent_english_and_visible_danish() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "sdg" / "packs" / "multi_turn_dialogue" / "configs" / "base_da.yaml"
    cfg = read_yaml(cfg_path)

    specs = multi_turn_build._seed_specs(cfg["generation"])
    variants_per_seed = cfg["generation"]["seed_expansion"]["variants_per_seed"]
    archetype_count = len(specs) // variants_per_seed
    first_round = specs[:archetype_count]
    second_round = specs[archetype_count : archetype_count * 2]

    assert archetype_count >= 20
    assert len(specs) == archetype_count * variants_per_seed
    assert len({spec["archetype_id"] for spec in first_round}) == archetype_count
    assert {spec["scenario_variant_index"] for spec in first_round} == {0}
    assert len({spec["archetype_id"] for spec in second_round}) == archetype_count
    assert {spec["scenario_variant_index"] for spec in second_round} == {1}
    assert {spec["dialogue_language"] for spec in specs} == {"da"}
    assert {spec["latent_language"] for spec in specs} == {"en"}
    assert {spec["source_language"] for spec in specs} == {"da"}
    assert {spec["intent_trajectory_hint"] for spec in specs} == multi_turn_build.CONSISTENTCHAT_INTENT_NAMES
    assert "advisory_interaction" in {spec["subtask_trajectory_hint"] for spec in specs}
    assert "role_simulation_interaction" in {spec["subtask_trajectory_hint"] for spec in specs}
    assert "interactive_quiz_game" in {spec["subtask_trajectory_hint"] for spec in specs}
    assert "stress_aware_planning_interaction" in {spec["subtask_trajectory_hint"] for spec in specs}
    assert {spec["scenario_variant"]["target_user_turns"] for spec in specs} == {3, 4, 5}
    assert {spec["work_session"]["top_level_intent"] for spec in specs} == multi_turn_build.CONSISTENTCHAT_INTENT_NAMES
    assert "bounded_artifact_completion_session" in {spec["work_session"]["work_session_type"] for spec in specs}
    assert "supportive_next_step_session" in {spec["work_session"]["work_session_type"] for spec in specs}
    assert specs[0]["seed_id"].endswith("::v00")
    assert specs[0]["success_criteria"]
    assert "required_final_markers" not in specs[0]


def test_dynaword_danish_config_samples_broad_source_packs() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "sdg"
        / "packs"
        / "multi_turn_dialogue"
        / "configs"
        / "dynaword_public_rules_da.yaml"
    )
    cfg = read_yaml(cfg_path)
    sources = [source for source in cfg["generation"]["seed_sources"] if source.get("type") == "source_pack"]
    source_ids = {source["id"] for source in sources}
    config_names = {source["documents"]["config_name"] for source in sources}

    assert "dynaword_tidsskrift_dk_da" in source_ids
    assert "tidsskrift-dk" in config_names
    assert {
        "retsinformationdk",
        "skat",
        "domsdatabasen",
        "retspraksis",
        "eur-lex-sum-da",
        "tidsskrift-dk",
        "miljoeportalen",
        "fm-udgivelser",
        "ai-aktindsigt",
        "ft",
        "ep",
        "municipality_meetings",
        "danske-taler",
        "health_hovedstaden",
    } <= config_names


def test_source_pack_seed_source_pairs_documents_with_personas(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    personas_path = tmp_path / "personas.jsonl"
    write_jsonl(
        [
            {
                "id": "doc-1",
                "source": "skat",
                "created": "2025-01-01",
                "text": "Dette er et synligt dansk kildeuddrag om frister og oplysninger. " * 20,
            },
            {
                "id": "doc-2",
                "source": "ignored",
                "text": "Denne kilde skal filtreres fra. " * 20,
            },
        ],
        documents_path,
    )
    write_jsonl(
        [
            {
                "uuid": "persona-1",
                "age": 37,
                "occupation": "bookkeeper",
                "persona": "A careful person who wants practical checklists.",
                "professional_persona": "Works with small-business administration.",
            },
            {
                "uuid": "persona-2",
                "age": 12,
                "persona": "This minor persona should be filtered by min_age.",
            },
        ],
        personas_path,
    )
    generation = {
        "languages": ["da"],
        "latent_language": "en",
        "seed_expansion": {"variants_per_seed": 1},
        "seed_sources": [
            {
                "type": "source_pack",
                "id": "test_public_rules",
                "family": "grounded_dialogue",
                "documents": {
                    "path": str(documents_path),
                    "text_field": "text",
                    "id_field": "id",
                    "source_field": "source",
                    "created_field": "created",
                    "min_chars": 100,
                    "max_chars": 500,
                    "max_records": 4,
                    "include_values": {"source": ["skat"]},
                },
                "personas": {
                    "path": str(personas_path),
                    "id_field": "uuid",
                    "persona_fields": ["persona", "professional_persona"],
                    "min_age": 18,
                    "max_records": 4,
                },
            }
        ],
    }

    specs = multi_turn_build._seed_specs(generation)

    assert len(specs) == 1
    spec = specs[0]
    assert spec["source_grounding"]["pack_id"] == "test_public_rules"
    assert spec["source_grounding"]["document_id"] == "doc-1"
    assert spec["source_grounding"]["source_name"] == "skat"
    assert spec["persona_context"]["id"] == "persona-1"
    assert spec["family"] == "grounded_dialogue"
    assert spec["requires_epistemic_hedging"] is True
    assert spec["external_sources"][0]["kind"] == "source_excerpt"
    assert spec["external_sources"][1]["kind"] == "persona_source"
    assert spec["source_payload_plan"]["strategy"] in multi_turn_build.DEFAULT_SOURCE_PAYLOAD_STRATEGIES
    assert "This minor persona" not in spec["persona_context"]["persona"]


def test_source_excerpt_trimming_preserves_document_formatting() -> None:
    text = (
        "\r\n"
        "§ 1. Overskrift\r\n"
        "\r\n"
        "  Stk. 1. Første linje med indrykning.\r\n"
        "  - punkt A\r\n"
        "  - punkt B\r\n"
        "\r\n"
        "§ 2. Næste afsnit med mere tekst.\r\n"
    )

    trimmed = multi_turn_build._trim_source_excerpt(text, max_chars=1_000)

    assert trimmed.startswith("§ 1. Overskrift\n\n")
    assert "  Stk. 1. Første linje med indrykning.\n  - punkt A\n  - punkt B" in trimmed
    assert "§ 1. Overskrift Stk. 1." not in trimmed


def test_source_pack_payload_strategy_varies_across_seed_expansion(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    personas_path = tmp_path / "personas.jsonl"
    write_jsonl(
        [
            {
                "id": "doc-1",
                "source": "skat",
                "created": "2025-01-01",
                "text": "Dette er et længere dansk kildeuddrag om frister, beløb og dokumentation. " * 20,
            }
        ],
        documents_path,
    )
    write_jsonl(
        [{"uuid": "persona-1", "age": 42, "persona": "Praktisk bruger med behov for korte svar."}],
        personas_path,
    )
    generation = {
        "languages": ["da"],
        "latent_language": "en",
        "seed_expansion": {"variants_per_seed": 6},
        "seed_sources": [
            {
                "type": "source_pack",
                "id": "payload_public_rules",
                "family": "grounded_dialogue",
                "source_payload_strategies": ["full_excerpt", "selected_clauses", "staged_excerpts"],
                "source_payload_lengths": ["short", "medium", "long"],
                "source_payload_styles": ["fenced_text", "blockquote", "plain_delimited"],
                "source_payload_label_styles": ["excerpt", "pasted_text", "document_passage"],
                "documents": {
                    "path": str(documents_path),
                    "text_field": "text",
                    "id_field": "id",
                    "source_field": "source",
                    "created_field": "created",
                    "min_chars": 100,
                    "max_chars": 500,
                    "max_records": 1,
                },
                "personas": {
                    "path": str(personas_path),
                    "id_field": "uuid",
                    "persona_fields": ["persona"],
                    "min_age": 18,
                    "max_records": 1,
                },
            }
        ],
    }

    strategies = {spec["source_payload_plan"]["strategy"] for spec in multi_turn_build._seed_specs(generation)}
    lengths = {spec["source_payload_plan"]["payload_length"] for spec in multi_turn_build._seed_specs(generation)}
    styles = {spec["source_payload_plan"]["paste_style"] for spec in multi_turn_build._seed_specs(generation)}
    label_styles = {spec["source_payload_plan"]["label_style"] for spec in multi_turn_build._seed_specs(generation)}

    assert strategies <= {"full_excerpt", "selected_clauses", "staged_excerpts"}
    assert len(strategies) > 1
    assert lengths <= {"short", "medium", "long"}
    assert styles <= {"fenced_text", "blockquote", "plain_delimited"}
    assert label_styles <= {"excerpt", "pasted_text", "document_passage"}
    assert len(label_styles) > 1


def test_selected_source_payloads_use_readable_passages() -> None:
    excerpt = "\n\n".join(
        [
            (
                "13.12.2016 DA Den Europæiske Unions Tidende LI 337/3. Denne indledning beskriver "
                "retsgrundlaget og forklarer, at læseren skal se bestemmelsen i sammenhæng med resten af teksten."
            ),
            (
                "Artikel 1 fastsætter, at de synlige krav skal vurderes ud fra dokumentets ordlyd. Passagen er lang "
                "nok til at fungere som selvstændigt kildeuddrag og ikke kun som en citation."
            ),
            (
                "Artikel 2 beskriver dokumentation og opfølgning. Den nævner, at manglende omgivende afsnit kan være "
                "relevante, når man skal lave en praktisk tjekliste."
            ),
        ]
    )

    payload = multi_turn_build._source_payload_text_for_strategy(
        "selected_clauses",
        excerpt,
        seed_text="doc-1",
    )
    row = {
        "messages": [
            {
                "role": "user",
                "content": f"Kildeuddrag 1:\n```text\n{payload}\n```",
            }
        ]
    }
    broken_row = {
        "messages": [
            {
                "role": "user",
                "content": "Kildeuddrag 1:\n```text\nUddrag 1:\n13.\n\nUddrag 2:\n2016 DA\n```",
            }
        ]
    }

    assert "Uddrag 1: 13." not in payload
    assert multi_turn_build._source_payloads_readable(row)["passed"] is True
    assert multi_turn_build._source_payloads_readable(broken_row)["passed"] is False


def test_source_surface_failures_are_not_assistant_repairable() -> None:
    seed_spec = source_seed_spec_for_tests()
    messages = [
        {
            "role": "user",
            "content": "Kildeuddrag 1:\n```text\nUddrag 1:\nDag 1 Kørselspraktik Logbog jf.\n```",
        },
        {"role": "assistant", "content": "Jeg laver en tjekliste."},
    ]
    review = {
        "selection": {"accepted": False},
        "review": {"repair_actions": ["Fix source issue."]},
    }

    assert multi_turn_build._unrepairable_surface_failures(messages, seed_spec) == ["source_payloads_readable"]
    assert multi_turn_build._needs_repair(review, messages, seed_spec) is False


def test_missing_visible_artifact_failures_are_not_assistant_repairable() -> None:
    seed_spec = {**source_seed_spec_for_tests(), "source_grounding": None}
    messages = [
        {"role": "user", "content": "Her er min kladde. Kan du rette den?"},
        {"role": "assistant", "content": "Her er en rettet version."},
    ]
    review = {
        "selection": {"accepted": False},
        "review": {"repair_actions": ["Ask for the draft."]},
    }

    assert multi_turn_build._unrepairable_surface_failures(messages, seed_spec) == ["visible_artifact_claims_grounded"]
    assert multi_turn_build._needs_repair(review, messages, seed_spec) is False


def test_long_source_payloads_insert_larger_contiguous_text() -> None:
    paragraphs = [
        (
            f"Afsnit {index} beskriver en del af dokumentet med konkrete oplysninger, forbehold og sammenhæng. "
            "Teksten er lang nok til at ligne et realistisk dokumentuddrag, hvor brugeren kan have brug for hjælp "
            "til at udlede struktur, ukendte felter og næste skridt."
        )
        for index in range(1, 35)
    ]
    excerpt = "\n\n".join(paragraphs)

    selected = multi_turn_build._source_payload_text_for_strategy(
        "selected_clauses",
        excerpt,
        seed_text="doc-long",
        payload_length="long",
    )
    window = multi_turn_build._source_payload_text_for_strategy(
        "long_contiguous_excerpt",
        excerpt,
        seed_text="doc-long",
        payload_length="long",
    )

    assert len(selected) > 3000
    assert len(window) > 6000


def test_source_payload_paste_styles_restore_and_verify() -> None:
    seed_spec = source_seed_spec_for_tests()
    seed_spec["source_payload_plan"] = {
        "strategy": "full_excerpt",
        "payload_length": "medium",
        "paste_style": "plain_delimited",
        "label_style": "pasted_text",
    }
    altered = [
        {
            "role": "user",
            "content": "Kildeuddrag 1 begynder:\nModellen ændrede teksten.\nKildeuddrag 1 slutter.",
        },
        {"role": "assistant", "content": "Jeg svarer kun ud fra teksten."},
    ]

    restored = multi_turn_build._restore_source_payloads_in_messages(seed_spec, altered)

    assert "Modellen ændrede teksten" not in restored[0]["content"]
    assert "Indsat tekst 1 begynder:" in restored[0]["content"]
    assert "Pressenævnets kendelse i sag nr. 2024-10933" in restored[0]["content"]
    assert multi_turn_build._source_payloads_readable({"messages": restored})["passed"] is True


def test_source_payload_readability_supports_blockquote_style() -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Kildeuddrag 1:\n"
                    "> Dette er et længere kildeuddrag med nok almindelig tekst til at være læsbart.\n"
                    "> Det består af flere sætninger og er ikke bare et sagsnummer eller en dato."
                ),
            }
        ]
    }

    assert multi_turn_build._source_payloads_readable(row)["passed"] is True


def test_source_payload_visible_label_can_vary() -> None:
    slot = multi_turn_build._source_payload_slot(
        index=1,
        turn_index=0,
        strategy="full_excerpt",
        payload_length="medium",
        paste_style="fenced_text",
        label_style="document_passage",
        text=(
            "Dette dokumentafsnit indeholder en sammenhængende forklaring med nok almindelige ord til, "
            "at læsbarhedskontrollen kan behandle det som et reelt indsat tekststykke."
        ),
    )

    formatted = multi_turn_build._format_source_payload_slot(slot)

    assert formatted.startswith("Dokumentpassage 1:\n```text\n")
    assert "Kildeuddrag" not in formatted
    assert multi_turn_build._source_payloads_readable(
        {"messages": [{"role": "user", "content": formatted}]}
    )["passed"] is True


def test_source_pack_persona_projection_uses_surface_signals_only(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.jsonl"
    personas_path = tmp_path / "personas.jsonl"
    write_jsonl(
        [
            {
                "id": "doc-1",
                "source": "retsinformationdk",
                "created": "2025-01-01",
                "text": "Dette er et synligt dansk kildeuddrag om regler og afgørelse. " * 20,
            }
        ],
        documents_path,
    )
    write_jsonl(
        [
            {
                "uuid": "persona-1",
                "age": 28,
                "occupation": "fast_food_or_counter_worker",
                "education_level": "high_school",
                "city": "Madison",
                "country": "USA",
                "persona": "Mary Alberti is a bullet-journal aficionado from Madison, Wisconsin.",
                "professional_persona": "Mary wants to become a restaurant manager in the Midwest.",
                "skills_and_expertise_list": ["Cash handling", "Inventory management", "Customer service"],
            }
        ],
        personas_path,
    )
    generation = {
        "languages": ["da"],
        "latent_language": "en",
        "seed_expansion": {"variants_per_seed": 1},
        "seed_sources": [
            {
                "type": "source_pack",
                "id": "projected_public_rules",
                "family": "grounded_dialogue",
                "documents": {
                    "path": str(documents_path),
                    "text_field": "text",
                    "id_field": "id",
                    "source_field": "source",
                    "created_field": "created",
                    "min_chars": 100,
                    "max_chars": 500,
                    "max_records": 1,
                },
                "personas": {
                    "path": str(personas_path),
                    "id_field": "uuid",
                    "min_age": 18,
                    "max_records": 1,
                    "projection": {
                        "enabled": True,
                        "target_locale": "da-DK",
                        "occupation_field": "occupation",
                        "education_field": "education_level",
                        "age_field": "age",
                        "include_skill_tags": True,
                        "skill_tags_field": "skills_and_expertise_list",
                        "max_skill_tags": 2,
                    },
                },
            }
        ],
    }

    spec = multi_turn_build._seed_specs(generation)[0]
    persona = spec["persona_context"]
    persona_blob = str(persona)

    assert persona["projection_mode"] == "surface"
    assert persona["surface"]["role_archetype"] == "fast food or counter worker"
    assert persona["surface"]["age_band"] == "adult_25_34"
    assert persona["surface"]["target_locale"] == "da-DK"
    assert persona["surface"]["skill_tags"] == ["Cash handling", "Inventory management"]
    assert "Mary" not in persona_blob
    assert "Madison" not in persona_blob
    assert "Wisconsin" not in persona_blob
    assert "USA" not in persona_blob
    assert "Midwest" not in persona_blob
    assert "Projected surface persona" in spec["seed_blueprint"]["user_persona"]


def test_source_task_planner_turns_source_and_persona_into_archetype() -> None:
    class SourceTaskPlannerLLM(LLM):
        def __init__(self):
            super().__init__(model="fake-source-task-planner", base_url="https://example.invalid", api_key_env=None)
            self.prompt = ""

        def chat(self, messages, **gen):
            self.prompt = "\n".join(message["content"] for message in messages)
            return "\n".join(
                [
                    "SOURCE_TASK_DOMAIN: tax_checklist_from_visible_excerpt",
                    "SOURCE_TASK_FAMILY: grounded_dialogue",
                    "SOURCE_TASK_INTENT_TRAJECTORY: Information Retrieval Interaction",
                    "SOURCE_TASK_SUBTASK_TRAJECTORY: source_grounded_checklist",
                    "SOURCE_TASK_DIFFICULTY_AXIS: visible tax excerpt with missing dates and current-status uncertainty",
                    "SOURCE_TASK_PERSONA: Danish freelancer who wants a cautious checklist, inspired by the sampled persona.",
                    "SOURCE_TASK_GOAL: understand what the pasted tax excerpt can and cannot answer",
                    "SOURCE_TASK_QUERY_TRAJECTORY: paste excerpt and ask meaning; add dates and amount; request checklist",
                    "SOURCE_TASK_SUCCESS_CRITERIA:",
                    "- uses only the visible excerpt for source-specific claims",
                    "- marks current status and unseen rules as unknown",
                    "- produces a practical checklist",
                    "SOURCE_TASK_VOLATILE_ITEMS: current tax guidance; unseen surrounding rules",
                    "SOURCE_TASK_BLUEPRINT:",
                    "The user should paste the excerpt first, reveal dates later, and ask for a final checklist.",
                ]
            )

    seed_spec = {
        "seed_id": "source-seed::v00",
        "archetype_id": "source-seed",
        "scenario_variant": multi_turn_build._scenario_variant(0, variant_count=1),
        "scenario_variant_index": 0,
        "scenario_variant_count": 1,
        "family": "grounded_dialogue",
        "domain": "skat",
        "seed_intent": "plan from source",
        "seed_blueprint": {"domain": "skat", "user_persona": "generic"},
        "success_criteria": ["use visible source"],
        "intent_trajectory_hint": "Information Retrieval Interaction",
        "subtask_trajectory_hint": "source_grounded_synthesis",
        "query_trajectory_hint": [],
        "source_methods": list(multi_turn_build.DEFAULT_SOURCE_METHODS),
        "source_language": "da",
        "difficulty_axis": "source boundary",
        "requires_epistemic_hedging": True,
        "volatile_items": ["current status"],
        "dialogue_language": "da",
        "latent_language": "en",
        "source_grounding": {
            "pack_id": "dynaword_public_rules_da",
            "source_name": "skat",
            "document_id": "doc-1",
            "title": "Tax excerpt",
            "created": "2025",
            "excerpt": "Synligt dansk kildeuddrag om skat.",
        },
        "persona_context": {
            "source": "nvidia/Nemotron-Personas-USA",
            "persona": "A practical administrative worker from the sampled persona source.",
        },
        "external_sources": [],
    }
    seed_spec["work_session"] = multi_turn_build._work_session_contract(seed_spec)
    llm = SourceTaskPlannerLLM()

    planned = multi_turn_build._generate_source_task_plan_if_needed(
        llm,
        seed_spec,
        {"source_task_temperature": 0.9},
    )

    assert "Sampled persona/context" in llm.prompt
    assert "Synligt dansk kildeuddrag om skat." in llm.prompt
    assert "Do not copy US locations" in llm.prompt
    assert "source excerpt and metadata are immutable inputs" in llm.prompt
    assert "Do not invent, rename, anonymize" in llm.prompt
    assert "Do not copy the sampled persona's name" in llm.prompt
    assert "abstract them into a localized Danish user role" in llm.prompt
    assert planned["domain"] == "tax_checklist_from_visible_excerpt"
    assert planned["intent_trajectory_hint"] == "Information Retrieval Interaction"
    assert planned["subtask_trajectory_hint"] == "source_grounded_checklist"
    assert planned["query_trajectory_hint"] == [
        "paste the source excerpt and ask for initial source grounded checklist",
        "add user-owned context, constraint, or intended use",
        "ask for a revised or final source-bounded deliverable with unknowns separated",
    ]
    assert "the assistant only treats source details as known after the relevant source payload is visible" in planned["success_criteria"]
    assert "the assistant does not assume the pasted excerpt is complete, current, or exhaustive" in planned["success_criteria"]
    assert "use visible source" in planned["success_criteria"]
    assert planned["seed_blueprint"]["source_context"]["source_pack"] == "dynaword_public_rules_da"
    assert planned["seed_blueprint"]["source_context"]["source_text_available_to_visible_dialogue_only_via_injected_payload"] is True
    assert "visible_source_excerpt" not in planned["seed_blueprint"]
    assert planned["source_task_plan"]["blueprint"].startswith("The user should paste")
    assert planned["work_session"]["top_level_intent"] == "Information Retrieval Interaction"


def test_source_task_parser_accepts_markdown_labels_and_configured_intent_fallback() -> None:
    seed_spec = {
        "seed_blueprint": {"user_persona": "generic"},
        "source_grounding": {
            "pack_id": "dynaword_public_rules_da",
            "source_name": "retsinformationdk",
            "document_id": "doc-1",
            "title": "",
            "created": "",
            "excerpt": "Synligt kildeuddrag.",
        },
        "persona_context": {},
        "success_criteria": ["use visible source"],
        "intent_trajectory_hint": "Information Retrieval Interaction",
        "subtask_trajectory_hint": "source_grounded_synthesis",
        "scenario_variant": multi_turn_build._scenario_variant(0, variant_count=1),
    }
    plan = multi_turn_build._parse_source_task_plan(
        "\n".join(
            [
                "**SOURCE_TASK_DOMAIN**",
                "Medieetik og pressekundskab",
                "**SOURCE_TASK_FAMILY**: retsinformationdk",
                "**SOURCE_TASK_INTENT_TRAJECTORY**: Explain and extract obligations from a regulatory decision",
                "**SOURCE_TASK_SUBTASK_TRAJECTORY**",
                "Identificere kritikpunkter",
                "**SOURCE_TASK_PERSONA**: Praktisk bruger",
                "**SOURCE_TASK_GOAL**: Forstå et synligt uddrag",
                "**SOURCE_TASK_QUERY_TRAJECTORY**: paste excerpt; ask checklist",
                "**SOURCE_TASK_SUCCESS_CRITERIA**:",
                "- use visible source only",
                "- preserve uncertainty",
                "**SOURCE_TASK_VOLATILE_ITEMS**: current status; unseen rules",
                "**SOURCE_TASK_BLUEPRINT**:",
                "The user should paste the excerpt first.",
            ]
        )
    )
    planned = multi_turn_build._apply_source_task_plan(seed_spec, plan)

    assert plan["domain"] == "medieetik_og_pressekundskab"
    assert plan["subtask_trajectory"] == "identificere_kritikpunkter"
    assert planned["family"] == "grounded_dialogue"
    assert planned["intent_trajectory_hint"] == "Information Retrieval Interaction"
    assert planned["query_trajectory_hint"] == [
        "paste the source excerpt and ask for initial identificere kritikpunkter",
        "add user-owned context, constraint, or intended use",
        "ask for a revised or final source-bounded deliverable with unknowns separated",
    ]
    assert "use visible source" in planned["success_criteria"]


def test_source_grounding_generation_prompts_withhold_source_reference() -> None:
    seed_spec = source_seed_spec_for_tests()
    blueprint = {"artifact": "Hidden blueprint artifact."}
    intent_model = {"artifact": "Intent artifact."}
    skeleton = [
        {"role": "user", "dialogue_act": "paste_source", "state_delta": "source visible"},
        {"role": "assistant", "dialogue_act": "explain_source", "state_delta": "bounded explanation"},
    ]

    scenario_prompt = "\n".join(message["content"] for message in multi_turn_build._scenario_instance_messages(seed_spec))
    blueprint_prompt = "\n".join(message["content"] for message in multi_turn_build._blueprint_messages(seed_spec))
    user_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._user_simulator_messages(seed_spec, blueprint, intent_model, skeleton)
    )

    for prompt in [scenario_prompt, blueprint_prompt]:
        assert "Source-grounding rule" in prompt
        assert "Pressenævnets kendelse i sag nr. 2024-10933" not in prompt
        assert "BPLN.dk får kritik" not in prompt
        assert "Source context for task shape only" in prompt
        assert "do not invent, rename, anonymize, or alter source-specific" in prompt
        assert "Persona context is only a diversity vector" in prompt
    assert "Source-pack user-simulation rule" in user_prompt
    assert "do not reproduce, paraphrase as if quoted, shorten, translate, or repair" in user_prompt
    assert "{{SOURCE_PAYLOAD_1}}" in user_prompt
    assert "Pressenævnets kendelse i sag nr. 2024-10933" not in user_prompt
    assert "BPLN.dk får kritik" not in user_prompt
    assert "Source visibility plan" in scenario_prompt
    assert "Source visibility plan" in blueprint_prompt
    assert "Sampled source visibility plan" in user_prompt


def test_user_turn_generation_injects_source_payload_directly() -> None:
    class PlaceholderUserLLM(LLM):
        def __init__(self):
            super().__init__(model="fake-placeholder-user", base_url="https://example.invalid", api_key_env=None)

        def chat(self, messages, **gen):
            return (
                "USER 1: Jeg har et uddrag her.\n\n{{SOURCE_PAYLOAD_1}}\n\n"
                "Kan du forklare, hvad kritikken går på?"
            )

    seed_spec = source_seed_spec_for_tests()
    skeleton = [{"role": "user", "dialogue_act": "paste_source", "state_delta": "source visible"}]

    turns = multi_turn_build._generate_user_turns(
        PlaceholderUserLLM(),
        seed_spec,
        {"artifact": "Hidden blueprint artifact."},
        {"artifact": "Intent artifact."},
        skeleton,
        {"user_temperature": 0.7},
    )

    assert "{{SOURCE_PAYLOAD_1}}" not in turns[0]
    assert "Uddrag 1:" in turns[0]
    assert "Pressenævnets kendelse i sag nr. 2024-10933" in turns[0]
    assert "BPLN.dk får kritik" in turns[0]


def test_user_turn_generation_regenerates_wrong_turn_count() -> None:
    class RegeneratingUserLLM(LLM):
        def __init__(self):
            super().__init__(model="fake-regenerate-user", base_url="https://example.invalid", api_key_env=None)
            self.prompts: list[str] = []

        def chat(self, messages, **gen):
            self.prompts.append(messages[-1]["content"])
            if "Regenerate all user turns" in messages[-1]["content"]:
                return "\n".join(
                    [
                        "USER 1: Første besked.",
                        "USER 2: Anden besked.",
                        "USER 3: Tredje besked.",
                    ]
                )
            return "USER 1: Første besked.\nAnden og tredje besked blev slået sammen."

    skeleton = [
        {"role": "user", "dialogue_act": "initial", "state_delta": "first"},
        {"role": "assistant", "dialogue_act": "respond", "state_delta": "response"},
        {"role": "user", "dialogue_act": "followup", "state_delta": "second"},
        {"role": "assistant", "dialogue_act": "respond", "state_delta": "response"},
        {"role": "user", "dialogue_act": "final_request", "state_delta": "third"},
        {"role": "assistant", "dialogue_act": "final", "state_delta": "done"},
    ]
    llm = RegeneratingUserLLM()

    turns = multi_turn_build._generate_user_turns(
        llm,
        {
            "domain": "test",
            "dialogue_language": "da",
            "latent_language": "en",
            "scenario_variant": {"id": "test"},
            "work_session": {"fit": "Problem Solving Interaction > planning"},
            "query_trajectory_hint": [],
            "success_criteria": ["three user turns"],
        },
        {"artifact": "Blueprint."},
        {"artifact": "Intent.", "user_query_plan_steps": []},
        skeleton,
        {"user_temperature": 0.7},
    )

    assert turns == ["Første besked.", "Anden besked.", "Tredje besked."]
    assert len(llm.prompts) == 2
    assert "Incomplete draft output" in llm.prompts[1]


def test_source_payload_restore_preserves_exact_inserted_text_after_polish() -> None:
    seed_spec = source_seed_spec_for_tests()
    altered = [
        {
            "role": "user",
            "content": 'Kildeuddrag 1:\n```text\nPressenævnets sag blev omskrevet af modellen.\n```',
        },
        {"role": "assistant", "content": "Jeg svarer kun ud fra teksten."},
    ]

    restored = multi_turn_build._restore_source_payloads_in_messages(seed_spec, altered)

    assert "Pressenævnets sag blev omskrevet" not in restored[0]["content"]
    assert "Pressenævnets kendelse i sag nr. 2024-10933" in restored[0]["content"]
    assert "BPLN.dk får kritik" in restored[0]["content"]


def test_staged_source_payloads_are_inserted_across_turns() -> None:
    seed_spec = source_seed_spec_for_tests()
    seed_spec["source_grounding"]["excerpt"] = "Første del handler om fristen. Anden del handler om dokumentation."
    seed_spec["source_payload_plan"] = multi_turn_build._source_payload_plan_for_seed(
        {"source_payload_strategies": ["staged_excerpts"]},
        archetype_id="source-seed",
        variant_index=0,
    )
    turns = [
        "Jeg starter med første del.\n\n{{SOURCE_PAYLOAD_1}}\n\nHvad betyder den?",
        "Okay, det giver mening.",
        "Her er næste del.\n\n{{SOURCE_PAYLOAD_2}}\n\nKan du lave en samlet liste?",
    ]

    injected = multi_turn_build._inject_source_payloads(seed_spec, turns)

    assert "Første del handler om fristen." in injected[0]
    assert "Anden del handler om dokumentation." in injected[2]
    assert "{{SOURCE_PAYLOAD_" not in "\n".join(injected)


def test_source_grounding_review_prompts_compare_against_original_reference() -> None:
    seed_spec = source_seed_spec_for_tests()
    blueprint = {"artifact": "Hidden blueprint artifact."}
    intent_model = {"artifact": "Intent artifact."}
    skeleton = [
        {"role": "user", "dialogue_act": "paste_source", "state_delta": "source visible"},
        {"role": "assistant", "dialogue_act": "explain_source", "state_delta": "bounded explanation"},
    ]
    messages = [
        {"role": "user", "content": "Her er uddraget: Pressenævnets kendelse i sag nr. 2024-10933."},
        {"role": "assistant", "content": "Det kan jeg forklare ud fra det synlige uddrag."},
    ]

    review_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._review_messages(seed_spec, blueprint, intent_model, skeleton, messages)
    )
    strict_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._strict_review_messages(seed_spec, blueprint, intent_model, skeleton, messages)
    )

    for prompt in [review_prompt, strict_prompt]:
        assert "Source-pack review rule" in prompt
        assert "Pressenævnets kendelse i sag nr. 2024-10933" in prompt
        assert "Fail source_boundary if a visible user payload or assistant answer mutates" in prompt
        assert "not assistant-visible evidence" in prompt
        assert "before that fact appears in prior visible user text" in prompt
        assert "irrelevant foreign persona names, places, currency, institutions" in prompt


def test_source_grounding_skeleton_prompt_keeps_assistant_steps_procedural() -> None:
    seed_spec = source_seed_spec_for_tests()
    blueprint = {
        "artifact": "Hidden source fact: the case number is 2024-10933 and the publication title is source-owned."
    }
    intent_model = {"artifact": "USER_QUERY_PLAN: paste source; ask for checklist"}

    skeleton_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._skeleton_messages(seed_spec, blueprint, intent_model)
    )

    assert "Source-pack skeleton rule" in skeleton_prompt
    assert "plan source visibility according to the sampled source payload plan" in skeleton_prompt
    assert "Sampled source visibility plan" in skeleton_prompt
    assert "Do not put hidden source facts into assistant dialogue acts" in skeleton_prompt
    assert "Bad assistant purpose" in skeleton_prompt
    assert "Good assistant purpose" in skeleton_prompt


def test_source_grounding_is_kept_in_hidden_verification() -> None:
    seed_spec = source_seed_spec_for_tests()
    blueprint = {"domain": "source_case", "artifact": "Hidden blueprint artifact."}
    intent_model = {
        "trajectory": "Information Retrieval Interaction",
        "subtask_trajectory": "source_grounded_synthesis",
        "user_query_plan_steps": ["paste source"],
    }
    skeleton = [
        {"role": "user", "dialogue_act": "paste_source", "state_delta": "source visible"},
        {"role": "assistant", "dialogue_act": "explain_source", "state_delta": "bounded explanation"},
    ]
    messages = [
        {"role": "user", "content": "Her er uddraget: Pressenævnets kendelse i sag nr. 2024-10933."},
        {"role": "assistant", "content": "Det kan jeg forklare ud fra det synlige uddrag."},
    ]
    review = {"review": {}, "selection": {"accepted": True, "score": 0.95}}

    row = multi_turn_build._row_from_generated_parts(
        row_id="source-row-1",
        seed_spec=seed_spec,
        blueprint=blueprint,
        intent_model=intent_model,
        skeleton=skeleton,
        messages=messages,
        review=review,
    )

    source_grounding = row["hidden"]["verification"]["source_grounding"]
    assert source_grounding["document_id"] == "AA014830"
    assert source_grounding["excerpt"].startswith("Pressenævnets kendelse i sag nr. 2024-10933")


def test_scenario_variants_drive_variable_turn_count() -> None:
    seed_spec = {
        "domain": "event_planning",
        "family": "general_chat",
        "seed_intent": "plan an event",
        "source_language": "da",
        "dialogue_language": "da",
        "latent_language": "en",
        "difficulty_axis": "variable turns",
        "success_criteria": ["keep the backup plan visible"],
        "intent_trajectory_hint": "planning_interaction",
        "query_trajectory_hint": ["informative request", "add constraints", "ask for final plan"],
        "scenario_variant": {
            "id": "longer_test_variant",
            "target_user_turns": 5,
            "first_turn_shape": "opening includes concrete event facts",
            "user_context": "organizer",
            "information_flow": "constraints accumulate over five user turns",
        },
        "seed_blueprint": {
            "domain": "event_planning",
            "hidden_goal": "plan a community event",
            "success_criteria": ["keep the backup plan visible"],
        },
    }
    blueprint = {"artifact": "Hidden goal: plan a community event."}
    intent_model = {"artifact": "INTENT_TRAJECTORY: planning_interaction"}
    skeleton = [
        {"role": "user", "dialogue_act": f"user_step_{index}", "state_delta": "user move"}
        if index % 2
        else {"role": "assistant", "dialogue_act": f"assistant_step_{index}", "state_delta": "assistant move"}
        for index in range(1, 11)
    ]

    skeleton_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._skeleton_messages(seed_spec, blueprint, intent_model)
    )
    user_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._user_simulator_messages(seed_spec, blueprint, intent_model, skeleton)
    )

    assert "Build a 10-step skeleton" in skeleton_prompt
    assert "user, assistant, user, assistant, user, assistant, user, assistant, user, assistant" in skeleton_prompt
    assert "The 5 user skeleton steps" in skeleton_prompt
    assert "USER 5:" in user_prompt
    assert "Use one labeled block per user turn" in user_prompt
    assert "Only sparse_opening may be a very short opening" in user_prompt


def test_skeleton_generation_repairs_wrong_step_count() -> None:
    class SkeletonRepairLLM(LLM):
        def __init__(self):
            super().__init__(model="fake-skeleton-repair", base_url="https://example.invalid", api_key_env=None)
            self.prompts: list[str] = []

        def chat(self, messages, **gen):
            prompt = messages[-1]["content"]
            self.prompts.append(prompt)
            if "Rewrite the draft skeleton" in prompt:
                return "\n".join(
                    [
                        "STEP 1: user | initial_request | visible task context appears",
                        "STEP 2: assistant | produce_initial_work | first useful output appears",
                        "STEP 3: user | add_constraint | new requirement becomes visible",
                        "STEP 4: assistant | revise_work | revised work product appears",
                        "STEP 5: user | ask_comparison | comparison criterion becomes visible",
                        "STEP 6: assistant | compare_options | comparison output appears",
                        "STEP 7: user | request_final | final format requirement becomes visible",
                        "STEP 8: assistant | finalize | final deliverable appears",
                    ]
                )
            return "\n".join(
                [
                    "STEP 1: user | initial_request | visible task context appears",
                    "STEP 2: assistant | produce_initial_work | first useful output appears",
                    "STEP 3: user | add_constraint | new requirement becomes visible",
                    "STEP 4: assistant | revise_work | revised work product appears",
                    "STEP 5: user | ask_comparison | comparison criterion becomes visible",
                    "STEP 6: assistant | compare_options | comparison output appears",
                    "STEP 7: user | request_final | final format requirement becomes visible",
                ]
            )

    seed_spec = {
        "domain": "editing",
        "dialogue_language": "da",
        "latent_language": "en",
        "scenario_variant": {"target_user_turns": 4},
        "work_session": {
            "top_level_intent": "Transaction Interaction",
            "subtask_trajectory": "refinement_interaction",
            "work_session_type": "bounded_artifact_completion_session",
        },
    }
    blueprint = {"artifact": "Hidden goal: revise a short application text."}
    intent_model = {"artifact": "USER_QUERY_PLAN: initial request; constraint; comparison; final"}
    llm = SkeletonRepairLLM()

    skeleton = multi_turn_build._generate_skeleton(
        llm,
        seed_spec,
        blueprint,
        intent_model,
        {"skeleton_temperature": 0.4},
    )

    assert len(skeleton) == 8
    assert [step["role"] for step in skeleton] == multi_turn_build._expected_skeleton_roles(seed_spec)
    assert len(llm.prompts) == 2
    assert "Required role order: user, assistant, user, assistant" in llm.prompts[1]


def test_skeleton_generation_falls_back_to_intent_plan_after_failed_repairs() -> None:
    class BadSkeletonLLM(LLM):
        def __init__(self):
            super().__init__(model="fake-bad-skeleton", base_url="https://example.invalid", api_key_env=None)
            self.prompts: list[str] = []

        def chat(self, messages, **gen):
            self.prompts.append(messages[-1]["content"])
            return "\n".join(
                [
                    "STEP 1: user | collapsed_initial | first and second user moves are merged",
                    "STEP 2: assistant | collapsed_response | first and second assistant moves are merged",
                    "STEP 3: user | collapsed_followup | third and fourth user moves are merged",
                    "STEP 4: assistant | collapsed_final | third and fourth assistant moves are merged",
                ]
            )

    seed_spec = {
        "domain": "editing",
        "dialogue_language": "da",
        "latent_language": "en",
        "scenario_variant": {"target_user_turns": 4},
        "work_session": {
            "top_level_intent": "Transaction Interaction",
            "subtask_trajectory": "refinement_interaction",
            "work_session_type": "bounded_artifact_completion_session",
        },
    }
    blueprint = {"artifact": "Hidden goal: revise a short application text."}
    intent_model = {
        "artifact": "USER_QUERY_PLAN: paste draft; add tone constraint; request comparison; final polish",
        "user_query_plan_steps": ["paste draft", "add tone constraint", "request comparison", "final polish"],
    }
    llm = BadSkeletonLLM()

    skeleton = multi_turn_build._generate_skeleton(
        llm,
        seed_spec,
        blueprint,
        intent_model,
        {"skeleton_temperature": 0.4},
    )

    assert len(llm.prompts) == 3
    assert [step["role"] for step in skeleton] == multi_turn_build._expected_skeleton_roles(seed_spec)
    assert skeleton[0]["dialogue_act"] == "paste_draft"
    assert skeleton[-1]["dialogue_act"] == "finalize_work_product"


def test_visible_entity_grounding_uses_visible_user_terms() -> None:
    score = multi_turn_build._visible_entity_grounding_score(
        ["Kan vi tage til Lissabon for 1800 dollars med to børn?"],
        ["Jeg kan lave en plan for Lissabon og behandle 1800 dollars som et skøn for familien."],
    )

    assert score > 0


def test_assistant_turn_prompt_uses_prefix_without_future_user_facts() -> None:
    seed_spec = {
        "domain": "travel_planning",
        "dialogue_language": "en",
        "requires_epistemic_hedging": True,
        "volatile_items": ["current flight prices"],
    }
    skeleton = [
        {"role": "user", "dialogue_act": "vague_initial_request", "state_delta": "ask for help"},
        {"role": "assistant", "dialogue_act": "clarify_missing_slots", "state_delta": "ask for budget and constraints"},
        {"role": "user", "dialogue_act": "reveal_scope", "state_delta": "give trip length"},
        {"role": "assistant", "dialogue_act": "draft_plan", "state_delta": "outline with caveats"},
        {"role": "user", "dialogue_act": "format_request", "state_delta": "ask for a checklist"},
        {"role": "assistant", "dialogue_act": "finalize", "state_delta": "produce grounded checklist"},
    ]
    user_turns = [
        "Can you help me plan a trip?",
        "Make it four days.",
        "Put the final version in a checklist.",
    ]
    assistant_turns = ["What budget and constraints should I plan around?"]

    prompt_messages = multi_turn_build._assistant_turn_messages(
        seed_spec,
        skeleton,
        user_turns,
        assistant_turns,
        assistant_index=1,
        review=None,
    )
    prompt = "\n".join(message["content"] for message in prompt_messages)

    assert "Can you help me plan a trip?" in prompt
    assert "USER 2: Make it four days." in prompt
    assert "USER 1: vague_initial_request" in prompt
    assert "ASSISTANT 1: clarify_missing_slots" in prompt
    assert "USER 2: reveal_scope" in prompt
    assert "USER 3: format_request" not in prompt
    assert "ASSISTANT 3: finalize" not in prompt
    assert "Put the final version in a checklist." not in prompt
    assert "Visible transcript prefix" in prompt
    assert "ConsistentChat-aligned work-session contract" in prompt
    assert "practical_solution_session" in prompt
    assert "work-session contract as generic task-shape guidance" in prompt
    assert "produce useful work immediately" in prompt
    assert "Avoid repeated broad clarification turns" in prompt
    assert "avoid zero-information openings" in prompt
    assert "ASSISTANT 2: draft_plan" in prompt
    assert "outline with caveats" not in prompt
    assert "ASSISTANT 3: finalize | produce grounded checklist" not in prompt
    assert "secret peanut allergy" not in prompt
    assert "$1800" not in prompt
    assert "Hidden blueprint" not in prompt


def test_generator_and_reviewer_share_general_epistemic_policy() -> None:
    seed_spec = {
        "domain": "travel_planning",
        "dialogue_language": "en",
        "latent_language": "en",
        "success_criteria": ["separate estimates from known values"],
        "intent_trajectory_hint": "planning_interaction",
        "requires_epistemic_hedging": True,
        "volatile_items": ["seed-specific current fare marker"],
    }
    skeleton = [
        {"role": "user", "dialogue_act": "vague_initial_request", "state_delta": "ask for help"},
        {"role": "assistant", "dialogue_act": "clarify_missing_slots", "state_delta": "ask for constraints"},
    ]
    user_turns = ["Can you help me plan a trip?"]
    assistant_turns: list[str] = []
    blueprint = {"artifact": "Hidden goal: plan a trip."}
    intent_model = {"artifact": "INTENT_TRAJECTORY: planning_interaction"}
    messages = [
        {"role": "user", "content": "Can you help me plan a trip?"},
        {"role": "assistant", "content": "What dates, budget, and constraints should I plan around?"},
    ]

    assistant_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._assistant_turn_messages(
            seed_spec,
            skeleton,
            user_turns,
            assistant_turns,
            assistant_index=0,
            review=None,
        )
    )
    reviewer_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._review_messages(seed_spec, blueprint, intent_model, skeleton, messages)
    )
    policy = multi_turn_build._epistemic_policy_instruction().strip()

    assert policy in assistant_prompt
    assert policy in reviewer_prompt
    assert "stable public and general knowledge" in assistant_prompt
    assert "prefer careful epistemics throughout" in reviewer_prompt
    assert "freshly verified, complete, current, official, or guaranteed" in assistant_prompt
    assert "company-policy" in reviewer_prompt
    assert "do not invent receipt dates" in assistant_prompt
    assert "appeal windows" in reviewer_prompt
    assert "complaint bodies" in assistant_prompt
    assert "success likelihood" in reviewer_prompt
    assert "institution names" in assistant_prompt
    assert "named-institution" in reviewer_prompt
    assert "Across domains, exact facts" in assistant_prompt
    assert "background memory rather than verified lookup" in assistant_prompt
    assert "compatibility guarantees" in assistant_prompt
    assert "local market ranges" in reviewer_prompt
    assert "not only product comparisons" in assistant_prompt
    assert "source-dependent tasks in any domain" in reviewer_prompt
    assert "assistant-filled exact fields" in reviewer_prompt
    assert "hedged background generalization" in reviewer_prompt
    assert "Do not let an assistant claim pass because the fact appears later" in reviewer_prompt
    assert "Reject foreshadowing when an assistant turn anticipates a later user constraint" in reviewer_prompt
    assert "Outcome review must reject vague conversations" in reviewer_prompt
    assert "REVIEWER outcome" in reviewer_prompt
    assert "only ask broad slot-filling questions" in reviewer_prompt
    assert "Reject zero-information first turns" in reviewer_prompt
    assert "Reject assistant turns that use emojis" in reviewer_prompt
    assert "checkmark/cross/warning symbols" in reviewer_prompt
    assert "seed-specific current fare marker" not in assistant_prompt
    assert "seed-specific current fare marker" not in reviewer_prompt


def test_generator_and_reviewer_share_discourse_quality_policy() -> None:
    seed_spec = {
        "domain": "meal_planning",
        "family": "general_chat",
        "seed_intent": "make a weekly lunch plan",
        "source_language": "da",
        "dialogue_language": "da",
        "latent_language": "en",
        "difficulty_axis": "semantic recovery",
        "success_criteria": ["use future-oriented planning language"],
        "intent_trajectory_hint": "advisory_interaction",
        "query_trajectory_hint": ["broad meal planning request", "reveal constraints"],
        "seed_blueprint": {
            "domain": "meal_planning",
            "hidden_goal": "make a weekly lunch plan",
            "success_criteria": ["use future-oriented planning language"],
        },
    }
    blueprint = {"artifact": "Hidden goal: make a weekly lunch plan."}
    intent_model = {"artifact": "INTENT_TRAJECTORY: advisory_interaction"}
    skeleton = [
        {"role": "user", "dialogue_act": "vague_initial_request", "state_delta": "ask for meal plan"},
        {"role": "assistant", "dialogue_act": "clarify_missing_slots", "state_delta": "ask for timing"},
    ]
    messages = [
        {"role": "user", "content": "Kan du hjælpe med en madplan?"},
        {"role": "assistant", "content": "Hvilke dage og hensyn skal jeg planlægge efter?"},
    ]

    user_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._user_simulator_messages(seed_spec, blueprint, intent_model, skeleton)
    )
    assistant_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._assistant_turn_messages(
            seed_spec,
            skeleton,
            ["Kan du hjælpe med en madplan?"],
            [],
            assistant_index=0,
            review=None,
        )
    )
    reviewer_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._review_messages(seed_spec, blueprint, intent_model, skeleton, messages)
    )
    skeleton_prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._skeleton_messages(seed_spec, blueprint, intent_model)
    )
    policy = multi_turn_build._discourse_quality_instruction().strip()

    assert "the immediately prior assistant step must have visibly introduced" in skeleton_prompt
    assert "STEP 1: user | initial_user_request" in skeleton_prompt
    assert policy in user_prompt
    assert policy in assistant_prompt
    assert policy in reviewer_prompt
    assert "silently going along with the oddity" in reviewer_prompt
    assert "recover by asking a concise clarification" in assistant_prompt
    assert "Assistant turns should not use emojis" in assistant_prompt
    assert "checkmark/cross/warning symbols" in assistant_prompt
    assert "Use light, functional formatting" in assistant_prompt
    assert "structurally valid Markdown" in assistant_prompt
    assert "Use bold sparingly" in assistant_prompt
    assert "Avoid decorative formatting" in assistant_prompt
    assert "may be vague, partial, or reasonably informative" in user_prompt
    assert "do not make every conversation start with underspecification" in user_prompt
    assert "The first user turn should normally contain at least one useful work payload" in user_prompt
    assert "underspecify the first turn" not in user_prompt
    assert "phrase late constraints as new information" in user_prompt
    assert "should not pretend that content was present" in assistant_prompt
    assert "include a short visible payload in that same turn" in user_prompt
    assert "If a needed artifact is missing, ask for it" in assistant_prompt
    assert "preserve missing details as unknown fields" in assistant_prompt
    assert "time estimates as rough suggestions to try" in assistant_prompt
    assert "root causes and compatibility explanations as hypotheses" in assistant_prompt
    assert "Technical troubleshooting should distinguish visible evidence from hypotheses" in assistant_prompt
    assert "Reject missing visible artifacts" in reviewer_prompt
    assert "Reject unsupported quoted edits" in reviewer_prompt
    assert "unsupported reaction" in reviewer_prompt
    assert "invented words" in reviewer_prompt
    assert "natural, ordinary Danish" in user_prompt
    assert "do not invent official product" in user_prompt
    assert "official localized term is uncertain" in reviewer_prompt
    assert "Do not mix English words into Danish prose" in user_prompt
    assert "do not invent literal compounds" in user_prompt
    assert "light functional formatting" in reviewer_prompt
    assert "Bold should be sparse" in reviewer_prompt
    assert "verification instruction is not a verified claim" in reviewer_prompt


def test_strict_review_prompt_and_parser_reject_source_dependent_overreach() -> None:
    seed_spec = {
        "domain": "product_comparison",
        "dialogue_language": "da",
        "latent_language": "en",
        "success_criteria": ["keep exact specs unknown unless supplied"],
        "intent_trajectory_hint": "comparison_interaction",
        "difficulty_axis": "source-boundary audit",
        "work_session": {
            "work_session_type": "grounded_synthesis_session",
        },
    }
    blueprint = {"artifact": "Hidden goal: compare two laptops without live specs."}
    intent_model = {"artifact": "INTENT_TRAJECTORY: Information Retrieval Interaction"}
    skeleton = [
        {"role": "user", "dialogue_act": "ask_comparison", "state_delta": "names options"},
        {"role": "assistant", "dialogue_act": "compare_with_unknowns", "state_delta": "gives framework"},
    ]
    messages = [
        {"role": "user", "content": "Jeg mangler præcise specs og priser."},
        {"role": "assistant", "content": "Model A har 64 Wh og koster 10.999 kr."},
    ]

    prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._strict_review_messages(seed_spec, blueprint, intent_model, skeleton, messages)
    )
    strict = multi_turn_build._parse_strict_review(
        "\n".join(
            [
                "STRICT_SELECTION: FAIL",
                "STRICT_SCORE: 0.35",
                "STRICT_CHECK source_boundary: FAIL - exact specs were filled without visible support",
                "STRICT_CHECK language_quality: PASS - clean Danish",
                "STRICT_CHECK factuality: FAIL - suspicious unsourced numbers",
                "STRICT_CHECK contradiction: PASS - no contradiction",
                "STRICT_CHECK format: PASS - restrained format",
                "STRICT_REPAIR_ACTIONS: Replace exact specs with unknown fields and verification steps.",
            ]
        ),
        threshold=0.9,
    )

    assert "Run a strict audit of this dialogue" in prompt
    assert "Stable public knowledge is allowed" in prompt
    assert "material, reject-worthy issues" in prompt
    assert "Do not fail merely because the assistant suggests a reasonable new option" in prompt
    assert "allow direct paraphrases and clearly marked interpretations" in prompt
    assert "Do not fail normal multi-turn revision after newly visible user constraints" in prompt
    assert "Entertainment rows are not exempt" in prompt
    assert "Do not fail source_boundary merely because the assistant tells the user to check documentation" in prompt
    assert "Allow light functional formatting" in prompt
    assert "structurally valid Markdown spacing" in prompt
    assert "Bold should be sparse" in prompt
    assert "strict audit" in prompt
    assert strict["accepted"] is False
    assert strict["checks"][0]["passed"] is False
    assert strict["repair_actions"]


def test_strict_review_parser_accepts_verification_advice() -> None:
    strict = multi_turn_build._parse_strict_review(
        "\n".join(
            [
                "STRICT_SELECTION: PASS",
                "STRICT_SCORE: 0.93",
                "STRICT_CHECK source_boundary: PASS - the assistant asks the user to check release notes and package metadata, without claiming what they contain",
                "STRICT_CHECK language_quality: PASS - clean Danish",
                "STRICT_CHECK factuality: PASS - no suspicious facts",
                "STRICT_CHECK contradiction: PASS - no contradiction",
                "STRICT_CHECK format: PASS - light functional formatting",
                "STRICT_REPAIR_ACTIONS: none",
            ]
        ),
        threshold=0.9,
    )

    assert strict["accepted"] is True
    assert strict["checks"][0]["passed"] is True


def test_strict_review_parser_accepts_comma_delimited_fields() -> None:
    strict = multi_turn_build._parse_strict_review(
        "\n".join(
            [
                "STRICT_SELECTION, FAIL",
                "STRICT_SCORE, 0.4",
                "STRICT_CHECK source_boundary, FAIL - unhedged source-dependent claim",
                "STRICT_CHECK language_quality, PASS - clean Danish",
                "STRICT_CHECK factuality, PASS - no suspicious facts",
                "STRICT_CHECK contradiction, PASS - no contradiction",
                "STRICT_CHECK format, PASS - light functional formatting",
                "STRICT_REPAIR_ACTIONS, Hedge the claim.",
            ]
        ),
        threshold=0.9,
    )

    assert strict["accepted"] is False
    assert strict["score"] == 0.4
    assert strict["checks"][0]["passed"] is False
    assert strict["repair_actions"] == ["Hedge the claim."]


def test_strict_review_parser_accepts_whitespace_fields_and_none_dash_actions() -> None:
    strict = multi_turn_build._parse_strict_review(
        "\n".join(
            [
                "STRICT_SELECTION PASS",
                "STRICT_SCORE 1.0",
                "STRICT_CHECK source_boundary PASS - source-bounded",
                "STRICT_CHECK language_quality PASS - clean Danish",
                "STRICT_CHECK factuality PASS - no suspicious facts",
                "STRICT_CHECK contradiction PASS - no contradiction",
                "STRICT_CHECK format PASS - light functional formatting",
                "STRICT_REPAIR_ACTIONS NONE \u2013 all checks passed",
            ]
        ),
        threshold=0.9,
    )

    assert strict["accepted"] is True
    assert strict["repair_actions"] == []


def test_surface_polish_prompt_and_parser_preserve_turn_structure() -> None:
    seed_spec = {
        "dialogue_language": "da",
    }
    messages = [
        {"role": "user", "content": "Jeg har en mærkelig fejl."},
        {"role": "assistant", "content": "Din Python-miljøpludselige ændrede sig."},
        {"role": "user", "content": "Kan du gøre det kort?"},
        {"role": "assistant", "content": "Ja."},
    ]

    prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._surface_polish_messages(seed_spec, messages)
    )
    polished = multi_turn_build._parse_polished_messages(
        "\n".join(
            [
                "USER 1: Jeg har en mærkelig fejl.",
                "ASSISTANT 1: Dit Python-miljø har sandsynligvis ændret sig.",
                "USER 2: Kan du gøre det kort?",
                "ASSISTANT 2: Ja.",
            ]
        ),
        messages,
    )

    assert "Surface-edit this dialogue" in prompt
    assert "USER 1:, ASSISTANT 1:, USER 2:, ASSISTANT 2:" in prompt
    assert "generic description rather than a precise-sounding invented label" in prompt
    assert "accidental English words in prose" in prompt
    assert "No turn content may contain USER n: or ASSISTANT n: labels" in prompt
    assert "Remove emojis, decorative emoji bullets, and checkmark/cross/warning symbols" in prompt
    assert "Keep functional paragraph breaks" in prompt
    assert "structurally valid" in prompt
    assert "Use bold sparingly" in prompt
    assert "Remove excessive emphasis" in prompt
    assert "do not create ad hoc" in prompt
    assert "Check Danish agreement" in prompt
    assert polished is not None
    assert [message["role"] for message in polished] == ["user", "assistant", "user", "assistant"]
    assert "Python-miljø har sandsynligvis ændret sig" in polished[1]["content"]
    assert multi_turn_build._parse_polished_messages("Polished without labels.", messages) is None
    assert (
        multi_turn_build._parse_polished_messages(
            "USER 1: Jeg har en fejl. ASSISTANT 1: Her er et svar.",
            messages,
        )
        is None
    )


def test_parse_polished_messages_preserves_paragraph_breaks() -> None:
    messages = [
        {"role": "user", "content": "Kan du lave et kort notat?"},
        {"role": "assistant", "content": "Første afsnit.\n\nAndet afsnit."},
    ]

    polished = multi_turn_build._parse_polished_messages(
        "\n".join(
            [
                "USER 1: Kan du lave et kort notat?",
                "ASSISTANT 1: Første afsnit.",
                "",
                "Andet afsnit.",
            ]
        ),
        messages,
    )

    assert polished is not None
    assert polished[1]["content"] == "Første afsnit.\n\nAndet afsnit."


def test_surface_polish_reformat_prompt_restores_exact_labels() -> None:
    seed_spec = {
        "dialogue_language": "da",
    }
    messages = [
        {"role": "user", "content": "Første"},
        {"role": "assistant", "content": "Andet"},
    ]
    prompt = "\n".join(
        message["content"]
        for message in multi_turn_build._reformat_surface_polish_messages(seed_spec, "Første\nAndet", messages)
    )

    assert "USER 1:, ASSISTANT 1:" in prompt
    assert "Do not combine turns" in prompt
    assert "must not contain other turn labels" in prompt


def test_surface_polish_defaults_to_non_english_dialogues() -> None:
    assert multi_turn_build._should_surface_polish({"dialogue_language": "da"}, {}) is True
    assert multi_turn_build._should_surface_polish({"dialogue_language": "en"}, {}) is False
    assert multi_turn_build._should_surface_polish({"dialogue_language": "da"}, {"surface_polish": False}) is False


def test_no_role_label_leak_blocks_embedded_labels() -> None:
    clean_row = {"messages": [{"role": "user", "content": "Kan du hjælpe?"}]}
    leaking_row = {"messages": [{"role": "user", "content": "Kan du hjælpe? ASSISTANT 1: Ja."}]}

    assert multi_turn_build._no_role_label_leak(clean_row)["passed"] is True
    assert multi_turn_build._no_role_label_leak(leaking_row)["passed"] is False


def test_assistant_format_restrained_allows_markdown_and_blocks_emojis() -> None:
    clean_row = {"messages": [{"role": "assistant", "content": "Her er tre korte punkter:\n- Første\n- Andet"}]}
    light_markdown_row = {
        "messages": [
            {
                "role": "assistant",
                "content": "## Kort plan\n\n| Trin | Gør |\n| --- | --- |\n| 1 | Tjek kilden |\n\n**Kort sagt:** brug det som udkast.",
            }
        ]
    }
    repeated_separator_row = {"messages": [{"role": "assistant", "content": "Plan\n---\nDetaljer\n---"}]}
    heading_stack_row = {"messages": [{"role": "assistant", "content": "# A\n## B\n### C\nTekst"}]}
    emoji_row = {"messages": [{"role": "assistant", "content": "God ide ✅"}]}

    assert multi_turn_build._assistant_format_restrained(clean_row)["passed"] is True
    assert multi_turn_build._assistant_format_restrained(light_markdown_row)["passed"] is True
    assert multi_turn_build._assistant_format_restrained(repeated_separator_row)["passed"] is True
    assert multi_turn_build._assistant_format_restrained(heading_stack_row)["passed"] is True
    assert multi_turn_build._assistant_format_restrained(emoji_row)["passed"] is False


def test_visible_artifact_claim_requires_payload() -> None:
    missing_payload = {
        "messages": [
            {
                "role": "user",
                "content": "Her er jobopslaget og mit nuværende udkast. Kan du gøre ansøgningen skarpere?",
            },
            {"role": "assistant", "content": "Jeg kan godt lave en version."},
        ]
    }
    visible_payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Her er jobopslaget og mit udkast.\n\n"
                    'Jobopslag: "Vi søger en koordinator, der kan planlægge leverancer og skrive til kunder."\n'
                    'Udkast: "Jeg har arbejdet med planlægning og kundedialog i tre år."'
                ),
            },
            {"role": "assistant", "content": "Her er en strammere version."},
        ]
    }

    assert multi_turn_build._visible_artifact_claims_grounded(missing_payload)["passed"] is False
    assert multi_turn_build._visible_artifact_claims_grounded(visible_payload)["passed"] is True


def test_visible_artifact_claim_does_not_match_inside_unrelated_words() -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": "Her er konteksten. Jeg søger hjælp til at prioritere tre opgaver i næste uge.",
            },
            {"role": "assistant", "content": "Det kan jeg hjælpe med."},
        ]
    }

    assert multi_turn_build._visible_artifact_claims_grounded(row)["passed"] is True


def test_reaction_reference_must_exist_in_visible_transcript() -> None:
    unsupported_reaction = {
        "messages": [
            {"role": "user", "content": "Kan du hjælpe med min ansøgning?"},
            {"role": "assistant", "content": "Ja, send gerne udkastet."},
            {"role": "user", "content": 'Kan du fjerne "styrer logistik" og gøre tonen mere rolig?'},
            {"role": "assistant", "content": "Jeg fjerner den formulering."},
        ]
    }
    grounded_reaction = {
        "messages": [
            {"role": "user", "content": 'Udkast: "Jeg styrer logistik og skriver til kunder."'},
            {"role": "assistant", "content": "Den sætning kan gøres mere konkret."},
            {"role": "user", "content": 'Kan du fjerne "styrer logistik" og skrive det mere præcist?'},
            {"role": "assistant", "content": "Ja, jeg omskriver den del."},
        ]
    }

    assert multi_turn_build._reaction_references_grounded(unsupported_reaction)["passed"] is False
    assert multi_turn_build._reaction_references_grounded(grounded_reaction)["passed"] is True


def test_review_pass_with_negative_words_still_passes_when_explicit() -> None:
    assert multi_turn_build._reviewer_passed("PASS - does not assert hidden facts and avoids contradiction") is True
    assert multi_turn_build._reviewer_passed("No unsupported assumptions introduced.") is True
    assert multi_turn_build._reviewer_passed("No semantic oddities, contradictory premises, or unsupported references are present.") is True
    assert multi_turn_build._reviewer_passed("No recovery situations were needed.") is True
    assert multi_turn_build._reviewer_passed("The assistant does not fabricate the school's policy.") is True
    assert multi_turn_build._reviewer_passed("The assistant introduces unsupported policy details.") is False


def test_review_parser_accepts_pass_score_and_none_required_actions() -> None:
    seed_spec = {
        "success_criteria": ["uses phone/router contrast", "mentions VPN", "keeps the 30-minute deadline in view"],
        "difficulty_axis": "wrong-assumption recovery under time pressure",
        "intent_trajectory_hint": "troubleshooting_interaction",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: assistant 3",
                "SCORE: PASS",
                "INTENT_TRAJECTORY: troubleshooting_interaction",
                "SUCCESS_CRITERIA: uses phone/router contrast; mentions VPN; keeps the 30-minute deadline in view",
                "TURN_EVIDENCE_AUDIT: ASSISTANT 1: PASS - supported by USER 1; ASSISTANT 2: PASS - supported by USER 1-2; ASSISTANT 3: PASS - supported by USER 1-3",
                "REVIEWER coherence: The assistant incorporates a late-revealed clue.",
                "REVIEWER naturalness: Language is conversational.",
                "REVIEWER grounding: All claims are grounded in user-provided information. No unsupported assumptions introduced.",
                "REVIEWER source_boundary: No source invention.",
                "REVIEWER language_quality: Clean language.",
                "REVIEWER recovery: Successfully incorporates late-revealed clue.",
                "REVIEWER format: Final response satisfies the requested structure.",
                "REPAIR_ACTIONS: None needed. All success criteria satisfied.",
            ]
        ),
        seed_spec,
        threshold=0.8,
    )

    assert review["selection"]["accepted"] is True
    assert review["selection"]["score"] == 1.0
    assert review["review"]["review_decision"] == "accept"
    assert review["review"]["turn_evidence_audit"]
    assert review["review"]["repair_actions"] == []
    assert all(reviewer["passed"] for reviewer in review["review"]["reviewers"])


def test_review_parser_accepts_multiline_turn_evidence_audit() -> None:
    seed_spec = {
        "success_criteria": ["audit all turns"],
        "difficulty_axis": "multiline audit",
        "intent_trajectory_hint": "Problem Solving Interaction",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: ACCEPT",
                "SCORE: 0.96",
                "INTENT_TRAJECTORY: Problem Solving Interaction",
                "SUCCESS_CRITERIA:",
                "- audit all turns: YES",
                "TURN_EVIDENCE_AUDIT:",
                "ASSISTANT 1: PASS; ASSISTANT 2: PASS",
                "REVIEWER coherence: PASS",
                "The dialogue is coherent.",
                "REVIEWER naturalness: PASS",
                "REVIEWER grounding: PASS",
                "REVIEWER source_boundary: PASS",
                "REVIEWER language_quality: PASS",
                "REVIEWER recovery: PASS",
                "REVIEWER outcome: PASS",
                "REVIEWER format: PASS",
                "REPAIR_ACTIONS: None",
            ]
        ),
        seed_spec,
        threshold=0.8,
        expected_assistant_turns=2,
    )

    assert review["selection"]["accepted"] is True
    assert "ASSISTANT 1: PASS" in review["review"]["turn_evidence_audit"]
    assert review["review"]["reviewers"][0]["finding"] == "The dialogue is coherent."


def test_preference_parser_accepts_space_label_for_rejected_answer() -> None:
    parsed = multi_turn_build._parse_preference_pair(
        "\n".join(
            [
                "FAILURE MODE: stale context",
                "CHOSEN ACTION: revise from visible source",
                "REJECTED ACTION: invent missing facts",
                "REJECTED ANSWER:",
                "Det betyder, at du bare kan bruge alle reglerne uden at tjekke kilden.",
            ]
        )
    )

    assert parsed["failure_mode"] == "stale context"
    assert parsed["chosen_action"] == "revise from visible source"
    assert parsed["rejected_action"] == "invent missing facts"
    assert parsed["rejected_answer"] == "Det betyder, at du bare kan bruge alle reglerne uden at tjekke kilden."


def test_preference_parser_uses_unlabeled_body_as_fallback_answer() -> None:
    parsed = multi_turn_build._parse_preference_pair(
        "\n".join(
            [
                "FAILURE_MODE: generic_answer",
                "CHOSEN_ACTION: source-bounded final answer",
                "REJECTED_ACTION: broad unsupported advice",
                "Her er et generelt svar uden at bruge uddraget.",
            ]
        )
    )

    assert parsed["failure_mode"] == "generic_answer"
    assert parsed["rejected_answer"] == "Her er et generelt svar uden at bruge uddraget."


def test_review_parser_blocks_accept_when_any_reviewer_fails() -> None:
    seed_spec = {
        "success_criteria": ["avoid future leakage"],
        "difficulty_axis": "turn-causal grounding",
        "intent_trajectory_hint": "planning_interaction",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: ACCEPT",
                "SCORE: 0.98",
                "INTENT_TRAJECTORY: planning_interaction",
                "SUCCESS_CRITERIA: avoid future leakage",
                "TURN_EVIDENCE_AUDIT: ASSISTANT 1: PASS - supported; ASSISTANT 2: FAIL - uses a fact that appears only in USER 3; ASSISTANT 3: PASS - supported",
                "REVIEWER coherence: PASS - coherent",
                "REVIEWER naturalness: PASS - natural",
                "REVIEWER grounding: FAIL - ASSISTANT 2 leaks future user information",
                "REVIEWER source_boundary: PASS - no source issue",
                "REVIEWER language_quality: PASS - clean language",
                "REVIEWER recovery: PASS - revises",
                "REVIEWER format: PASS - alternates",
                "REPAIR_ACTIONS: Regenerate assistant turns causally.",
            ]
        ),
        seed_spec,
        threshold=0.8,
    )

    assert review["selection"]["accepted"] is False
    assert review["review"]["review_decision"] == "repair"


def test_review_parser_blocks_accept_when_audit_is_missing() -> None:
    seed_spec = {
        "success_criteria": ["include evidence audit"],
        "difficulty_axis": "review strictness",
        "intent_trajectory_hint": "planning_interaction",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: ACCEPT",
                "SCORE: 0.95",
                "INTENT_TRAJECTORY: planning_interaction",
                "SUCCESS_CRITERIA: include evidence audit",
                "REVIEWER coherence: PASS - coherent",
                "REVIEWER naturalness: PASS - natural",
                "REVIEWER grounding: PASS - grounded",
                "REVIEWER source_boundary: PASS - no source issue",
                "REVIEWER language_quality: PASS - clean",
                "REVIEWER recovery: PASS - revises",
                "REVIEWER format: PASS - alternates",
                "REPAIR_ACTIONS: none",
            ]
        ),
        seed_spec,
        threshold=0.8,
    )

    assert review["selection"]["accepted"] is False
    assert review["review"]["review_decision"] == "repair"


def test_review_parser_blocks_accept_when_language_quality_fails() -> None:
    seed_spec = {
        "success_criteria": ["clean language"],
        "difficulty_axis": "language quality",
        "intent_trajectory_hint": "troubleshooting_interaction",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: ACCEPT",
                "SCORE: 0.95",
                "INTENT_TRAJECTORY: troubleshooting_interaction",
                "SUCCESS_CRITERIA: clean language",
                "TURN_EVIDENCE_AUDIT: ASSISTANT 1: PASS - supported; ASSISTANT 2: PASS - supported; ASSISTANT 3: PASS - supported",
                "REVIEWER coherence: PASS - coherent",
                "REVIEWER naturalness: PASS - natural",
                "REVIEWER grounding: PASS - grounded",
                "REVIEWER source_boundary: PASS - no source issue",
                "REVIEWER language_quality: FAIL - corrupted command fragment in Danish answer",
                "REVIEWER recovery: PASS - revises",
                "REVIEWER format: PASS - alternates",
                "REPAIR_ACTIONS: Regenerate corrupted assistant turn.",
            ]
        ),
        seed_spec,
        threshold=0.8,
    )

    assert review["selection"]["accepted"] is False
    assert review["review"]["review_decision"] == "repair"


def test_review_parser_blocks_accept_when_outcome_fails() -> None:
    seed_spec = {
        "success_criteria": ["complete a useful work-session output"],
        "difficulty_axis": "task completion",
        "intent_trajectory_hint": "Transaction Interaction",
        "subtask_trajectory_hint": "artifact_revision",
        "domain": "tenant_email_revision",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: ACCEPT",
                "SCORE: 0.95",
                "INTENT_TRAJECTORY: Transaction Interaction",
                "SUBTASK_TRAJECTORY: artifact_revision",
                "SUCCESS_CRITERIA: complete a useful work-session output",
                "TURN_EVIDENCE_AUDIT: ASSISTANT 1: PASS - supported; ASSISTANT 2: PASS - supported; ASSISTANT 3: PASS - supported",
                "REVIEWER coherence: PASS - coherent",
                "REVIEWER naturalness: PASS - natural",
                "REVIEWER grounding: PASS - grounded",
                "REVIEWER source_boundary: PASS - no source issue",
                "REVIEWER language_quality: PASS - clean",
                "REVIEWER recovery: PASS - revises",
                "REVIEWER outcome: FAIL - the dialogue gives generic advice but never revises or completes the email",
                "REVIEWER format: PASS - alternates",
                "REPAIR_ACTIONS: Produce the revised email artifact.",
            ]
        ),
        seed_spec,
        threshold=0.8,
    )

    assert review["selection"]["accepted"] is False
    assert review["review"]["review_decision"] == "repair"


def test_review_parser_uses_expected_assistant_turn_count_for_audit() -> None:
    seed_spec = {
        "success_criteria": ["audit all assistant turns"],
        "difficulty_axis": "variable turn count",
        "intent_trajectory_hint": "planning_interaction",
    }
    review = multi_turn_build._parse_review(
        "\n".join(
            [
                "SELECTION: ACCEPT",
                "SCORE: 0.95",
                "INTENT_TRAJECTORY: planning_interaction",
                "SUCCESS_CRITERIA: audit all assistant turns",
                "TURN_EVIDENCE_AUDIT: ASSISTANT 1: PASS - supported; ASSISTANT 2: PASS - supported",
                "REVIEWER coherence: PASS - coherent",
                "REVIEWER naturalness: PASS - natural",
                "REVIEWER grounding: PASS - grounded",
                "REVIEWER source_boundary: PASS - no source issue",
                "REVIEWER language_quality: PASS - clean",
                "REVIEWER recovery: PASS - revises",
                "REVIEWER format: PASS - alternates",
                "REPAIR_ACTIONS: none",
            ]
        ),
        seed_spec,
        threshold=0.8,
        expected_assistant_turns=2,
    )

    assert review["selection"]["accepted"] is True


def test_mds_scores_include_reviewed_task_completion() -> None:
    row = _mds_candidate("work-session", family="general_chat", domain="event_planning")
    row["hidden"]["review"] = {
        "reviewers": [
            {"name": "outcome", "passed": True, "finding": "completed useful work"},
        ]
    }

    scored = multi_turn_build._with_mds_scores(row)
    local_scores = scored["hidden"]["selection"]["mds"]["local_scores"]

    assert local_scores["task_completion"] == 0.95
    assert "task_completion" in local_scores


def test_first_turn_substance_score_penalizes_empty_help_requests() -> None:
    empty = "Jeg skal forklare et forskningsabstract, men forstår det ikke helt. Kan du hjælpe?"
    concrete = "Her er abstractet: \"A cross-sectional study of 1,200 adolescents...\" Kan du forklare det kort?"
    prior_attempt = "Jeg får ModuleNotFoundError i Python efter at have opdateret requests-forge fra 0.8 til 0.9."

    assert multi_turn_build._first_turn_substance_score(empty) < 0.5
    assert multi_turn_build._first_turn_substance_score(concrete) > 0.7
    assert multi_turn_build._first_turn_substance_score(prior_attempt) > 0.7


def test_candidate_seed_spec_adds_instance_variation() -> None:
    seed_spec = {
        "seed_id": "seed::v00",
    }

    first = multi_turn_build._candidate_seed_spec(seed_spec, candidate_index=0)
    second = multi_turn_build._candidate_seed_spec(seed_spec, candidate_index=1)

    assert first["scenario_instance"]["id"] == "artifact_payload"
    assert second["scenario_instance"]["id"] == "constraint_payload"
    assert first["scenario_instance"]["seed_id"] == "seed::v00"
    assert "fresh scenario" in first["scenario_instance"]["instruction"]


def test_scenario_instance_prompt_fans_out_before_blueprint() -> None:
    seed_spec = {
        "seed_id": "seed::v00",
        "family": "general_chat",
        "domain": "travel_planning",
        "source_language": "da",
        "dialogue_language": "da",
        "latent_language": "en",
        "difficulty_axis": "vary the concrete planning case",
        "success_criteria": ["keep prices visibly uncertain"],
        "intent_trajectory_hint": "Problem Solving Interaction",
        "subtask_trajectory_hint": "planning_interaction",
        "seed_blueprint": {
            "domain": "travel_planning",
            "hidden_goal": "make a bounded trip plan",
            "success_criteria": ["keep prices visibly uncertain"],
        },
        "scenario_variant": {
            "id": "artifact_payload",
            "first_turn_shape": "opening includes concrete task material",
        },
        "scenario_instance": {
            "id": "constraint_payload",
            "variation_axis": "change the visible constraints",
        },
    }

    prompt = "\n".join(message["content"] for message in multi_turn_build._scenario_instance_messages(seed_spec))

    assert "ConsistentChat diversity fanout layer before APIGen-MT-style blueprint generation" in prompt
    assert "the seed is an archetype, not the final scenario" in prompt
    assert "Treat the seed query trajectory as a pattern, not a fixed script" in prompt
    assert "Do not write the dialogue" in prompt
    assert "visible first-turn payload" in prompt


def test_generate_scenario_instance_adds_materialized_artifact() -> None:
    seed_spec = {
        "seed_id": "seed::v00",
        "family": "general_chat",
        "domain": "travel_planning",
        "source_language": "da",
        "dialogue_language": "da",
        "latent_language": "en",
        "difficulty_axis": "vary the concrete planning case",
        "success_criteria": ["keep prices visibly uncertain"],
        "intent_trajectory_hint": "Problem Solving Interaction",
        "subtask_trajectory_hint": "planning_interaction",
        "seed_blueprint": {
            "domain": "travel_planning",
            "hidden_goal": "make a bounded trip plan",
            "success_criteria": ["keep prices visibly uncertain"],
        },
        "scenario_instance": {
            "id": "constraint_payload",
            "variation_axis": "change the visible constraints",
        },
    }

    materialized = multi_turn_build._generate_scenario_instance(FakeDialogueLLM(), seed_spec, {"scenario_temperature": 0.9})

    assert materialized is not seed_spec
    assert "artifact" in materialized["scenario_instance"]
    assert "Concrete user situation" in materialized["scenario_instance"]["artifact"]
    assert "artifact" not in seed_spec["scenario_instance"]


def test_mds_selection_writes_candidate_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        multi_turn_build,
        "load_clients",
        lambda model_refs: {role: FakeDialogueLLM() for role in model_refs},
    )

    cfg_path = Path(__file__).resolve().parents[1] / "sdg" / "packs" / "multi_turn_dialogue" / "configs" / "base.yaml"
    cfg = read_yaml(cfg_path)
    cfg["reuse_completed"] = False
    cfg["generation"]["count"] = 3
    cfg["generation"]["candidate_multiplier"] = 1
    cfg["generation"]["include_preference_pairs"] = False
    cfg["generation"]["shuffle_seeds"] = False

    result = build(cfg)
    rows = read_jsonl(result.artifacts["dataset"].path)
    candidates = read_jsonl(result.artifacts["candidates"].path)
    rejected_candidates = read_jsonl(result.artifacts["rejected_candidates"].path)
    selection_report = read_json(result.artifacts["selection_report"].path)

    assert len(candidates) == 3
    assert len(rows) == cfg["generation"]["count"]
    assert len(rejected_candidates) == len(candidates) - len(rows)
    assert selection_report["status"] == "completed"
    assert selection_report["selection_complete"] is True
    assert selection_report["shortfall"] == 0


def test_mds_selection_stratifies_families_before_filling_bins() -> None:
    candidates = [
        _mds_candidate(f"general-{index}", family="general_chat", domain=f"general_{index}")
        for index in range(6)
    ]
    candidates.extend(
        _mds_candidate(f"grounded-{index}", family="grounded_dialogue", domain=f"grounded_{index}")
        for index in range(2)
    )

    selected, rejected, report = multi_turn_build._select_rows_mds(candidates, target_count=4)
    selected_families = [row["meta"]["family"] for row in selected]

    assert selected_families.count("grounded_dialogue") == 2
    assert selected_families.count("general_chat") == 2
    assert len(rejected) == 4
    assert report["families"]["grounded_dialogue"] == 2


def _mds_candidate(row_id: str, *, family: str, domain: str) -> dict[str, object]:
    messages = [
        {"role": "user", "content": f"{domain} vague request"},
        {"role": "assistant", "content": f"{domain} clarifying question"},
        {"role": "user", "content": f"{domain} added details"},
        {"role": "assistant", "content": f"{domain} final answer"},
    ]
    skeleton = [
        {"role": "user", "dialogue_act": "vague_initial_request", "state_delta": "ask"},
        {"role": "assistant", "dialogue_act": "clarify_missing_slots", "state_delta": "clarify"},
        {"role": "user", "dialogue_act": "reveal_details", "state_delta": "details"},
        {"role": "assistant", "dialogue_act": "finalize", "state_delta": "answer"},
    ]
    return {
        "id": row_id,
        "messages": messages,
        "hidden": {
            "blueprint": {
                "evidence_boundary": {
                    "available_information": "visible conversation and stable general background knowledge",
                    "generation_hidden_state": "hidden blueprint facts are not assistant-visible evidence",
                    "unavailable_information": [],
                    "requires_hedging": False,
                    "volatile_items": [],
                    "dialogue_language": "en",
                }
            },
            "intent_model": {
                "user_query_plan_steps": [
                    f"{domain} vague request",
                    f"{domain} added details",
                ]
            },
            "skeleton": skeleton,
            "selection": {
                "accepted": True,
                "score": 0.9,
            },
        },
        "meta": {
            "family": family,
            "domain": domain,
            "intent_trajectory": "planning_interaction",
        },
    }


def test_multi_turn_dialogue_pack_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        multi_turn_build,
        "load_clients",
        lambda model_refs: {role: FakeDialogueLLM() for role in model_refs},
    )

    cfg_path = Path(__file__).resolve().parents[1] / "sdg" / "packs" / "multi_turn_dialogue" / "configs" / "base.yaml"
    cfg = read_yaml(cfg_path)
    cfg["generation"]["count"] = 1
    cfg["generation"]["candidate_multiplier"] = 1
    cfg["generation"]["shuffle_seeds"] = False

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id
    assert first.pack == "multi_turn_dialogue"
    assert {
        "blueprints",
        "candidates",
        "candidate_assistant_turns",
        "candidate_blueprints",
        "candidate_final_rows",
        "candidate_intent_models",
        "candidate_messages",
        "candidate_polished_messages",
        "candidate_reviews",
        "candidate_scenario_instances",
        "candidate_skeletons",
        "candidate_user_turns",
        "dataset",
        "intent_models",
        "preference_pairs",
        "rejected_candidates",
        "selection_report",
        "skeletons",
    } <= set(first.artifacts)

    loaded = load(first.run_id)
    rows = read_jsonl(loaded.artifacts["dataset"].path)
    blueprints = read_jsonl(loaded.artifacts["blueprints"].path)
    candidate_blueprints = read_jsonl(loaded.artifacts["candidate_blueprints"].path)
    candidate_final_rows = read_jsonl(loaded.artifacts["candidate_final_rows"].path)
    candidate_reviews = read_jsonl(loaded.artifacts["candidate_reviews"].path)
    candidates = read_jsonl(loaded.artifacts["candidates"].path)
    intent_models = read_jsonl(loaded.artifacts["intent_models"].path)
    rejected_candidates = read_jsonl(loaded.artifacts["rejected_candidates"].path)
    skeletons = read_jsonl(loaded.artifacts["skeletons"].path)
    preference_pairs = read_jsonl(loaded.artifacts["preference_pairs"].path)
    selection_report = read_json(loaded.artifacts["selection_report"].path)

    assert len(rows) == cfg["generation"]["count"]
    assert len(candidates) == cfg["generation"]["count"] * cfg["generation"]["candidate_multiplier"]
    assert len(candidate_blueprints) == len(candidates)
    assert len(candidate_final_rows) == len(candidates)
    assert len(candidate_reviews) == len(candidates)
    assert len(rejected_candidates) == len(candidates) - len(rows)
    assert len(blueprints) == len(rows)
    assert len(intent_models) == len(rows)
    assert len(skeletons) == len(rows)
    assert len(preference_pairs) == len(rows)
    assert selection_report["accepted_rows"] == len(rows)
    assert selection_report["candidate_rows"] == len(candidates)
    assert selection_report["preference_pairs"] == len(preference_pairs)

    row = rows[0]
    assert "blueprint" in row["hidden"]
    assert "evidence_boundary" in row["hidden"]["blueprint"]
    assert "intent_model" in row["hidden"]
    assert "skeleton" in row["hidden"]
    assert row["hidden"]["intent_model"]["trajectory"] == "Problem Solving Interaction"
    assert row["hidden"]["intent_model"]["subtask_trajectory"] == "planning_interaction"
    assert row["hidden"]["intent_model"]["work_session_type"] == "practical_solution_session"
    assert row["hidden"]["blueprint"]["work_session"]["top_level_intent"] == "Problem Solving Interaction"
    assert row["meta"]["subtask_trajectory"] == "planning_interaction"
    assert row["meta"]["work_session_type"] == "practical_solution_session"
    assert row["hidden"]["selection"]["mds"]["local_scores"]["task_completion"] > 0
    assert row["hidden"]["selection"]["mds"]["selected"] is True
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][-1]["role"] == "assistant"
    assert "prompt" not in row
    assert "target" not in row
    assert row["meta"]["language"] == "en"
    assert row["meta"]["language_mode"] == "same_language"
    assert "APIGen-MT" in row["meta"]["source_methods"]
    assert "ConsistentChat" in row["meta"]["source_methods"]

    pair = preference_pairs[0]
    assert pair["chosen"]["role"] == "assistant"
    assert pair["rejected"]["role"] == "assistant"
    assert pair["hidden"]["failure_mode"] == "ignores_late_constraint"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0
    assert verification["dataset_checks"]["rows"] == len(rows)

    summary = summarize(first.run_id)
    assert summary["rows"] == cfg["generation"]["count"]
    assert summary["preference_pairs"] == cfg["generation"]["count"]
    assert summary["metrics"]["checks"]["messages_alternate"]["failed"] == 0
    assert summary["metrics"]["checks"]["evidence_boundary_respected"]["failed"] == 0

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert published["preference_pairs"] == cfg["generation"]["count"]
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "preference_pairs.parquet").exists()
    assert (out_dir / "selection_report.json").exists()
    assert (out_dir / "dataset_checks.json").exists()
