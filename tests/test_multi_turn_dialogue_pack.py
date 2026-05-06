from __future__ import annotations

from pathlib import Path

from sdg.commons.model import LLM
from sdg.commons.run import load
from sdg.commons.store import read_jsonl
from sdg.commons.utils import read_json, read_yaml
from sdg.packs.multi_turn_dialogue import build as multi_turn_build
from sdg.packs.multi_turn_dialogue.build import build, publish, summarize, verify


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
    assert "Avoid markdown emphasis" in assistant_prompt
    assert "Use restrained plain-text formatting" in assistant_prompt
    assert "may be vague, partial, or reasonably informative" in user_prompt
    assert "do not make every conversation start with underspecification" in user_prompt
    assert "The first user turn should normally contain at least one useful work payload" in user_prompt
    assert "underspecify the first turn" not in user_prompt
    assert "phrase late constraints as new information" in user_prompt
    assert "should not pretend that content was present" in assistant_prompt
    assert "include a short visible payload in that same turn" in user_prompt
    assert "If a needed artifact is missing, ask for it" in assistant_prompt
    assert "preserve missing details as unknown fields" in assistant_prompt
    assert "Reject missing visible artifacts" in reviewer_prompt
    assert "Reject unsupported quoted edits" in reviewer_prompt
    assert "unsupported reaction" in reviewer_prompt
    assert "invented words" in reviewer_prompt
    assert "natural, ordinary Danish" in user_prompt
    assert "do not invent official product" in user_prompt
    assert "official localized term is uncertain" in reviewer_prompt
    assert "Do not mix English words into Danish prose" in user_prompt
    assert "do not invent literal compounds" in user_prompt
    assert "markdown-heavy assistant turns" in reviewer_prompt
    assert "horizontal rules" in reviewer_prompt


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
    assert "Entertainment rows are not exempt" in prompt
    assert "strict audit" in prompt
    assert strict["accepted"] is False
    assert strict["checks"][0]["passed"] is False
    assert strict["repair_actions"]


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
    assert "Remove emojis and decorative emoji bullets" in prompt
    assert "Remove excessive markdown bold/italic emphasis" in prompt
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


def test_assistant_format_restrained_blocks_markdown_and_emojis() -> None:
    clean_row = {"messages": [{"role": "assistant", "content": "Her er tre korte punkter:\n- Første\n- Andet"}]}
    markdown_row = {"messages": [{"role": "assistant", "content": "**Plan**\n---\n| A | B |"}]}
    emoji_row = {"messages": [{"role": "assistant", "content": "God ide ✅"}]}

    assert multi_turn_build._assistant_format_restrained(clean_row)["passed"] is True
    assert multi_turn_build._assistant_format_restrained(markdown_row)["passed"] is False
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
    assert set(first.artifacts) == {
        "blueprints",
        "candidates",
        "dataset",
        "intent_models",
        "preference_pairs",
        "rejected_candidates",
        "selection_report",
        "skeletons",
    }

    loaded = load(first.run_id)
    rows = read_jsonl(loaded.artifacts["dataset"].path)
    blueprints = read_jsonl(loaded.artifacts["blueprints"].path)
    candidates = read_jsonl(loaded.artifacts["candidates"].path)
    intent_models = read_jsonl(loaded.artifacts["intent_models"].path)
    rejected_candidates = read_jsonl(loaded.artifacts["rejected_candidates"].path)
    skeletons = read_jsonl(loaded.artifacts["skeletons"].path)
    preference_pairs = read_jsonl(loaded.artifacts["preference_pairs"].path)
    selection_report = read_json(loaded.artifacts["selection_report"].path)

    assert len(rows) == cfg["generation"]["count"]
    assert len(candidates) == cfg["generation"]["count"] * cfg["generation"]["candidate_multiplier"]
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
