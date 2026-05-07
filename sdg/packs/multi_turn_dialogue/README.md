# Multi-Turn Dialogue Pack

`multi_turn_dialogue` is a blueprint-first pack for realistic multi-turn synthetic conversations.

The pack follows the research pattern from the provided plan:

- APIGen-MT-style blueprint-first hidden-state design before dialogue generation
- ConsistentChat-style intent modeling before dialogue skeleton generation
- Socratic-style user simulation where information is revealed incrementally
- Evidence review and repair for turn-level grounding/source-boundary failures
- ACT-style preference pairs for action mistakes such as premature answers or missed clarifications
- MDS-style dialogue-level selection over candidate conversations

## What the pack emits

Each accepted dataset row has:

- `hidden.blueprint`: latent user goal, constraints, generator-only facts, assistant-visible context requirements, conversation flow, and success criteria
- `hidden.intent_model`: ConsistentChat top-level interaction type, finer subtask trajectory, work-session type, user-query plan, information flow, role interaction, and guardrails
- `hidden.skeleton`: the planned role/action trajectory, including visible state progress and work-product progress
- `messages`: the generated user/assistant dialogue
- `hidden.review`: reviewer checks for coherence, naturalness, grounding, and recovery
- `hidden.verification`: success criteria and dialogue acts used for review and structural checks
- `hidden.selection`: reviewer acceptance plus MDS global-bin and local-structure selection metadata

Every row carries an evidence boundary: hidden blueprint facts are generator state, not assistant evidence. The assistant may use stable public and general knowledge as background knowledge, including broadly known public procedures, but should use careful epistemics throughout and keep public knowledge high-level unless the visible conversation supplies the controlling source. Even public background knowledge should not sound freshly verified, current, official, complete, or guaranteed when precision matters. Claims that depend on current/live data, a specific provider, a named institution, an official current source, a jurisdiction, a private organization, the user's personal situation, or a pasted/private source must either be visible in the conversation or be framed as uncertain, general, illustrative, or source-dependent. Deadlines, appeal windows, complaint bodies, fees, official URLs, rights/entitlements, and success likelihood should not be stated as verified without visible source support. Visible source, policy, form, or excerpt text controls the task once supplied; the assistant should not contradict it or invent exceptions.

Rows also carry a discourse-quality requirement. User turns should be semantically plausible human turns, not literal blueprint translations. If a visible user turn contains odd wording, a likely typo or translation artifact, a contradictory premise, or an ambiguity that materially changes the task, the assistant should recover by asking a concise clarification, stating a reasonable assumption, or gently reframing the request instead of silently going along with the oddity. Danish output should use ordinary idiomatic Danish and avoid invented words or awkward calques.

Assistant turns should use restrained plain text rather than emojis, decorative emoji bullets, markdown emphasis, horizontal rules, decorative separators, or long heading stacks unless the user explicitly asks for a formatted artifact.

## Work-Session Layer

The pack keeps ConsistentChat's nine top-level intent categories as the primary taxonomy. Under each top-level intent, it adds a work-session contract that describes the useful LLM work the dialogue should accomplish.

Examples:

- `Problem Solving Interaction > planning > practical_solution_session`
- `Educational Interaction > misconception correction > learning_work_session`
- `Information Retrieval Interaction > source-grounded explanation > grounded_synthesis_session`
- `Transaction Interaction > artifact revision > bounded_artifact_completion_session`
- `Emotional Support Interaction > low-pressure next step > supportive_next_step_session`

This layer is not a competing taxonomy. It is a deliverable objective: the conversation should usually transform visible material, produce or revise an artifact, run a visible calculation, explain a supplied item, compare visible options, provide rehearsal feedback, or create a bounded plan/decision frame/next-step output. For softer exploratory or support seeds, the final output can be a structured frame, message draft, low-pressure checklist, or concrete next step rather than a formal artifact.

First turns should be concise when appropriate, but not empty. The pack now prefers first user turns with at least one useful work payload: a concrete fact, number, deadline, draft, excerpt, named option, role/context, prior attempt, error message, or constraint. Pure openings such as "can you help me understand this?" without any task material should be repaired or ranked below payload-bearing openings. `sparse_opening` remains a sampleable type, but it must still contain one concrete payload item; the other opener variants require richer payload rules.

## Language Plan

The pack supports English and Danish visible conversations through language-specific seed sources and `generation.languages`.

By default, latent planning artifacts stay in English (`generation.latent_language: en`) while user and assistant turns are generated in the target visible language. This keeps blueprint, ConsistentChat intent artifacts, skeletons, and review outputs easier to parse and compare across languages, while still producing Danish `messages` rows. Danish generation uses localized seed cards rather than translating English cards implicitly. Rows record `source_language`, `prompt_language`, `reasoning_language`, `target_language`, and `language_mode` in `meta`.

Private/source facts are handled through the same visible-language evidence boundary: source-specific rules must appear in prior user-visible text before the assistant can apply them as facts.

The build also writes separate artifacts:

- `blueprints.jsonl`
- `intent_models.jsonl`
- `skeletons.jsonl`
- `candidates.jsonl`
- `rejected_candidates.jsonl`
- `dataset.jsonl`
- `preference_pairs.jsonl`
- `selection_report.json`

## Current scope

The implementation uses the model roles configured in `.env` through the standard commons model layer. Model calls produce staged text artifacts: a latent blueprint, an intent model, a skeleton, simulated user turns, assistant turns, review findings, and contrastive rejected answers. The pack parses those artifacts into the repo's JSONL storage format.

MDS selection happens after candidate generation. The pack bins candidates by the ConsistentChat user-query trajectory plus the realized skeleton trajectory, stratifies across accepted families, then ranks within bins using entity grounding, information progress, query-answer form consistency, evidence-boundary compliance, reviewed task completion, and first-turn substance.

First-stage seeds can come from handwritten seed-source cards in `seed_sources/handwritten.yaml` or `seed_sources/handwritten_da.yaml`, or from `source_pack` adapters. A handwritten card defines an archetype: seed domain, family, persona, latent goal, ConsistentChat query-trajectory hint, difficulty axis, success criteria, and optional epistemic-risk metadata. A `source_pack` normalizes document excerpts and sampled personas into source/persona pairs, samples a source-payload visibility strategy, then an LLM `source_task_planner` turns each pair into a source-grounded archetype before the normal blueprint stage. Persona projection can be enabled on a persona source to pass only surface-level behavioral signals, such as role archetype, age band, education level, communication style, task posture, and practical constraints, instead of raw biography columns. The pack maps each archetype onto ConsistentChat's nine top-level interaction types, then keeps a finer `subtask_trajectory` underneath for task-specific patterns such as planning, troubleshooting, refinement, comparison, source-grounded extraction, and form completion.

The source-pack path keeps deterministic work limited to source normalization, filtering, excerpt trimming, source-payload strategy sampling, persona projection, and metadata preservation. Task choice is deliberately stochastic: the planner sees the source excerpt, source metadata, scenario variant, sampled payload plan, and projected persona context, then chooses a realistic user situation, missing facts, late-turn reveal, useful deliverable, and success criteria. Source-pack user simulation may expose a full excerpt, a long contiguous passage, selected clauses, a user summary plus exact quotes, or staged excerpts across turns. The assistant is still only allowed to use source facts after they appear in visible user text. Persona sources such as `nvidia/Nemotron-Personas-USA` are treated as diversity seeds only; for Danish conversations the planner is instructed to localize naturally and not copy irrelevant US names, locations, biographies, institutions, laws, or currency into the dialogue.

Candidate generation now adds an explicit scenario fanout stage before the APIGen-MT-style blueprint. The scenario fanout treats the handwritten card as an archetype, then asks the model to instantiate a fresh concrete scenario with different visible payloads, names, quantities, artifacts, stakeholder context, prior attempts, and disclosure timing where the success criteria allow it. The hidden blueprint is then generated from that row-specific scenario instance rather than directly from the handwritten seed alone.

`generation.seed_expansion.variants_per_seed` expands each archetype into deterministic scenario variants before candidate generation. Variants change first-turn informativeness, user-owned facts, visible artifacts, quantities, prior attempts, stakeholder context, disclosure timing, work-session focus, and dialogue length. Current variants target 3-5 user/assistant exchanges, producing 6-10 visible messages while preserving the archetype's evidence boundary and success criteria.

Each generated candidate also receives a candidate-level scenario instance. This gives repeated candidates from the same archetype a fresh variation axis, such as artifact payload, constraint payload, prior-attempt payload, or named-options payload. The model then materializes that axis into a compact scenario artifact before blueprint generation. The intent is to keep the seed as an archetype rather than a fixed conversation script.

Seed expansion is ordered by variant round, not by archetype. Small runs see broad domain coverage first; larger runs then cover additional variants of each archetype. The base configs keep full-length staged generations and raise model read timeouts for long blueprint, skeleton, dialogue, review, and preference calls. The generator and reviewer use one shared general epistemic policy rather than seed-specific string checks. This keeps seed acquisition separate from dialogue generation so later adapters can normalize log clusters, reversed instruct datasets, Magpie-style intent seeds, localized seed banks, or document excerpts into the same shape.

The Danish handwritten seed bank currently has 38 archetypes expanded to 304 seed specs by the default `variants_per_seed: 8`, covering all nine ConsistentChat top-level interaction types. The Dynaword source-grounded config samples 14 Danish source packs with up to 512 document records per source: `retsinformationdk`, `skat`, `domsdatabasen`, `eur-lex-sum-da`, `retspraksis`, `tidsskrift-dk`, `miljoeportalen`, `fm-udgivelser`, `ai-aktindsigt`, `ft`, `ep`, `municipality_meetings`, `danske-taler`, and `health_hovedstaden`, each paired with projected surface personas.

The first families are:

- `general_chat`: planning, troubleshooting, and refinement dialogues
- `grounded_dialogue`: document-grounded conversations where private facts must be surfaced in visible user/source text before use

Tool execution should stay in `tool_use` until that pack has real executable validators.
This pack should not emit tool-call messages, API schemas, executable actions, tool observations, or agent-environment state.

## Model roles

The base config binds every role to the `openai` endpoint alias:

- `models.blueprint_writer`
- `models.intent_modeler`
- `models.skeleton_writer`
- `models.user_simulator`
- `models.assistant_teacher`
- `models.reviewer`
- `models.preference_writer`

## Start here

```bash
uv run sdg build sdg/packs/multi_turn_dialogue/configs/base.yaml
uv run sdg build sdg/packs/multi_turn_dialogue/configs/base_da.yaml
uv run sdg build sdg/packs/multi_turn_dialogue/configs/dynaword_public_rules_da.yaml
uv run sdg verify <run-id>
uv run sdg summarize <run-id>
uv run sdg publish <run-id>
```
