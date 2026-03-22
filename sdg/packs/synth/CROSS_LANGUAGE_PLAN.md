# Cross-Language Plan

## Goal

Make `synth` language-aware without forking the pack.

The core use case is cross-language memorization:

- source corpus in one language
- prompt in another language
- reasoning in either language
- target in either language

Example:

- source corpus: English
- prompt: Danish
- reasoning: English or Danish
- target: Danish

## Principles

- One pack, not `synth_da`.
- One explicit run-level language plan.
- No optional per-row language overrides.
- Fail on unknown languages.
- Keep source language and output languages separate.

## Current blockers

- Retrieval uses lexical overlap on the visible prompt.
- Verification uses lexical overlap between visible fields and source evidence.
- Prompt templates are English-only.
- Tokenization is still too English-shaped.
- Local corpora do not carry explicit source language by default.

## Implementation order

1. Add explicit `source_language` and `language_plan` plumbing.
2. Add row metadata for `source`, `prompt`, `reasoning`, and `target` languages.
3. Make question, reasoning, and answer prompts language-aware.
4. Use source-side retrieval when the row is cross-language.
5. Add a source-language canonical answer for cross-language verification.
6. Generalize tokenization and lexical resources by language.
7. Move prompts and presets into language-specific data files.

## Status

- Done:
  - explicit `source_language` and `language_plan`
  - row metadata for `source`, `prompt`, `reasoning`, and `target` languages
  - language-aware instructions in question, reasoning, answer, and judge prompts
  - model-generated visible language-choice reasoning via `delivery_note`
  - source-side retrieval for cross-language rows
  - hidden `source_target` for cross-language support and coverage checks
  - hidden `source_reasoning` for cross-language reasoning grounding checks
  - shared Unicode-safe tokenization for retrieval and filter checks
  - basic `en`/`da` lexical resources for stopwords and person detection
  - cross-language judge split into explicit `language_match` and `language_natural` checks, with derived `language_quality`
  - Danish cross-language probe config and live validation
- Remaining:
  - language-specific prompt and preset files
  - decide whether cross-language language quality needs stronger filtering beyond the current judge

## Notes

- `same_language` is not a separate codepath choice in config. It is the derived mode when all language plan values match the source language.
- Corpora and languages stay orthogonal.
  - English corpus + Danish outputs is valid.
  - Danish corpus + English outputs is valid.
- Current Danish probe result:
  - `2026-03-20T130056+0000-27d99302`
  - 12 candidates, 11 kept, 1 rejected
  - kept rows passed `language_quality`
  - rejected rows still failed on `coverage_not_supported`, not language quality
  - explicit language prompting cleaned up the visible Danish outputs I spot-checked, but the judge still did not actively reject any row on language checks in this probe
