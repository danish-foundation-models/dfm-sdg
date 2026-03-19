# dfm-sdg

This repository is bootstrapping a thin SDG commons plus method packs.

Current Phase 0 scope:

- `sdg.commons.*` provides run tracking, artifact IO, publishing helpers, model adapters, and pack discovery.
- `sdg.packs.demo` is a small arithmetic pack that proves the contract end to end.
- `sdg.packs.pleias_synth` now has a real memory-core setup path for Wikipedia Vital Articles plus structured Wikipedia and Wikidata enrichment, with an initial memorization/retrieval generator on top.
- model endpoints are now loaded by the commons from `.env` as named reusable endpoints, so one step can bind several models against the same or different backends.
- `sdg` CLI exposes `build`, `verify`, `summarize`, `publish`, `compare`, and `list-packs`.

Example commands:

```bash
uv run sdg list-packs
uv run sdg build sdg/packs/demo/configs/base.yaml
uv run sdg verify <run-id>
uv run sdg publish <run-id>
uv run sdg build sdg/packs/pleias_synth/configs/base.yaml
uv run sdg build sdg/packs/pleias_synth/configs/smoke.yaml
```

For model-backed PleIAs steps, define named endpoints in `.env` and bind pack roles to them in config.

Development checks:

```bash
uv run ruff check .
uv run pytest
```

The project now targets Python 3.13.

Ruff stays intentionally moderate here and checks:

- import and syntax issues
- import ordering
- simple Python modernizations
- a small set of bug-prone patterns
- a few low-friction cleanup and correctness checks
