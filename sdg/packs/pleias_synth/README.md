# PleIAs SYNTH Pack

This pack is the first real method pack for the framework.

Current implementation scope:

- memory-core source loading from Wikipedia Vital Articles or local JSON/JSONL
- cached title discovery from the Vital Articles pages
- batched article fetches from Wikipedia with provenance and license metadata
- structured Wikipedia enrichment from categories, outgoing article links, and templates
- optional Wikidata enrichment for structured metadata
- cleaning, sectioning, chunking, and lightweight lexical indexing
- memorization row generation from lead facts with lexical support retrieval
- teacher-side fact bundles with supporting claims and structured context
- LLM-backed query generation, answer generation, and judge filtering through commons endpoint configs in `.env`
- dedicated backreasoning generation with a stronger sectioned `reasoning` column
- normalized persona loading with a starter preset, plus support for inline or external persona datasets
- normalized query-profile loading with structured fields like channel, fluency, register, urgency, query shape, and exemplars
- verification and publication of memory-core artifacts

Not implemented yet:

- grounded RAG, math, editing, creative, and dialogue generators
- teacher-model training
- local embedding and reranking support
- broader multi-document fact-bundle construction beyond the current memorization path
