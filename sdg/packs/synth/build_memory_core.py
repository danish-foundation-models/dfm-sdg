from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from sdg.commons import Artifact, store
from sdg.commons.utils import write_json
from sdg.packs.synth.memorization_text import lexical_tokens
from sdg.packs.synth.sources import load_sources


class ChunkSettings(TypedDict):
    size: int
    overlap: int


def build_memory_core(cfg: dict[str, Any], outputs_dir: Path) -> dict[str, Any]:
    docs = load_sources(cfg)
    cleaned_docs = clean_corpus(docs)
    sections = sectionize(cleaned_docs)
    chunk_settings = _chunk_settings(cfg)
    chunks = chunk_docs(sections, chunk_settings["size"], chunk_settings["overlap"])
    enriched_chunks = enrich_entities(chunks, cleaned_docs)
    index = build_index(enriched_chunks)
    source_table = _make_source_table(cleaned_docs)
    artifacts = write_memory_core(enriched_chunks, source_table, index, outputs_dir)

    return {
        "docs": cleaned_docs,
        "sections": sections,
        "chunks": enriched_chunks,
        "index": index,
        "source_table": source_table,
        "artifacts": artifacts,
    }


def clean_corpus(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_docs: list[dict[str, Any]] = []
    for doc in docs:
        paragraphs = [line.strip() for line in doc["text"].splitlines()]
        text = "\n".join(line for line in paragraphs if line)
        cleaned_docs.append({**doc, "text": text})
    return cleaned_docs


def sectionize(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for doc in docs:
        raw_sections = [part.strip() for part in doc["text"].split("\n\n") if part.strip()]
        if not raw_sections:
            raw_sections = [doc["text"]]

        for index, text in enumerate(raw_sections):
            sections.append(
                {
                    "id": f"{doc['id']}-section-{index:03d}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "text": text,
                    "source": doc["source"],
                    "url": doc["url"],
                    "license": doc["license"],
                    "meta": dict(doc["meta"]),
                }
            )
    return sections


def chunk_docs(sections: list[dict[str, Any]], size: int, overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = size - overlap

    for section in sections:
        words = section["text"].split()
        if not words:
            continue

        chunk_index = 0
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if not window:
                continue

            chunks.append(
                {
                    "id": f"{section['id']}-chunk-{chunk_index:03d}",
                    "section_id": section["id"],
                    "doc_id": section["doc_id"],
                    "source_id": section["doc_id"],
                    "title": section["title"],
                    "text": " ".join(window),
                    "source": section["source"],
                    "url": section["url"],
                    "license": section["license"],
                    "meta": {**dict(section["meta"]), "word_count": len(window)},
                }
            )
            chunk_index += 1

            if start + size >= len(words):
                break

    return chunks


def enrich_entities(chunks: list[dict[str, Any]], docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_lookup = {doc["id"]: doc for doc in docs}
    enriched: list[dict[str, Any]] = []
    for chunk in chunks:
        entity_candidates = [word for word in chunk["title"].split() if word[:1].isupper()]
        meta = dict(chunk["meta"])
        meta["entity_candidates"] = entity_candidates
        doc_meta = dict(doc_lookup[chunk["doc_id"]]["meta"])
        if "wikidata" in doc_meta:
            meta["wikidata"] = doc_meta["wikidata"]
        if "wikidata_id" in doc_meta:
            meta["wikidata_id"] = doc_meta["wikidata_id"]
        if "structured_wikipedia" in doc_meta:
            meta["structured_wikipedia"] = doc_meta["structured_wikipedia"]
        enriched.append({**chunk, "meta": meta})
    return enriched


def build_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "lexical-v2",
        "chunks": {
            chunk["id"]: {
                "title": chunk["title"],
                "tokens": lexical_tokens(chunk["text"], limit=128),
                "url": chunk["url"],
            }
            for chunk in chunks
        },
    }


def write_memory_core(
    chunks: list[dict[str, Any]],
    source_table: list[dict[str, Any]],
    index: dict[str, Any],
    outputs_dir: Path,
) -> dict[str, Artifact]:
    chunks_path = store.write_jsonl(chunks, outputs_dir / "memory_chunks.jsonl")
    source_table_path = store.write_jsonl(source_table, outputs_dir / "source_table.jsonl")
    index_path = write_json(index, outputs_dir / "retrieval_index.json")
    manifest_path = write_json(
        {
            "source_count": len(source_table),
            "chunk_count": len(chunks),
            "index_type": index["type"],
            "sources_with_wikidata": sum(1 for row in source_table if row.get("meta", {}).get("wikidata_id")),
            "sources_with_structured_wikipedia": sum(
                1 for row in source_table if row.get("meta", {}).get("structured_wikipedia")
            ),
        },
        outputs_dir / "memory_manifest.json",
    )

    return {
        "memory_chunks": Artifact(
            name="memory_chunks",
            path=str(chunks_path),
            kind="jsonl",
            meta={"rows": len(chunks)},
        ),
        "source_table": Artifact(
            name="source_table",
            path=str(source_table_path),
            kind="jsonl",
            meta={"rows": len(source_table)},
        ),
        "retrieval_index": Artifact(
            name="retrieval_index",
            path=str(index_path),
            kind="blob",
            meta={"type": index["type"]},
        ),
        "memory_manifest": Artifact(
            name="memory_manifest",
            path=str(manifest_path),
            kind="blob",
            meta={"source_count": len(source_table), "chunk_count": len(chunks)},
        ),
    }


def _make_source_table(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "url": doc["url"],
            "license": doc["license"],
            "meta": dict(doc["meta"]),
        }
        for doc in docs
    ]


def _chunk_settings(cfg: dict[str, Any]) -> ChunkSettings:
    memory_cfg = cfg.get("memory_core")
    if memory_cfg is None:
        return {"size": 80, "overlap": 20}

    assert isinstance(memory_cfg, dict), "memory_core config must be a mapping"
    size = memory_cfg.get("chunk_size", 80)
    overlap = memory_cfg.get("chunk_overlap", 20)
    assert isinstance(size, int) and size > 0, "memory_core chunk_size must be a positive integer"
    assert isinstance(overlap, int) and overlap >= 0, "memory_core chunk_overlap must be a non-negative integer"
    assert overlap < size, "memory_core chunk_overlap must be smaller than chunk_size"
    return {"size": size, "overlap": overlap}
