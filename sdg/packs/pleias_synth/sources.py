from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sdg.commons import store
from sdg.commons.utils import artifacts_root, iso_timestamp, read_json, write_json

USER_AGENT = "dfm-sdg/0.1 (synthetic-data-research)"
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def load_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    memory_cfg = cfg.get("memory_core", {})
    source = memory_cfg.get("source")

    if source == "wikipedia_vital_articles":
        return load_wikipedia_vital_articles(memory_cfg)

    inline_docs = memory_cfg.get("inline_docs", [])
    if inline_docs:
        return [_normalize_doc(doc, idx) for idx, doc in enumerate(inline_docs)]

    source_path = memory_cfg.get("source_path")
    if source_path:
        path = Path(source_path).expanduser().resolve()
        if path.suffix == ".jsonl":
            return [_normalize_doc(doc, idx) for idx, doc in enumerate(store.read_jsonl(path))]
        if path.suffix == ".json":
            docs = read_json(path)
            return [_normalize_doc(doc, idx) for idx, doc in enumerate(docs)]
        return [
            _normalize_doc(
                {
                    "id": path.stem,
                    "title": path.stem.replace("_", " ").title(),
                    "text": path.read_text(),
                    "source": str(path),
                    "url": str(path),
                    "license": memory_cfg.get("default_license", "unknown"),
                    "meta": {"dataset": "local_file"},
                },
                0,
            )
        ]

    raise ValueError("memory_core requires a supported source configuration")


def load_wikipedia_vital_articles(memory_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    language = memory_cfg.get("language", "en")
    level = int(memory_cfg.get("vital_level", 4))
    max_articles = memory_cfg.get("max_articles")
    refresh = bool(memory_cfg.get("refresh", False))
    batch_size = int(memory_cfg.get("fetch_batch_size", 20))
    request_pause = float(memory_cfg.get("request_pause_seconds", 0.0))
    expand_with = set(memory_cfg.get("expand_with", []))
    cache_dir = _cache_dir(language, level)

    title_entries = load_vital_title_entries(
        language=language,
        level=level,
        cache_dir=cache_dir,
        refresh=refresh,
        request_pause=request_pause,
    )
    if max_articles:
        title_entries = title_entries[: int(max_articles)]

    docs = load_wikipedia_docs(
        language=language,
        level=level,
        title_entries=title_entries,
        cache_dir=cache_dir,
        refresh=refresh,
        batch_size=batch_size,
        request_pause=request_pause,
    )

    if "structured_wikipedia" in expand_with:
        docs = attach_structured_wikipedia(
            docs,
            language=language,
            cache_dir=cache_dir,
            refresh=refresh,
            batch_size=batch_size,
            request_pause=request_pause,
        )

    if "wikidata" in expand_with:
        docs = attach_wikidata(
            docs,
            cache_dir=cache_dir,
            refresh=refresh,
            batch_size=min(batch_size, 25),
            request_pause=request_pause,
        )

    return docs


def attach_structured_wikipedia(
    docs: list[dict[str, Any]],
    *,
    language: str,
    cache_dir: Path,
    refresh: bool,
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / "structured_wikipedia.json"
    existing_by_title: dict[str, Any] = {}

    if cache_path.exists() and not refresh:
        existing_by_title = read_json(cache_path)

    missing_titles = sorted({doc["title"] for doc in docs if doc["title"] not in existing_by_title})
    if missing_titles:
        existing_by_title.update(
            fetch_structured_wikipedia(
                language=language,
                titles=missing_titles,
                batch_size=batch_size,
                request_pause=request_pause,
            )
        )
        write_json(existing_by_title, cache_path)

    enriched_docs: list[dict[str, Any]] = []
    for doc in docs:
        meta = dict(doc.get("meta") or {})
        meta["structured_wikipedia"] = existing_by_title.get(doc["title"])
        enriched_docs.append({**doc, "meta": meta})

    return enriched_docs


def load_vital_title_entries(
    *,
    language: str,
    level: int,
    cache_dir: Path,
    refresh: bool,
    request_pause: float,
) -> list[dict[str, str]]:
    cache_path = cache_dir / "titles.json"
    if cache_path.exists() and not refresh:
        return read_json(cache_path)["titles"]

    root_page = f"Wikipedia:Vital articles/Level/{level}"
    titles = discover_vital_title_entries(language, root_page, request_pause=request_pause)
    write_json(
        {
            "language": language,
            "vital_level": level,
            "retrieved_at": iso_timestamp(),
            "titles": titles,
        },
        cache_path,
    )
    return titles


def discover_vital_title_entries(language: str, root_page: str, *, request_pause: float) -> list[dict[str, str]]:
    pages_to_visit = [root_page]
    visited_pages: set[str] = set()
    seen_titles: set[str] = set()
    title_entries: list[dict[str, str]] = []

    while pages_to_visit:
        page = pages_to_visit.pop(0)
        if page in visited_pages:
            continue
        visited_pages.add(page)

        parsed = wikipedia_api_json(
            language,
            {
                "action": "parse",
                "format": "json",
                "formatversion": "2",
                "page": page,
                "prop": "links",
            },
        )["parse"]

        for link in parsed.get("links", []):
            title = link["title"]
            if link.get("ns") == 0 and link.get("exists") and title not in seen_titles:
                seen_titles.add(title)
                title_entries.append({"title": title, "listing_page": page})

            if (
                link.get("ns") == 4
                and link.get("exists")
                and title.startswith(f"{root_page}/")
                and title not in visited_pages
            ):
                pages_to_visit.append(title)

        _maybe_sleep(request_pause)

    return title_entries


def load_wikipedia_docs(
    *,
    language: str,
    level: int,
    title_entries: list[dict[str, str]],
    cache_dir: Path,
    refresh: bool,
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / "pages.jsonl"
    existing_by_title: dict[str, dict[str, Any]] = {}

    if cache_path.exists() and not refresh:
        existing_by_title = {doc["title"]: doc for doc in store.read_jsonl(cache_path)}

    missing_titles = [entry["title"] for entry in title_entries if entry["title"] not in existing_by_title]
    if missing_titles:
        new_docs = fetch_wikipedia_docs(
            language=language,
            level=level,
            titles=missing_titles,
            batch_size=batch_size,
            request_pause=request_pause,
        )
        existing_by_title.update({doc["title"]: doc for doc in new_docs})

    docs: list[dict[str, Any]] = []
    missing_after_fetch: list[str] = []

    for entry in title_entries:
        title = entry["title"]
        doc = existing_by_title.get(title)
        if doc is None:
            missing_after_fetch.append(title)
            continue
        meta = dict(doc.get("meta") or {})
        meta["dataset"] = "wikipedia_vital_articles"
        meta["language"] = language
        meta["vital_level"] = level
        meta["listing_page"] = entry["listing_page"]
        docs.append({**doc, "meta": meta})

    if missing_after_fetch:
        raise ValueError(f"Missing fetched documents for {len(missing_after_fetch)} titles")

    store.write_jsonl(docs, cache_path)
    return docs


def fetch_wikipedia_docs(
    *,
    language: str,
    level: int,
    titles: list[str],
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for batch in _batched(titles, batch_size):
        response = wikipedia_api_json(
            language,
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "extracts|pageprops|info",
                "inprop": "url",
                "titles": "|".join(batch),
                "redirects": "1",
                "explaintext": "1",
                "exsectionformat": "plain",
                "exlimit": "max",
            },
        )

        pages = {
            page["title"]: page
            for page in response["query"]["pages"]
            if "missing" not in page
        }

        for title in batch:
            page = pages.get(title)
            if page is None:
                continue

            pageprops = page.get("pageprops", {})
            meta = {
                "dataset": "wikipedia_vital_articles",
                "language": language,
                "vital_level": level,
                "pageid": page.get("pageid"),
                "lastrevid": page.get("lastrevid"),
                "length": page.get("length"),
                "touched": page.get("touched"),
                "retrieved_at": iso_timestamp(),
                "wikidata_id": pageprops.get("wikibase_item"),
                "wikibase_shortdesc": pageprops.get("wikibase-shortdesc"),
                "license_url": WIKIPEDIA_LICENSE_URL,
            }
            docs.append(
                {
                    "id": _slug(page["title"]),
                    "title": page["title"],
                    "text": page.get("extract", ""),
                    "source": page.get("canonicalurl") or page.get("fullurl"),
                    "url": page.get("canonicalurl") or page.get("fullurl"),
                    "license": WIKIPEDIA_LICENSE,
                    "meta": meta,
                }
            )

        _maybe_sleep(request_pause)

    return docs


def attach_wikidata(
    docs: list[dict[str, Any]],
    *,
    cache_dir: Path,
    refresh: bool,
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / "wikidata.json"
    existing_entities: dict[str, Any] = {}

    if cache_path.exists() and not refresh:
        existing_entities = read_json(cache_path)

    missing_ids = sorted(
        {
            doc["meta"]["wikidata_id"]
            for doc in docs
            if doc.get("meta", {}).get("wikidata_id") and doc["meta"]["wikidata_id"] not in existing_entities
        }
    )

    if missing_ids:
        existing_entities.update(
            fetch_wikidata_entities(
                entity_ids=missing_ids,
                batch_size=batch_size,
                request_pause=request_pause,
            )
        )
        write_json(existing_entities, cache_path)

    enriched_docs: list[dict[str, Any]] = []
    for doc in docs:
        meta = dict(doc.get("meta") or {})
        wikidata_id = meta.get("wikidata_id")
        if wikidata_id:
            meta["wikidata"] = existing_entities.get(wikidata_id)
        enriched_docs.append({**doc, "meta": meta})

    return enriched_docs


def fetch_structured_wikipedia(
    *,
    language: str,
    titles: list[str],
    batch_size: int,
    request_pause: float,
) -> dict[str, Any]:
    structured_by_title: dict[str, Any] = {}

    for batch in _batched(titles, batch_size):
        batch_structured: dict[str, dict[str, Any]] = {}
        continue_args: dict[str, Any] = {}

        while True:
            response = wikipedia_api_json(
                language,
                {
                    "action": "query",
                    "format": "json",
                    "formatversion": "2",
                    "prop": "categories|links|templates",
                    "titles": "|".join(batch),
                    "redirects": "1",
                    "cllimit": "max",
                    "clshow": "!hidden",
                    "pllimit": "max",
                    "tllimit": "max",
                    **continue_args,
                },
            )

            for page in response["query"]["pages"]:
                if "missing" in page:
                    continue

                entry = batch_structured.setdefault(
                    page["title"],
                    {
                        "categories": [],
                        "outgoing_links": [],
                        "templates": [],
                        "infobox_templates": [],
                    },
                )

                categories = page.get("categories", [])
                category_titles = [row["title"].removeprefix("Category:") for row in categories]
                entry["categories"] = _merge_unique(entry["categories"], category_titles)

                links = page.get("links", [])
                article_links = [
                    row["title"]
                    for row in links
                    if row.get("ns") == 0 and "missing" not in row
                ]
                entry["outgoing_links"] = _merge_unique(entry["outgoing_links"], article_links)

                templates = page.get("templates", [])
                template_titles = [row["title"] for row in templates if "missing" not in row]
                entry["templates"] = _merge_unique(entry["templates"], template_titles)
                infobox_templates = [
                    title
                    for title in template_titles
                    if title.startswith("Template:Infobox")
                ]
                entry["infobox_templates"] = _merge_unique(entry["infobox_templates"], infobox_templates)

            if "continue" not in response:
                break

            continue_args = response["continue"]
            _maybe_sleep(request_pause)

        structured_by_title.update(batch_structured)
        _maybe_sleep(request_pause)

    return structured_by_title


def fetch_wikidata_entities(
    *,
    entity_ids: list[str],
    batch_size: int,
    request_pause: float,
) -> dict[str, Any]:
    entities: dict[str, Any] = {}

    for batch in _batched(entity_ids, batch_size):
        response = wikidata_api_json(
            {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(batch),
                "props": "labels|descriptions|aliases|sitelinks",
                "languages": "en",
            }
        )

        for entity_id in batch:
            raw = response["entities"].get(entity_id)
            if raw is None or "missing" in raw:
                continue

            entities[entity_id] = {
                "id": entity_id,
                "label": _lang_value(raw.get("labels", {})),
                "description": _lang_value(raw.get("descriptions", {})),
                "aliases": [alias["value"] for alias in raw.get("aliases", {}).get("en", [])],
                "sitelinks": {
                    name: value.get("title")
                    for name, value in raw.get("sitelinks", {}).items()
                    if name.endswith("wiki")
                },
            }

        _maybe_sleep(request_pause)

    return entities


def wikipedia_api_json(language: str, params: dict[str, Any]) -> dict[str, Any]:
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    return _api_json(base_url, params)


def wikidata_api_json(params: dict[str, Any]) -> dict[str, Any]:
    return _api_json("https://www.wikidata.org/w/api.php", params)


def _api_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    request_obj = Request(f"{base_url}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request_obj) as response:
        return json.loads(response.read().decode("utf-8"))


def _cache_dir(language: str, level: int) -> Path:
    target = artifacts_root() / "external" / "wikipedia" / language / f"vital_level_{level}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _normalize_doc(doc: dict[str, Any], index: int) -> dict[str, Any]:
    doc_id = doc.get("id") or f"doc-{index:05d}"
    meta = dict(doc.get("meta") or {})
    if "dataset" not in meta:
        meta["dataset"] = "inline"

    return {
        "id": doc_id,
        "title": doc.get("title") or doc_id,
        "text": doc.get("text", ""),
        "source": doc.get("source", "inline"),
        "url": doc.get("url", doc.get("source", "inline")),
        "license": doc.get("license", "unknown"),
        "meta": meta,
    }


def _lang_value(values: dict[str, Any], language: str = "en") -> str | None:
    if language in values:
        return values[language]["value"]
    return None


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _maybe_sleep(request_pause: float) -> None:
    if request_pause > 0:
        time.sleep(request_pause)


def _slug(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "_")


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(existing)
    for value in incoming:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged
