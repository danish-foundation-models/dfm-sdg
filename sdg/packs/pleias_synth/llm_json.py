from __future__ import annotations

import json
import re
from typing import Any

from sdg.commons.model import LLM


def parse_json_response(text: Any) -> dict[str, Any]:
    payload = _extract_json_object(str(text).strip())
    parsed = json.loads(payload)
    assert isinstance(parsed, dict), "LLM response must decode to a JSON object"
    return parsed


def chat_json(llm: LLM, messages: list[dict[str, str]], *, temperature: float) -> dict[str, Any]:
    response = llm.chat(messages, temperature=temperature)
    try:
        return parse_json_response(response)
    except (AssertionError, json.JSONDecodeError):
        repaired = llm.chat(_repair_json_messages(response), temperature=0.0)
        return parse_json_response(repaired)


async def achat_json(llm: LLM, messages: list[dict[str, str]], *, temperature: float) -> dict[str, Any]:
    response = await llm.achat(messages, temperature=temperature)
    try:
        return parse_json_response(response)
    except (AssertionError, json.JSONDecodeError):
        repaired = await llm.achat(_repair_json_messages(response), temperature=0.0)
        return parse_json_response(repaired)


def _repair_json_messages(response: Any) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair malformed JSON. "
                "Return only valid JSON with the same top-level object structure. "
                "Do not add commentary."
            ),
        },
        {
            "role": "user",
            "content": str(response),
        },
    ]


def _extract_json_object(content: str) -> str:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    if content.startswith("{") and content.endswith("}"):
        return content

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match is None:
        return content
    return match.group(0)
