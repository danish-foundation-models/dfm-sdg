from __future__ import annotations

from typing import TypedDict

Record = dict[str, object]
Meta = dict[str, object]


class Persona(TypedDict):
    persona_id: str
    source: str
    name: str
    intent: str
    knowledge_level: str
    tone: str
    question_style: str
    answer_granularity: str
    constraints: list[str]
    tags: list[str]
    preferred_angles: list[str]
    meta: Meta


class QueryProfile(TypedDict):
    profile_id: str
    source: str
    name: str
    weight: int
    channel: str
    fluency: str
    register: str
    urgency: str
    query_shape: str
    noise_level: str
    instructions: str
    exemplars: list[str]
    tags: list[str]
    meta: Meta


class AssistantStyle(TypedDict):
    style_id: str
    source: str
    name: str
    tone: str
    detail_level: str
    structure: str
    voice: str
    formatting_style: str
    punctuation_style: str
    instructions: str
    exemplars: list[str]
    tags: list[str]
    meta: Meta


class QueryPlan(TypedDict):
    bundle: Record
    persona: Persona
    query_angle: str
    query_profile: QueryProfile
    assistant_style: AssistantStyle
