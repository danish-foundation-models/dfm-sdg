from __future__ import annotations

import asyncio

import httpx

from sdg.commons.model import (
    LLM,
    Embedder,
    Reranker,
    load_clients,
    load_endpoints,
    resolve_client,
)


def test_named_endpoints_support_multiple_models(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "SDG_ENDPOINT__OPENAI__BASE_URL=https://api.openai.com/v1",
                "SDG_ENDPOINT__OPENAI__API_KEY=test-openai-key",
                "SDG_ENDPOINT__OPENAI__DEFAULT_MODEL=gpt-4.1-mini",
                "SDG_ENDPOINT__LOCAL__BASE_URL=http://localhost:8000/v1",
                "SDG_ENDPOINT__LOCAL__API_KEY=dummy",
            ]
        )
    )

    endpoints = load_endpoints(env_path)
    assert endpoints["openai"].base_url == "https://api.openai.com/v1"
    assert endpoints["openai"].default_model == "gpt-4.1-mini"
    assert endpoints["openai"].max_concurrency == 4
    assert endpoints["local"].base_url == "http://localhost:8000/v1"

    models = load_clients(
        {
            "query_teacher": {"endpoint": "openai", "model": "gpt-4.1-mini"},
            "answer_teacher": {"endpoint": "openai", "model": "gpt-4.1"},
            "embedder": {"endpoint": "local", "type": "embedder", "model": "bge-small"},
            "reranker": {"endpoint": "local", "type": "reranker", "model": "bge-reranker-v2"},
        },
        env_path=env_path,
    )

    assert isinstance(models["query_teacher"], LLM)
    assert isinstance(models["answer_teacher"], LLM)
    assert isinstance(models["embedder"], Embedder)
    assert isinstance(models["reranker"], Reranker)

    assert models["query_teacher"].model == "gpt-4.1-mini"
    assert models["answer_teacher"].model == "gpt-4.1"
    assert models["embedder"].base_url == "http://localhost:8000/v1"


def test_legacy_openai_env_still_resolves(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-key",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_MODEL_NAME=gpt-4.1-mini",
            ]
        )
    )

    endpoints = load_endpoints(env_path)
    assert endpoints["openai"].default_model == "gpt-4.1-mini"

    client = resolve_client("openai", role="query_teacher", env_path=env_path)
    assert isinstance(client, LLM)
    assert client.model == "gpt-4.1-mini"


def test_sync_chat_retries_after_rate_limit() -> None:
    calls = {"count": 0}

    class RetryTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after-ms": "0"},
                    json={"error": "rate limited"},
                    request=request,
                )

            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        transport=RetryTransport(),
    )

    content = llm.chat([{"role": "user", "content": "hello"}], temperature=0.0)
    assert content == '{"target":"ok"}'
    assert calls["count"] == 2


def test_async_chat_retries_after_rate_limit() -> None:
    calls = {"count": 0}

    class RetryAsyncTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after": "0"},
                    json={"error": "rate limited"},
                    request=request,
                )

            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_retries=1,
        async_transport=RetryAsyncTransport(),
    )

    content = asyncio.run(llm.achat([{"role": "user", "content": "hello"}], temperature=0.0))
    assert content == '{"target":"ok"}'
    assert calls["count"] == 2


def test_async_chat_respects_shared_semaphore_limit() -> None:
    class ConcurrencyAsyncTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.current = 0
            self.max_seen = 0
            self.lock = asyncio.Lock()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            async with self.lock:
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)

            await asyncio.sleep(0.02)

            async with self.lock:
                self.current -= 1

            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"target":"ok"}'}}]},
                request=request,
            )

    transport = ConcurrencyAsyncTransport()
    llm = LLM(
        model="test-model",
        base_url="https://example.com/v1",
        max_concurrency=1,
        async_transport=transport,
    )

    async def run_requests() -> None:
        await asyncio.gather(
            llm.achat([{"role": "user", "content": "one"}], temperature=0.0),
            llm.achat([{"role": "user", "content": "two"}], temperature=0.0),
        )

    asyncio.run(run_requests())
    assert transport.max_seen == 1
