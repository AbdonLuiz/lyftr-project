import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def llm_calls() -> list[dict]:
    return []


@pytest.fixture
def llm_behavior() -> dict:
    return {"mode": "ok", "delay": 0.05}


@pytest.fixture
def client(tmp_path, monkeypatch, llm_calls, llm_behavior):
    install_fake_completion(monkeypatch, llm_calls, llm_behavior)
    settings = build_settings(tmp_path)
    application = create_app(settings)
    with TestClient(application) as test_client:
        yield test_client


def build_settings(tmp_path, *, simulation: bool = True, keys: str = "sk-test-1,sk-test-2") -> Settings:
    database = tmp_path / "app.db"
    return Settings(
        llm_api_keys=keys,
        llm_model="openai/gpt-4o-mini",
        llm_models="",
        database_url="sqlite:///" + database.as_posix(),
        allowed_origins="http://localhost:3000",
        enable_failure_simulation=simulation,
        default_system_prompt="You are a helpful assistant.",
        llm_timeout_seconds=5,
        cache_ttl_seconds=3600,
    )


def install_fake_completion(monkeypatch, llm_calls: list[dict], llm_behavior: dict) -> None:
    from litellm.exceptions import InternalServerError, RateLimitError

    async def fake_acompletion(**kwargs):
        llm_calls.append(kwargs)
        await asyncio.sleep(llm_behavior.get("delay", 0.05))
        api_key = kwargs.get("api_key", "")
        mode = llm_behavior.get("mode", "ok")
        if mode == "rate-limit-first" and api_key.endswith("1"):
            raise RateLimitError(message="quota exceeded", llm_provider="openai", model=kwargs.get("model", "m"))
        if mode == "server-error-first" and api_key.endswith("1"):
            raise InternalServerError(message="upstream 500", llm_provider="openai", model=kwargs.get("model", "m"))
        if mode == "all-quota":
            raise RuntimeError("quota exceeded for this project")
        messages = kwargs["messages"]
        reply = f"reply:{messages[-1]['content']}"
        prompt_tokens = sum(len(item["content"]) for item in messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=len(reply)),
            model=kwargs["model"],
            _hidden_params={"cache_hit": False},
        )

    monkeypatch.setattr("app.llm.acompletion", fake_acompletion)
