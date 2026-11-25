import json

import pytest
from fastapi.testclient import TestClient

from app.db import resolve_sqlite_path
from app.main import create_app
from app.simulation import parse_simulated_failures
from tests.conftest import build_settings, install_fake_completion


def test_sqlite_url_paths():
    assert resolve_sqlite_path("sqlite:///./data/app.db") == "./data/app.db"
    assert resolve_sqlite_path("sqlite:////data/app.db") == "/data/app.db"
    assert resolve_sqlite_path("sqlite:///C:/data/app.db") == "C:/data/app.db"
    with pytest.raises(ValueError):
        resolve_sqlite_path("postgres://localhost/db")


def test_simulation_header_parser():
    assert parse_simulated_failures("key_1=429", enabled=False) == {}
    assert parse_simulated_failures(None, enabled=True) == {}
    assert parse_simulated_failures("key_1=429", enabled=True) == {"key_1": 429}
    assert parse_simulated_failures("key_1=429,key_2=500", enabled=True) == {"key_1": 429, "key_2": 500}
    assert parse_simulated_failures("key_1=429, key_2=502", enabled=True) == {"key_1": 429, "key_2": 502}
    assert parse_simulated_failures("*=429", enabled=True) == {"*": 429}
    assert parse_simulated_failures("key_1=200,key_2=503", enabled=True) == {"key_2": 503}
    assert parse_simulated_failures("not-a-pair", enabled=True) == {}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_validation_and_unknown_session(client):
    missing = client.post("/chat", json={"session_id": "s-1"})
    assert missing.status_code == 422
    assert "detail" in missing.json()

    empty = client.post("/chat", json={"session_id": "s-1", "message": "   "})
    assert empty.status_code == 422

    too_long = client.post("/chat", json={"session_id": "s-1", "message": "a" * 4001})
    assert too_long.status_code == 422

    bad_session = client.post("/chat", json={"session_id": "", "message": "hi"})
    assert bad_session.status_code == 422

    long_session = client.post("/chat", json={"session_id": "s" * 129, "message": "hi"})
    assert long_session.status_code == 422

    malformed = client.post("/chat", content=b"{", headers={"content-type": "application/json"})
    assert malformed.status_code == 422
    assert "Traceback" not in malformed.text

    unknown = client.get("/sessions/missing/messages")
    assert unknown.status_code == 404
    assert unknown.json() == {"detail": "session not found"}

    still_missing = client.get("/sessions/s-1/messages")
    assert still_missing.status_code == 404


def test_live_reply_metadata_and_trimmed_message(client, llm_calls):
    response = client.post("/chat", json={"session_id": "s-trim", "message": "  Paris?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s-trim"
    assert body["reply"] == "reply:Paris?"
    meta = body["metadata"]
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["model"] == "openai/gpt-4o-mini"
    assert isinstance(meta["latency_ms"], int)
    assert meta["input_tokens"] > 0
    assert meta["output_tokens"] > 0
    assert meta["key_id"] == "key_1"
    assert meta["fallback_attempts"] == 0
    assert llm_calls[0]["api_key"] == "sk-test-1"
    assert llm_calls[0]["num_retries"] == 0
    assert llm_calls[0]["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert llm_calls[0]["messages"][-1] == {"role": "user", "content": "Paris?"}

    history = client.get("/sessions/s-trim/messages")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert messages[0] == {"role": "user", "content": "Paris?"}
    assert "metadata" not in messages[0]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "reply:Paris?"
    assert messages[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert messages[1]["metadata"]["input_tokens"] == meta["input_tokens"]
    assert messages[1]["metadata"]["output_tokens"] == meta["output_tokens"]


def test_cache_hit_is_identical_and_faster(client, llm_calls):
    first = client.post("/chat", json={"session_id": "s-a", "message": "What is the capital of France?"})
    second = client.post("/chat", json={"session_id": "s-b", "message": "What is the capital of France?"})
    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert second_body["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second_body["reply"] == first_body["reply"]
    assert second_body["metadata"]["input_tokens"] == first_body["metadata"]["input_tokens"]
    assert second_body["metadata"]["output_tokens"] == first_body["metadata"]["output_tokens"]
    assert second_body["metadata"]["key_id"] is None
    assert second_body["metadata"]["fallback_attempts"] == 0
    assert second_body["metadata"]["latency_ms"] < first_body["metadata"]["latency_ms"]
    assert len(llm_calls) == 1

    different = client.post("/chat", json={"session_id": "s-c", "message": "What is the capital of Spain?"})
    assert different.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert different.json()["reply"] != first_body["reply"]
    assert len(llm_calls) == 2


def test_system_prompt_changes_the_cache_key_and_is_remembered(client, llm_calls):
    first = client.post(
        "/chat",
        json={"session_id": "s-prompt", "message": "Hello", "system_prompt": "Be brief."},
    )
    assert first.status_code == 200
    assert llm_calls[0]["messages"][0]["content"] == "Be brief."

    follow = client.post("/chat", json={"session_id": "s-prompt", "message": "Again"})
    assert follow.status_code == 200
    assert llm_calls[1]["messages"][0]["content"] == "Be brief."
    assert llm_calls[1]["messages"][1] == {"role": "user", "content": "Hello"}
    assert llm_calls[1]["messages"][2]["role"] == "assistant"
    assert llm_calls[1]["messages"][3] == {"role": "user", "content": "Again"}
    assert follow.json()["metadata"]["input_tokens"] > first.json()["metadata"]["input_tokens"]

    other = client.post("/chat", json={"session_id": "s-other", "message": "Hello"})
    assert other.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert llm_calls[-1]["messages"][0]["content"] == "You are a helpful assistant."

    cached = client.post("/chat", json={"session_id": "s-other-2", "message": "Hello"})
    assert cached.json()["metadata"]["resolution_source"] == "LLM_CACHE"


def test_sessions_stay_separate(client, llm_calls):
    client.post("/chat", json={"session_id": "s-one", "message": "alpha"})
    client.post("/chat", json={"session_id": "s-two", "message": "beta"})
    client.post("/chat", json={"session_id": "s-two", "message": "beta-follow"})
    sent = llm_calls[-1]["messages"]
    assert all(item["content"] != "alpha" for item in sent)
    history_one = client.get("/sessions/s-one/messages").json()["messages"]
    history_two = client.get("/sessions/s-two/messages").json()["messages"]
    assert [item["content"] for item in history_one] == ["alpha", "reply:alpha"]
    assert [item["content"] for item in history_two] == ["beta", "reply:beta", "beta-follow", "reply:beta-follow"]


def test_simulated_failover_uses_the_next_key(client, llm_calls):
    response = client.post(
        "/chat",
        json={"session_id": "s-fail", "message": "Need a live answer"},
        headers={"X-Simulate-Failure": "key_1=429"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["key_id"] == "key_2"
    assert body["metadata"]["fallback_attempts"] == 1
    assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body["reply"] == "reply:Need a live answer"
    assert "Traceback" not in response.text
    assert [call["api_key"] for call in llm_calls] == ["sk-test-2"]

    server_error = client.post(
        "/chat",
        json={"session_id": "s-fail-500", "message": "Another live answer"},
        headers={"X-Simulate-Failure": "key_1=500"},
    )
    assert server_error.status_code == 200
    assert server_error.json()["metadata"]["key_id"] == "key_2"
    assert server_error.json()["metadata"]["fallback_attempts"] == 1


def test_simulated_total_failure_does_not_store_history(client, llm_calls):
    response = client.post(
        "/chat",
        json={"session_id": "s-down", "message": "This should not be stored"},
        headers={"X-Simulate-Failure": "*=429"},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert response.headers["content-type"].startswith("application/json")
    assert llm_calls == []
    missing = client.get("/sessions/s-down/messages")
    assert missing.status_code == 404

    both = client.post(
        "/chat",
        json={"session_id": "s-down-2", "message": "Still nothing"},
        headers={"X-Simulate-Failure": "key_1=429,key_2=503"},
    )
    assert both.status_code == 503
    assert client.get("/sessions/s-down-2/messages").status_code == 404


def test_real_provider_errors_fail_over(client, llm_behavior, llm_calls):
    llm_behavior["mode"] = "rate-limit-first"
    quota = client.post("/chat", json={"session_id": "s-quota", "message": "quota path"})
    assert quota.status_code == 200
    assert quota.json()["metadata"]["key_id"] == "key_2"
    assert quota.json()["metadata"]["fallback_attempts"] == 1
    assert quota.json()["reply"] == "reply:quota path"

    llm_behavior["mode"] = "server-error-first"
    failed = client.post("/chat", json={"session_id": "s-500", "message": "server path"})
    assert failed.status_code == 200
    assert failed.json()["metadata"]["key_id"] == "key_2"

    llm_behavior["mode"] = "all-quota"
    down = client.post("/chat", json={"session_id": "s-all", "message": "everyone is out"})
    assert down.status_code == 503
    assert down.json() == {"detail": "all providers unavailable"}
    assert "Traceback" not in down.text
    assert client.get("/sessions/s-all/messages").status_code == 404


def test_simulation_header_is_ignored_when_disabled(tmp_path, monkeypatch, llm_calls, llm_behavior):
    install_fake_completion(monkeypatch, llm_calls, llm_behavior)
    application = create_app(build_settings(tmp_path, simulation=False))
    with TestClient(application) as disabled:
        response = disabled.post(
            "/chat",
            json={"session_id": "s-ignore", "message": "Use the first key"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_1"
    assert response.json()["metadata"]["fallback_attempts"] == 0
    assert llm_calls[0]["api_key"] == "sk-test-1"


def test_cors_allows_the_frontend_origin(client):
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_history_metadata_is_json_not_a_string(client):
    client.post("/chat", json={"session_id": "s-json", "message": "store me"})
    payload = client.get("/sessions/s-json/messages").json()
    metadata = payload["messages"][1]["metadata"]
    assert isinstance(metadata, dict)
    assert isinstance(json.dumps(metadata), str)
