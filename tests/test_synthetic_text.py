import json

import pytest
from pydantic import ValidationError

pytest.importorskip("httpx")

import httpx

from lfm_audio_rl.data import load_dataset
from lfm_audio_rl.synthetic.config import EngineConfig, TextConfig
from lfm_audio_rl.synthetic.text import (
    OpenRouter,
    Scene,
    example,
    generate_text,
    parse_response,
    payload,
)


def completion(items, finish="stop"):
    return {
        "id": "test-response",
        "model": "test-returned-model",
        "provider": "test-provider",
        "usage": {"total_tokens": 100},
        "choices": [
            {"finish_reason": finish, "message": {"content": json.dumps({"items": items})}}
        ],
    }


def test_openrouter_cache_retry_and_dataset_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    calls = []
    config = TextConfig(version="v1", model="test/model", count=2, batch_size=2)
    scenes = [
        {"item": "cup", "color": "red", "place": "kitchen"},
        {"item": "book", "color": "blue", "place": "library"},
    ]

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["authorization"] == "Bearer test-secret"
        assert body["provider"]["require_parameters"]
        assert body["response_format"]["json_schema"]["strict"]
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"code": 429}})
        return httpx.Response(200, json=completion(scenes))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = generate_text(config, tmp_path, client, sleep=lambda _: None)
        assert generate_text(config, tmp_path, client) == result
        # Resume an interrupted publication without another API call.
        (tmp_path / "dataset.json").unlink()
        assert generate_text(config, tmp_path, client) == result
    assert len(calls) == 2
    rows, _ = load_dataset(tmp_path)
    assert rows[0].answer == "red"
    assert rows[0].provenance["reward_protocol"] == "exact_answer"
    assert rows[0].provenance["returned_model"] == "test-returned-model"
    assert all("test-secret" not in p.read_text() for p in tmp_path.rglob("*.json*"))
    with pytest.raises(ValueError, match="different configuration"):
        generate_text(config.model_copy(update={"seed": 3}), tmp_path)


@pytest.mark.parametrize(
    "status,body,expected_calls",
    [(401, {"error": {"code": 401}}, 1), (200, {"error": {"code": 503}}, 2)],
)
def test_terminal_errors_and_request_budget(tmp_path, monkeypatch, status, body, expected_calls):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body)

    config = TextConfig(version="v1", model="test/model", max_requests=2, max_attempts=3)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = OpenRouter(config, tmp_path, client, sleep=lambda _: None)
        with pytest.raises(RuntimeError):
            api.complete(payload(config, 0, 1), 0)
    assert len(calls) == expected_calls
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == expected_calls


@pytest.mark.parametrize("finish", ["length", "error", "content_filter"])
def test_incomplete_outputs_are_rejected(finish):
    with pytest.raises(ValueError, match="Incomplete"):
        parse_response(completion([], finish), "grounded_qa")


def test_bad_schema_and_cache_tampering(tmp_path, monkeypatch):
    with pytest.raises(ValidationError):
        parse_response(
            completion([{"prompt": "Hello", "answer": "Hi", "unexpected": True}]), "dialogue"
        )
    with pytest.raises(ValidationError):
        Scene(item="<audio>", color="red", place="kitchen")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = TextConfig(version="v1", model="test/model", count=1)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json=completion([{"item": "cup", "color": "red", "place": "kitchen"}])
            )
        )
    ) as client:
        api = OpenRouter(config, tmp_path, client)
        request = payload(config, 0, 1)
        api.complete(request, 0)
        cache = tmp_path / "responses/000000.json"
        value = json.loads(cache.read_text())
        value["response"]["model"] = "tampered"
        cache.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="hash"):
            api.complete(request, 0)


def test_scene_split_ignores_version_color_and_provider():
    config = TextConfig(version="v1", model="test/model")
    a = example(Scene(item="cup", color="red", place="kitchen"), config, 0, {})
    b = example(
        Scene(item="cup", color="blue", place="kitchen"),
        config.model_copy(update={"version": "v2", "seed": 6}),
        1,
        {"model": "other"},
    )
    assert (a.semantic_group, a.split) == (b.semantic_group, b.split)
    assert a.id != b.id


def test_engine_controls_cannot_be_silently_ignored():
    with pytest.raises(ValidationError, match="does not apply"):
        EngineConfig(
            name="echo", engine="echo-tts", python="python", source_dir=".", max_new_tokens=300
        )
    with pytest.raises(ValidationError, match="requires"):
        EngineConfig(
            name="qwen",
            engine="qwen3-tts",
            python="python",
            source_dir=".",
            qwen_mode="voice_clone",
        )
