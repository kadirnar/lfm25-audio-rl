"""Bounded OpenRouter requests, cached responses, and validated text datasets."""

import json
import os
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, field_validator

from ..config import StrictModel
from ..data import Example, digest, load_dataset, normalize
from .config import TextConfig
from .storage import atomic_json, publish, run_directory


class Scene(StrictModel):
    item: str = Field(min_length=1, max_length=40)
    color: Literal[
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "white",
        "black",
        "brown",
        "gray",
    ]
    place: str = Field(min_length=1, max_length=50)

    @field_validator("item", "place")
    @classmethod
    def simple_phrase(cls, value):
        value = " ".join(value.lower().split())
        if not re.fullmatch(r"[a-z]+(?: [a-z]+){0,5}", value):
            raise ValueError("Use a short English noun phrase containing letters and spaces")
        if set(value.split()) & {
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "purple",
            "pink",
            "white",
            "black",
            "brown",
            "gray",
            "grey",
        }:
            raise ValueError("Keep color words in the color field, not item or place")
        return value


class Dialogue(StrictModel):
    prompt: str = Field(min_length=5, max_length=300)
    answer: str = Field(min_length=1, max_length=300)

    @field_validator("prompt", "answer")
    @classmethod
    def plain_speech(cls, value):
        value = " ".join(value.split())
        if not normalize(value) or len(value.split()) > 40 or any(c in value for c in "<>[]{}"):
            raise ValueError("Use plain spoken text without tags or markup")
        return value


class SceneBatch(StrictModel):
    items: list[Scene] = Field(min_length=1, max_length=50)


class DialogueBatch(StrictModel):
    items: list[Dialogue] = Field(min_length=1, max_length=50)


def payload(config: TextConfig, batch: int, count: int):
    cls = SceneBatch if config.mode == "grounded_qa" else DialogueBatch
    instruction = (
        "Invent fictional scenes. Each has an item (a singular noun phrase without 'the'), "
        "one color, and a place (a noun phrase without 'the'). Avoid color words in item/place."
        if config.mode == "grounded_qa"
        else "Write varied, short user questions and helpful spoken answers. These are unverified "
        "synthetic SFT examples. Avoid specialist advice and claims about current events."
    )
    provider = {"require_parameters": True}
    if config.providers:
        provider.update(order=config.providers, allow_fallbacks=False)
    return {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "Create English synthetic speech training data. "
                + instruction
                + " Return only the requested JSON. Use plain words, no markup or speaker tags.",
            },
            {
                "role": "user",
                "content": f"Generate {count} distinct items about {config.topics[batch % len(config.topics)]}. Batch {batch}, variation seed {config.seed}. Keep every spoken turn under forty words.",
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "synthetic_batch",
                "strict": True,
                "schema": cls.model_json_schema(),
            },
        },
        "provider": provider,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": (config.seed + batch) % (2**31),
        "stream": False,
    }


def retry_delay(header: str | None, attempt: int):
    if header:
        try:
            return min(60.0, max(0.0, float(header)))
        except ValueError:
            try:
                return min(
                    60.0,
                    max(0.0, (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds()),
                )
            except (TypeError, ValueError):
                pass
    return min(30, 2**attempt)


class OpenRouter:
    def __init__(self, config: TextConfig, output: Path, client=None, sleep=time.sleep):
        self.config, self.output, self.sleep = config, output, sleep
        self.client = client

    def complete(self, request: dict, batch: int):
        cache = self.output / "responses" / f"{batch:06d}.json"
        cache.parent.mkdir(exist_ok=True)
        request_hash = digest(request)
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved["request_hash"] != request_hash or saved["response_hash"] != digest(
                saved["response"]
            ):
                raise ValueError("OpenRouter cache hash mismatch")
            return saved["response"]
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY before generating text")
        ledger = self.output / "requests.jsonl"
        count = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        for attempt in range(self.config.max_attempts):
            if count >= self.config.max_requests:
                raise RuntimeError(
                    "OpenRouter request limit reached; use a new run/config for a larger budget"
                )
            # Reserve before sending; crashes and unsuccessful calls count against the limit.
            with ledger.open("a") as f:
                f.write(
                    json.dumps({"number": count, "batch": batch, "request_hash": request_hash})
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
            count += 1
            try:
                if self.client is None:
                    with httpx.Client(timeout=self.config.timeout_seconds) as client:
                        response = client.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            json=request,
                            headers={
                                "Authorization": f"Bearer {key}",
                                "X-OpenRouter-Title": "lfm25-audio-rl",
                            },
                        )
                else:
                    response = self.client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=request,
                        headers={"Authorization": f"Bearer {key}"},
                    )
            except httpx.TransportError:
                if attempt + 1 == self.config.max_attempts:
                    raise RuntimeError(
                        "OpenRouter transport failure; reserved requests may have been billed"
                    ) from None
                self.sleep(retry_delay(None, attempt))
                continue
            try:
                body = response.json()
            except ValueError:
                body = {}
            status = response.status_code
            error = body.get("error") if isinstance(body, dict) else None
            if status == 200 and isinstance(error, dict):
                try:
                    status = int(error.get("code", 502))
                except (ValueError, TypeError):
                    status = 502
            if status in {408, 429, 500, 502, 503, 504} and attempt + 1 < self.config.max_attempts:
                self.sleep(retry_delay(response.headers.get("Retry-After"), attempt))
                continue
            if status != 200 or error:
                raise RuntimeError(
                    f"OpenRouter request failed (status {status}); check account, model and provider settings"
                )
            # Store even malformed/length-limited completions so a resume does not buy them again.
            atomic_json(
                cache,
                {
                    "request_hash": request_hash,
                    "request": request,
                    "response_hash": digest(body),
                    "response": body,
                },
            )
            return body
        raise RuntimeError("OpenRouter attempts exhausted")


def parse_response(body, mode):
    try:
        choice = body["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
            raise ValueError("Incomplete or refused completion")
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenRouter returned no complete text choice") from exc
    cls = SceneBatch if mode == "grounded_qa" else DialogueBatch
    return cls.model_validate_json(content).items


def example(item, config: TextConfig, batch: int, response: dict):
    if isinstance(item, Scene):
        semantic = {"task": "color_recall", "item": item.item, "place": item.place}
        prompt = f"In this story, the {item.item} is {item.color} and is in the {item.place}. What color is the {item.item}? Reply with only the color."
        answer, task = item.color, "color_recall"
    else:
        semantic = {"task": "dialogue", "prompt": normalize(item.prompt)}
        prompt, answer, task = item.prompt, item.answer, "dialogue"
    group = digest(semantic)
    bucket = int(group[:8], 16) % 100
    return Example(
        id=digest({"group": group, "prompt": prompt, "answer": answer})[:24],
        semantic_group=group,
        split="train" if bucket < 80 else "validation" if bucket < 90 else "test",
        task=task,
        prompt=prompt,
        answer=answer,
        provenance={
            "synthetic": True,
            "generator": "openrouter-v1",
            "version": config.version,
            "mode": config.mode,
            "reward_protocol": "exact_answer" if task == "color_recall" else "open_ended",
            "source_fields": item.model_dump(),
            "seed": config.seed,
            "batch": batch,
            "requested_model": config.model,
            "returned_model": response.get("model"),
            "provider": response.get("provider"),
            "response_id": response.get("id"),
            "response_hash": digest(response),
        },
    )


def generate_text(config: TextConfig, output: Path, client=None, sleep=time.sleep):
    with run_directory(output, {"stage": "text-v1", "config": config.model_dump()}):
        if (output / "dataset.json").exists():
            return load_dataset(output)[1]
        api = OpenRouter(config, output, client, sleep)
        rows, groups, usage = [], set(), []
        # One batch per request at best; this also bounds cached duplicate-only batches.
        for batch in range(config.max_requests):
            count = min(config.batch_size, config.count - len(rows))
            response = api.complete(payload(config, batch, count), batch)
            items = parse_response(response, config.mode)
            usage.append({"batch": batch, "id": response.get("id"), "usage": response.get("usage")})
            for item in items[:count]:
                row = example(item, config, batch, response)
                if row.semantic_group in groups:
                    continue
                groups.add(row.semantic_group)
                rows.append(row)
            if len(rows) == config.count:
                return publish(
                    output,
                    rows,
                    config.model_dump(),
                    usage=usage,
                    reward_protocol="exact_answer"
                    if config.mode == "grounded_qa"
                    else "open_ended",
                )
        raise RuntimeError("Not enough unique examples within the request limit")
