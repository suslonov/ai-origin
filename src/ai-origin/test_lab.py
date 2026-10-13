"""Offline behavioral tests: real LangGraph execution, fake paid model calls.

No student data or credentials are needed. Temporary directories also let us
test restart recovery without touching the user's actual experiment history.
"""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from graphs import evaluation_content, evaluation_instructions
from metrics import quality_metrics, select_samples
from provider import OpenAIProvider, ProviderError, make_request, parse_response, text_model
from schemas import CandidateInput, CandidateRename, ImprovementInput, RunInput, SampleInput
from service import REPO_PROMPT, Service
from storage import Store


def sample(**changes: Any) -> SampleInput:
    return SampleInput.model_validate({"title": "Example", "text": "My own answer.", "lang": "en",
        "manual_probability": 20, "manual_low": 10, "manual_high": 30, **changes})


class FakeProvider:
    configured = True

    def __init__(self, replies: list | None = None) -> None:
        self.requests: list[dict] = []
        self.replies = replies or []
        self.on_call = None

    def models(self, refresh: bool = False) -> list[dict]:
        return [{"id": "test-model", "efforts": ["auto", "high"]}]

    def validate_model(self, config: Any) -> None:
        if config.model != "test-model":
            raise ValueError("Model unavailable")

    def complete(self, request: dict) -> dict:
        self.requests.append(copy.deepcopy(request))
        if self.on_call:
            self.on_call()
        value = self.replies.pop(0) if self.replies else {"ai_origin_probability": .25, "ai_origin_explanation": "A quoted observation."}
        if isinstance(value, Exception):
            raise value
        if "status" in value:
            return value
        return {"status": "completed", "model": "test-model-snapshot", "usage": {"input_tokens": 100, "output_tokens": 20},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}


@pytest.fixture
def lab(tmp_path: Path) -> Service:
    instance = Service(tmp_path / "data", FakeProvider())
    yield instance
    instance.queue.join()
    instance.close()


def wait_run(lab: Service, **config: Any) -> dict:
    run = lab.start_run(RunInput(model="test-model", **config))
    lab.queue.join()
    return lab.store.get("runs", run["id"])


def test_validation_and_language_defaults() -> None:
    assert sample(lang=None).lang == "en"
    assert sample(lang="").lang == "en"
    for changes in ({"manual_low": 25}, {"manual_high": 5}, {"manual_probability": float("nan")},
                    {"materials_mode": "use"}, {"text": "   "}):
        with pytest.raises(ValidationError):
            sample(**changes)


def test_balanced_selection_is_seeded_and_stratified() -> None:
    samples = []
    for lang in ["en", "he"]:
        for score in [5, 50, 95]:
            for index in range(8):
                samples.append({**sample(lang=lang, manual_probability=score, manual_low=0, manual_high=100).model_dump(), "id": f"{lang}-{score}-{index}"})
    config = RunInput(model="test-model", strategy="balanced", percent=25, seed=19)
    selected = select_samples(samples, config)
    assert len(selected) == 12
    assert selected == select_samples(list(reversed(samples)), config)
    assert len({s["id"] for s in selected}) == 12
    assert all(sum(s["lang"] == lang and s["manual_probability"] == score for s in selected) == 2 for lang in ["en", "he"] for score in [5, 50, 95])
    assert len(select_samples(samples, config.model_copy(update={"percent": 100}))) == 48
    assert len(select_samples(samples, config.model_copy(update={"percent": .1}))) == 1


@pytest.mark.parametrize("split", ["holdout", "test"])
def test_manual_selection_respects_reserved_datasets(split: str) -> None:
    samples = [{**sample(split=split).model_dump(), "id": "held"}]
    with pytest.raises(ValueError):
        select_samples(samples, RunInput(model="test-model", strategy="manual", sample_ids=["held"]))
    assert len(select_samples(samples, RunInput(model="test-model", strategy="manual", sample_ids=["held"], split=split))) == 1


def test_metrics_separate_estimates_from_known_origin() -> None:
    rows = []
    for manual, predicted, origin in [(0, 60, "human"), (100, 80, "ai"), (50, 40, "unknown")]:
        rows.append({"status": "completed", "sample": sample(manual_probability=manual, manual_low=manual, manual_high=manual, origin=origin).model_dump(), "model_probability": predicted})
    rows.append({"status": "failed", "sample": sample().model_dump()})
    result = quality_metrics(rows, 50)
    assert result["mae_pp"] == 30
    assert result["bias_pp"] == 10
    assert result["completed"] == 3 and result["failed"] == 1
    assert result["known_origin"]["count"] == 2
    assert result["known_origin"]["fp"] == 1
    assert result["known_origin"]["accuracy"] == .5
    assert result["known_origin"]["brier"] == pytest.approx(.2)
    empty = quality_metrics([], 50)
    assert empty["mae_pp"] is None and empty["known_origin"]["false_positive_rate"] is None


def test_range_metric_uses_distance_outside_interval() -> None:
    rows = [{"status": "completed", "sample": sample().model_dump(), "model_probability": p} for p in [10, 25, 40]]
    result = quality_metrics(rows, 50)
    assert result["within_range"] == pytest.approx(2 / 3)
    assert result["range_distance_pp"] == pytest.approx(10 / 3)


def test_no_label_leakage_and_material_modes() -> None:
    for mode in ["use", "unavailable", "irrelevant"]:
        value = sample(notes="SECRET LABEL NOTE", origin="human", materials="COURSE SECRET", materials_mode=mode).model_dump()
        content = evaluation_content(value)
        instructions = evaluation_instructions("original policy", value)
        assert "manual_probability" not in content and "SECRET LABEL NOTE" not in content
        assert "origin" not in json.loads(content)
        assert ("COURSE SECRET" in content) == (mode == "use")
        assert f"Materials mode: {mode}" in instructions


def test_evaluation_snapshots_and_backtest(lab: Service) -> None:
    saved = lab.save_sample(sample())
    run = wait_run(lab)
    assert run["status"] == "completed"
    assert run["name"] == "test-model · Current repository prompt · Development"
    assert run["results"][0]["model_probability"] == 25
    assert run["prompt"]["text"] == REPO_PROMPT.read_text()
    assert run["results"][0]["request"]["instructions"].startswith(run["prompt"]["text"])
    lab.save_sample(sample(text="Edited later", manual_probability=90, manual_low=80, manual_high=100), saved["id"], saved["revision"])
    candidate = lab.save_candidate(CandidateInput(name="Candidate", prompt="Different prompt"))
    repeat = wait_run(lab, name="Backtest", prompt_id=candidate["id"], reference_run_id=run["id"])
    assert repeat["name"] == "Backtest · Development"
    assert repeat["config"]["split"] == "development"
    assert repeat["results"][0]["sample"]["text"] == "My own answer."
    assert repeat["results"][0]["sample"]["manual_probability"] == 20
    assert repeat["comparison"]["paired_samples"] == 1
    assert repeat["comparison"]["mae_change_pp"] == 0
    assert repeat["prompt"]["text"] == "Different prompt"


def test_stale_sample_update_rejected(lab: Service) -> None:
    saved = lab.save_sample(sample())
    lab.save_sample(sample(title="Changed"), saved["id"], saved["revision"])
    with pytest.raises(ValueError, match="another window"):
        lab.save_sample(sample(), saved["id"], saved["revision"])


def test_bad_response_saved_without_zero_score(lab: Service) -> None:
    lab.save_sample(sample())
    lab.provider.replies = [{"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}]
    run = wait_run(lab)
    assert run["status"] == "completed_with_errors"
    assert run["results"][0]["status"] == "failed"
    assert "model_probability" not in run["results"][0]
    assert run["metrics"]["overall"]["mae_pp"] is None
    assert run["results"][0]["raw_response"]["status"] == "incomplete"


def test_fatal_config_error_stops_further_calls(lab: Service) -> None:
    for _ in range(3):
        lab.save_sample(sample())
    lab.provider.replies = [ProviderError("Unsupported effort", 400)]
    run = wait_run(lab)
    assert run["status"] == "failed" and len(lab.provider.requests) == 1
    assert run["metrics"]["overall"]["pending"] == 2


def test_cancel_preserves_inflight_result(lab: Service) -> None:
    for _ in range(3):
        lab.save_sample(sample())
    entered, release = threading.Event(), threading.Event()
    lab.provider.on_call = lambda: (entered.set(), release.wait(timeout=3))
    run = lab.start_run(RunInput(model="test-model"))
    assert entered.wait(timeout=3)
    lab.cancel_run(run["id"])
    release.set()
    lab.queue.join()
    result = lab.store.get("runs", run["id"])
    assert result["status"] == "cancelled"
    assert result["metrics"]["overall"]["completed"] == 1
    assert len(lab.provider.requests) == 1


def test_improver_sees_development_only_and_creates_candidate(lab: Service) -> None:
    lab.save_sample(sample(text="DEVELOPMENT SAMPLE"))
    lab.save_sample(sample(text="HOLDOUT SECRET", split="holdout"))
    lab.save_sample(sample(text="TEST SECRET", split="test"))
    run = wait_run(lab, split="all")
    original = REPO_PROMPT.read_bytes()
    lab.provider.replies = [{"prompt": "A revised reusable prompt", "rationale": "One focused adjustment."}]
    job = lab.start_improvement(ImprovementInput(model="test-model", run_id=run["id"], name="v1"))
    lab.queue.join()
    result = lab.store.get("improvements", job["id"])
    assert result["status"] == "completed"
    content = lab.provider.requests[-1]["input"][0]["content"]
    assert "HOLDOUT SECRET" not in content and "TEST SECRET" not in content and "DEVELOPMENT SAMPLE" in content
    assert json.loads(content)["development_metrics"]["total"] == 1
    candidate = lab.candidate_detail(result["candidate_id"])
    assert candidate["prompt"] == "A revised reusable prompt" and candidate["diff"]
    renamed = lab.rename_candidate(candidate["id"], CandidateRename(name="Clearer name"))
    assert renamed["name"] == "Clearer name" and renamed["prompt"] == candidate["prompt"]
    assert lab.prompt(candidate["id"])["name"] == "Clearer name"
    lab.delete_candidate(candidate["id"])
    with pytest.raises(FileNotFoundError):
        lab.candidate_detail(candidate["id"])
    assert not (lab.store.root / "candidates" / f"{candidate['id']}.txt").exists()
    assert REPO_PROMPT.read_bytes() == original


@pytest.mark.parametrize("split", ["holdout", "test"])
def test_reserved_only_run_cannot_improve(lab: Service, split: str) -> None:
    lab.save_sample(sample(split=split))
    run = wait_run(lab, split=split)
    assert run["name"].endswith(" · " + {"holdout": "Holdout", "test": "Test"}[split])
    with pytest.raises(ValueError, match="development"):
        lab.start_improvement(ImprovementInput(model="test-model", run_id=run["id"], name="v1"))


def test_restart_marks_unfinished_jobs_and_process_lock(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with pytest.raises(RuntimeError, match="Another"):
        Store(tmp_path)
    store.write("runs", {"id": "run_0123456789abcdef", "created_at": "2026-01-01", "status": "running", "results": [{"saved": True}]})
    store.close()
    restarted = Store(tmp_path)
    run = restarted.get("runs", "run_0123456789abcdef")
    assert run["status"] == "interrupted" and run["results"] == [{"saved": True}]
    with pytest.raises(ValueError):
        restarted.get("samples", "../../secret")
    restarted.close()


def test_csv_escapes_formulas(lab: Service) -> None:
    lab.save_sample(sample(title="=SUM(1,2)"))
    run = wait_run(lab)
    assert "'=SUM(1,2)" in lab.csv_results(run["id"])


def test_credentials_and_effort_do_not_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_GPT_KEY", "test-primary")
    monkeypatch.setenv("OPENAI_API_KEY", "test-fallback")
    provider = OpenAIProvider()
    assert provider._key == "test-primary"
    request = make_request(RunInput(model="test-model").model_dump(), "policy", "text", "evaluation")
    assert "reasoning" not in request and "test-primary" not in json.dumps(request)
    assert request["store"] is False and request["truncation"] == "disabled"
    with pytest.raises(ValueError, match="refused"):
        parse_response({"status": "completed", "output": [{"content": [{"type": "refusal", "refusal": "No"}]}]})


@pytest.mark.parametrize("strategy", ["all", "balanced", "manual"])
def test_test_dataset_stays_separate(lab: Service, strategy: str) -> None:
    records = {split: lab.save_sample(sample(split=split)) for split in ["development", "holdout", "test"]}
    run = wait_run(lab, split="test", strategy=strategy, percent=100,
                   sample_ids=[records["test"]["id"]] if strategy == "manual" else [])
    assert [row["sample"]["id"] for row in run["results"]] == [records["test"]["id"]]
    assert set(run["metrics"]["split"]) == {"test"}
    assert run["metrics"]["overall"]["completed"] == 1
    assert {s["split"] for s in select_samples(list(records.values()), RunInput(model="test-model", split="all"))} == {"development", "holdout", "test"}


@pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini-2025-04-16", "o9-preview",
                                   "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06",
                                   "chatgpt-4o-latest", "ft:gpt-4o-mini:org:custom:id",
                                   "ft:o3:org:custom:id"])
def test_excluded_models_never_reach_provider(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider()
    def forbidden(*args, **kwargs):
        pytest.fail("Excluded model must be rejected before any HTTP request")
    monkeypatch.setattr(provider, "_request", forbidden)
    assert not text_model(model)
    with pytest.raises(ValueError, match="disabled"):
        provider.validate_model(RunInput(model=model))
    with pytest.raises(ProviderError, match="disabled"):
        provider.complete({"model": model})


def test_model_discovery_keeps_supported_families(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider()
    ids = ["o3", "gpt-4o-mini", "ft:gpt-4o:org:custom:id", "gpt-4.1", "gpt-5-mini", "gpt-6-astra"]
    monkeypatch.setattr(provider, "_request", lambda *args: {"data": [{"id": model} for model in ids]})
    assert {m["id"] for m in provider.models()} == {"gpt-4.1", "gpt-5-mini", "gpt-6-astra"}
