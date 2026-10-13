"""Validate every browser input and every model output at the boundary.

The UI and files use 0–100 percentages; the existing evaluation prompt returns
0–1. Only graphs.save_result converts that model result into a percentage.
Keeping this distinction explicit avoids a silent 100× scoring error.
"""
from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InputModel(BaseModel):
    # Unknown fields are usually a typo, not something to silently discard.
    # NaN/Infinity must not enter history or contaminate quality metrics.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class SampleInput(InputModel):
    # 'origin' is provenance, whereas manual_probability is an assessment.
    # An uncertain human judgment must not become a verified accuracy label.
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    lang: str = Field(default="en", pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")
    manual_probability: float = Field(ge=0, le=100)
    manual_low: float = Field(ge=0, le=100)
    manual_high: float = Field(ge=0, le=100)
    origin: Literal["unknown", "human", "ai", "mixed"] = "unknown"
    notes: str = Field(default="", max_length=10_000)
    materials_mode: Literal["use", "unavailable", "irrelevant"] = "irrelevant"
    materials: str = Field(default="", max_length=200_000)
    assignment: str = Field(default="", max_length=30_000)
    split: Literal["development", "holdout", "test"] = "development"

    @field_validator("lang", mode="before")
    @classmethod
    def default_language(cls, value: str | None) -> str:
        return (value or "en").strip() or "en"

    @model_validator(mode="after")
    def check_context(self) -> Self:
        # These cross-field constraints cannot be expressed by HTML min/max.
        if not self.manual_low <= self.manual_probability <= self.manual_high:
            raise ValueError("Manual range must contain the manual probability.")
        if self.materials_mode == "use" and not self.materials:
            raise ValueError("Add course materials or choose unavailable / irrelevant.")
        return self


class ModelInput(InputModel):
    model: str = Field(default="gpt-6-luna", min_length=1, max_length=150)
    effort: Literal["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"] = "auto"
    max_output_tokens: int = Field(default=8192, ge=512, le=65536)


class RunInput(ModelInput):
    # A reference run deliberately overrides strategy/split with its saved
    # sample snapshots. That makes candidate backtests a paired comparison.
    name: str = Field(default="", max_length=200)
    prompt_id: str = "repo"
    strategy: Literal["all", "balanced", "manual"] = "all"
    percent: float = Field(default=100, gt=0, le=100)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    sample_ids: list[str] = Field(default_factory=list, max_length=10_000)
    split: Literal["development", "holdout", "test", "all"] = "development"
    threshold: float = Field(default=50, gt=0, lt=100)
    reference_run_id: str | None = None


class ImprovementInput(ModelInput):
    run_id: str
    name: str = Field(min_length=1, max_length=200)
    guidance: str = Field(default="", max_length=10_000)
    max_output_tokens: int = Field(default=16384, ge=512, le=65536)


class CandidateInput(InputModel):
    name: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=100_000)
    parent_id: str = "repo"
    rationale: str = Field(default="", max_length=20_000)


class CandidateReview(InputModel):
    status: Literal["proposed", "approved", "rejected"]
    note: str = Field(default="", max_length=10_000)


class EvaluationOutput(InputModel):
    ai_origin_explanation: str = Field(min_length=1)
    ai_origin_probability: float = Field(ge=0, le=1)


class ImprovementOutput(InputModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    rationale: str = Field(min_length=1, max_length=20_000)


def output_schema(kind: str) -> dict:
    """Small strict JSON schema supported by the Responses API.

    Use Pydantic again after parsing: provider adherence is not a substitute for
    local validation, particularly for refusals or token-limit interruptions.
    """
    fields = ({"ai_origin_explanation": {"type": "string"},
               "ai_origin_probability": {"type": "number", "minimum": 0, "maximum": 1}}
              if kind == "evaluation" else
              {"prompt": {"type": "string"}, "rationale": {"type": "string"}})
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
