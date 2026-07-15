"""
The Anthropic API call.

Takes a Findings object and returns a written diagnosis, plus usage
metadata (tokens in/out and an approximate cost). All the work is in
the prompt; this file is a thin wrapper around the SDK with:

  - Token/cost tracking (returned alongside the diagnosis text).
  - Retry with exponential backoff on transient errors (rate limits,
    5xx, connection errors). Non-transient errors (auth, 400) fail fast.

Public API:
  diagnose(findings, ...) -> DiagnosisResult
    .text: markdown diagnosis
    .usage: UsageInfo (tokens, cost estimate, model)
    .__str__ / __format__: returns .text so the object still prints cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic, APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .findings import Findings
from .prompts import build_prompt

DEFAULT_MODEL = "claude-sonnet-5"
# Set generously so extended-reasoning models (Sonnet 5+) don't burn the entire
# budget on internal thinking and return empty text. Observed: 2000 was too low.
DEFAULT_MAX_TOKENS = 4000

# USD per million tokens. Update when pricing changes; check docs.claude.com
# for current numbers. These are used only for the human-readable cost
# estimate on DiagnosisResult — not for anything programmatic.
MODEL_PRICING_PER_MTOK = {
    # Sonnet 5 pricing as of late 2025 / early 2026. Confirm against the
    # current docs before quoting these numbers to a stakeholder.
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}


class DiagnosisError(Exception):
    """Raised when the diagnosis API call fails in a way the caller should know about."""


@dataclass
class UsageInfo:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None if the model isn't in the pricing table.

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class DiagnosisResult:
    text: str
    usage: UsageInfo

    def __str__(self) -> str:
        # So `print(diagnose(...))` still works.
        return self.text

    def __format__(self, spec: str) -> str:
        return format(self.text, spec)


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float | None:
    pricing = MODEL_PRICING_PER_MTOK.get(model)
    if not pricing:
        return None
    return (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000


# Retry policy: 4 attempts, exponential backoff, capped at 30s between tries.
# We retry on transient errors only. Auth failures and 400s are not retried.
_RETRY_EXCEPTIONS = (RateLimitError, APIConnectionError, APIStatusError)


def _is_retryable_status(exc: BaseException) -> bool:
    """
    APIStatusError covers all non-2xx responses. We only want to retry 5xx
    and 429. tenacity doesn't natively know that, so this predicate does.
    """
    if isinstance(exc, RateLimitError) or isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status is not None and (status >= 500 or status == 429)
    return False


@retry(
    retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def _call_with_retry(client: Anthropic, **kwargs):
    """Wrapped API call. tenacity handles the retry loop."""
    try:
        return client.messages.create(**kwargs)
    except APIStatusError as e:
        if _is_retryable_status(e):
            raise  # let tenacity retry
        # Non-retryable 4xx (auth, bad request). Fail fast.
        raise DiagnosisError(f"Anthropic API returned {e.status_code}: {e}") from e


def diagnose(
    findings: Findings,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
) -> DiagnosisResult:
    """
    Generate a written diagnosis from a Findings object.

    Returns a DiagnosisResult with .text (markdown) and .usage
    (token counts + cost estimate). str(result) returns the text,
    so callers that just want the string still work.

    Raises DiagnosisError on missing API key or non-retryable API failures.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise DiagnosisError(
            "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment "
            "variable or pass api_key= to diagnose()."
        )

    client = Anthropic(api_key=key)
    system_prompt, user_prompt = build_prompt(findings.to_dict())

    try:
        response = _call_with_retry(
            client,
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except DiagnosisError:
        raise
    except Exception as e:
        raise DiagnosisError(f"Anthropic API call failed after retries: {e}") from e

    text_parts = [block.text for block in response.content if hasattr(block, "text")]
    if not text_parts:
        raise DiagnosisError("API returned no text content.")
    text = "\n".join(text_parts).strip()

    in_tok = getattr(response.usage, "input_tokens", 0)
    out_tok = getattr(response.usage, "output_tokens", 0)
    usage = UsageInfo(
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=_estimate_cost(model, in_tok, out_tok),
    )

    return DiagnosisResult(text=text, usage=usage)
