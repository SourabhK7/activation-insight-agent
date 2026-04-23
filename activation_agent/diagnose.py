"""
The Anthropic API call.

Takes a Findings object and returns a written diagnosis (markdown string).
All the work is in the prompt; this file is a thin wrapper around the SDK.
"""

from __future__ import annotations

import os

from anthropic import Anthropic

from .findings import Findings
from .prompts import build_prompt


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 2000


class DiagnosisError(Exception):
    """Raised when the diagnosis API call fails in a way the caller should know about."""


def diagnose(
    findings: Findings,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
) -> str:
    """
    Generate a written diagnosis from a Findings object.

    Parameters
    ----------
    findings : Findings
        Structured findings from the analysis pipeline.
    model : str
        Anthropic model to use. Defaults to Sonnet (see README for rationale).
    max_tokens : int
        Max tokens in the response. 2000 is generous for this format.
    api_key : str, optional
        Anthropic API key. If None, read from ANTHROPIC_API_KEY env var.

    Returns
    -------
    str
        Markdown diagnosis.

    Raises
    ------
    DiagnosisError
        If the API key is missing or the API call fails.
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
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        raise DiagnosisError(f"Anthropic API call failed: {e}") from e

    # Response content is a list of blocks; we only emit text.
    text_parts = [block.text for block in response.content if hasattr(block, "text")]
    if not text_parts:
        raise DiagnosisError("API returned no text content.")

    return "\n".join(text_parts).strip()
