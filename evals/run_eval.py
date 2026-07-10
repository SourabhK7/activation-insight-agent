"""
End-to-end evaluation harness.

For each of N runs, per arm:
  1. Generate a fresh synthetic funnel (different seed each run to test
     robustness across data variants).
  2. Run the arm end-to-end to produce a diagnosis.
  3. Score the diagnosis with the LLM judge, given the ground truth for
     that specific dataset.
  4. Record everything to results/raw_runs.jsonl.

At the end, produce results/latest.md with aggregate scores (mean, stdev,
per-criterion breakdown) for each arm.

Usage:
    python evals/run_eval.py --n 10
    python evals/run_eval.py --n 10 --seed-base 1000
    python evals/run_eval.py --n 3 --arms structured    # just one arm

Requires ANTHROPIC_API_KEY. Each run costs a handful of cents; N=10 across
both arms is roughly a dollar depending on model choice.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic

# Allow `python evals/run_eval.py` from the repo root without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activation_agent import synthesize  # noqa: E402
from activation_agent.__main__ import DEFAULT_ATTRIBUTES, DEFAULT_STEP_ORDER, _build_findings  # noqa: E402
from activation_agent.diagnose import diagnose  # noqa: E402

from evals.judge import JudgeError, score  # noqa: E402
from evals.naive_baseline import build_naive_prompt  # noqa: E402


RESULTS_DIR = Path(__file__).parent / "results"
RAW_RUNS_PATH = RESULTS_DIR / "raw_runs.jsonl"
LATEST_PATH = RESULTS_DIR / "latest.md"

ARM_STRUCTURED = "structured"
ARM_NAIVE = "naive"

DEFAULT_DIAGNOSIS_MODEL = "claude-sonnet-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


def _compute_ground_truth(events, step_order, attribute_columns) -> Dict[str, Any]:
    """
    Ground truth for the judge = the true findings computed by the
    (deterministic, pandas) analysis pipeline for this specific dataset.
    Both arms are graded against the same ground truth.
    """
    findings = _build_findings(events, step_order, attribute_columns)
    return findings.to_dict()


def _run_structured_arm(events, findings_dict, model, api_key) -> str:
    """Arm A: the current design — structured findings JSON → Claude."""
    findings = _build_findings(events, DEFAULT_STEP_ORDER, DEFAULT_ATTRIBUTES)
    return diagnose(findings, model=model, api_key=api_key)


def _run_naive_arm(events, model, api_key) -> str:
    """Arm B: naive baseline — aggregated counts, LLM computes rates."""
    system_prompt, user_prompt = build_naive_prompt(
        events, DEFAULT_STEP_ORDER, DEFAULT_ATTRIBUTES
    )
    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text_parts = [b.text for b in resp.content if hasattr(b, "text")]
    return "\n".join(text_parts).strip()


def _append_raw_run(entry: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-arm mean, stdev, per-criterion means."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        if r.get("status") != "ok":
            continue
        by_arm.setdefault(r["arm"], []).append(r)

    summary: Dict[str, Any] = {}
    for arm, arm_runs in by_arm.items():
        totals = [r["total"] for r in arm_runs]
        per_crit: Dict[str, List[int]] = {}
        for r in arm_runs:
            for k, v in r["per_criterion"].items():
                per_crit.setdefault(k, []).append(int(v["score"]))
        summary[arm] = {
            "n": len(arm_runs),
            "total_mean": round(statistics.mean(totals), 2) if totals else None,
            "total_stdev": round(statistics.stdev(totals), 2) if len(totals) > 1 else 0.0,
            "per_criterion_mean": {
                k: round(statistics.mean(v), 2) for k, v in per_crit.items()
            },
        }
    return summary


def _write_latest(summary: Dict[str, Any], n_planned: int, seed_base: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Latest evaluation results",
        "",
        f"Generated: {ts}",
        f"Runs per arm: {n_planned} (seed base: {seed_base})",
        "",
        "## Aggregate scores (0–12 per run)",
        "",
        "| Arm | N | Mean total | Stdev |",
        "|---|---:|---:|---:|",
    ]
    for arm, s in sorted(summary.items()):
        lines.append(f"| {arm} | {s['n']} | {s['total_mean']} | {s['total_stdev']} |")

    lines.extend(["", "## Per-criterion mean scores (0–2)", ""])
    crits = [
        "criterion_1_mobile_payment",
        "criterion_2_paid_social",
        "criterion_3_cx_region",
        "criterion_4_no_fabrication",
        "criterion_5_numerical_accuracy",
        "criterion_6_calibrated_language",
    ]
    header = "| Arm | " + " | ".join(c.replace("criterion_", "c").split("_", 1)[0] for c in crits) + " |"
    sep = "|---" * (len(crits) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for arm, s in sorted(summary.items()):
        row = [arm]
        for c in crits:
            row.append(str(s["per_criterion_mean"].get(c, "—")))
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Raw runs",
        "",
        f"See `results/raw_runs.jsonl` for the full per-run judge output (diagnosis text, scorecard, reasons).",
        "",
        "## What the numbers mean",
        "",
        "The design thesis under test: *sending pre-computed findings to the LLM (Arm A) beats sending aggregated counts and asking the LLM to compute rates itself (Arm B) — mostly because the LLM is unreliable at arithmetic.*",
        "",
        "Look at criterion 5 (numerical accuracy) first — that's the direct test of the thesis. Total scores are a secondary summary.",
        "",
        "Small N caveats apply. See `../rubric.md` for limitations of LLM-as-judge.",
    ])
    LATEST_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_one(
    seed: int,
    arm: str,
    diagnosis_model: str,
    judge_model: str,
    n_users: int,
    api_key: str,
) -> Dict[str, Any]:
    """Run a single (seed, arm) trial and return the row for raw_runs.jsonl."""
    events = synthesize.generate(n_users=n_users, seed=seed)
    ground_truth = _compute_ground_truth(events, DEFAULT_STEP_ORDER, DEFAULT_ATTRIBUTES)

    row_base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "seed": seed,
        "n_users": n_users,
        "diagnosis_model": diagnosis_model,
        "judge_model": judge_model,
    }

    try:
        if arm == ARM_STRUCTURED:
            diagnosis_text = _run_structured_arm(events, ground_truth, diagnosis_model, api_key)
        elif arm == ARM_NAIVE:
            diagnosis_text = _run_naive_arm(events, diagnosis_model, api_key)
        else:
            raise ValueError(f"unknown arm: {arm}")
    except Exception as e:
        return {**row_base, "status": "arm_failed", "error": str(e)}

    try:
        scorecard = score(
            diagnosis_text=diagnosis_text,
            ground_truth=ground_truth,
            model=judge_model,
            api_key=api_key,
        )
    except JudgeError as e:
        return {
            **row_base,
            "status": "judge_failed",
            "error": str(e),
            "diagnosis": diagnosis_text,
        }

    return {
        **row_base,
        "status": "ok",
        "diagnosis": diagnosis_text,
        "per_criterion": scorecard.per_criterion,
        "total": scorecard.total,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python evals/run_eval.py")
    parser.add_argument("--n", type=int, default=10, help="Runs per arm.")
    parser.add_argument("--seed-base", type=int, default=1000, help="Seed for run 1; incremented per run.")
    parser.add_argument("--n-users", type=int, default=20000, help="Users per synthetic dataset. Smaller = faster + cheaper.")
    parser.add_argument("--diagnosis-model", default=DEFAULT_DIAGNOSIS_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--arms",
        default=f"{ARM_STRUCTURED},{ARM_NAIVE}",
        help="Comma-separated arms to run (structured,naive).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete results/raw_runs.jsonl before starting.",
    )
    args = parser.parse_args(argv)

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset and RAW_RUNS_PATH.exists():
        RAW_RUNS_PATH.unlink()

    all_runs: List[Dict[str, Any]] = []
    for i in range(args.n):
        seed = args.seed_base + i
        for arm in arms:
            print(f"[{i+1}/{args.n}] arm={arm} seed={seed} ...", file=sys.stderr, flush=True)
            t0 = time.time()
            entry = run_one(
                seed=seed,
                arm=arm,
                diagnosis_model=args.diagnosis_model,
                judge_model=args.judge_model,
                n_users=args.n_users,
                api_key=api_key,
            )
            entry["duration_s"] = round(time.time() - t0, 2)
            _append_raw_run(entry)
            all_runs.append(entry)
            status = entry.get("status", "?")
            total = entry.get("total", "-")
            print(f"    → {status} total={total} ({entry['duration_s']}s)", file=sys.stderr)

    summary = _aggregate(all_runs)
    _write_latest(summary, n_planned=args.n, seed_base=args.seed_base)

    print("\nDone.", file=sys.stderr)
    print(f"Raw runs: {RAW_RUNS_PATH}", file=sys.stderr)
    print(f"Summary:  {LATEST_PATH}", file=sys.stderr)
    print("\n" + LATEST_PATH.read_text(encoding="utf-8"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
