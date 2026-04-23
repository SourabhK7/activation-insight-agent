"""
CLI entry point for the Activation Insight Agent.

Subcommands:
  generate-data    Generate synthetic funnel data for testing.
  run              Run the full pipeline: analyze a funnel CSV and produce a diagnosis.
  analyze          Run only the analysis (no LLM call). Useful for debugging / no-API-key runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

from . import synthesize
from .cohorts import find_divergent_segments
from .findings import Findings
from .funnel import compute_funnel


DEFAULT_STEP_ORDER = [
    "landing_page_view",
    "product_page_view",
    "add_to_cart",
    "shipping_info_entered",
    "payment_info_entered",
    "purchase_complete",
]

DEFAULT_ATTRIBUTES = ["device", "acquisition_source", "country_code", "signup_week"]


def _build_findings(
    events: pd.DataFrame,
    step_order: List[str],
    attribute_columns: List[str],
) -> Findings:
    overall = compute_funnel(events, step_order)
    divergent = find_divergent_segments(events, step_order, attribute_columns)

    # Count segments that were too small per attribute, for context.
    small_counts: dict[str, int] = {}
    for attr in attribute_columns:
        if attr not in events.columns:
            continue
        seg_sizes = events[events["step_name"] == step_order[0]].groupby(attr)["user_id"].nunique()
        small_counts[attr] = int((seg_sizes < 500).sum())

    analyzed = [a for a in attribute_columns if a in events.columns]

    return Findings(
        overall=overall,
        divergent_segments=divergent,
        attribute_columns_analyzed=analyzed,
        small_segment_counts=small_counts,
    )


def cmd_generate_data(args: argparse.Namespace) -> int:
    df = synthesize.generate(n_users=args.n_users, seed=args.seed)
    out = Path(args.output)
    synthesize.write(df, out)
    print(f"Generated {len(df):,} events for {args.n_users:,} users → {out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    events = pd.read_csv(args.data)
    step_order = args.steps.split(",") if args.steps else DEFAULT_STEP_ORDER
    attrs = args.attributes.split(",") if args.attributes else DEFAULT_ATTRIBUTES

    findings = _build_findings(events, step_order, attrs)
    out_dict = findings.to_dict()

    if args.output:
        Path(args.output).write_text(json.dumps(out_dict, indent=2, default=str))
        print(f"Findings written to {args.output}")
    else:
        print(json.dumps(out_dict, indent=2, default=str))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Import lazily so `generate-data` and `analyze` work without the anthropic package.
    from .diagnose import diagnose, DiagnosisError

    events = pd.read_csv(args.data)
    step_order = args.steps.split(",") if args.steps else DEFAULT_STEP_ORDER
    attrs = args.attributes.split(",") if args.attributes else DEFAULT_ATTRIBUTES

    print(f"Analyzing {len(events):,} events...", file=sys.stderr)
    findings = _build_findings(events, step_order, attrs)
    print(
        f"Found {len(findings.divergent_segments)} divergent segments across "
        f"{len(findings.attribute_columns_analyzed)} attributes. "
        f"Generating diagnosis via Anthropic API...",
        file=sys.stderr,
    )

    try:
        diagnosis = diagnose(findings, model=args.model)
    except DiagnosisError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(diagnosis)
        print(f"Diagnosis written to {args.output}", file=sys.stderr)
    print(diagnosis)
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m activation_agent",
        description="Activation Insight Agent: analyze a funnel and generate a diagnosis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate-data
    p_gen = sub.add_parser("generate-data", help="Generate synthetic funnel data.")
    p_gen.add_argument("--n-users", type=int, default=50000, help="Number of users to simulate.")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_gen.add_argument("--output", default="data/sample_funnel.csv", help="Output CSV path.")
    p_gen.set_defaults(func=cmd_generate_data)

    # analyze (no LLM)
    p_an = sub.add_parser("analyze", help="Run analysis only; output findings as JSON.")
    p_an.add_argument("--data", required=True, help="Input CSV.")
    p_an.add_argument("--steps", help="Comma-separated funnel step order.")
    p_an.add_argument("--attributes", help="Comma-separated attribute columns for segmentation.")
    p_an.add_argument("--output", help="Output JSON path. If omitted, prints to stdout.")
    p_an.set_defaults(func=cmd_analyze)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run full pipeline: analysis + LLM diagnosis.")
    p_run.add_argument("--data", required=True, help="Input CSV.")
    p_run.add_argument("--steps", help="Comma-separated funnel step order.")
    p_run.add_argument("--attributes", help="Comma-separated attribute columns.")
    p_run.add_argument("--output", help="Output markdown path for the diagnosis.")
    p_run.add_argument(
        "--model",
        default="claude-sonnet-4-5",
        help="Anthropic model to use. Check docs.claude.com for current model IDs.",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
