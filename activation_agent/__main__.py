"""
CLI entry point for the Activation Insight Agent.

Subcommands:
  generate-data    Generate synthetic funnel data for testing.
  run              Full funnel pipeline: analyze a funnel CSV and produce a diagnosis.
  analyze          Funnel analysis only, no LLM. Useful for debugging / no-API-key runs.
  retention        Retention curve analysis + LLM narrative.
  ab-readout       A/B test analysis + LLM readout.
  anomaly          Rate/mix anomaly decomposition + LLM narrative.

All LLM-producing subcommands accept --format {full, exec, slack} to pick
output length/register. Every number in every output is computed in Python
and quoted by the LLM; the LLM never does arithmetic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
    *,
    detection_mode: str = "threshold",
) -> Findings:
    overall = compute_funnel(events, step_order)
    divergent = find_divergent_segments(
        events, step_order, attribute_columns, detection_mode=detection_mode
    )

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
    if args.preset == "b2b-trial":
        df = synthesize.generate_b2b_trial(n_users=args.n_users, seed=args.seed)
    else:
        df = synthesize.generate(n_users=args.n_users, seed=args.seed)
    out = Path(args.output)
    synthesize.write(df, out)
    print(
        f"Generated {len(df):,} events for {args.n_users:,} users "
        f"(preset={args.preset}) -> {out}"
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    events = pd.read_csv(args.data)
    step_order = args.steps.split(",") if args.steps else DEFAULT_STEP_ORDER
    attrs = args.attributes.split(",") if args.attributes else DEFAULT_ATTRIBUTES

    findings = _build_findings(events, step_order, attrs, detection_mode=args.detection_mode)
    out_dict = findings.to_dict()

    if args.output:
        Path(args.output).write_text(json.dumps(out_dict, indent=2, default=str))
        print(f"Findings written to {args.output}")
    else:
        print(json.dumps(out_dict, indent=2, default=str))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Import lazily so `generate-data` and `analyze` work without the anthropic package.
    from .diagnose import DiagnosisError, diagnose

    events = pd.read_csv(args.data)
    step_order = args.steps.split(",") if args.steps else DEFAULT_STEP_ORDER
    attrs = args.attributes.split(",") if args.attributes else DEFAULT_ATTRIBUTES

    print(f"Analyzing {len(events):,} events (detection={args.detection_mode})...", file=sys.stderr)
    findings = _build_findings(events, step_order, attrs, detection_mode=args.detection_mode)
    print(
        f"Found {len(findings.divergent_segments)} divergent segments across "
        f"{len(findings.attribute_columns_analyzed)} attributes. "
        f"Generating diagnosis via Anthropic API (format={args.format})...",
        file=sys.stderr,
    )

    try:
        result = diagnose(findings, model=args.model, output_format=args.format)
    except DiagnosisError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(result.text)
        print(f"Diagnosis written to {args.output}", file=sys.stderr)

    u = result.usage
    cost_str = f"~${u.cost_usd:.4f}" if u.cost_usd is not None else "unknown (model not in pricing table)"
    print(
        f"Tokens: in={u.input_tokens:,} out={u.output_tokens:,} · cost: {cost_str} · model: {u.model}",
        file=sys.stderr,
    )

    print(result.text)
    return 0


def _dataclass_to_dict(obj) -> dict:
    """Recursively convert nested dataclasses to plain dicts for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _print_usage(result, prefix: str = "") -> None:
    u = result.usage
    cost_str = f"~${u.cost_usd:.4f}" if u.cost_usd is not None else "unknown (model not in pricing table)"
    print(
        f"{prefix}Tokens: in={u.input_tokens:,} out={u.output_tokens:,} · cost: {cost_str} · model: {u.model}",
        file=sys.stderr,
    )


def cmd_retention(args: argparse.Namespace) -> int:
    from .diagnose import DiagnosisError, run_llm
    from .prompts import build_retention_prompt
    from .retention import compute_retention

    signups = pd.read_csv(args.signups)
    activity = pd.read_csv(args.activity)
    days = [int(d) for d in args.days.split(",")] if args.days else None

    findings = compute_retention(
        signups=signups,
        activity=activity,
        days=days,
        data_pull_date=args.data_pull_date,
        cohort_column=args.cohort_column,
    )
    findings_dict = _dataclass_to_dict(findings)

    if args.no_llm:
        print(json.dumps(findings_dict, indent=2, default=str))
        return 0

    print(
        f"Retention curve computed at days {findings.days_measured}. "
        f"{len(findings.cohorts)} cohort curves. Generating narrative (format={args.format})...",
        file=sys.stderr,
    )
    system_prompt, user_prompt = build_retention_prompt(findings_dict, output_format=args.format)
    try:
        result = run_llm(system_prompt, user_prompt, model=args.model)
    except DiagnosisError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(result.text)
        print(f"Retention narrative written to {args.output}", file=sys.stderr)
    _print_usage(result)
    print(result.text)
    return 0


def cmd_ab_readout(args: argparse.Namespace) -> int:
    from .ab_test import analyze_binary, analyze_continuous
    from .diagnose import DiagnosisError, run_llm
    from .prompts import build_ab_prompt

    if args.spec:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    else:
        # Minimal CLI: binary metric only. For continuous, use --spec.
        spec = {
            "metric_kind": "binary",
            "metric_name": args.metric_name or "conversion",
            "hypothesis": args.hypothesis or "",
            "control_n": args.control_n,
            "control_successes": args.control_successes,
            "treatment_n": args.treatment_n,
            "treatment_successes": args.treatment_successes,
        }

    if spec["metric_kind"] == "binary":
        result_obj = analyze_binary(
            control_n=spec["control_n"],
            control_successes=spec["control_successes"],
            treatment_n=spec["treatment_n"],
            treatment_successes=spec["treatment_successes"],
            metric_name=spec.get("metric_name", "conversion"),
            hypothesis=spec.get("hypothesis", ""),
        )
    elif spec["metric_kind"] == "continuous":
        result_obj = analyze_continuous(
            control_n=spec["control_n"],
            control_mean=spec["control_mean"],
            control_stddev=spec["control_stddev"],
            treatment_n=spec["treatment_n"],
            treatment_mean=spec["treatment_mean"],
            treatment_stddev=spec["treatment_stddev"],
            metric_name=spec.get("metric_name", "metric"),
            hypothesis=spec.get("hypothesis", ""),
        )
    else:
        print(f"ERROR: unknown metric_kind {spec['metric_kind']!r}", file=sys.stderr)
        return 1

    result_dict = _dataclass_to_dict(result_obj)

    if args.no_llm:
        print(json.dumps(result_dict, indent=2, default=str))
        return 0

    print(
        f"A/B analysis complete: {result_obj.directional_interpretation}, "
        f"p={result_obj.p_value:.4f}. Generating readout (format={args.format})...",
        file=sys.stderr,
    )
    system_prompt, user_prompt = build_ab_prompt(result_dict, output_format=args.format)
    try:
        llm = run_llm(system_prompt, user_prompt, model=args.model)
    except DiagnosisError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(llm.text)
        print(f"A/B readout written to {args.output}", file=sys.stderr)
    _print_usage(llm)
    print(llm.text)
    return 0


def cmd_anomaly(args: argparse.Namespace) -> int:
    from .anomaly import analyze_anomaly
    from .diagnose import DiagnosisError, run_llm
    from .prompts import build_anomaly_prompt

    before = pd.read_csv(args.before)
    after = pd.read_csv(args.after)
    attrs = args.attributes.split(",")

    findings = analyze_anomaly(
        before=before,
        after=after,
        attributes=attrs,
        metric_name=args.metric_name,
        before_label=args.before_label,
        after_label=args.after_label,
    )
    findings_dict = _dataclass_to_dict(findings)

    if args.no_llm:
        print(json.dumps(findings_dict, indent=2, default=str))
        return 0

    print(
        f"Decomposition computed across {len(findings.decompositions)} attributes. "
        f"Aggregate delta: {findings.aggregate_delta:+.4f}. "
        f"Generating narrative (format={args.format})...",
        file=sys.stderr,
    )
    system_prompt, user_prompt = build_anomaly_prompt(findings_dict, output_format=args.format)
    try:
        llm = run_llm(system_prompt, user_prompt, model=args.model)
    except DiagnosisError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(llm.text)
        print(f"Anomaly narrative written to {args.output}", file=sys.stderr)
    _print_usage(llm)
    print(llm.text)
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
    p_gen.add_argument(
        "--preset",
        choices=["ecommerce", "b2b-trial"],
        default="ecommerce",
        help="Which synthetic funnel shape to generate. 'ecommerce' is the default "
             "6-step e-commerce checkout; 'b2b-trial' is a B2B SaaS free-trial funnel "
             "with different attributes (plan_tier, region, initial_seats_claimed).",
    )
    p_gen.set_defaults(func=cmd_generate_data)

    # analyze (no LLM)
    p_an = sub.add_parser("analyze", help="Run analysis only; output findings as JSON.")
    p_an.add_argument("--data", required=True, help="Input CSV.")
    p_an.add_argument("--steps", help="Comma-separated funnel step order.")
    p_an.add_argument("--attributes", help="Comma-separated attribute columns for segmentation.")
    p_an.add_argument("--output", help="Output JSON path. If omitted, prints to stdout.")
    p_an.add_argument(
        "--detection-mode",
        choices=["threshold", "statistical"],
        default="threshold",
        help="How to flag divergent segments. 'threshold' (default) uses a fixed "
             "4pp magnitude gate. 'statistical' also requires a two-proportion "
             "z-test to clear a Bonferroni-corrected alpha across all segments scanned.",
    )
    p_an.set_defaults(func=cmd_analyze)

    # run (full pipeline)
    p_run = sub.add_parser("run", help="Run full pipeline: analysis + LLM diagnosis.")
    p_run.add_argument("--data", required=True, help="Input CSV.")
    p_run.add_argument("--steps", help="Comma-separated funnel step order.")
    p_run.add_argument("--attributes", help="Comma-separated attribute columns.")
    p_run.add_argument("--output", help="Output markdown path for the diagnosis.")
    p_run.add_argument(
        "--model",
        default="claude-sonnet-5",
        help="Anthropic model to use. Check docs.claude.com for current model IDs.",
    )
    p_run.add_argument(
        "--detection-mode",
        choices=["threshold", "statistical"],
        default="threshold",
        help="How to flag divergent segments. 'threshold' (default) uses a fixed "
             "4pp magnitude gate. 'statistical' also requires a two-proportion "
             "z-test to clear a Bonferroni-corrected alpha across all segments scanned.",
    )
    p_run.add_argument(
        "--format",
        choices=["full", "exec", "slack"],
        default="full",
        help="Output register/length. 'full' (default) is the multi-section markdown "
             "diagnosis. 'exec' is a three-bullet exec summary under 100 words. "
             "'slack' is a short casual-but-precise message.",
    )
    p_run.set_defaults(func=cmd_run)

    # retention
    p_ret = sub.add_parser("retention", help="Retention curve analysis + LLM narrative.")
    p_ret.add_argument("--signups", required=True, help="CSV with user_id, signup_date, optional cohort attrs.")
    p_ret.add_argument("--activity", required=True, help="CSV with user_id, timestamp (activity events).")
    p_ret.add_argument("--days", help="Comma-separated day-N points. Default: 1,3,7,14,30.")
    p_ret.add_argument("--data-pull-date", help="ISO date used as the cutoff for observing activity. Default: max event date.")
    p_ret.add_argument("--cohort-column", help="Signup-DataFrame column to break out cohorts by (e.g., 'acquisition_source').")
    p_ret.add_argument("--model", default="claude-sonnet-5")
    p_ret.add_argument("--output", help="Write narrative to file. Default: stdout.")
    p_ret.add_argument("--format", choices=["full", "exec", "slack"], default="full")
    p_ret.add_argument("--no-llm", action="store_true", help="Skip LLM; print structured findings as JSON.")
    p_ret.set_defaults(func=cmd_retention)

    # ab-readout
    p_ab = sub.add_parser("ab-readout", help="A/B test analysis + LLM readout.")
    p_ab.add_argument("--spec", help="JSON file with the full experiment spec. Overrides individual flags.")
    p_ab.add_argument("--control-n", type=int)
    p_ab.add_argument("--control-successes", type=int)
    p_ab.add_argument("--treatment-n", type=int)
    p_ab.add_argument("--treatment-successes", type=int)
    p_ab.add_argument("--metric-name", help="Metric name (used in the readout narrative).")
    p_ab.add_argument("--hypothesis", help="One-line hypothesis being tested.")
    p_ab.add_argument("--model", default="claude-sonnet-5")
    p_ab.add_argument("--output", help="Write readout to file. Default: stdout.")
    p_ab.add_argument("--format", choices=["full", "exec", "slack"], default="full")
    p_ab.add_argument("--no-llm", action="store_true", help="Skip LLM; print computed result as JSON.")
    p_ab.set_defaults(func=cmd_ab_readout)

    # anomaly
    p_an2 = sub.add_parser("anomaly", help="Rate/mix anomaly decomposition + LLM narrative.")
    p_an2.add_argument("--before", required=True, help="CSV: attribute columns + n + successes for the before window.")
    p_an2.add_argument("--after", required=True, help="Same schema for the after window.")
    p_an2.add_argument("--attributes", required=True, help="Comma-separated attribute columns to decompose by.")
    p_an2.add_argument("--metric-name", required=True, help="Name of the metric (e.g., 'signup_rate').")
    p_an2.add_argument("--before-label", default="before")
    p_an2.add_argument("--after-label", default="after")
    p_an2.add_argument("--model", default="claude-sonnet-5")
    p_an2.add_argument("--output", help="Write narrative to file. Default: stdout.")
    p_an2.add_argument("--format", choices=["full", "exec", "slack"], default="full")
    p_an2.add_argument("--no-llm", action="store_true", help="Skip LLM; print structured findings as JSON.")
    p_an2.set_defaults(func=cmd_anomaly)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
