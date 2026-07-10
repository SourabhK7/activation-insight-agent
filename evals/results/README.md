# Evaluation results

This directory is populated by `python evals/run_eval.py`.

- `latest.md` — human-readable aggregate summary from the most recent run.
- `raw_runs.jsonl` — one line per (arm, seed) run, containing the full diagnosis text, per-criterion scores, and judge reasons. Auditable — if a specific score looks wrong, open the file and inspect the row.

Both files are checked in so results are visible without running the harness. To regenerate:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/run_eval.py --n 10 --reset
```
