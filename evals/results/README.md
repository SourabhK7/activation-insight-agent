# Evaluation results

`python evals/run_eval.py` writes these files.

- `latest.md` is the readable summary of the most recent run.
- `raw_runs.jsonl` has one line per (arm, seed) run with the full diagnosis, the score for each criterion, and the judge's reasons. If a score looks wrong, you can open the file and check that row.

Both are checked in so you can see the results without running anything. To regenerate:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/run_eval.py --n 10 --reset
```
