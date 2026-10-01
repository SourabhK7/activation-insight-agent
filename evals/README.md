# Evals

This tests the main design choice in the repo: does it actually help to keep the arithmetic away from the LLM?

## The setup

Two versions, the same synthetic data, and the same rubric.

The structured version is what the repo does: raw events go to pandas, pandas computes the findings, and Claude writes the diagnosis from the findings JSON.

The naive version is the comparison: raw events go to pandas, pandas only aggregates step counts and per-segment counts, and Claude has to compute the conversion rates itself and then diagnose. I tried to make this a fair baseline rather than a strawman. Both versions get the same information, and the only difference is who computes the rates.

An LLM judge (Claude) scores both diagnoses against a 6-criterion rubric ([rubric.md](rubric.md)). The judge gets the ground truth for each run's dataset, so criterion 5 (numerical accuracy) is graded against the real numbers and not against the judge's own arithmetic.

## Why I built it

The repo started from the claim that doing the math in pandas avoids a whole class of bugs where the LLM confidently computes the wrong conversion rate. That's testable, so I tested it.

If the structured version clearly wins, especially on criterion 5, the claim holds. If they score the same, the extra structure isn't buying accuracy and the code could be simpler. Either answer is useful. (It turned out to be the second one, at least for accuracy. See [`results/latest.md`](results/latest.md).)

## Running it

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python evals/run_eval.py --n 10
```

Options:

- `--n 10` is runs per version (default 10). Each run uses a fresh seed, so the comparison isn't riding on one lucky dataset.
- `--n-users 20000` is users per synthetic dataset. Smaller is faster and cheaper. The default is smaller than the 50k demo on purpose, since 10 runs times 2 versions adds up.
- `--diagnosis-model claude-sonnet-5` is the model writing the diagnoses.
- `--judge-model claude-sonnet-5` is the model grading them. It can be the same model. Using a different one is easier to defend but costs more.
- `--arms structured,naive` lets you rerun only one version.
- `--reset` deletes `results/raw_runs.jsonl` before starting.

With `--n 10 --n-users 20000` on Sonnet 5 it costs about $1 and takes 4 to 8 minutes, depending on API latency.

## Output

`results/latest.md` is the readable summary: mean total per version, standard deviation, and per-criterion means.

`results/raw_runs.jsonl` has one line per run with the full diagnosis, the judge's scores, and its reasons for each criterion. If a score looks off, you can open the file and read exactly why it got that score.

## Reading the results

Start with criterion 5, since that's the direct test. A big gap on criterion 5 with little difference elsewhere would mean the structured design specifically fixes the numbers without hurting anything else.

Check the variance too. With 10 runs the interval on each mean is wide, so I'd treat a difference under about 1 point (out of 12) as noise.

Criterion 4 (no fabrication) is the next one I'd look at. If the naive version computes its own rates and then also invents patterns to explain them, the two problems stack.

## Limitations

The full list is in [rubric.md](rubric.md#limitations-of-llm-as-judge). The short version:

1. An LLM judge can be wrong, and it shares training data with the models it's grading.
2. 10 runs is small. This is a directional check, not a benchmark.
3. The ground truth for criterion 5 comes from the same pandas pipeline the structured version uses, which gives it a slight edge whenever the "right" number depends on exact rounding. That's why criterion 5 allows 1 point of tolerance, so rounding can't decide it.
4. The naive baseline is fair but not the most sophisticated option. A real data scientist with the aggregated counts would compute the rates in a notebook. This compares "counts plus the LLM does the math" with "findings plus the LLM only writes," which is the tradeoff in this repo. It doesn't cover an agent that writes and runs its own code, which is a different design.

None of this is a general claim about LLM arithmetic. It's about this pipeline, tested with a rubric built for this task.
