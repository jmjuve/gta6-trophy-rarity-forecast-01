# gta6-trophy-rarity-forecast-01

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jmjuve/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jmjuve/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PS5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy — scored by pinball loss on the logit scale.

## How it works

Exophase rates are a ratio — earners over players tracked — and both move. On launch night the tracked population was 638 ultra-hardcore players, so nearly every trophy sat at the same ~1.1%; by the next morning the denominator had exploded and most rates had collapsed to 0.03–0.73%. So the program:

1. **Anchors on the last read** of the case history.
2. **Fits the recent trend only** — least squares of logit rate against ln(1 + hours since release) over the last 24 h of reads. Older reads carry the launch-night collapse, which must never be extrapolated forward.
3. **Shrinks that trend toward a per-trophy-class prior**, read from each trophy's own description and tier at run time: early story climbs fastest, the final mission / platinum / 100% completion drift down (nobody finishes a ~200-hour game in week one while the denominator grows), long grinds stay near zero, quick activities climb moderately.
4. **Treats the floor specially**: a trophy at the 2-decimal floor (0.00–0.05%) gets its median blended toward its class's week-one level, because a logit drift from −9 cannot express that *some* players will earn it by the target.
5. **Calibrates the spread per class** (0.40 logits for near-frozen grinds up to 0.70 for vague hidden misc trophies), widened a little with the horizon, with the fit's residual noise, and more when the history was flat — the flat history hides the next Exophase refresh jump. That is the edge over the house persistence baseline, whose spread is 0.5·√days ≈ 1.06 logits at the 4.52-day final horizon.

Deterministic (no clock, no randomness), standard library only, offline, ~0.3 s per case. `main.ts` is the whole entry.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory and writes one `<id>.out` per case to the output directory: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`.

## Validation

- Practice cases p00/p01: valid schema (55 rows, monotone quantiles in [0,100], ≤65,536 B), byte-identical across two runs, ~0.2–0.4 s per case.
- On a dilution-based estimate of the practice answers (launch-night earners grown modestly, denominator 638 → 3,868): loss ≈ 0.91 vs persistence baseline ≈ 1.28. The real practice answers are known only to the server — this is an estimate, not a measurement.
- On three synthetic final cases (steady progression / denominator flood / fast progression, with Exophase refresh jumps and per-class progression curves): loss 0.33 / 0.44 / 0.45 vs persistence 0.76 / 0.86 / 0.72 and the curve baseline 0.92 / 0.80 / 1.71.

<sub>Entered on [make.fun](https://make.fun/?ref=vt2v2aqg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
