# Statistical baselines

The baselines establish how much can be predicted without neural
representations. A learned embedding model is useful only if it improves
probabilistic forecasts on later matches.

The rolling hierarchy is:

```text
global → context → player → venue → vs_pace → matchup
```

`vs_arm_pace` and `vs_nation_arm_pace` are still computed for diagnostics and
player-card fallbacks, but they do not parent the matchup forecast. A first
full-chain eval showed `vs_pace` best and nation/arm chaining hurt matchup.

## What one dataset row means

The canonical dataset is not a collection of player career averages. Its
primary training grain is one recorded delivery:

```text
historical context before delivery
  + batter identity
  + bowler identity
  + venue and match context
  -> outcomes on the current delivery
```

Inputs currently used by the baseline are:

- canonical batter and bowler IDs
- gender and club/international team type
- venue
- first innings, chase, or super over
- powerplay, middle, or death phase
- wicket-pressure bucket: 0–2, 3–5, or 6–9 wickets lost

The outcome heads are:

- batter runs: classes 0 through 7
- extras: classes 0 through 6 and a 7+ class
- delivery legality: legal, wide, or no-ball
- whether the batter was dismissed
- whether the wicket was credited to the bowler

Runs, wickets, legal balls, and all other state stored on the delivery are
pre-delivery values. Current-delivery outcomes are targets only.

Related canonical tables describe matches, innings, declared players, wickets,
replacements, reviews, aliases, source provenance, and chronological split
assignments. See [`data-foundation.md`](data-foundation.md).

## Historical update policy

Predictions for date `D` use every delivery strictly before `D`. There is no
recent-match window.

Cricsheet has match dates but generally no start times. All matches on the same
calendar date are therefore predicted first and added to history only
afterwards. One same-day match can never leak into another.

The evaluation is rolling:

1. Training dates initialize the historical aggregates.
2. Each validation date is predicted, measured, then added to history.
3. Each test date is predicted using training, validation, and earlier test
   dates, then added to history.

This mirrors the intended real system: when predicting a future fixture, all
completed historical matches are available.

## Baseline hierarchy

Each level uses empirical-Bayes smoothing toward the less specific level:

1. **Global** — all historical deliveries.
2. **Context** — gender, team type, innings group, phase, and wicket pressure.
3. **Player** — separate batter and bowler evidence, shrunk to context.
4. **Venue** — venue evidence blended with the player forecast.
5. **Matchup** — direct batter-bowler evidence, shrunk to the player/venue
   forecast.

For categorical outcomes, posterior probabilities are:

```text
(observed class counts + prior strength × prior probabilities)
----------------------------------------------------------------
                 observations + prior strength
```

Binary outcomes use the equivalent beta-binomial posterior mean.

The initial smoothing strengths are intentionally fixed rather than optimized
against the test set:

- context: 2,000 deliveries
- player: 200 deliveries
- venue: 500 deliveries
- direct matchup: 60 deliveries

These values can later be tuned on validation data only.

## Metrics

The evaluator reports:

- multiclass log loss and Brier score
- binary log loss and Brier score
- expected-runs absolute error
- calibration error
- historical coverage and mean prior evidence

Log loss is the primary metric because the system needs complete, calibrated
outcome distributions for match simulation—not only the most likely class.

## Initial full-corpus result

The rolling test period contains 276,439 deliveries.

For batter runs, test log loss improves from:

```text
global   1.28033
context  1.24633
player   1.23710
venue    1.23702
matchup  1.23691
```

The direct matchup layer adds only a very small runs improvement and slightly
worsens dismissal/wicket forecasts relative to the player layer. This is not a
failure: only about 34.4% of test deliveries have any prior direct matchup, and
the mean is just 3.7 previous deliveries.

That sparsity is the central reason to learn representations. The future model
must generalize from how a batter performs against many bowlers and how a
bowler performs against many batters, instead of relying on a tiny direct-pair
sample.

## Reproduce

```bash
cric-evaluate-baselines \
  --dataset artifacts/canonical \
  --output artifacts/baselines/metrics.json
```

The metrics artifact is generated locally and excluded from Git.
