# Player representation learning

The first neural model learns reusable batting and bowling vectors directly
from delivery-outcome prediction. It is deliberately simpler than a generative
diffusion model so improvements over the statistical baselines can be
attributed to the player representations.

## Identity and role vectors

Separate train-only batting and bowling vocabularies map each Cricsheet person
ID to a role index:

```text
canonical player ID
├── batting role index → 32-dimensional batting embedding
└── bowling role index → 32-dimensional bowling embedding
```

An all-rounder receives gradients in both tables. A player known only as a
batter cannot accidentally receive an untrained bowling vector. Exported rows
include role-specific delivery and match counts plus first and last train dates.

Index zero is a learned role-specific unknown vector. During training, five
percent of known player IDs are independently replaced by it. Known venues use
the same five-percent dropout policy to train `UNK_VENUE`. This teaches useful
cold-start fallbacks without leaking validation or test identities.

## Encoded model data

`cric-build-model-data` derives all vocabularies and numeric normalization
statistics from the training split only.

Categorical inputs:

- batter and bowler player indices
- venue
- powerplay, middle, or death phase
- gender
- club or international team type
- first innings, chase, or super over
- wickets-lost bucket

Numeric pre-delivery inputs:

- score
- wickets
- innings progress
- current run rate
- target runs remaining
- required run rate
- whether a target exists
- `log1p` prior batter, bowler, venue, and direct-matchup delivery evidence

The dataset builder also runs the historical empirical-Bayes model in strict
date order. Residual priors are configurable with `--baseline-level`. The
current embedding-stress experiment uses **context**-level probabilities
(gender, team type, innings group, phase, wicket pressure) so player identity
cannot hide inside the prior. Matchup-level residuals remain available for
delivery-forecast comparisons. All matches on one date are predicted before
that date is added to history.

Targets match the statistical baseline:

- batter runs: 0 through 7
- extras amount: 0 through 6 and 7+
- legality: legal, wide, or no-ball
- batter dismissal
- bowler-attributed wicket

The test dataset is encoded but must not be evaluated until the architecture,
loss weights, and training choices are frozen using validation results.

## Architecture

For each delivery, the interaction network receives:

```text
batting vector
bowling vector
element-wise product
absolute vector difference
venue/context embeddings
projected numeric state
```

The element-wise product and difference let the network learn compatibility
between styles without storing a direct pair embedding. An unseen
batter-bowler pair can therefore still be predicted from both players' broader
histories.

The shared interaction trunk feeds five residual heads:

- categorical batter-runs residual logits
- categorical extras residual logits
- categorical legality residual logits
- batter-dismissal residual logit
- bowler-wicket residual logit

Each final prediction is:

```text
final logits = log empirical-Bayes probability + neural residual
```

Residual heads start near zero, so the initial network reproduces the
calibrated historical forecast. The embeddings must learn incremental signal
instead of relearning global outcome frequencies.

All heads are trained jointly with ordinary cross-entropy or binary
cross-entropy. Class weighting is intentionally avoided in the first model
because it changes probability calibration. Rare-event discrimination and
calibration are evaluated explicitly.

## Evaluation policy

The first model is trained only on the training split and selected only on
validation metrics. The current matched protocol is operational prequential:
neural parameters remain frozen during validation while empirical-Bayes
features update after each complete date under the same policy used by the
baseline evaluator. Metrics include:

- runs log loss and Brier score
- expected-runs error
- dismissal and wicket log loss/Brier score
- legality and extras log loss

The residual model can consume any hierarchy level from the empirical-Bayes
baselines. The matchup residual improved validation runs log loss from about
`1.24255` to `1.23048`, but a no-player retrain matched that gain. The
context-only residual redesign then improved from context EB `1.25074` to
`1.23266` with players, versus `1.23417` without players—an embedding gap of
only `~0.0015`, below the `0.005` bar. Embeddings therefore still do not earn
their keep under delivery multiclass residual learning. See
[`validation-analysis.md`](validation-analysis.md).

A neural model is not considered useful merely because its embeddings look
plausible; it must improve held-out probability forecasts without harming
calibration, and player IDs must be necessary for that improvement.

After architecture selection, a final historical model can be retrained using
all matches available before a real fixture. New players remain unknown until
they accumulate data and the model is retrained.

## Checkpoint and export

The best checkpoint owns:

- model and training configuration
- model weights
- complete player and venue vocabularies
- data-manifest hashes
- training evidence counts
- validation metrics

`cric-export-embeddings` writes checkpoint-bound player vectors to Parquet with
Cricsheet IDs, display names, role evidence, and checkpoint provenance. This
prevents embeddings from being paired with a newer or incompatible mapping.

## Why diffusion is deferred

Diffusion is also neural modelling, but its strength is iterative generation of
complex high-dimensional distributions. Delivery outcomes here are compact
categorical targets where calibration and attribution matter more.

A conditional generative model may later simulate complete innings or player
score distributions. It can be conditioned on the validated player embeddings;
it should not replace the simpler model before those embeddings have proved
predictive value.

## Commands

```bash
cric-build-model-data \
  --canonical artifacts/canonical \
  --output artifacts/model-data-context-residual \
  --baseline-level context

cric-train-nxt \
  --model-data artifacts/model-data-context-residual \
  --output artifacts/checkpoints/context-residual-full \
  --baseline-metrics artifacts/baselines/metrics.json \
  --device mps

cric-export-embeddings \
  --checkpoint artifacts/checkpoints/context-residual-full/best.pt \
  --output artifacts/embeddings
```
