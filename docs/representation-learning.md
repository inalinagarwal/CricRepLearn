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
percent of known player IDs are independently replaced by it, teaching a useful
cold-start fallback without leaking validation or test identities.

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

The shared interaction trunk feeds five heads:

- categorical batter-runs logits
- categorical extras logits
- categorical legality logits
- batter-dismissal logit
- bowler-wicket logit

All heads are trained jointly with ordinary cross-entropy or binary
cross-entropy. Class weighting is intentionally avoided in the first model
because it changes probability calibration. Rare-event discrimination and
calibration are evaluated explicitly.

## Evaluation policy

The first model is trained only on the training split and selected only on
validation metrics. Frozen neural results and rolling historical-baseline
results are labelled separately until matched frozen and operational
prequential protocols are implemented. Metrics include:

- runs log loss and Brier score
- expected-runs error
- dismissal and wicket log loss/Brier score
- legality and extras log loss

The strongest current validation runs baseline is approximately `1.24241` log
loss. A neural model is not considered useful merely because its embeddings
look plausible; it must improve held-out probability forecasts without harming
calibration.

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
  --output artifacts/model-data

cric-train-representations \
  --model-data artifacts/model-data \
  --output artifacts/checkpoints/representations \
  --baseline-metrics artifacts/baselines/metrics.json

cric-export-embeddings \
  --checkpoint artifacts/checkpoints/representations/best.pt \
  --output artifacts/embeddings
```
