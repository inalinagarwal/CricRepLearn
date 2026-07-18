# CricRepLearn

CricRepLearn is an experimental system for learning contextual T20 cricket
player representations from ball-by-ball data.

The intended system learns separate role-specific representations for batting
and bowling. These representations can then be combined with the players,
venue, competition, phase, match state, and other available context for a
future fixture. The longer-term goal is to predict player performance and
select a constrained fantasy-cricket XI.

## Project status

This repository is being rebuilt as a reproducible research project. The new
data foundation currently contains:

- Stable player identity through Cricsheet registry UUIDs
- Canonical match, squad, delivery, wicket, and player-alias tables
- Correct treatment of wides, no-balls, and pre-delivery innings state
- Source deduplication with hashes and revision metadata
- Match-level chronological train, validation, and test assignments
- Tests for identity, wicket attribution, legal balls, state, and leakage
- Rolling global, contextual, player, venue, and matchup baselines

Representation learning is now implemented as a dual-role residual multitask
network. Full validation training and architecture ablations are the current
research stage; opportunity modelling and Dream11 optimization come next.

## Planned pipeline

1. Build a canonical delivery dataset from all available T20 matches.
2. Resolve players through stable Cricsheet registry identifiers.
3. Create chronological train, validation, and test splits.
4. Establish statistical player and matchup baselines.
5. Train and evaluate contextual batting and bowling representations.
6. Model expected player opportunities, including batting position and overs.
7. Predict performance for arbitrary future fixtures.
8. Convert predictions to fantasy points and optimize the final XI.

## Repository layout

```text
src/cric_rep_learn/data/     Canonical ingestion and chronological splitting
src/cric_rep_learn/baselines/ Rolling empirical-Bayes baselines and metrics
src/cric_rep_learn/representations/ Dual-role neural player embeddings
tests/                       Data correctness and leakage tests
docs/                        Design and data documentation
```

Raw data, generated datasets, and model artifacts are intentionally excluded
from Git. They can be rebuilt locally and will later be managed through a
versioned data/artifact workflow.

## Environment

Python 3.11 or newer is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Build the canonical dataset

Place downloaded Cricsheet JSON folders below `data/`, then run:

```bash
cric-build-data --input data --output artifacts/canonical
```

The build writes compressed Parquet tables plus metadata, skipped-match
diagnostics, and a complete source manifest. Existing output is never replaced
unless `--overwrite` is supplied explicitly.

Create leakage-safe chronological assignments:

```bash
cric-split-data \
  --matches artifacts/canonical/matches.parquet \
  --output artifacts/canonical/split_manifest.parquet
```

Validate the complete serialized dataset and run the automated tests:

```bash
cric-validate-data --dataset artifacts/canonical
pytest
```

Evaluate the statistical baselines:

```bash
cric-evaluate-baselines \
  --dataset artifacts/canonical \
  --output artifacts/baselines/metrics.json
```

## Weather

Weather can be added, but it is not part of Cricsheet. It remains a separate,
provenance-tracked table keyed by match ID. A reliable historical join requires:

- verified venue latitude, longitude, and timezone
- scheduled local start time (Cricsheet generally supplies only the date)
- an historical weather provider
- the observation time and retrieval provenance

The canonical schemas reserve explicit venue-location and match-weather tables.
Weather will only be used when its timestamp would have been available before
the prediction, preventing forecast leakage.

See [`docs/data-foundation.md`](docs/data-foundation.md) for table definitions,
identity rules, validation guarantees, and the planned weather join.

See [`docs/statistical-baselines.md`](docs/statistical-baselines.md) for the
delivery targets, historical update policy, smoothing hierarchy, and initial
full-corpus results.

See [`docs/representation-learning.md`](docs/representation-learning.md) for
the dual-role embedding architecture, leakage controls, targets, checkpoint
format, and why diffusion is deferred until after predictive validation.

See [`docs/project-guide.md`](docs/project-guide.md) for the complete product
direction, code map, training guide, current limitations, and decisions that
still need review.

See [`docs/validation-analysis.md`](docs/validation-analysis.md) for the first
validation calibration, subgroup, and ablation results on the residual model.

## Data

Ball-by-ball match data is sourced from
[Cricsheet](https://cricsheet.org/). Cricsheet data is not redistributed in
this repository. Follow Cricsheet's licensing and attribution requirements
when downloading or using its datasets.
