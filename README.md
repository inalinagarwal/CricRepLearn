# CricRepLearn

CricRepLearn is an experimental system for learning contextual T20 cricket
player representations from ball-by-ball data.

The intended system learns separate role-specific representations for batting
and bowling. These representations can then be combined with the players,
venue, competition, phase, match state, and other available context for a
future fixture. The longer-term goal is to predict player performance and
select a constrained fantasy-cricket XI.

## Project status

This repository is an early research prototype. It currently contains:

- Cricsheet JSON parsing and delivery-level feature preparation
- Separate batter and bowler embedding tables
- Ball-outcome model training
- Embedding export
- Prototype fixture-level matchup inference

The current model and example rankings have not yet been validated against
strong statistical baselines or a chronological test set. They should not be
treated as reliable forecasts.

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
build_ball_universe.py       Parse Cricsheet JSON into delivery rows
create_playerid.py           Create current player and venue mappings
context_aware_dataset.py     Add innings-state features
prepare_trainingdata.py      Build current train/validation datasets
model.py                     Representation model
dataset.py                   PyTorch dataset
train.py                     Model training
export_embeddings.py         Export learned role embeddings
match_predictor.py           Prototype fixture matchup inference
predict_match.py             Fixture prediction CLI
examples/                    Small example fixture definitions
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
python -m pip install -r requirements.txt
```

## Current prototype commands

Train the current model:

```bash
python train.py
```

Export its embeddings:

```bash
python export_embeddings.py
```

Run the example fixture:

```bash
python predict_match.py --match examples/ind_vs_eng.json
```

These commands document the existing prototype. The data preparation,
evaluation, and modelling pipeline will be revised as the project develops.

## Data

Ball-by-ball match data is sourced from
[Cricsheet](https://cricsheet.org/). Cricsheet data is not redistributed in
this repository. Follow Cricsheet's licensing and attribution requirements
when downloading or using its datasets.
