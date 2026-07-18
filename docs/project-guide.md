# CricRepLearn: project direction and code guide

This document explains what we are building, what has been implemented, how
the code works, what the current model can and cannot claim, and which
decisions still need review.

## 1. The actual product goal

The long-term product is a reusable pre-match system for any T20 fixture:

1. Take the expected players from two teams.
2. Represent every player separately as a batter and as a bowler.
3. Combine those representations with the opposing players and match context.
4. Predict complete player-performance distributions, not only averages.
5. Convert simulated performances into Dream11 points.
6. Select the best legal XI under role, team, captain, vice-captain, and credit
   constraints.

For an example such as Rohit Sharma against Australia, the eventual system
should evaluate Rohit against each likely Australian bowler, at the expected
venue and in the expected match conditions. It should then account for how
many balls Rohit is likely to face from each bowler. The same process applies
to every batter and bowler in the fixture.

The system is generic. MI versus RCB, India versus England, and any other T20
fixture use the same trained components.

## 2. An important interpretation of an embedding

A player embedding is not intended to literally store a venue, weather report,
or one direct matchup result. It is a learned vector describing reusable
performance tendencies inferred across many historical interactions.

The final prediction combines:

```text
player tendencies
+ opposing player tendencies
+ their learned interaction
+ venue and match context
+ strictly historical statistical evidence
+ current pre-delivery state
```

This distinction matters. A static Rohit batting vector can be reused at
Canberra or Mumbai, but venue, weather, opposition, innings state, and the
prediction date are separate inputs. This lets the model distinguish enduring
player tendencies from temporary context.

All-rounders have two independent vectors. Hardik Pandya's batting evidence
updates his batting vector; his bowling evidence updates his bowling vector.
Both vectors retain the same canonical Cricsheet person ID.

## 3. What is implemented today

The repository currently has three working layers:

### Layer A: canonical data

Raw Cricsheet JSON is converted into stable Parquet tables. Player identity,
delivery legality, wicket attribution, pre-delivery state, source provenance,
and chronological splits are validated.

### Layer B: statistical baselines

A rolling hierarchical empirical-Bayes model predicts delivery outcomes from
global, context, player, venue, and direct-matchup history. It establishes the
minimum performance that a neural model must beat.

### Layer C: neural representations

A PyTorch residual multitask network learns separate 32-dimensional batting
and bowling vectors. It starts from the empirical-Bayes probabilities and
learns only a correction:

```text
final logits = baseline logits + neural residual logits
```

This is the current research stage. Opportunity modelling, complete innings
simulation, fantasy scoring, XI optimization, and weather ingestion have not
yet been implemented.

## 4. Data scale and current splits

The current encoded delivery dataset contains:

```text
training:     2,225,144 deliveries
validation:    278,408 deliveries
test:          276,439 deliveries
total:       2,779,991 deliveries
```

The train-only role vocabularies currently contain:

```text
batters: 8,759
bowlers: 6,454
venues:    544
```

There are 9,358 unique people across the two role vocabularies. A player may
appear in one or both role tables.

Complete calendar dates are assigned to one split:

```text
every training date < every validation date < every test date
```

The test period remains reserved until architecture and training choices are
frozen using validation.

## 5. What one model example means

The training grain is one recorded delivery.

Inputs available before the delivery include:

- canonical batter and bowler IDs
- venue
- gender and club/international match type
- first innings, chase, or super over
- powerplay, middle, or death phase
- wickets-lost pressure bucket
- score and wickets before the delivery
- innings progress
- current run rate
- target presence, runs remaining, and required rate
- prior batter, bowler, venue, and matchup evidence
- rolling empirical-Bayes probabilities

Targets observed on the delivery are:

- batter runs: classes 0 through 7
- extras: classes 0 through 6 and 7+
- legality: legal, wide, or no-ball
- batter dismissed: yes or no
- wicket credited to the bowler: yes or no

Current-delivery outcomes are never included in the inputs.

## 6. Canonical data layer

### Stable identity

Cricsheet's `info.registry.people` mapping supplies the canonical
eight-character person ID. Names are aliases for display only. The strict
parser rejects unregistered participants instead of silently creating duplicate
identities.

### Canonical tables

- `matches.parquet`: match date, venue, teams, toss, result, event, source
  revision, and source hash.
- `match_players.parquet`: people listed for each team. This is not assumed to
  be a confirmed starting XI.
- `innings.parquet`: innings targets, penalties, super overs, powerplays,
  absent-hurt players, and miscounted-over metadata.
- `deliveries.parquet`: one row per delivery attempt, including illegal balls,
  pre-delivery state, participant IDs, runs, extras, phase, and wicket counts.
- `wickets.parquet`: one row per dismissal with team-wicket and
  bowler-attribution semantics kept separate.
- `player_aliases.parquet`: known display names and observation dates for each
  canonical person ID.
- `replacements.parquet`: substitutions and role replacements attached to the
  delivery where they occurred.
- `reviews.parquet`: decision reviews attached to their event time.
- `source_manifest.parquet`: every source file, hash, data version, revision,
  and deduplication decision.
- `split_manifest.parquet`: one chronological split assignment per match.

### Critical state semantics

`score_before`, `wickets_before`, and `legal_balls_before` describe the state
before the current delivery. Wides and no-balls increment the attempt count but
not the legal-ball count. Fractional over notation such as `5.3` means five
completed overs and three balls, not 5.3 decimal overs.

Run-outs and non-bowler dismissals remain team wickets but are not credited to
the bowler. This distinction supplies two different neural targets:
`batter_dismissed` and `bowler_wicket`.

## 7. Statistical baseline

The historical model predicts from progressively more specific evidence:

1. Global historical outcomes.
2. Match context: gender, team type, innings group, phase, and wicket pressure.
3. Batter and bowler histories.
4. Venue history blended with the player forecast.
5. Direct batter-bowler history.

Sparse levels shrink toward their parent:

```text
posterior =
    (observed counts + prior strength × prior probabilities)
    / (observations + prior strength)
```

Current equivalent sample sizes are:

```text
context: 2,000
player:    200
venue:     500
matchup:    60
```

These are initial fixed choices and may later be tuned on validation only.

### Rolling history policy

For date `D`, predictions use deliveries strictly before `D`. Because
Cricsheet generally lacks start times, every match on date `D` is predicted
before any result from date `D` updates history.

The operational evaluation proceeds as:

```text
predict complete date → measure complete date → update complete date
```

Validation and test are therefore realistic prequential evaluations: the model
may use all genuinely completed earlier dates.

### Current baseline result

On 276,439 test deliveries, batter-runs log loss was:

```text
global   1.28033
context  1.24633
player   1.23710
venue    1.23702
matchup  1.23691
```

Only about 34.4% of test deliveries have previous direct-matchup evidence, with
an average of only 3.7 prior balls. The tiny matchup improvement demonstrates
why direct historical averages are insufficient and why reusable player
representations may help.

## 8. Leakage-safe neural feature generation

`baseline_features.py` replays all 2,779,991 deliveries chronologically. For
each delivery it stores:

- eight batter-run probabilities
- eight extras probabilities
- three legality probabilities
- batter-dismissal probability
- bowler-wicket probability
- `log1p` prior batter, bowler, venue, and matchup delivery counts

These values are calculated before updating the delivery's calendar date.
They are joined back to encoded examples by:

```text
match_id + innings + attempt_index_in_innings
```

This gives the neural model a calibrated starting forecast and explicit sample
size. Embedding magnitude does not have to become an unreliable proxy for how
many observations a player has.

## 9. Neural architecture

### Role vocabularies and cold start

Batters and bowlers use separate train-only vocabularies. This prevents a
player known only as a batter from receiving a random, untrained bowling
vector.

Index zero is a learned role-specific unknown vector. During training, 5% of
known batter IDs and 5% of known bowler IDs are independently replaced by
unknown. This ID dropout teaches the model a genuine cold-start fallback.
Five percent of known venue IDs are likewise replaced by `UNK_VENUE`, which
trains the venue fallback used for later unseen grounds.

Validation and test identities never create new random rows. An unseen batter
uses `UNK_BATTER`; an unseen bowler uses `UNK_BOWLER`.

### Interaction network

For a delivery, the network concatenates:

```text
32d batting embedding
32d bowling embedding
element-wise batting × bowling product
absolute batting − bowling difference
venue embedding
phase embedding
gender embedding
team-type embedding
innings-group embedding
wicket-pressure embedding
projected numeric and evidence features
```

There is deliberately no direct batter-bowler pair embedding. A pair embedding
would memorize frequent matchups and fail on unseen pairs. Product and
difference interactions require the model to compose the two reusable player
representations.

The trunk is:

```text
input
→ Linear(256)
→ LayerNorm
→ SiLU
→ Dropout(0.10)
→ Linear(128)
→ LayerNorm
→ SiLU
→ Dropout(0.10)
```

Five output heads produce neural residuals for runs, extras, legality,
dismissal, and bowler wicket.

For multiclass outcomes:

```text
final_logits = log(baseline_probabilities) + residual_logits
```

For binary outcomes:

```text
final_logit = logit(baseline_probability) + residual_logit
```

Residual head weights initialize close to zero. Before learning, predictions
are therefore close to the calibrated baseline.

### Objective and regularization

The five heads use unweighted cross-entropy or binary cross-entropy at natural
event frequency. Class weighting, focal loss, and oversampling are avoided
because the output probabilities must remain calibrated for simulation.

Current regularization:

- AdamW optimizer
- learning rate `3e-4`
- trunk weight decay `1e-4`
- embedding weight decay `1e-3`
- dropout `0.10`
- ID dropout `0.05`
- gradient clipping at norm `5`
- early stopping after two validation epochs without improvement

## 10. How training and evaluation work

Training examples are shuffled, but every rolling feature was computed in
chronological order beforehand. Model parameters are updated only from the
training split.

After each epoch:

1. The complete validation split is evaluated without gradient updates.
2. Per-head log loss, Brier score, and expected-runs error are reported.
3. The checkpoint is replaced only when total validation loss improves.
4. Training stops early after two non-improving validation epochs.

The neural parameters remain frozen while validation baseline features update
through earlier validation dates. This is the matched operational prequential
comparison used by the baseline.

The test split must not be evaluated merely to decide architecture, dimensions,
loss weights, or epoch count.

## 11. Running training yourself on the M1 GPU

The earlier assistant-started run has been stopped. In a normal terminal at the
repository root, run:

```bash
source .venv/bin/activate

caffeinate -i cric-train-representations \
  --model-data artifacts/model-data \
  --output artifacts/checkpoints/representations-residual-mps-user \
  --baseline-metrics artifacts/baselines/metrics.json \
  --epochs 8 \
  --batch-size 4096 \
  --device mps
```

`caffeinate -i` prevents idle sleep while the command runs. The trainer prints
training and validation completion percentages in 5% steps, followed by one
JSON metric summary per epoch.

If MPS is unavailable, the explicit `--device mps` argument raises an error
instead of silently training on CPU.

After training:

```bash
cric-export-embeddings \
  --checkpoint artifacts/checkpoints/representations-residual-mps-user/best.pt \
  --output artifacts/embeddings-residual-mps-user
```

The best checkpoint, history, exported Parquet vectors, and metadata are under
`artifacts/`, which is intentionally excluded from Git.

## 12. How to read the epoch output

Important validation fields are:

- `runs_log_loss`: primary batter-run probability metric; lower is better.
- `runs_brier`: squared probability error across run classes; lower is better.
- `runs_expected_mae`: error in expected batter runs per delivery.
- `extras_log_loss`: extras distribution quality.
- `legality_log_loss`: legal/wide/no-ball distribution quality.
- `batter_dismissal_log_loss` and `batter_dismissal_brier`.
- `bowler_wicket_log_loss` and `bowler_wicket_brier`.
- `total_loss`: unweighted sum used for early stopping.

The matchup baseline consumed by the residual model has validation runs log
loss of approximately `1.24255`. The strongest baseline level is venue at
`1.24241`. A useful neural model should improve held-out metrics without
degrading calibration. One metric alone is not sufficient.

The completed MPS run stopped after epoch 4 and selected epoch 2. Against the
matchup baseline, validation runs log loss improved from `1.24255` to
`1.23048`. Extras, legality, dismissal, and bowler-wicket log loss also
improved. Total five-head loss improved from approximately `2.15226` to
`2.13323`.

Expected-runs MAE moved slightly in the wrong direction, from approximately
`1.09351` to `1.09579`. Calibration ECE also worsened slightly.

Retrain ablations changed the interpretation:

- Residual over baseline helps versus a standalone neural model (`1.23403`).
- A residual model trained with player IDs forced unknown reached `1.22933`,
  matching or beating the full dual-role model.
- A later context-only residual redesign still failed the embedding test:
  players `1.23266` versus no-players `1.23417` (gap `~0.0015`, below the
  `0.005` bar), while both beat context EB `1.25074`.
- Current exported embeddings are therefore **not** validated as the source of
  predictive gain. See [`validation-analysis.md`](validation-analysis.md).

The earlier standalone one-epoch result of `1.23883` preceded the matched
residual architecture and cold-start corrections and is retained only as an
obsolete experiment.

## 13. Checkpoint and embedding export

`best.pt` stores:

- model weights
- model and training configuration
- role and venue vocabularies
- encoded-data manifest and hashes
- best epoch
- validation metrics
- baseline validation reference

The exported Parquet file has one row per learned player-role combination:

- role and role index
- canonical player ID and display name
- role delivery and match counts
- first and last training date
- embedding norm
- 32-dimensional vector

A player who batted and bowled has two rows. Export is tied to the checkpoint's
own mappings so vectors cannot accidentally be interpreted with a newer
vocabulary.

## 14. Code map

### Repository configuration

- `pyproject.toml`: package metadata, Python dependencies, development tools,
  and all `cric-*` console commands.
- `.gitignore`: excludes raw data, generated Parquet files, checkpoints, and
  local environments.
- `README.md`: short project entry point and setup instructions.

### `src/cric_rep_learn/data/`

- `schema.py`: defines every Arrow schema and the canonical schema version.
- `parser.py`: converts one Cricsheet JSON match into canonical rows; resolves
  IDs, reconstructs pre-delivery state, classifies phases, and parses wickets,
  reviews, replacements, penalties, targets, and edge cases.
- `build.py`: discovers JSON files, resolves duplicate match revisions, hashes
  sources, buffers rows, and writes canonical Parquet tables and manifests.
- `split.py`: assigns complete dates and matches to chronological
  train/validation/test periods and validates isolation.
- `validate.py`: checks IDs, references, run arithmetic, legal-ball state,
  attempt counters, wicket attribution, and reconstructed innings state.
- `__init__.py`: exposes the public data parser types.

### `src/cric_rep_learn/baselines/`

- `historical.py`: stores rolling outcome counts, builds hierarchy keys,
  performs empirical-Bayes smoothing, and returns probabilities plus evidence.
- `metrics.py`: streaming multiclass and binary log loss, Brier score,
  expected-value error, and calibration calculations.
- `evaluate.py`: reads deliveries in date order, predicts complete dates before
  updates, evaluates every baseline level, and writes metrics JSON.
- `__init__.py`: exposes baseline public classes.

### `src/cric_rep_learn/representations/`

- `baseline_features.py`: generates strictly historical per-delivery baseline
  probabilities and evidence for residual learning.
- `data.py`: builds train-only role/venue vocabularies, fits numeric
  normalization on training only, joins rolling features, writes encoded split
  Parquet files, and loads them into PyTorch tensors.
- `model.py`: defines role embeddings, context embeddings, interaction trunk,
  residual heads, ID dropout, and baseline-logit composition.
- `train.py`: chooses CPU/CUDA/MPS, seeds training, displays batch percentage,
  computes multitask loss and metrics, applies early stopping, and saves the
  best checkpoint.
- `export.py`: loads a checkpoint, extracts role vectors, and writes
  checkpoint-bound Parquet and JSON metadata.
- `__init__.py`: exposes the representation model and configuration.

### `tests/`

- `test_parser.py`: parser state, illegal-ball, wicket-credit, identity,
  penalties, replacements, reviews, and unknown-fielder edge cases.
- `test_split.py`: complete-date and complete-match chronological isolation.
- `test_baselines.py`: unseen fallback, smoothing behavior, metrics, and
  same-date no-leakage policy.
- `test_representations.py`: tensor layout, role-specific train-only
  vocabulary, unknown-role learning, model shapes and gradients, and exact
  baseline recovery when the neural residual is zero.

## 15. What is deliberately not implemented yet

### Opportunity

Delivery quality alone does not predict fantasy totals. We still need:

- probability of being selected in the XI
- batting position
- probability of reaching the crease
- expected balls faced
- expected bowling overs and phase allocation
- substitute and impact-player rules

### Match simulation

Dream11 needs match-level correlated outcomes, not independent expected values.
The simulator must update score, wickets, legal balls, strike, target state,
batting availability, and bowling allocation after each sampled event.

### Fantasy optimization

After simulation we need the exact platform scoring rules and constraints:

- role minimums and maximums
- maximum players from one real team
- credits
- captain and vice-captain multipliers
- lineup announcement and substitution rules

The optimizer should maximize an explicit objective such as expected points,
cash-game floor, or tournament upside.

### Weather

Cricsheet does not provide reliable match weather or usually even scheduled
start time. Weather needs a separate provenance-tracked pipeline with verified
venue coordinates, timezone, start time, provider timestamp, and historical
availability rules.

## 16. Known risks and open technical work

1. A static career embedding can mix player development, decline, and role
   changes. A later model should add a strictly historical recent-form or
   time-conditioned residual without replacing the career vector.
2. The dismissal and bowler-wicket heads are independent and can assign small
   probability to impossible combinations. A joint dismissal-attribution head
   should be tested before full simulation.
3. Runs, extras, and legality heads are not yet constrained as one coherent
   event generator. Simulation will require legal combinations, for example no
   batter runs on a wide and nonzero extras on a no-ball.
4. Validation currently uses the operational rolling protocol. A second frozen
   transfer protocol should be reported for research clarity.
5. Calibration should eventually use a dedicated late-validation calibration
   period rather than the same data used for architecture selection.
6. Confidence intervals and player-macro metrics are not yet implemented.
7. Embeddings may encode era, gender, competition, or observation frequency.
   Norm and principal-component audits are required before interpretation.
8. Team selection requires confirmed squads and roles from an external fixture
   source; Cricsheet history does not provide future lineups.
9. MPS is substantially faster on this machine but seeded MPS execution is not
   guaranteed to be bit-for-bit deterministic.
10. Checkpoints produced before the role-specific residual architecture are
    incompatible with the current model and should be treated as obsolete
    experiments.

## 17. Minimum experiments before moving downstream

The current residual model should be compared against:

1. Context and baseline only, with no player IDs.
2. Batter embedding only.
3. Bowler embedding only.
4. Both embeddings without product/difference interaction.
5. Both embeddings with the current interaction.
6. Standalone neural logits versus baseline-residual logits.
7. No venue.
8. No detailed match state.
9. No ID dropout.
10. Embedding dimensions 16, 32, and 64.
11. Independent versus joint dismissal attribution.
12. Real IDs versus shuffled-player-ID control.

These tests determine whether the learned vectors contain transferable player
signal rather than merely increasing model capacity.

## 18. Proposed roadmap from here

### Milestone 1: validate representations

- complete the MPS training run
- inspect validation metrics and calibration
- export embeddings
- run core ablations
- audit cold-start and evidence buckets
- freeze architecture before one test evaluation

### Milestone 2: fixture and opportunity model

- ingest future squads and likely XIs
- model batting order and balls faced
- model bowling allocation by phase
- attach predictions to a clear `as_of_date`

### Milestone 3: coherent match simulator

- sample legal delivery outcomes
- update match state and strike
- produce player score, wicket, catch, and economy distributions
- model correlations between opposing players

### Milestone 4: Dream11 layer

- encode current scoring and roster rules
- calculate expected points and uncertainty
- optimize XI, captain, and vice-captain
- support conservative and high-upside objectives

### Milestone 5: external context

- verified weather
- pitch and venue history
- confirmed lineup information
- injuries and role changes

## 19. Decisions to review and correct

The following assumptions should be explicitly confirmed or changed:

1. T20-only data is the initial training universe.
2. Men's, women's, club, and international T20s may share one model with
   context indicators.
3. All history before the prediction cutoff is available; recent form is an
   additional dynamic feature, not a replacement window.
4. Career batting and bowling vectors remain separate.
5. The primary representation test is improved calibrated delivery
   probabilities, not visually plausible nearest neighbours.
6. The first downstream target is Dream11, but representation and simulation
   components should remain platform-independent.
7. Weather is deferred until timestamp and venue provenance are reliable.
8. Prediction should eventually use confirmed or probabilistic opportunity,
   rather than assuming every squad member receives equal involvement.

Corrections to these assumptions should be made before the opportunity and
simulation layers are designed.
