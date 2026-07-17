# Canonical data foundation

The representation model must not depend directly on filenames, player names,
or mutable CSV row numbers. This layer converts Cricsheet JSON into stable,
model-independent Parquet tables.

## Identity

`info.registry.people` maps every observed person name to Cricsheet's stable
eight-character identifier. That identifier is the canonical player key.

Names remain aliases for display and search only. The strict build rejects a
participant missing from the registry instead of silently creating a second
identity. `--allow-unregistered` exists for investigation, but its match-scoped
fallback IDs are not suitable for final model training.

One player ID can later index both role-specific tables:

```text
player_id
├── batting embedding
└── bowling embedding
```

The amount of evidence and confidence for each role is tracked independently.

## Tables

### `matches.parquet`

One row per match with date, event, venue, teams, toss, result, source revision,
and source hash. `source_dataset` records where the file was downloaded from;
it is not treated as the competition label.

### `match_players.parquet`

Players associated with each team and match. Cricsheet's `players` list may
include substitutes and replacements, so the field is deliberately named
`listed_in_match_squad`, not `starting_xi`.

### `innings.parquet`

One row per innings, including repeated super overs, raw and parsed targets,
pre/post innings penalties, absent-hurt players, recorded powerplays, and
miscounted-over metadata. Fractional target notation such as `5.3` is parsed as
five completed overs plus three deliveries, not as a decimal number.

### `player_aliases.parquet`

Every player ID/name combination with first and last observation dates and
match count.

### `deliveries.parquet`

One row per recorded delivery, including illegal deliveries. Important fields:

- batter, bowler, and non-striker canonical IDs
- runs and each extras category
- pre-delivery score, wickets, and legal-ball count
- attempt and legal-ball counters within both the innings and source over
- legal-delivery flag
- boundary and dismissal outcomes
- bowler-attributable wickets
- phase and phase derivation source

Pre-delivery state is calculated before applying the current delivery, avoiding
target leakage.

### `wickets.parquet`

One row per dismissal. Team wickets and bowler-attributable wickets remain
separate. Run-outs, retired hurt, obstructing the field, and timed out are not
credited to the bowler.

### `replacements.parquet` and `reviews.parquet`

Temporal match/role substitutions and decision reviews remain attached to the
delivery where they occurred. This prevents later impact players or replacement
roles from being exposed as pre-match information.

### `source_manifest.parquet`

Every discovered source file, SHA-256 hash, data version, revision, and
deduplication decision.

### `split_manifest.parquet`

One train/validation/test assignment per match. Complete calendar dates stay in
one split:

```text
all training dates < all validation dates < all test dates
```

This supports honest historical backtesting and prevents deliveries from one
match appearing in multiple splits.

## Weather

Cricsheet generally supplies a venue and calendar date, but not:

- venue coordinates
- timezone
- scheduled start time
- weather observations

Weather therefore remains external, rather than being guessed during parsing.
The canonical schemas define:

- `VENUE_LOCATION_SCHEMA` for verified venue coordinates and timezone
- `MATCH_WEATHER_SCHEMA` for timestamped provider observations or forecasts

A historical weather pipeline will:

1. Resolve each venue to a verified latitude, longitude, and timezone.
2. Acquire the actual scheduled start time from a fixture source.
3. Fetch hourly historical observations around the playing window.
4. Store provider, retrieval timestamp, temporal resolution, and whether the
   value was an observation or forecast.
5. Use only information that would have been available before the prediction
   cutoff.

Date-level weather estimates may be used for exploratory analysis, but they
must be marked as daily resolution and should not be presented as precise match
conditions.

## Rebuilding

```bash
cric-build-data --input data --output artifacts/canonical
cric-validate-data --dataset artifacts/canonical
cric-split-data \
  --matches artifacts/canonical/matches.parquet \
  --output artifacts/canonical/split_manifest.parquet
```

Generated tables are excluded from Git. Their source hashes and metadata make
the build auditable without committing gigabytes of data.
