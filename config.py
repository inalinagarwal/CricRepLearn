"""Project paths and vocabulary — loaded from processed artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
DATA = ROOT / "data"

BALL_UNIVERSE = PROCESSED / "ball_universe.csv"
TRAIN_CSV = PROCESSED / "train.csv"
VAL_CSV = PROCESSED / "val.csv"
SCALER_PATH = PROCESSED / "scaler.pkl"
CHECKPOINT_PATH = ROOT / "checkpoints" / "model.pt"
EMBEDDINGS_DIR = PROCESSED / "embeddings"

BATTER_TO_ID = PROCESSED / "batter_to_id.json"
BOWLER_TO_ID = PROCESSED / "bowler_to_id.json"
VENUE_TO_ID = PROCESSED / "venue_to_id.json"

# Cricsheet folder name -> human label (folder names come from build_ball_universe)
LEAGUE_ALIASES = {
    "t20i": "t20s_json",
    "t20": "t20s_json",
    "international": "t20s_json",
    "india": "t20s_json",
    "ipl": "ipl_json",
    "bbl": "bbl_json",
    "psl": "psl_json",
    "cpl": "cpl_json",
    "bpl": "bpl_json",
    "lpl": "lpl_json",
    "mlc": "mlc_json",
    "ilt20": "ilt_json",
    "sa20": "sat_json",
}

# T20 phase boundaries (over number, 1-indexed in cricket terms)
PHASE_PP_END = 6
PHASE_MIDDLE_END = 16
N_PHASES = 3

# Ball outcome classes: 0-6 runs off bat, 7 = wicket
OUTCOME_RUN_VALUES = [0, 1, 2, 3, 4, 5, 6, 0]
N_OUTCOMES = 8

# Typical balls per phase in a T20 innings (for weighting)
PHASE_BALL_WEIGHTS = [36, 60, 24]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_vocab():
    """Return name<->id maps and counts for model init."""
    batter_to_id = load_json(BATTER_TO_ID)
    bowler_to_id = load_json(BOWLER_TO_ID)
    venue_to_id = load_json(VENUE_TO_ID)

    # Must match sorted order used in create_playerid.py
    leagues = sorted({p.name for p in DATA.iterdir() if p.is_dir() and p.name.endswith("_json")})
    league_to_id = {name: idx for idx, name in enumerate(leagues)}

    return {
        "batter_to_id": batter_to_id,
        "bowler_to_id": bowler_to_id,
        "venue_to_id": venue_to_id,
        "league_to_id": league_to_id,
        "id_to_batter": {v: k for k, v in batter_to_id.items()},
        "id_to_bowler": {v: k for k, v in bowler_to_id.items()},
        "id_to_venue": {v: k for k, v in venue_to_id.items()},
        "id_to_league": {v: k for k, v in league_to_id.items()},
        "n_batters": len(batter_to_id),
        "n_bowlers": len(bowler_to_id),
        "n_venues": len(venue_to_id),
        "n_leagues": len(league_to_id),
    }


def resolve_league(name: str, league_to_id: dict) -> str:
    """Map user input ('T20I', 'ipl', 't20s_json') to a league folder key."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in league_to_id:
        return key
    if key in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[key]
    # try suffix match
    for alias, folder in LEAGUE_ALIASES.items():
        if alias in key or key in alias:
            return folder
    raise ValueError(
        f"Unknown league '{name}'. Use one of: {sorted(league_to_id.keys())}"
    )
