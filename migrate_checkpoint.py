"""Convert legacy best_model.pth (state_dict only) to checkpoints/model.pt."""

from typing import Dict, Optional

import torch
from pathlib import Path

from config import CHECKPOINT_PATH, load_vocab


def migrate(
    legacy_path: Path = Path("best_model.pth"),
    out_path: Path = CHECKPOINT_PATH,
    legacy_vocab: Optional[Dict] = None,
):
    if legacy_vocab is None:
        # Dimensions used when the original model was trained
        legacy_vocab = {
            "n_batters": 9965,
            "n_bowlers": 7358,
            "n_venues": 580,
            "n_leagues": 18,
        }

    vocab = load_vocab()
    legacy_vocab["league_to_id"] = vocab["league_to_id"]

    state = torch.load(legacy_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": state,
            "vocab": legacy_vocab,
            "migrated_from": str(legacy_path),
        },
        out_path,
    )
    print(f"Migrated {legacy_path} -> {out_path}")
    print(f"Vocab: {legacy_vocab}")


if __name__ == "__main__":
    migrate()
