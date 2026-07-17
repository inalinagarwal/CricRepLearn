"""Export trained batter and bowler embedding matrices to disk."""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from config import CHECKPOINT_PATH, EMBEDDINGS_DIR, load_vocab
from model import CricketRepModel


def load_model(checkpoint_path: Path = CHECKPOINT_PATH) -> Tuple[CricketRepModel, dict]:
    vocab = load_vocab()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    meta = ckpt.get("vocab", {})
    if "league_to_id" in meta:
        vocab["league_to_id"] = meta["league_to_id"]

    model = CricketRepModel(
        n_batters=meta.get("n_batters", vocab["n_batters"]),
        n_bowlers=meta.get("n_bowlers", vocab["n_bowlers"]),
        n_venues=meta.get("n_venues", vocab["n_venues"]),
        n_leagues=meta.get("n_leagues", vocab["n_leagues"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def export(checkpoint_path: Path = CHECKPOINT_PATH, out_dir: Path = EMBEDDINGS_DIR):
    model, vocab = load_model(checkpoint_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    batter_vecs = model.get_batter_embeddings().numpy()
    bowler_vecs = model.get_bowler_embeddings().numpy()

    np.savez_compressed(
        out_dir / "embeddings.npz",
        batter=batter_vecs,
        bowler=bowler_vecs,
    )

    with open(out_dir / "batter_index.json", "w", encoding="utf-8") as f:
        json.dump(vocab["batter_to_id"], f)

    with open(out_dir / "bowler_index.json", "w", encoding="utf-8") as f:
        json.dump(vocab["bowler_to_id"], f)

    print(f"Batter embeddings: {batter_vecs.shape}")
    print(f"Bowler embeddings: {bowler_vecs.shape}")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export player embeddings")
    parser.add_argument(
        "--checkpoint",
        default=str(CHECKPOINT_PATH),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--out",
        default=str(EMBEDDINGS_DIR),
        help="Output directory",
    )
    args = parser.parse_args()
    export(Path(args.checkpoint), Path(args.out))
