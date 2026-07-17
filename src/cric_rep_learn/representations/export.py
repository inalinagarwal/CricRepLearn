"""Export checkpoint-owned batting and bowling embeddings with evidence metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .model import ModelConfig, PlayerRepresentationModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_embeddings(checkpoint_path: Path, output_dir: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = PlayerRepresentationModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    batting = model.batting_embeddings().numpy()
    bowling = model.bowling_embeddings().numpy()
    rows = []
    for role, players, vectors in (
        ("batting", checkpoint["vocab"]["batters"], batting),
        ("bowling", checkpoint["vocab"]["bowlers"], bowling),
    ):
        for player in players:
            index = int(player["index"])
            vector = vectors[index]
            rows.append(
                {
                    "role": role,
                    "role_index": index,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "deliveries": int(player["deliveries"]),
                    "matches": int(player["matches"]),
                    "first_date": player["first_date"],
                    "last_date": player["last_date"],
                    "learned": True,
                    "embedding_norm": float(np.linalg.norm(vector)),
                    "embedding": vector.tolist(),
                }
            )

    vector_type = pa.list_(pa.float32(), model_config.player_dim)
    schema = pa.schema(
        [
            ("role", pa.string()),
            ("role_index", pa.int32()),
            ("player_id", pa.string()),
            ("player_name", pa.string()),
            ("deliveries", pa.int64()),
            ("matches", pa.int64()),
            ("first_date", pa.string()),
            ("last_date", pa.string()),
            ("learned", pa.bool_()),
            ("embedding_norm", pa.float32()),
            ("embedding", vector_type),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "player_embeddings.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output_path, compression="zstd")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "embedding_rows": len(rows),
        "unique_players": len({row["player_id"] for row in rows}),
        "batters": len(checkpoint["vocab"]["batters"]),
        "bowlers": len(checkpoint["vocab"]["bowlers"]),
        "embedding_dimension": model_config.player_dim,
        "best_epoch": checkpoint["best_epoch"],
        "validation_metrics": checkpoint["validation_metrics"],
        "baseline_validation_runs_log_loss": checkpoint.get("baseline_validation_runs_log_loss"),
        "identity": "Cricsheet registry person ID",
        "roles": "separate train-only batting and bowling vocabularies",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/representations/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/embeddings"))
    args = parser.parse_args()
    print(
        json.dumps(
            export_embeddings(args.checkpoint, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
