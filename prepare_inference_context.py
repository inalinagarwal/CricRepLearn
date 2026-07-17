"""Write processed/inference_context.json (scaled phase medians for match inference)."""

import json

import joblib
import pandas as pd

from config import PROCESSED, SCALER_PATH, TRAIN_CSV


def main():
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(
        TRAIN_CSV,
        usecols=[
            "phase",
            "current_score",
            "current_wickets",
            "balls_remaining",
            "current_run_rate",
        ],
        nrows=800_000,
    )
    states = {}
    cols = [
        "current_score",
        "current_wickets",
        "balls_remaining",
        "current_run_rate",
    ]
    for phase in (0, 1, 2):
        raw = df[df["phase"] == phase][cols].median()
        states[str(phase)] = scaler.transform([raw.values])[0].tolist()

    out = PROCESSED / "inference_context.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"phase_states": states}, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
