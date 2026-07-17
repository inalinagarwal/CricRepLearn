import pandas as pd
import torch

df = pd.read_csv(
    "processed/train.csv",
    low_memory=False
)

counts = (
    df["outcome"]
    .value_counts()
    .sort_index()
)

print(counts)

weights = 1.0 / counts

weights = weights / weights.sum()

print("\nWeights:")
print(weights)