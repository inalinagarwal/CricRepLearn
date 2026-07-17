"""Train batter/bowler embeddings from all T20 ball-by-ball data."""

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_PATH, load_vocab
from dataset import CricketDataset
from model import CricketRepModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(dataset: CricketDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.df["outcome"].value_counts().sort_index()
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def train(
    batch_size: int = 4096,
    epochs: int = 10,
    lr: float = 1e-3,
    num_workers: int = 0,
):
    device = get_device()
    print(f"Device: {device}")

    vocab = load_vocab()
    train_ds = CricketDataset(split="train")
    val_ds = CricketDataset(split="val")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = CricketRepModel(
        n_batters=vocab["n_batters"],
        n_bowlers=vocab["n_bowlers"],
        n_venues=vocab["n_venues"],
        n_leagues=vocab["n_leagues"],
    ).to(device)

    class_weights = compute_class_weights(train_ds, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batter, bowler, venue, league, phase, numeric, target = batch
            batter = batter.to(device)
            bowler = bowler.to(device)
            venue = venue.to(device)
            league = league.to(device)
            phase = phase.to(device)
            numeric = numeric.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(batter, bowler, venue, league, phase, numeric)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"batch {batch_idx}/{len(train_loader)} "
                    f"loss {loss.item():.4f}"
                )

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batter, bowler, venue, league, phase, numeric, target = batch
                batter = batter.to(device)
                bowler = bowler.to(device)
                venue = venue.to(device)
                league = league.to(device)
                phase = phase.to(device)
                numeric = numeric.to(device)
                target = target.to(device)

                logits = model(batter, bowler, venue, league, phase, numeric)
                val_loss += criterion(logits, target).item()
                preds = logits.argmax(dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        print(
            f"\nEpoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}\n"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": {
                        "n_batters": vocab["n_batters"],
                        "n_bowlers": vocab["n_bowlers"],
                        "n_venues": vocab["n_venues"],
                        "n_leagues": vocab["n_leagues"],
                        "league_to_id": vocab["league_to_id"],
                    },
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved checkpoint -> {CHECKPOINT_PATH}")

    print(f"Done. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    train()
