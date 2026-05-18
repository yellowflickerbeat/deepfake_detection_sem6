import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import config
from dataset import (
    build_dataset_manifests,
    LAVDFDataset
)
from model_saff import SAFFOnlyModel


# ==========================================
# EVALUATION
# ==========================================
def evaluate(model, loader, criterion):
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            frames = batch["frames"].to(config.DEVICE)
            mel = batch["mel"].to(config.DEVICE)
            labels = batch["label"].float().to(config.DEVICE)

            with autocast("cuda"):
                outputs = model(frames, mel)
                loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs)
    }

    return metrics


# ==========================================
# TRAINING
# ==========================================
def train():
    print("Building manifests...")
    manifests = build_dataset_manifests()

    print("\nCreating datasets...")
    train_dataset = LAVDFDataset(manifests["train"])
    val_dataset = LAVDFDataset(manifests["dev"])
    test_dataset = LAVDFDataset(manifests["test"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    batch = next(iter(train_loader))

    print("\nBatch sanity check:")
    print("Frames:", batch["frames"].shape)
    print("Mel:", batch["mel"].shape)
    print("Labels:", batch["label"][:8])

    model = SAFFOnlyModel().to(config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=1e-4
    )

    scaler = GradScaler("cuda")

    best_score = 0

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            frames = batch["frames"].to(config.DEVICE)
            mel = batch["mel"].to(config.DEVICE)
            labels = batch["label"].float().to(config.DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                outputs = model(frames, mel)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        current_score = (
            val_metrics["auc"] +
            val_metrics["f1"]
        ) / 2

        if current_score > best_score:
            best_score = current_score

            torch.save(
                model.state_dict(),
                "best_saff_model.pth"
            )

            print("Best SAFF model saved.")

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(
            "best_saff_model.pth",
            map_location=config.DEVICE
        )
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nFINAL SAFF TEST RESULTS")
    print(
        f"Loss: {test_metrics['loss']:.4f} | "
        f"Accuracy: {test_metrics['accuracy']:.4f} | "
        f"F1: {test_metrics['f1']:.4f} | "
        f"AUC: {test_metrics['auc']:.4f}"
    )


if __name__ == "__main__":
    train()