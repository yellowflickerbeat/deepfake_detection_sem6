import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from torch.utils.data import Dataset
from config import config


# ==========================================
# LABEL LOGIC
# ==========================================
def compute_label(entry):
    modify_audio = entry.get("modify_audio", False)
    modify_video = entry.get("modify_video", False)
    n_fakes = entry.get("n_fakes", 0)

    return 1 if (
        modify_audio
        or modify_video
        or n_fakes > 0
    ) else 0


# ==========================================
# LOAD METADATA
# ==========================================
def load_metadata_records(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    records = []

    if isinstance(metadata, list):
        for entry in metadata:
            file_rel = entry.get("file")
            split = entry.get("split")

            if not file_rel:
                continue

            records.append({
                "file": file_rel,
                "video_id": Path(file_rel).stem,
                "split": split,
                "label": compute_label(entry)
            })

    elif isinstance(metadata, dict):
        for key, entry in metadata.items():
            file_rel = entry.get("file")

            if not file_rel:
                continue

            records.append({
                "file": file_rel,
                "video_id": Path(file_rel).stem,
                "split": entry.get("split"),
                "label": compute_label(entry)
            })

    return pd.DataFrame(records)


# ==========================================
# BUILD MANIFESTS
# ==========================================
def build_dataset_manifests():
    metadata_source = (
        config.METADATA_MIN_PATH
        if config.METADATA_MIN_PATH.exists()
        else config.METADATA_PATH
    )

    metadata_df = load_metadata_records(
        metadata_source
    )

    # keep only valid files
    full_paths = metadata_df["file"].apply(
        lambda p: (
            config.DATA_ROOT / p
        ).exists()
    )

    metadata_df = metadata_df[
        full_paths
    ].reset_index(drop=True)

    manifests = {}

    for split in ["train", "dev", "test"]:
        split_df = metadata_df[
            metadata_df["split"] == split
        ][[
            "file",
            "video_id",
            "label"
        ]].copy()

        split_df = split_df.reset_index(
            drop=True
        )

        real_count = (
            split_df["label"] == 0
        ).sum()

        fake_count = (
            split_df["label"] == 1
        ).sum()

        print(f"\n{split.upper()} SPLIT")
        print(f"Real: {real_count}")
        print(f"Fake: {fake_count}")
        print(f"Total: {len(split_df)}")

        manifests[split] = split_df

    return manifests


# ==========================================
# DATASET
# ==========================================
class LAVDFDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_id = row["video_id"]
        label = row["label"]

        frame_path = (
            config.CACHE_ROOT
            / "frames"
            / f"{video_id}.npz"
        )

        mel_path = (
            config.CACHE_ROOT
            / "mels"
            / f"{video_id}.npz"
        )

        if not frame_path.exists():
            raise FileNotFoundError(
                f"Missing frames: {frame_path}"
            )

        if not mel_path.exists():
            raise FileNotFoundError(
                f"Missing mel: {mel_path}"
            )

        # load compressed features
        frames = np.load(
            frame_path
        )["frames"]

        mel = np.load(
            mel_path
        )["mel"]

        frames = torch.tensor(
            frames,
            dtype=torch.float32
        )

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return {
            "frames": frames,
            "mel": mel,
            "label": label
        }