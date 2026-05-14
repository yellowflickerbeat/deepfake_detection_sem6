import json
import cv2
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import config


# ===================================
# EXTERNAL DRIVE CACHE LOCATION
# ===================================
CACHE_DIR = Path(
    "/media/ccps/30EC7E2FEC7DF008/lavdf2/cached_features"
)

FRAMES_DIR = CACHE_DIR / "frames"
MELS_DIR = CACHE_DIR / "mels"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
MELS_DIR.mkdir(parents=True, exist_ok=True)


# ===================================
# SETTINGS
# ===================================
NUM_FRAMES = 4
IMG_SIZE = 96
AUDIO_SR = 16000
N_MELS = 128
MAX_AUDIO_TIME_STEPS = 250


# ===================================
# LABEL LOGIC
# ===================================
def compute_label(entry):
    modify_audio = entry.get("modify_audio", False)
    modify_video = entry.get("modify_video", False)
    n_fakes = entry.get("n_fakes", 0)

    return 1 if (
        modify_audio
        or modify_video
        or n_fakes > 0
    ) else 0


# ===================================
# LOAD METADATA
# ===================================
def load_metadata():
    metadata_path = (
        config.METADATA_MIN_PATH
        if config.METADATA_MIN_PATH.exists()
        else config.METADATA_PATH
    )

    print(f"Using metadata: {metadata_path}")

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

    df = pd.DataFrame(records)

    print(f"Total metadata records: {len(df)}")
    return df


# ===================================
# FRAME EXTRACTION
# ===================================
def sample_indices(total_frames):
    if total_frames <= 0:
        return np.zeros(NUM_FRAMES, dtype=int)

    indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    )

    return indices.astype(int)


def extract_frames(video_path):
    frames = []

    try:
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        indices = sample_indices(total_frames)

        for idx in indices:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(idx)
            )

            ret, frame = cap.read()

            if not ret:
                frame = np.zeros(
                    (IMG_SIZE, IMG_SIZE, 3),
                    dtype=np.uint8
                )
            else:
                frame = cv2.resize(
                    frame,
                    (IMG_SIZE, IMG_SIZE)
                )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

            frame = frame.astype(np.uint8)

            frames.append(frame)

        cap.release()

    except:
        frames = [
            np.zeros(
                (IMG_SIZE, IMG_SIZE, 3),
                dtype=np.float16
            )
            for _ in range(NUM_FRAMES)
        ]

    frames = np.stack(frames)

    frames = np.transpose(
        frames,
        (0, 3, 1, 2)
    )

    return frames.astype(np.float16)


# ===================================
# AUDIO EXTRACTION
# ===================================
def extract_mel(video_path):
    try:
        y, _ = librosa.load(
            str(video_path),
            sr=AUDIO_SR,
            mono=True,
            duration=config.AUDIO_DURATION
        )

    except:
        y = np.zeros(
            int(
                AUDIO_SR *
                config.AUDIO_DURATION
            )
        )

    target_len = int(
        AUDIO_SR *
        config.AUDIO_DURATION
    )

    if len(y) < target_len:
        y = np.pad(
            y,
            (0, target_len - len(y))
        )
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=AUDIO_SR,
        n_mels=N_MELS
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    if mel.shape[1] < MAX_AUDIO_TIME_STEPS:
        mel = np.pad(
            mel,
            (
                (0, 0),
                (
                    0,
                    MAX_AUDIO_TIME_STEPS -
                    mel.shape[1]
                )
            )
        )
    else:
        mel = mel[:, :MAX_AUDIO_TIME_STEPS]

    return mel.astype(np.float16)


# ===================================
# MAIN
# ===================================
def main():
    print(f"Dataset path: {config.DATA_ROOT}")
    print(f"Saving cache to: {CACHE_DIR}")

    df = load_metadata()

    processed = 0
    skipped = 0

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Precomputing"
    ):
        video_path = (
            config.DATA_ROOT /
            row["file"]
        )

        if not video_path.exists():
            skipped += 1
            continue

        video_id = row["video_id"]

        frame_path = (
            FRAMES_DIR /
            f"{video_id}.npz"
        )

        mel_path = (
            MELS_DIR /
            f"{video_id}.npz"
        )

        if frame_path.exists() and mel_path.exists():
            continue

        try:
            frames = extract_frames(video_path)
            mel = extract_mel(video_path)

            np.savez_compressed(
                frame_path,
                frames=frames
            )

            np.savez_compressed(
                mel_path,
                mel=mel
            )

            processed += 1

        except:
            skipped += 1

    print("\nDone")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()