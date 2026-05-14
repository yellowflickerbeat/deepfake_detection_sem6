from pathlib import Path
import torch


class Config:
    # -----------------------------
    # Project root
    # -----------------------------
    PROJECT_ROOT = Path.cwd()

    # -----------------------------
    # Original dataset location
    # -----------------------------
    DATA_ROOT = (
        PROJECT_ROOT
        / "lavdf_extracted"
        / "archive"
        / "LAV-DF"
    )

    # -----------------------------
    # Metadata files
    # -----------------------------
    METADATA_PATH = DATA_ROOT / "metadata.json"
    METADATA_MIN_PATH = DATA_ROOT / "metadata.min.json"

    # -----------------------------
    # External cache location
    # -----------------------------
    CACHE_ROOT = Path(
        "/media/ccps/30EC7E2FEC7DF008/lavdf2/cached_features"
    )

    # -----------------------------
    # Training params
    # -----------------------------
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    EPOCHS = 5
    LR = 1e-4

    # -----------------------------
    # Feature settings
    # -----------------------------
    NUM_FRAMES = 4
    IMG_SIZE = 112
    AUDIO_DURATION = 2.0
    AUDIO_SR = 16000
    N_MELS = 128
    MAX_AUDIO_TIME_STEPS = 400

    # -----------------------------
    # Hardware
    # -----------------------------
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()

print(f"Dataset path: {config.DATA_ROOT}")
print(f"Dataset exists: {config.DATA_ROOT.exists()}")

print(f"Cache path: {config.CACHE_ROOT}")