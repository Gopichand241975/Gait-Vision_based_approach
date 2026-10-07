"""Central configuration. Edit values here only; other files import from this."""
from pathlib import Path
import torch

# ---------- Paths (relative to this file, so the folder can be moved) ----------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"            # dataset/<person_id>/*.mp4
SKELETON_RAW_DIR = ROOT / "skeletons_raw" # per-video keypoints (T x 17 x 3), .npy
PROCESSED_DIR = ROOT / "processed"        # cleaned/normalised sequences
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"