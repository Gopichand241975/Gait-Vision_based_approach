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

VIDEO_EXTS = (".mp4", ".avi", ".mov")

# ---------- Pose estimation ----------
# YOLOv8-pose (ultralytics) is used instead of HRNet; both output 17 COCO keypoints.
POSE_MODEL = "yolov8n-pose.pt"   # downloaded automatically on first run
POSE_CONF = 0.25                 # minimum person-detection confidence
NUM_JOINTS = 17
NUM_CHANNELS = 3                 # x, y, confidence

# COCO 17 keypoint skeleton (graph edges)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders and arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

# ---------- Sequences ----------
SEQ_LEN = 60        # frames per training sample (GaitGraph uses 60)
SEQ_STRIDE = 30     # step between windows when cutting long videos
MIN_VALID_FRAMES = 20   # discard videos with fewer detected frames

# ---------- Gallery / probe split ----------
# First GALLERY_PER_PERSON videos of each person (sorted by name) = gallery,
# the remaining videos = probes.
GALLERY_PER_PERSON = 1

# Subject split (only used if you switch to CASIA-B style data)
TRAIN_RATIO = 0.6   # fraction of people used for training, the rest for testing

# ---------- Model ----------
FEATURE_DIM = 128   # size of the gait feature vector
DROPOUT = 0.1

# ---------- Training ----------
BATCH_SIZE = 16     # lower this if you see "CUDA out of memory"
LR = 1e-3
WEIGHT_DECAY = 1e-4
TOTAL_ITERS = 2000
SAVE_EVERY = 200    # checkpoint interval (iterations)
LOG_EVERY = 20
NUM_WORKERS = 2
SEED = 42

# ---------- Smoke test overrides (used by selftest.py) ----------
SMOKE_ITERS = 5
SMOKE_BATCH_SIZE = 2


def get_device():
    """cuda if available, otherwise cpu. Never hardcode .cuda() elsewhere."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (DATASET_DIR, SKELETON_RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)