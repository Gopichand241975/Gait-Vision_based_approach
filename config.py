"""Central configuration. Edit values here only; every other file imports from here.

Folder layout (all relative to this project folder):
    dataset/S01/LINE/videos/S01_LINE_Cam-P.mp4   <- YOU put the videos here (see GUIDE)
    skeletons_raw/                    <- pose_extract.py output (T x 17 x 3, pixels)
    processed/                        <- preprocess.py output (normalised skeletons)
    checkpoints/                      <- train.py output (model weights)
    results/skeleton/                 <- skeleton summary + pose check images
    results/training/                 <- training log + split used
    results/recognition/              <- gait recognition results (features, predictions, accuracy)
"""
import os
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parent
# GAIT_ROOT lets selftest.py run in a temporary folder. Normally it is the project folder.
ROOT = Path(os.environ.get("GAIT_ROOT", str(CODE_DIR)))

DATASET_DIR = ROOT / "dataset"
SKELETON_RAW_DIR = ROOT / "skeletons_raw"
PROCESSED_DIR = ROOT / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
RESULTS_SKELETON_DIR = RESULTS_DIR / "skeleton"
RESULTS_TRAIN_DIR = RESULTS_DIR / "training"
RESULTS_RECOG_DIR = RESULTS_DIR / "recognition"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ---------- Pose estimation (17 COCO joints only) ----------
# YOLOv8-pose is used instead of HRNet; both give the 17 COCO keypoints (x, y, confidence).
POSE_MODEL = "yolov8m-pose.pt"   # downloaded automatically on first run (needs internet)
POSE_CONF = 0.25                 # minimum person-detection confidence
BATCH_VIDEOS = 5                 # videos processed per pose_extract.py run
NUM_JOINTS = 17
NUM_CHANNELS = 3                 # x, y, confidence
MIN_VALID_FRAMES = 20            # skip videos with fewer frames containing a person

# COCO 17 joints: 0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear, 5 L-shoulder, 6 R-shoulder,
# 7 L-elbow, 8 R-elbow, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip, 13 L-knee, 14 R-knee,
# 15 L-ankle, 16 R-ankle
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),   # head and neck
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),          # shoulders and arms
    (5, 11), (6, 12), (11, 12),                       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),           # legs
]
# Parent joint of every joint (used for bone vectors). Every pair is also an edge above.
PARENT = [5, 0, 0, 1, 2, 11, 12, 5, 6, 7, 8, 12, 11, 11, 12, 13, 14]
LEFT_RIGHT_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

# ---------- Sequences ----------
SEQ_LEN = 60          # frames per sample (GaitGraph uses 60)
SEQ_STRIDE = 30       # step between windows at test time
SMOOTH_WINDOW = 5     # temporal smoothing of keypoints in preprocess.py (1 = off)
IN_CHANNELS = 7       # x, y, conf, velocity(2), bone(2)

# ---------- Gallery / probe split ----------
# Video names like  P01_STR_cam1.mp4  ->  person = folder name (P01), direction = 2nd part (STR),
# camera = 3rd part (cam1). Names like front_01.mp4 also work (direction = text before the first "_").
#
# "temporal" (DEFAULT): every video is cut in time. The first TRAIN_FRACTION of each video
#     (the first walking passes) = training + gallery; the rest of the video = probes (test).
#     Same person and directions, but walking passes never seen in training.
# "cross_direction": every video whose direction == TEST_DIRECTION (e.g. "CR2") = probes;
#     all other directions = training + gallery. Hardest test: a direction never seen in training.
# "per_direction": per person and direction, the first GALLERY_PER_DIRECTION videos = training +
#     gallery, the remaining videos = probes (needs several video files per direction).
# "videos": first GALLERY_PER_PERSON videos of each person. "subjects": TRAIN_RATIO of the people train.
SPLIT_MODE = "cross_direction"
TRAIN_FRACTION = 0.6        # only for "temporal"
TEST_DIRECTION = "CROSS-RX-RIGHT"   # only for "cross_direction" (directions: LINE, PERP, CROSS-RX-LEFT, CROSS-RX-RIGHT)
GALLERY_PER_DIRECTION = 6   # only for "per_direction"
GALLERY_PER_PERSON = 2      # only for "videos" / "subjects"
TRAIN_RATIO = 0.6           # only for "subjects"

# ---------- Model ----------
FEATURE_DIM = 128
DROPOUT = 0.1
COSINE_SCALE = 16.0

# ---------- Training ----------
BATCH_SIZE = 16      # lower this if you see "CUDA out of memory"
LR = 1e-3
WEIGHT_DECAY = 1e-4
TOTAL_ITERS = 3000
SAVE_EVERY = 200
LOG_EVERY = 20
NUM_WORKERS = 2
DATASET_REPEAT = 50  # one "epoch" = number of training videos x this
SEED = 42

# ---------- Smoke test ----------
SMOKE_ITERS = 5
SMOKE_BATCH_SIZE = 2


def get_device():
    """cuda if available, otherwise cpu. Nothing else in the code hardcodes a device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (DATASET_DIR, SKELETON_RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR,
              RESULTS_SKELETON_DIR, RESULTS_TRAIN_DIR, RESULTS_RECOG_DIR):
        d.mkdir(parents=True, exist_ok=True)