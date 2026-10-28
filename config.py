"""Central configuration. Edit values here only; every other file imports from here.

Folder layout (all relative to this project folder):
    dataset/S03/PERP/Videos/Cam-S/S03_PERP_Cam-S_T01.mp4   <- YOU put the clips here
    manifests/dataset_inventory.csv   <- verify_dataset.py   (every clip found, readable or not)
    manifests/camera_selection.csv    <- select_camera.py    (selected camera per subject + direction)
    manifests/dataset_split.csv       <- make_split.py       (train / val / test per clip)
    manifests/camera_review/          <- select_camera.py    (pictures for manual camera review)
    skeletons_raw/                    <- pose_extract.py output (T x 17 x 3, pixels)
    processed/                        <- preprocess.py output (normalised skeletons)
    checkpoints/                      <- train.py output (last.pt, best.pt, iter_XXXXXX.pt)
    results/skeleton/                 <- skeleton summary + pose check images
    results/training/                 <- training log, validation log, curves, split used
    results/recognition/              <- test results (predictions, metrics, confusion matrix)
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

# Manifests are decisions (selected camera, split). They live outside results/ so that
# cleaning results/ never deletes them.
MANIFEST_DIR = ROOT / "manifests"
INVENTORY_CSV = MANIFEST_DIR / "dataset_inventory.csv"
CAMERA_SELECTION_CSV = MANIFEST_DIR / "camera_selection.csv"
DATASET_SPLIT_CSV = MANIFEST_DIR / "dataset_split.csv"
CAMERA_REVIEW_DIR = MANIFEST_DIR / "camera_review"

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

# ---------- Dataset description (used to CHECK the data, never to assume it) ----------
# Clip names:  S03_PERP_Cam-S_T01.mp4 = subject_direction_camera_trial
EXPECTED_SUBJECTS = [f"S{i:02d}" for i in range(1, 8)]
EXPECTED_DIRECTIONS = ["LINE", "PERP", "CROSS-RX-LEFT", "CROSS-RX-RIGHT"]
EXPECTED_CAMERAS = ["Cam-P", "Cam-S"]
EXPECTED_CLIPS_PER_CAMERA = 10

# ---------- Split of the selected clips (per subject + direction group) ----------
SPLIT_SEED = 42                  # fixed seed -> the same dataset_split.csv every time
SPLIT_CLIPS = (6, 2, 2)          # train, validation, test clips per group of 10 clips

# ---------- Automatic camera selection (select_camera.py) ----------
# A side view shows the person walking ACROSS the image: shoulders overlap (small shoulder
# width / torso length) and the hips travel far sideways. These numbers are starting values and
# were NOT calibrated on your videos: always look at manifests/camera_review/ pictures.
CAM_SAMPLE_CLIPS = 3             # clips per camera that are inspected (first, middle, last trial)
CAM_FRAMES_PER_CLIP = 30         # frames sampled from each inspected clip
CAM_MIN_KP_CONF = 0.3            # joint confidence needed to use a shoulder / hip
CAM_MIN_VALID_FRAMES = 10        # fewer usable frames -> the camera is "unclear"
CAM_SHOULDER_FRONTAL = 0.8       # shoulder width / torso length of a fully front or back view
CAM_TRAVEL_FULL = 3.0            # hip travel (in torso lengths) that counts as a full side view
CAM_MIN_SIDE_SCORE = 0.5         # the chosen camera must reach at least this side-view score
CAM_MIN_MARGIN = 0.25            # and beat the other camera by at least this much

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

# ---------- Train / validation / test split ----------
# "manifest" (DEFAULT): reads manifests/dataset_split.csv made by make_split.py (6 / 2 / 2 clips per
#     subject + direction). Training clips = training data AND the gallery; validation clips pick
#     the best checkpoint during training; test clips are used once, by test.py.
# The older modes below do not use manifests and have no validation set:
# "temporal": the first TRAIN_FRACTION of every video = training + gallery, the rest = probes.
# "cross_direction": every video whose direction == TEST_DIRECTION = probes, the others = training.
# "per_direction": per person and direction, the first GALLERY_PER_DIRECTION videos = training.
# "videos": first GALLERY_PER_PERSON videos of each person. "subjects": TRAIN_RATIO of the people train.
SPLIT_MODE = "manifest"
TRAIN_FRACTION = 0.6        # only for "temporal"
TEST_DIRECTION = ""         # only for "cross_direction", e.g. "CROSS-RX-RIGHT"
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
SAVE_EVERY = 200      # also how often the validation clips are evaluated
LOG_EVERY = 20
NUM_WORKERS = 2
DATASET_REPEAT = 50  # one "epoch" = number of training videos x this
SEED = 42
BEST_CKPT = "best.pt"   # checkpoint with the best validation accuracy (used by test.py)

# ---------- Smoke test ----------
SMOKE_ITERS = 5
SMOKE_BATCH_SIZE = 2


def get_device():
    """cuda if available, otherwise cpu. Nothing else in the code hardcodes a device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs():
    for d in (DATASET_DIR, MANIFEST_DIR, SKELETON_RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR,
              RESULTS_SKELETON_DIR, RESULTS_TRAIN_DIR, RESULTS_RECOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
