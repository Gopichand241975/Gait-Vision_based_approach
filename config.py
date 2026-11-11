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
CAM_SAMPLE_CLIPS = 3
CAM_FRAMES_PER_CLIP = 30
CAM_MIN_KP_CONF = 0.3
CAM_MIN_VALID_FRAMES = 10
CAM_SHOULDER_FRONTAL = 0.8
CAM_TRAVEL_FULL = 3.0

# Auto-select only when the better camera is clearly better.
# If equal / too close -> needs_review and left out.
CAM_MIN_SIDE_SCORE = 0.50
CAM_MIN_MARGIN = 0.15

# NO fixed direction rule
USE_FIXED_CAMERA_POLICY = False
ALLOW_MANUAL_CAMERA_OVERRIDE = True    # needed for: python select_camera.py --set ...  (your own camera decisions)

# ---------- Pose estimation (17 COCO joints only) ----------
# Pose model: HRNet-W32 ONLY (17 COCO keypoints: x, y, confidence). There is no other pose backend.

# ---------- HRNet pose backend (hrnet_pose.py, pose_backend.py) ----------
# HRNet is TOP-DOWN: a person detector (torchvision Faster R-CNN, downloaded automatically by torchvision)
# gives a box, HRNet-W32 predicts the 17 COCO keypoints inside it. The box is then followed from frame to
# frame from the previous skeleton and the detector is only re-run every HRNET_REDETECT_EVERY frames (fast).
WEIGHTS_DIR = ROOT / "weights"
HRNET_WEIGHTS = WEIGHTS_DIR / "hrnet_w32_coco_256x192.pth"            # OpenMMLab copy of the COCO weights
HRNET_WEIGHTS_OFFICIAL = WEIGHTS_DIR / "pose_hrnet_w32_256x192.pth"   # or the official file, if you have it
HRNET_WEIGHTS_URL = ("https://download.openmmlab.com/mmpose/top_down/hrnet/"
                     "hrnet_w32_coco_256x192-c78dce93_20200708.pth")
HRNET_INPUT_W = 192              # network input 192 x 256 (width x height); heatmaps are 48 x 64
HRNET_INPUT_H = 256
HRNET_DET_SCORE = 0.6            # minimum person-detector score
HRNET_DET_MIN_SIZE = 540         # detector works on a smaller copy of the frame (short side, pixels): faster
HRNET_DET_MAX_SIZE = 960
HRNET_REDETECT_EVERY = 8         # re-run the person detector every N frames (1 = every frame, slowest)
HRNET_JOINT_CONF = 0.2           # a joint counts as "found" above this heatmap peak value
HRNET_MIN_JOINTS = 5             # fewer found joints = frame treated as "no person"
HRNET_FP16 = True                # half precision on GPU (faster, same result for practical purposes)
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
SPLIT_MODE = "direction_holdout"
# "direction_holdout" (DEFAULT, honest): whole walking DIRECTIONS are held out. The model is trained on
#     TRAIN_DIRECTIONS only, the best checkpoint is picked on VAL_DIRECTION and the final number is
#     measured on TEST_DIRECTION - directions the model has never seen. Needs only processed/.
#     (A random split of clips from the same subject + direction + session gives ~100% even for a
#     trivial bone-length baseline, so it cannot tell a good model from a bad one.)
TRAIN_DIRECTIONS = ["CROSS-RX-LEFT", "CROSS-RX-RIGHT"]   # only for "direction_holdout" ([] = every direction not held out)
VAL_DIRECTION = "PERP"       # only for "direction_holdout" ("" = no validation, last.pt is used)
TRAIN_FRACTION = 0.6        # only for "temporal"
TEST_DIRECTION = "LINE"   # "direction_holdout" and "cross_direction": the unseen test direction
GALLERY_PER_DIRECTION = 6   # only for "per_direction"
GALLERY_PER_PERSON = 2      # only for "videos" / "subjects"
TRAIN_RATIO = 0.6           # only for "subjects"

# ---------- Model ----------
# Slightly larger embedding and a bit more regularisation to improve subject separation.
FEATURE_DIM = 256
DROPOUT = 0.2
COSINE_SCALE = 24.0

# ---------- Training ----------
# Tuned for a small gait dataset: a lower LR + longer training helps the model settle without
# overfitting too early on a limited number of clips.
BATCH_SIZE = 12
LR = 2e-4
WEIGHT_DECAY = 8e-4
TOTAL_ITERS = 1500
SAVE_EVERY = 250
LOG_EVERY = 20
NUM_WORKERS = 2
DATASET_REPEAT = 120
SEED = 42
BEST_CKPT = "best.pt"
EARLY_STOP_PATIENCE = 4      # validations (every SAVE_EVERY iterations) without improvement
EARLY_STOP_MIN_DELTA = 0.00025

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
