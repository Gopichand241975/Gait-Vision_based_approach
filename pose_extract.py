"""Extract 17 COCO keypoints per frame from every video in dataset/<person_id>/.

Output: skeletons_raw/<person_id>/<video_name>.npy with shape (T, 17, 3)
        columns = x (pixels), y (pixels), confidence
"""
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import config

def load_model():
    from ultralytics import YOLO
    return YOLO(config.POSE_MODEL)  # downloads weights on first run

def pick_person(result):
    """Return a (17, 3) array for the largest detected person, or None."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    xy = result.keypoints.xy[i].cpu().numpy()            # (17, 2)
    if result.keypoints.conf is None:
        conf = np.ones((config.NUM_JOINTS, 1), dtype=np.float32)
    else:
        conf = result.keypoints.conf[i].cpu().numpy().reshape(-1, 1)
    return np.concatenate([xy, conf], axis=1).astype(np.float32)

def fill_missing(seq, valid):
    """Linearly interpolate frames where no person was found. Filled frames get conf=0."""
    t = len(seq)
    idx_valid = np.where(valid)[0]
    idx_all = np.arange(t)
    flat = seq.reshape(t, -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(idx_all, idx_valid, flat[idx_valid, c])
    out = out.reshape(seq.shape)
    out[~valid, :, 2] = 0.0
    return out
