"""Training-time augmentation on a (T, 17, 3) normalised skeleton window."""
import numpy as np

import config


def flip(seq):
    seq = seq.copy()
    seq[..., 0] *= -1
    for a, b in config.LEFT_RIGHT_PAIRS:
        seq[:, [a, b]] = seq[:, [b, a]]
    return seq


def augment(seq, rng):
    """Returns a new array with the same shape. rng is a numpy Generator."""
    seq = seq.copy()
    if rng.random() < 0.5:
        seq = flip(seq)
    xy = seq[..., :2]
    xy = xy * rng.uniform(0.85, 1.15)                                 # scale
    theta = np.deg2rad(rng.uniform(-8, 8))                             # small rotation
    c, s = np.cos(theta), np.sin(theta)
    xy = xy @ np.array([[c, s], [-s, c]], dtype=np.float32)
    xy = xy + rng.normal(0, 0.015, xy.shape)                           # keypoint noise
    seq[..., :2] = xy
    if rng.random() < 0.2:                                              # drop 1-2 joints occasionally
        joints = rng.choice(config.NUM_JOINTS, size=rng.integers(1, 3), replace=False)
        for j in joints:
            seq[:, j, :] = 0.0
    return seq.astype(np.float32)