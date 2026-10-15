"""Loading processed skeletons, gallery/probe/train split, and window helpers."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import augment
import config


def list_records():
    return [{"person": p.parent.name, "video": p.stem}
            for p in sorted(config.PROCESSED_DIR.glob("*/*.npy"))]


def record_path(rec):
    return config.PROCESSED_DIR / rec["person"] / (rec["video"] + ".npy")


def make_split(records, mode=None, seed=None):
    """Returns dict with train / gallery / probe record lists, train_persons and warnings."""
    mode = mode or config.SPLIT_MODE
    seed = config.SEED if seed is None else seed
    by_person = {}
    for r in records:
        by_person.setdefault(r["person"], []).append(r)
    for v in by_person.values():
        v.sort(key=lambda r: r["video"])
    persons = sorted(by_person)
    warnings = []

    def gallery_probe(vids):
        n = len(vids)
        g = n if n == 1 else min(config.GALLERY_PER_PERSON, n - 1)
        return vids[:g], vids[g:]

    train, gallery, probe = [], [], []
    if mode == "videos":
        train_persons = persons
        for p in persons:
            g, q = gallery_probe(by_person[p])
            if not q:
                warnings.append(f"{p} has only 1 video: it has no probe video")
            gallery += g
            probe += q
        train = list(gallery)
    elif mode == "subjects":
        shuffled = persons[:]
        random.Random(seed).shuffle(shuffled)
        n_train = min(max(2, int(round(len(persons) * config.TRAIN_RATIO))), len(persons) - 1)
        train_persons = sorted(shuffled[:n_train])
        for p in train_persons:
            train += by_person[p]
        for p in sorted(shuffled[n_train:]):
            g, q = gallery_probe(by_person[p])
            gallery += g
            probe += q
    else:
        raise ValueError(f"unknown SPLIT_MODE {mode}")
    return {"mode": mode, "train": train, "gallery": gallery, "probe": probe,
            "train_persons": train_persons, "warnings": warnings}


def window_of(seq, start, length):
    """Cut `length` frames from `start`. Short videos are repeated cyclically."""
    t = len(seq)
    if t >= length:
        return seq[start:start + length]
    return seq[np.arange(length) % t]


def window_starts(t, length, stride):
    if t <= length:
        return [0]
    starts = list(range(0, t - length + 1, stride))
    if starts[-1] != t - length:
        starts.append(t - length)
    return starts


def to_features(window):
    """(T, 17, 3) -> (7, T, 17) = x, y, conf, velocity(2), bone(2)."""
    xy, conf = window[..., :2], window[..., 2:3]
    vel = np.zeros_like(xy)
    vel[1:] = xy[1:] - xy[:-1]
    bone = xy - xy[:, config.PARENT, :]
    feat = np.concatenate([xy, conf, vel, bone], axis=-1)
    return np.ascontiguousarray(feat.transpose(2, 0, 1), dtype=np.float32)


class TrainDataset(Dataset):
    """Random 60-frame windows from the training videos, augmented."""

    def __init__(self, records, person_to_idx):
        self.items = [(np.load(record_path(r)).astype(np.float32), person_to_idx[r["person"]])
                      for r in records]
        if not self.items:
            raise ValueError("No training videos.")

    def __len__(self):
        return len(self.items) * config.DATASET_REPEAT

    def __getitem__(self, i):
        seq, label = self.items[i % len(self.items)]
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
        start = int(rng.integers(0, max(len(seq) - config.SEQ_LEN, 0) + 1))
        win = augment.augment(window_of(seq, start, config.SEQ_LEN), rng)
        return torch.from_numpy(to_features(win)), label