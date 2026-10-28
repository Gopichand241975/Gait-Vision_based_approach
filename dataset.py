"""Loading processed skeletons, the train / validation / test split, and window helpers."""
import csv
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset

import augment
import config


def list_records():
    return [{"person": p.parent.name, "video": p.stem, "direction": direction_of(p.stem)}
            for p in sorted(config.PROCESSED_DIR.glob("*/*.npy"))]


def record_path(rec):
    return config.PROCESSED_DIR / rec["person"] / (rec["video"] + ".npy")


def natural_key(text):
    """Sort walk_2 before walk_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def direction_of(video):
    """P01_STR_cam1 -> 'STR';  front_03 -> 'front';  names without '_' -> 'all'."""
    parts = video.split("_")
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    return "all"


def load_record(rec):
    """Load a processed skeleton sequence. If rec has a "span" [a, b] (fractions of the video),
    only that part of the video is returned (used by the "temporal" split)."""
    seq = np.load(record_path(rec)).astype(np.float32)
    span = rec.get("span")
    if span:
        seq = seq[int(round(span[0] * len(seq))):int(round(span[1] * len(seq)))]
    return seq


def read_split_manifest():
    """manifests/dataset_split.csv -> {clip_id: row}."""
    if not config.DATASET_SPLIT_CSV.exists():
        raise SystemExit(f"{config.DATASET_SPLIT_CSV} does not exist. Run verify_dataset.py, "
                         f"select_camera.py and make_split.py first.")
    with open(config.DATASET_SPLIT_CSV, newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def make_split(records, mode=None, seed=None):
    """Returns dict with train / val / test, gallery / probe record lists, train_persons and warnings.
    "manifest" mode: gallery = the training clips, probe = the test clips (validation clips in "val").
    The older modes have no validation clips (val is empty)."""
    mode = mode or config.SPLIT_MODE
    seed = config.SEED if seed is None else seed
    by_person = {}
    for r in records:
        by_person.setdefault(r["person"], []).append(r)
    for v in by_person.values():
        v.sort(key=lambda r: natural_key(r["video"]))
    persons = sorted(by_person)
    warnings = []

    def gallery_probe(vids, count=None):
        count = config.GALLERY_PER_PERSON if count is None else count
        n = len(vids)
        g = n if n == 1 else min(count, n - 1)
        return vids[:g], vids[g:]

    train, gallery, probe, val = [], [], [], []
    if mode == "manifest":
        table = read_split_manifest()
        by_clip = {r["video"]: r for r in records}
        parts = {"train": train, "val": val, "test": probe}
        missing = [c for c in table if c not in by_clip]
        for clip_id, row in table.items():
            rec = by_clip.get(clip_id)
            if rec is None:
                continue
            if rec["person"] != row["subject"]:
                warnings.append(f"{clip_id}: processed under {rec['person']} but the manifest says {row['subject']}")
            parts[row["split"]].append({**rec, "split": row["split"]})
        if missing:
            warnings.append(f"{len(missing)} clips of dataset_split.csv have no processed skeleton "
                            f"(pose failed or not run): {', '.join(missing[:3])} ...")
        unused = [r for r in records if r["video"] not in table]
        if unused:
            warnings.append(f"{len(unused)} processed skeletons are not in dataset_split.csv and are ignored")
        for part in parts.values():
            part.sort(key=lambda r: (r["person"], natural_key(r["video"])))
        train_persons = sorted({r["person"] for r in train})
        for name, part in (("validation", val), ("test", probe)):
            unseen = sorted({r["person"] for r in part} - set(train_persons))
            if unseen:
                warnings.append(f"{name} subjects without training clips: {unseen}")
        gallery = list(train)
    elif mode == "videos":
        train_persons = persons
        for p in persons:
            g, q = gallery_probe(by_person[p])
            if not q:
                warnings.append(f"{p} has only 1 video: it has no probe video")
            gallery += g
            probe += q
        train = list(gallery)
    elif mode == "temporal":
        train_persons = persons
        for p in persons:
            for r in by_person[p]:
                gallery.append({**r, "span": [0.0, config.TRAIN_FRACTION]})
                probe.append({**r, "span": [config.TRAIN_FRACTION, 1.0]})
        train = list(gallery)
    elif mode in ("per_direction", "cross_direction"):
        train_persons = persons
        for p in persons:
            by_dir = {}
            for r in by_person[p]:
                by_dir.setdefault(direction_of(r["video"]), []).append(r)
            for direction in sorted(by_dir):
                vids = by_dir[direction]
                if mode == "per_direction":
                    g, q = gallery_probe(vids, config.GALLERY_PER_DIRECTION)
                    if not q:
                        warnings.append(f"{p}/{direction} has only 1 video: no probe")
                elif direction == config.TEST_DIRECTION:
                    g, q = [], vids
                else:
                    g, q = vids, []
                gallery += g
                probe += q
        if mode == "cross_direction" and not probe:
            found = sorted({direction_of(r["video"]) for r in records})
            raise ValueError(f"TEST_DIRECTION '{config.TEST_DIRECTION}' not found. Directions: {found}")
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
    return {"mode": mode, "train": train, "val": val, "test": list(probe), "gallery": gallery,
            "probe": probe, "train_persons": train_persons, "warnings": warnings}


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
        self.items = [(load_record(r), person_to_idx[r["person"]]) for r in records]
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
