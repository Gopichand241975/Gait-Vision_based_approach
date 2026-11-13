"""Inference with a saved model: who is walking in a new clip?

    python predict.py path/to/clip.mp4                        # uses checkpoints/best.pt
    python predict.py path/to/skeleton_raw.npy --top 3        # a skeleton (T, 17, 3) in pixels
    python predict.py clip.mp4 --ckpt checkpoints/last.pt

The clip is turned into a skeleton (pose model), normalised like the training data, converted to a
gait feature, and compared with the features of the TRAINING clips stored in the checkpoint.
Needs the processed/ folder of the training clips (made by preprocess.py).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import config
import preprocess
from dataset import record_path
from evaluate import extract, video_feature
from model import GaitResGCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="video file (.mp4 ...) or raw skeleton file (.npy)")
    ap.add_argument("--ckpt", default=None, help="default: checkpoints/best.pt, else last.pt")
    ap.add_argument("--top", type=int, default=3, help="how many closest subjects to list")
    a = ap.parse_args()

    clip = Path(a.clip)
    if not clip.exists():
        raise SystemExit(f"{clip} does not exist")
    best = config.CHECKPOINT_DIR / config.BEST_CKPT
    ckpt_path = Path(a.ckpt) if a.ckpt else (best if best.exists() else config.CHECKPOINT_DIR / "last.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} does not exist. Train first (python train.py).")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gallery = ckpt["split"]["gallery"]
    missing = [r["video"] for r in gallery if not record_path(r).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} training skeletons are missing in {config.PROCESSED_DIR} "
                         f"(e.g. {missing[0]}). Run preprocess.py first.")

    device = config.get_device()
    model = GaitResGCN(num_classes=len(ckpt["split"]["train_persons"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if clip.suffix.lower() == ".npy":
        raw = np.load(clip)
    else:
        import pose_extract
        raw = pose_extract.extract_video(pose_extract.load_model(), clip)
        if raw is None:
            raise SystemExit("No person was found in enough frames of this clip.")
    if raw.ndim != 3 or raw.shape[1:] != (config.NUM_JOINTS, config.NUM_CHANNELS):
        raise SystemExit(f"Expected a skeleton of shape (T, 17, 3), got {raw.shape}")
    seq = preprocess.normalise(preprocess.trim_empty(raw))

    feat = video_feature(model, seq, device)
    fg = extract(model, gallery, device)
    dist = np.linalg.norm(fg - feat[None, :], axis=1)
    persons = [r["person"] for r in gallery]
    nearest = int(dist.argmin())
    per_subject = {}
    for p, d in zip(persons, dist):
        per_subject[p] = min(d, per_subject.get(p, 1e9))
    ranked = sorted(per_subject.items(), key=lambda kv: kv[1])[:a.top]

    print(f"checkpoint: {ckpt_path.name} (iteration {ckpt['iter']})   frames used: {len(seq)}")
    print(f"Predicted subject: {persons[nearest]}   (nearest training clip: {gallery[nearest]['video']}, "
          f"distance {dist[nearest]:.3f})")
    print("Closest subjects: " + ", ".join(f"{p} ({d:.3f})" for p, d in ranked))


if __name__ == "__main__":
    main()
