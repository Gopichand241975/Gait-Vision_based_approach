"""Step 3: raw pixel skeletons -> normalised skeletons.

Reads   skeletons_raw/<person>/<video>.npy      (T, 17, 3) pixels
Writes  processed/<person>/<video>.npy          (T, 17, 3) normalised

Normalisation: smooth over time, centre every frame on the hip midpoint, divide by the
person's median torso length (so camera distance and body size do not matter).
"""
import numpy as np

import config

EPS = 1e-3


def smooth(xy, window):
    if window <= 1 or len(xy) < window:
        return xy
    pad = window // 2
    padded = np.pad(xy, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty_like(xy)
    for j in range(xy.shape[1]):
        for c in range(xy.shape[2]):
            out[:, j, c] = np.convolve(padded[:, j, c], kernel, mode="valid")
    return out


def normalise(seq):
    seq = seq.astype(np.float32)
    xy = smooth(seq[..., :2], config.SMOOTH_WINDOW)
    conf = seq[..., 2:3]
    hip = xy[:, [11, 12]].mean(axis=1)
    shoulder = xy[:, [5, 6]].mean(axis=1)
    torso = np.linalg.norm(shoulder - hip, axis=1)
    good = torso[torso > EPS]
    scale = float(np.median(good)) if len(good) else 1.0
    xy = (xy - hip[:, None, :]) / max(scale, EPS)
    return np.concatenate([xy, conf], axis=2).astype(np.float32)


def main():
    config.ensure_dirs()
    files = sorted(config.SKELETON_RAW_DIR.glob("*/*.npy"))
    if not files:
        raise SystemExit(f"No skeletons in {config.SKELETON_RAW_DIR}. Run pose_extract.py first.")
    n = 0
    for p in files:
        seq = np.load(p)
        assert seq.ndim == 3 and seq.shape[1:] == (config.NUM_JOINTS, config.NUM_CHANNELS), \
            f"{p}: expected (T, 17, 3), got {seq.shape}"
        out = config.PROCESSED_DIR / p.parent.name / p.name
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, normalise(seq))
        n += 1
    print(f"Processed {n} skeleton sequences -> {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()