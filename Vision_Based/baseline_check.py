"""Sanity check WITHOUT any neural network (numpy only, runs in seconds, no GPU).

Shows why a random clip split gives ~100% and why the direction hold-out is the honest test.
Two trivial 1-nearest-neighbour baselines on the processed skeletons:
  static  = median bone lengths (body shape only, no walking motion)
  dynamic = per-joint movement statistics (std of position + mean speed)

    python baseline_check.py

Prints the accuracy of each baseline for (a) a random split of clips inside each subject+direction
(the old, too-easy protocol) and (b) leave-one-direction-out (new directions). Use these numbers
as the reference your GCN must beat. If the GCN does not beat them, say so honestly.
"""
import random
from pathlib import Path

import numpy as np

PROCESSED = Path(__file__).resolve().parent / "processed"
PARENT = [5, 0, 0, 1, 2, 11, 12, 5, 6, 7, 8, 12, 11, 11, 12, 13, 14]


def static(s):
    xy = s[..., :2]
    return np.median(np.linalg.norm(xy - xy[:, PARENT, :], axis=2), axis=0)


def dynamic(s):
    xy = s[..., :2]
    return np.concatenate([xy.std(0).ravel(), np.abs(np.diff(xy, axis=0)).mean(0).ravel()])


def nn_acc(Ftr, Ytr, Fte, Yte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    a, b = (Fte - mu) / sd, (Ftr - mu) / sd
    pred = Ytr[((a[:, None] - b[None]) ** 2).sum(2).argmin(1)]
    return float((pred == Yte).mean())


def main():
    files = sorted(PROCESSED.glob("*/*.npy"))
    if not files:
        raise SystemExit(f"No processed skeletons in {PROCESSED}")
    meta = []
    for p in files:
        parts = p.stem.split("_")          # S03_CROSS-RX-LEFT_Cam-P_T05
        meta.append((p.parent.name, parts[1]))
    Y = np.array([m[0] for m in meta])
    D = np.array([m[1] for m in meta])
    seqs = [np.load(p) for p in files]
    print(f"{len(files)} clips, {len(set(Y))} subjects, directions {sorted(set(D.tolist()))}, "
          f"chance {100 / len(set(Y)):.1f}%\n")
    for name, f in (("static  (bone lengths only)", static), ("dynamic (motion stats only)", dynamic)):
        X = np.array([f(s) for s in seqs])
        rng = random.Random(42)
        idx = {}
        for i, (y, d) in enumerate(meta):
            idx.setdefault((y, d), []).append(i)
        tr, te = [], []
        for v in idx.values():
            v = v[:]
            rng.shuffle(v)
            k = max(1, round(0.2 * len(v)))
            te += v[:k]
            tr += v[k:]
        print(f"{name}")
        print(f"   random clips inside the same direction (OLD protocol): {100 * nn_acc(X[tr], Y[tr], X[te], Y[te]):5.1f}%")
        res = []
        for d in sorted(set(D.tolist())):
            m = D == d
            res.append((d, nn_acc(X[~m], Y[~m], X[m], Y[m])))
        print("   leave-one-direction-out (NEW protocol): " +
              "   ".join(f"{d} {100 * a:.1f}%" for d, a in res) +
              f"   mean {100 * np.mean([a for _, a in res]):.1f}%\n")


if __name__ == "__main__":
    main()
