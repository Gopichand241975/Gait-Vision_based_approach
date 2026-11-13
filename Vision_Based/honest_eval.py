"""Honest evaluation: leave-one-DIRECTION-out, scored on short windows.

Why: the normal split (train/val/test clips from the same subject + direction + session) gives
100% even for a trivial bone-length baseline, so it cannot tell a good model from a bad one.
Here, for each fold, ALL clips of one walking direction are held out. The model never sees that
direction in training, and nothing is chosen using the held-out clips (fixed number of iterations,
no best-checkpoint picking, no early stopping on test data).

Scoring is per short window (SEQ_LEN frames, ~2 s) instead of the mean of a whole ~32 s clip, which
is much less forgiving. A clip-level score is also printed for comparison.

Put this file next to train.py and run:

    python honest_eval.py                 # all 4 folds, 1500 iterations each
    python honest_eval.py --iters 800     # faster
    python honest_eval.py --smoke         # 5 iterations, checks that everything runs

Output: results/honest/honest_report.txt and honest_folds.csv

NOTE: the held-out direction is often also a different camera (PERP = Cam-S, the others = Cam-P),
so a drop in accuracy mixes "new direction" and "new camera". That is a real-world difficulty,
not a bug, but say so when you report it.
"""
import argparse
import csv
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import (TrainDataset, direction_of, list_records, load_record, to_features,
                     window_of, window_starts)
from model import GaitResGCN

OUT = config.RESULTS_DIR / "honest"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@torch.no_grad()
def window_features(model, seq, device, stride):
    starts = window_starts(len(seq), config.SEQ_LEN, stride)
    x = np.stack([to_features(window_of(seq, s, config.SEQ_LEN)) for s in starts])
    return F.normalize(model(torch.from_numpy(x).to(device)), dim=1).cpu().numpy()


def train_fold(train_recs, persons, iters, bs, device, workers):
    p2i = {p: i for i, p in enumerate(persons)}
    ds = TrainDataset(train_recs, p2i)
    loader = DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True, drop_last=True,
                        num_workers=workers, pin_memory=device.type == "cuda")
    model = GaitResGCN(num_classes=len(persons)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=config.LR * 0.01)
    it = 0
    model.train()
    while it < iters:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model.classify(model(x)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % 100 == 0:
                print(f"    iter {it}/{iters} loss {loss.item():.3f}", flush=True)
            if it >= iters:
                break
    return model


def evaluate_fold(model, train_recs, test_recs, device):
    """Gallery = mean feature of every TRAINING clip. Probe = each test WINDOW (and each test clip)."""
    model.eval()
    gal_f, gal_p = [], []
    for r in train_recs:
        f = window_features(model, load_record(r), device, config.SEQ_STRIDE).mean(0)
        gal_f.append(f / (np.linalg.norm(f) + 1e-12))
        gal_p.append(r["person"])
    gal_f, gal_p = np.stack(gal_f), np.array(gal_p)

    win_true, win_pred, clip_true, clip_pred = [], [], [], []
    for r in test_recs:
        w = window_features(model, load_record(r), device, config.SEQ_LEN)  # non-overlapping windows
        pred = gal_p[((w[:, None] - gal_f[None]) ** 2).sum(2).argmin(1)]
        win_true += [r["person"]] * len(pred)
        win_pred += list(pred)
        m = w.mean(0)
        m /= np.linalg.norm(m) + 1e-12
        clip_true.append(r["person"])
        clip_pred.append(gal_p[((gal_f - m) ** 2).sum(1).argmin()])
    acc = lambda t, p: float(np.mean(np.array(t) == np.array(p)))
    return acc(win_true, win_pred), acc(clip_true, clip_pred), len(win_true), len(clip_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.batch_size, a.num_workers = config.SMOKE_ITERS, config.SMOKE_BATCH_SIZE, 0

    OUT.mkdir(parents=True, exist_ok=True)
    device = config.get_device()
    records = list_records()
    if not records:
        raise SystemExit(f"No processed skeletons in {config.PROCESSED_DIR}. Run preprocess.py first.")
    for r in records:
        r["direction"] = direction_of(r["video"])
    directions = sorted({r["direction"] for r in records})
    persons = sorted({r["person"] for r in records})
    chance = 1 / len(persons)
    print(f"device {device}  clips {len(records)}  subjects {len(persons)}  directions {directions}")

    rows = []
    for d in directions:
        train = [r for r in records if r["direction"] != d]
        test = [r for r in records if r["direction"] == d]
        missing = set(persons) - {r["person"] for r in train}
        if missing or not test:
            print(f"skip fold {d}: missing subjects {missing} or no test clips")
            continue
        print(f"\nFOLD hold out {d}: train {len(train)} clips, test {len(test)} clips")
        set_seed(config.SEED)
        model = train_fold(train, persons, a.iters, a.batch_size, device, a.num_workers)
        w_acc, c_acc, n_w, n_c = evaluate_fold(model, train, test, device)
        print(f"  window accuracy {w_acc:.3f} ({n_w} windows)   clip accuracy {c_acc:.3f} ({n_c} clips)")
        rows.append({"held_out_direction": d, "window_accuracy": round(w_acc, 4),
                     "clip_accuracy": round(c_acc, 4), "test_windows": n_w, "test_clips": n_c})

    if not rows:
        raise SystemExit("No fold could be run.")
    with open(OUT / "honest_folds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mw = float(np.mean([r["window_accuracy"] for r in rows]))
    mc = float(np.mean([r["clip_accuracy"] for r in rows]))
    lines = ["Honest evaluation: leave-one-direction-out", f"iterations per fold: {a.iters} (fixed, no selection on test)",
             f"chance level: {100 * chance:.1f}%", ""]
    lines += [f"  held out {r['held_out_direction']:<15} window acc {100 * r['window_accuracy']:5.1f}%   "
              f"clip acc {100 * r['clip_accuracy']:5.1f}%" for r in rows]
    lines += ["", f"MEAN window accuracy {100 * mw:.1f}%   MEAN clip accuracy {100 * mc:.1f}%",
              "", "Report the worst fold as well as the mean. If a fold is near chance, the model has not learned",
              "identity that transfers to a new direction / camera."]
    (OUT / "honest_report.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
