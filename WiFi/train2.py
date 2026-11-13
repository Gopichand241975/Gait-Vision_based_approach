#!/usr/bin/env python3
"""
train2.py - CSI window -> 2D pose, trained on BOTH cameras at once.
One shared CSI encoder (3 Conv1d layers -> 2-layer GRU) and one output head per camera (34 numbers each).

Usage (Anaconda Prompt, csi_env active):
    python train2.py "D:\\CSI_GAIT\\Training_clips\\clip_index.csv" ^
        --csi D:\\CSI_GAIT\\csi_cache_v2 --pose D:\\CSI_GAIT\\pose_cache --out D:\\CSI_GAIT\\runs\\two_run01

Quick pipeline test first (one session, 3 epochs, about a minute):
    python train2.py "D:\\CSI_GAIT\\Training_clips\\clip_index.csv" --only S01_PERP --epochs 3 --out D:\\CSI_GAIT\\runs\\test2

Sanity check against the old script (Cam-S only, should land close to train_basic.py):
    python train2.py "D:\\CSI_GAIT\\Training_clips\\clip_index.csv" --cameras Cam-S --out D:\\CSI_GAIT\\runs\\two_camS_only

What it does
  * The unit is the TRIAL (one CSI file). A trial may have a Cam-S clip, a Cam-P clip, or both. For every CSI window
    it looks up the pose of each camera that has a usable clip; a camera that is missing or excluded simply
    contributes nothing to the loss for that window (its weights are 0).
  * Clips with mean pose confidence below --min-conf, or where the tracked box jumps between frames, are left out
    (per camera, same rules as train_basic.py).
  * Split BY TRIAL inside every session, in trial order: first 7 trials train, the 8th validation, the rest test.
    The order is taken over the trial numbers of ALL selected cameras together, so both cameras of a trial are
    always on the same side. The split is decided before any clip is left out.
  * Example = 1 s window of filtered CSI (52 subcarriers x 100 samples). Targets = the hip-centred, torso-scaled
    pose of each camera at the window centre. CSI is normalised per subcarrier with TRAINING trials only.
  * Loss = sum over cameras of the confidence-weighted L1 (each camera normalised by its own weights).
  * Every camera is scored separately and against its OWN baselines: the average pose and the average pose of the
    same session. Model selection uses the mean validation loss of the heads.
  * --shift-aug S: random pose time shift (+-S seconds) in training, drawn separately for each camera.

Writes to --out: log.txt, history.csv, best.pt, summary.json, test_predictions.npz
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

JOINT_MIN = 0.3
LEGS = [13, 14, 15, 16]
GROUPS = {"head": [0, 1, 2, 3, 4], "arms": [5, 6, 7, 8, 9, 10], "hips": [11, 12], "legs": LEGS}
SWITCH_JUMP = 0.35
METRIC_KEYS = ["mean_err", "PCK@0.1", "PCK@0.2", "PCK@0.5", "legs_PCK@0.2", "legs_mean_err"] + [g + "_err" for g in GROUPS]


def as_path(s):
    return Path(str(s).replace("\\", "/"))


# ----------------------------------------------------------------------------------------------- data
def pose_quality(boxes, scores, det, fps, t0, t1):
    t = np.arange(len(det)) / fps
    win = (t >= t0) & (t <= t1) & det
    conf = float(scores[win].mean()) if win.any() else 0.0
    cx, cy, h = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2, boxes[:, 3] - boxes[:, 1]
    both = det[1:] & det[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = np.hypot(cx[1:] - cx[:-1], cy[1:] - cy[:-1]) / np.where(h[:-1] > 0, h[:-1], np.nan)
        inwin = (t[1:] >= t0 - 0.7) & (t[1:] <= t1 + 0.7)
        sw = int(np.nansum((jump > SWITCH_JUMP) & both & inwin))
    return conf, sw


def normalise_pose(kp, sc, det):
    """Hip-centred, torso-scaled pose per frame. Returns rel (F,34) and weights (F,17); weight 0 = do not use."""
    hip = (kp[:, 11] + kp[:, 12]) / 2
    sho = (kp[:, 5] + kp[:, 6]) / 2
    torso = np.linalg.norm(sho - hip, axis=1)
    ok = det & (sc[:, [5, 6, 11, 12]].min(axis=1) >= JOINT_MIN) & (torso > 5)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (kp - hip[:, None, :]) / torso[:, None, None]
    rel = np.clip(np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0), -5, 5)
    w = np.where(sc >= JOINT_MIN, sc, 0.0).astype(np.float32)
    w[~ok] = 0.0
    return rel.reshape(len(kp), 34).astype(np.float32), w


def split_of(pos):
    return "train" if pos < 7 else ("val" if pos == 7 else "test")


def load_trials(a, log):
    cams = a.cameras
    with open(a.clip_index, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["camera"] in cams]
    if a.only:
        rows = [r for r in rows if any(o in f"{r['subject']}_{r['path']}" for o in a.only)]
    # session -> trial number -> camera -> index row
    sessions = {}
    for r in rows:
        sessions.setdefault((r["subject"], r["path"]), {}).setdefault(int(r["trial"]), {})[r["camera"]] = r
    excl = {c: {"no pose file": 0, "no csi file": 0, "low confidence": 0, "person switch": 0} for c in cams}
    n_clips = {c: sum(1 for r in rows if r["camera"] == c) for c in cams}
    trials, split_moved = [], {c: 0 for c in cams}
    for (subj, path), by_trial in sorted(sessions.items()):
        order = sorted(by_trial)                           # trial numbers of all selected cameras together
        for pos, tn in enumerate(order):
            split = split_of(pos)                          # decided here, before any exclusion
            for c, r in by_trial[tn].items():              # how far this differs from a one-camera split
                alone = sorted(n for n, d in by_trial.items() if c in d)
                if split_of(alone.index(tn)) != split:
                    split_moved[c] += 1
            name = f"{subj}_{path}_T{tn:02d}"
            cf = Path(a.csi) / f"{name}.npz"
            if not cf.is_file():
                for c in by_trial[tn]:
                    excl[c]["no csi file"] += 1
                continue
            cam_data = {}
            for c, r in by_trial[tn].items():
                rel = as_path(r["clip"])
                npz = Path(a.pose) / rel.with_name(rel.stem + "_pose.npz")
                if not npz.is_file():
                    excl[c]["no pose file"] += 1
                    continue
                p = np.load(npz, allow_pickle=False)
                kp, sc, det, box, fps = p["keypoints"], p["scores"], p["detected"].astype(bool), p["boxes"], float(p["fps"])
                conf, sw = pose_quality(box, sc, det, fps, float(r["trial_start_in_clip_s"]), float(r["trial_end_in_clip_s"]))
                if conf < a.min_conf:
                    excl[c]["low confidence"] += 1
                    continue
                if sw:
                    excl[c]["person switch"] += 1
                    continue
                rel_pose, w = normalise_pose(kp, sc, det)
                cam_data[c] = {"pose": rel_pose, "w": w, "fps": fps, "clip_t0": float(r["trial_start_in_clip_s"]),
                               "csi_start_us": float(r["csi_t_start_us"])}
            if not cam_data:
                continue
            z = np.load(cf, allow_pickle=False)
            trials.append({"name": name, "split": split, "csi": z["csi"].astype(np.float32),
                           "t_us": z["t_us"].astype(np.int64), "cams": cam_data})
    for c in cams:
        kept = sum(1 for t in trials if c in t["cams"])
        log(f"{c}: kept {kept} of {n_clips[c]} clips; left out: "
            + (", ".join(f"{v} {k}" for k, v in excl[c].items() if v) or "none"))
    log(f"trials with at least one usable camera: {len(trials)}; "
        f"with both cameras: {sum(1 for t in trials if len(t['cams']) == len(cams))}")
    for c in cams:
        log(f"split check {c}: {split_moved[c]} trials sit in a different split than if {c} were split on its own "
            f"(0 = same split as a one-camera run)")
    return trials


class Windows:
    def __init__(self, trials, cams, W, stride):
        self.trials, self.cams, self.W = trials, cams, W
        self.items = []                                   # (trial index, csi start sample, frame per camera or -1)
        for ti, t in enumerate(trials):
            for s in range(0, len(t["csi"]) - W + 1, stride):
                centre_us = t["t_us"][s + W // 2]
                frames = []
                for c in cams:
                    d = t["cams"].get(c)
                    f = -1
                    if d is not None:
                        f = int(round((d["clip_t0"] + (centre_us - d["csi_start_us"]) / 1e6) * d["fps"]))
                        if not (0 <= f < len(d["pose"]) and d["w"][f].sum() > 0):
                            f = -1
                    frames.append(f)
                if max(frames) >= 0:
                    self.items.append((ti, s, frames))

    def __len__(self):
        return len(self.items)

    def targets(self):
        """Pose and weights of every window without the CSI: Y (N, cams, 34), Wt (N, cams, 17). Weight 0 = no pose."""
        n, nc = len(self.items), len(self.cams)
        Y, Wt = np.zeros((n, nc, 34), np.float32), np.zeros((n, nc, 17), np.float32)
        for k, (ti, _, frames) in enumerate(self.items):
            for c, f in enumerate(frames):
                if f >= 0:
                    d = self.trials[ti]["cams"][self.cams[c]]
                    Y[k, c], Wt[k, c] = d["pose"][f], d["w"][f]
        return Y, Wt

    def sessions(self):
        return np.array([self.trials[ti]["name"].rsplit("_T", 1)[0] for ti, _, _ in self.items])

    def batch(self, idx, mean, std, jitter=None, rng=None):
        """jitter = max frame shift per camera (list), or None."""
        nc = len(self.cams)
        X = np.empty((len(idx), self.W, mean.shape[0]), np.float32)
        Y = np.zeros((len(idx), nc, 34), np.float32)
        Wt = np.zeros((len(idx), nc, 17), np.float32)
        for k, i in enumerate(idx):
            ti, s, frames = self.items[i]
            t = self.trials[ti]
            X[k] = (t["csi"][s:s + self.W] - mean) / std
            for c, f in enumerate(frames):
                if f < 0:
                    continue
                d = t["cams"][self.cams[c]]
                if jitter and jitter[c]:
                    f2 = int(np.clip(f + rng.integers(-jitter[c], jitter[c] + 1), 0, len(d["pose"]) - 1))
                    if d["w"][f2].sum() > 0:
                        f = f2
                Y[k, c], Wt[k, c] = d["pose"][f], d["w"][f]
        return X, Y, Wt


# ---------------------------------------------------------------------------------------------- model
class TwoHeadCnnGru(nn.Module):
    """Shared CSI encoder; one small output head per camera."""

    def __init__(self, n_cams=2, n_sub=52, hid=128, out=34, drop=0.3):
        super().__init__()

        def block(i, o):
            return [nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU()]
        self.conv = nn.Sequential(*block(n_sub, 64), *block(64, 128), nn.MaxPool1d(2), *block(128, 128), nn.MaxPool1d(2))
        self.gru = nn.GRU(128, hid, num_layers=2, batch_first=True, dropout=drop)
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Dropout(drop), nn.Linear(hid, 128), nn.ReLU(), nn.Linear(128, out)) for _ in range(n_cams)])

    def forward(self, x):                                 # x: (B, time, subcarriers) -> (B, cams, 34)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.gru(h)
        z = o.mean(dim=1)
        return torch.stack([head(z) for head in self.heads], dim=1)


def weighted_l1(pred, y, w):
    """pred, y: (B, 34); w: (B, 17). One camera."""
    ww = w.repeat_interleave(2, dim=1)
    return (ww * (pred - y).abs()).sum() / (ww.sum() + 1e-6)


def loss_per_cam(pred, y, w):
    """pred, y: (B, cams, 34); w: (B, cams, 17) -> list of one loss per camera."""
    return [weighted_l1(pred[:, c], y[:, c], w[:, c]) for c in range(pred.shape[1])]


# -------------------------------------------------------------------------------------------- metrics
def metrics(pred, y, w):
    """pred, y: (N, 34) in torso lengths; w: (N, 17). Joints with weight 0 are ignored."""
    ok = w > 0
    if not ok.any():
        return {k: float("nan") for k in METRIC_KEYS}
    d = np.linalg.norm((pred - y).reshape(-1, 17, 2), axis=2)
    m = {"mean_err": float(d[ok].mean())}
    for th in (0.1, 0.2, 0.5):
        m[f"PCK@{th}"] = float((d[ok] < th).mean())
    lg = np.zeros_like(ok)
    lg[:, LEGS] = True
    lg &= ok
    m["legs_PCK@0.2"] = float((d[lg] < 0.2).mean()) if lg.any() else float("nan")
    m["legs_mean_err"] = float(d[lg].mean()) if lg.any() else float("nan")
    for g, js in GROUPS.items():
        gm = np.zeros_like(ok)
        gm[:, js] = True
        gm &= ok
        m[f"{g}_err"] = float(d[gm].mean()) if gm.any() else float("nan")
    return m


@torch.no_grad()
def predict(model, ds, mean, std, device, bs=256):
    model.eval()
    P, Y, Wt = [], [], []
    for s in range(0, len(ds), bs):
        idx = list(range(s, min(s + bs, len(ds))))
        X, y, w = ds.batch(idx, mean, std)
        P.append(model(torch.from_numpy(X).to(device)).cpu().numpy())
        Y.append(y)
        Wt.append(w)
    return np.concatenate(P), np.concatenate(Y), np.concatenate(Wt)


def mean_pose_of(Y, W):
    """Weighted average pose. Y (N,34), W (N,17) -> (34,)"""
    ww = np.repeat(W, 2, axis=1)
    return ((ww * Y).sum(0) / (ww.sum(0) + 1e-6)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_index")
    ap.add_argument("--csi", default=r"D:\CSI_GAIT\csi_cache_v2")
    ap.add_argument("--pose", default=r"D:\CSI_GAIT\pose_cache")
    ap.add_argument("--out", default=r"D:\CSI_GAIT\runs\two_run01")
    ap.add_argument("--cameras", nargs="+", default=["Cam-S", "Cam-P"], help="one output head per camera")
    ap.add_argument("--only", nargs="*", help="only sessions containing these texts, e.g. S01_PERP or S03")
    ap.add_argument("--min-conf", type=float, default=0.82)
    ap.add_argument("--window", type=int, default=100, help="CSI samples per example (100 = 1 s)")
    ap.add_argument("--stride", type=int, default=25, help="step between training windows (samples)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shift-aug", type=float, default=0.0, help="random pose time shift in training, seconds")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cams = a.cameras
    nc = len(cams)
    key = [c.replace("-", "") for c in cams]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(s=""):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  device {device}  args {vars(a)}")

    trials = load_trials(a, log)
    by = {s: [t for t in trials if t["split"] == s] for s in ("train", "val", "test")}
    names = {s: {t["name"] for t in by[s]} for s in by}
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"]) and not (names["val"] & names["test"]), "split leak"
    if not by["train"] or not by["val"]:
        raise SystemExit("not enough trials for train/validation - check --only and the paths")
    allc = np.concatenate([t["csi"] for t in by["train"]])
    mean, std = allc.mean(axis=0).astype(np.float32), np.maximum(allc.std(axis=0), 1e-3).astype(np.float32)
    ds = {"train": Windows(by["train"], cams, a.window, a.stride),
          "val": Windows(by["val"], cams, a.window, a.window // 2),
          "test": Windows(by["test"], cams, a.window, a.window // 2)}
    tg = {s: ds[s].targets() for s in ds}
    for s in ds:
        per = "  ".join(f"{c}: {int((tg[s][1][:, i].sum(1) > 0).sum())}" for i, c in enumerate(cams))
        log(f"{s:<5} {len(by[s]):>4} trials  {len(ds[s]):>6} windows   windows with a pose -> {per}")
    if len(ds["train"]) == 0 or len(ds["val"]) == 0:
        raise SystemExit("no windows could be built - pose and CSI times do not overlap?")

    # baselines, one pair per camera: (1) average training pose, (2) average training pose of the same session
    Ytr, Wtr = tg["train"]
    mean_pose = [mean_pose_of(Ytr[:, c], Wtr[:, c]) for c in range(nc)]
    tr_sess = ds["train"].sessions()
    sess_mean = [{s_: mean_pose_of(Ytr[tr_sess == s_, c], Wtr[tr_sess == s_, c]) for s_ in np.unique(tr_sess)} for c in range(nc)]

    def avg_baseline(split, c):
        Y, W = tg[split][0][:, c], tg[split][1][:, c]
        return metrics(np.tile(mean_pose[c], (len(Y), 1)), Y, W)

    def sess_baseline(split, c):
        Y, W = tg[split][0][:, c], tg[split][1][:, c]
        sn = ds[split].sessions()
        P = np.stack([sess_mean[c].get(x, mean_pose[c]) for x in sn])
        return metrics(P, Y, W)

    model = TwoHeadCnnGru(n_cams=nc, n_sub=mean.shape[0]).to(device)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()):,}  ({nc} heads: {', '.join(cams)})")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    jit = [int(round(a.shift_aug * np.mean([t["cams"][c]["fps"] for t in by["train"] if c in t["cams"]] or [0]))) for c in cams]
    for c in range(nc):
        b = avg_baseline("val", c)
        log(f"baseline (average pose) on validation, {cams[c]}: PCK@0.2 {b['PCK@0.2']:.3f}, mean error {b['mean_err']:.3f}")

    hist, best, best_ep = [], float("inf"), -1
    n = len(ds["train"])
    for ep in range(1, a.epochs + 1):
        model.train()
        order = rng.permutation(n)
        tl, nb, t0 = np.zeros(nc), 0, time.time()
        for s in range(0, n, a.batch):
            idx = order[s:s + a.batch]
            if len(idx) < 2:
                continue
            X, y, w = ds["train"].batch(idx, mean, std, jit, rng)
            X, y, w = (torch.from_numpy(v).to(device) for v in (X, y, w))
            ls = loss_per_cam(model(X), y, w)
            loss = sum(ls)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += np.array([l.item() for l in ls])
            nb += 1
        sched.step()
        tl = tl / max(nb, 1)
        P, Y, W = predict(model, ds["val"], mean, std, device)
        vls = [float(l) for l in loss_per_cam(torch.from_numpy(P), torch.from_numpy(Y), torch.from_numpy(W))]
        vl = float(np.mean(vls))
        vms = [metrics(P[:, c], Y[:, c], W[:, c]) for c in range(nc)]
        row = {"epoch": ep, "train_loss": float(tl.mean()), "val_loss": vl}
        for c in range(nc):
            row[f"train_loss_{key[c]}"] = float(tl[c])
            row[f"val_loss_{key[c]}"] = vls[c]
            row.update({f"val_{key[c]}_{k}": v for k, v in vms[c].items()})
        hist.append(row)
        mark = ""
        if vl < best:
            best, best_ep, mark = vl, ep, "  <- best"
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(a), "cameras": cams}, out / "best.pt")
        per = "   ".join(f"{cams[c]} train {tl[c]:.3f} val {vls[c]:.3f} PCK@0.2 {vms[c]['PCK@0.2']:.3f} legs {vms[c]['legs_PCK@0.2']:.3f}"
                         for c in range(nc))
        log(f"epoch {ep:>3}  val {vl:.4f}  {per}  ({time.time() - t0:.0f}s){mark}")
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        wr.writeheader()
        wr.writerows(hist)

    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    res = {"best_epoch": best_ep, "cameras": cams}
    pred = {s: predict(model, ds[s], mean, std, device) for s in ("val", "test") if len(ds[s])}
    if "test" in pred:
        P, Y, W = pred["test"]
        np.savez_compressed(out / "test_predictions.npz", pred=P, target=Y, weight=W, cameras=np.array(cams),
                            trial=np.array([ds["test"].trials[ti]["name"] for ti, _, _ in ds["test"].items]),
                            start=np.array([st for _, st, _ in ds["test"].items]))
    log(f"\nbest epoch {best_ep}. Scores in torso lengths (1.0 = shoulder-to-hip distance); PCK = share of joints within the distance")
    for c in range(nc):
        mm = {}
        for s in pred:
            P, Y, W = pred[s]
            mm[s] = (avg_baseline(s, c), metrics(P[:, c], Y[:, c], W[:, c]), sess_baseline(s, c))
        res[cams[c]] = {s: {"baseline": mm[s][0], "model": mm[s][1], "session_baseline": mm[s][2]} for s in mm}
        log(f"\n========== {cams[c]} ==========")
        log(f"{'':<16}{'baseline val':>14}{'model val':>11}{'baseline test':>15}{'model test':>12}")
        for k in ("mean_err", "PCK@0.1", "PCK@0.2", "PCK@0.5", "legs_PCK@0.2", "legs_mean_err"):
            row = f"{k:<16}"
            for s in ("val", "test"):
                if s in mm:
                    row += f"{mm[s][0][k]:>14.3f}{mm[s][1][k]:>11.3f}" if s == "val" else f"{mm[s][0][k]:>15.3f}{mm[s][1][k]:>12.3f}"
            log(row)
        if "test" in mm and "val" in mm:
            log(f"\nCHECK 1 ({cams[c]}) - stronger baseline: average pose of the SAME session. If the model only matches this,")
            log("it may be recognising the session from the CSI and not following the walking motion.")
            log(f"{'':<16}{'session base val':>17}{'model val':>11}{'session base test':>19}{'model test':>12}")
            for k in ("mean_err", "PCK@0.2", "legs_PCK@0.2", "legs_mean_err"):
                log(f"{k:<16}{mm['val'][2][k]:>17.3f}{mm['val'][1][k]:>11.3f}{mm['test'][2][k]:>19.3f}{mm['test'][1][k]:>12.3f}")
            log(f"\nCHECK 2 ({cams[c]}) - mean error per body part on the TEST trials (torso lengths; lower is better)")
            log(f"{'':<10}{'average pose':>14}{'session pose':>14}{'model':>9}")
            for g in GROUPS:
                log(f"{g:<10}{mm['test'][0][g + '_err']:>14.3f}{mm['test'][2][g + '_err']:>14.3f}{mm['test'][1][g + '_err']:>9.3f}")
    with open(out / "summary.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"\nsaved in {out}: best.pt, history.csv, summary.json, test_predictions.npz, log.txt")


if __name__ == "__main__":
    main()
