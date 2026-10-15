"""Smoke test of the whole chain using SYNTHETIC walking skeletons (no videos needed).

    python selftest.py          # preprocess -> train -> resume -> test, in a temp folder
    python selftest.py --pose   # also checks the YOLOv8-pose model downloads and runs

It runs in a temporary folder, so it never touches your real dataset or results.
Run it first on any new machine (CPU or GPU). If it prints PASSED, the code works there.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def synth_skeleton(t, person, rng):
    """Fake walking skeleton in pixels. Each person has own stride speed/amplitudes/proportions."""
    w = 0.14 + 0.05 * person
    leg_amp, arm_amp, leg_len = 0.30 + 0.12 * person, 0.25 + 0.05 * person, 0.85 + 0.10 * person
    phase = rng.uniform(0, 6.28)
    scale, x0 = 140.0, rng.uniform(50, 150)
    out = np.zeros((t, 17, 3), np.float32)
    for f in range(t):
        s = np.sin(w * f + phase)
        a_l, a_r = leg_amp * s, -leg_amp * s
        b_l, b_r = arm_amp * -s, arm_amp * s
        p = {0: (0, -1.45), 1: (-.06, -1.52), 2: (.06, -1.52), 3: (-.12, -1.48), 4: (.12, -1.48),
             5: (-.35, -1.0), 6: (.35, -1.0), 11: (-.25, 0), 12: (.25, 0)}
        for sh, el, wr, b in ((5, 7, 9, b_l), (6, 8, 10, b_r)):
            e = (p[sh][0] + .5 * np.sin(b), p[sh][1] + .5 * np.cos(b))
            p[el] = e
            p[wr] = (e[0] + .5 * np.sin(1.3 * b), e[1] + .5 * np.cos(1.3 * b))
        for hp, kn, an, a in ((11, 13, 15, a_l), (12, 14, 16, a_r)):
            k = (p[hp][0] + .9 * leg_len * np.sin(a), p[hp][1] + .9 * leg_len * np.cos(a))
            p[kn] = k
            p[an] = (k[0] + .9 * leg_len * np.sin(.7 * a), k[1] + .9 * leg_len * np.cos(.7 * a))
        for j, (x, y) in p.items():
            out[f, j, 0] = x0 + 2.0 * f + scale * x + rng.normal(0, 1.5)
            out[f, j, 1] = 250 + scale * y + rng.normal(0, 1.5)
            out[f, j, 2] = np.clip(rng.normal(0.9, 0.05), 0.5, 1.0)
    return out


def step(name, args, env):
    print(f"\n--- {name} ---", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=str(HERE), env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAILED at step: {name}")


def check_pose_model():
    print("\n--- pose model (YOLOv8-pose) ---", flush=True)
    sys.path.insert(0, str(HERE))
    import pose_extract
    model = pose_extract.load_model()
    frame = np.zeros((480, 640, 3), np.uint8)
    res = model(frame, verbose=False)[0]
    pick = pose_extract.pick_person(res)
    print("model loaded and ran on a blank frame; detection on blank frame:", pick)
    assert pick is None, "unexpected detection on a blank frame"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    a = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="gait_selftest_"))
    env = dict(os.environ, GAIT_ROOT=str(tmp))
    os.environ["GAIT_ROOT"] = str(tmp)
    sys.path.insert(0, str(HERE))
    import config
    import torch
    print("device:", config.get_device(), "| torch", torch.__version__, "| temp folder:", tmp)

    try:
        rng = np.random.default_rng(0)
        for person in range(3):
            for vid in range(4):
                d = tmp / "skeletons_raw" / f"person{person + 1}"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"walk{vid + 1}.npy", synth_skeleton(int(rng.integers(100, 160)), person, rng))
        print("created 3 people x 4 synthetic videos")

        step("preprocess", ["preprocess.py"], env)
        proc = sorted((tmp / "processed").glob("*/*.npy"))
        assert len(proc) == 12 and np.load(proc[0]).shape[1:] == (17, 3)
        assert np.isfinite(np.load(proc[0])).all(), "NaN in processed skeleton"

        step("train (100 iters)", ["train.py", "--iters", "100", "--batch_size", "4",
                                   "--num_workers", "0", "--save_every", "50"], env)
        step("train resume (to 120 iters)", ["train.py", "--resume", "--iters", "120",
                                             "--batch_size", "4", "--num_workers", "0"], env)
        ck = torch.load(tmp / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
        assert ck["iter"] == 120, f"resume failed, iter={ck['iter']}"
        step("test", ["test.py"], env)

        rec = tmp / "results" / "recognition"
        for f in ("features_gallery.npz", "features_probe.npz", "distance_matrix.csv",
                  "predictions.csv", "confusion_matrix.csv", "confusion_matrix.png",
                  "accuracy.json", "report.txt"):
            assert (rec / f).exists(), f"missing result file {f}"
        assert (tmp / "results" / "training" / "train_log.csv").exists()
        feats = np.load(rec / "features_probe.npz")["features"]
        assert feats.shape[1] == config.FEATURE_DIM and np.isfinite(feats).all()
        acc = json.loads((rec / "accuracy.json").read_text())
        print(f"\nsynthetic Rank-1 accuracy: {acc['rank1_accuracy_pct']}% "
              f"(only a sanity number; chance is 33%)")
        if a.pose:
            check_pose_model()
        print("\nSELFTEST PASSED")
    finally:
        if a.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()