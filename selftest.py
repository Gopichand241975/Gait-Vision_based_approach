"""Smoke test of the whole chain using SYNTHETIC data (no real videos needed).

    python selftest.py          # camera selection logic -> split -> preprocess -> train (+ validation,
                                # best checkpoint, resume) -> test (metrics) -> predict, in a temp folder
    python selftest.py --pose   # also checks the YOLOv8-pose model downloads and runs

It runs in a temporary folder, so it never touches your real dataset, manifests or results.
Run it first on any new machine (CPU or GPU). If it prints PASSED, the code works there.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE   # folder the steps run from; main() points it to a temporary copy of the code


def synth_skeleton(t, person, rng, direction=1):
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
            out[f, j, 0] = x0 + direction * 2.0 * f + direction * scale * x + rng.normal(0, 1.5)
            out[f, j, 1] = 250 + scale * y + rng.normal(0, 1.5)
            out[f, j, 2] = np.clip(rng.normal(0.9, 0.05), 0.5, 1.0)
    return out


def synth_view(kind, n, rng):
    """n keypoint frames (17, 3) of a person seen from the side ('side') or from the front ('front')."""
    frames, torso = [], 100.0
    for i in range(n):
        kp = np.zeros((17, 3), np.float32)
        cx = 100.0 + (6.0 if kind == "side" else 0.2) * torso * i / n      # sideways travel
        half = (0.05 if kind == "side" else 0.45) * torso     # half the shoulder width
        kp[5] = (cx - half, 200, 0.9)
        kp[6] = (cx + half, 200, 0.9)
        kp[11] = (cx - half / 2, 300, 0.9)
        kp[12] = (cx + half / 2, 300, 0.9)
        kp[:, :2] += rng.normal(0, 1.0, (17, 2))
        frames.append(kp)
    return frames


def make_code_copy(tmp):
    """Copy the code into the temp folder and force SPLIT_MODE = "manifest" in the copy, so the
    selftest works whatever split mode is set in your real config.py."""
    code = tmp / "code"
    shutil.copytree(HERE, code, ignore=shutil.ignore_patterns(
        "dataset", "skeletons_raw", "processed", "checkpoints", "results", "__pycache__",
        "gait", "venv", ".venv", ".git", "*.pt", "*.zip"))
    cfg = (code / "config.py").read_text()
    cfg = re.sub(r"(?m)^SPLIT_MODE = .*$", 'SPLIT_MODE = "manifest"', cfg)
    (code / "config.py").write_text(cfg)
    return code


def step(name, args, env):
    print(f"\n--- {name} ---", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=str(CODE), env=env)
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
    global CODE
    CODE = make_code_copy(tmp)
    import config
    import torch
    print("device:", config.get_device(), "| torch", torch.__version__, "| temp folder:", tmp)

    try:
        import csv
        import select_camera
        rng = np.random.default_rng(0)
        subjects, directions = ["S01", "S02", "S03"], ["LINE", "PERP"]
        sign = {"LINE": 1, "PERP": -1}
        side_cam = {}                                     # the truth used to fake the camera views
        for si, sj in enumerate(subjects):
            for di, d in enumerate(directions):
                side_cam[(sj, d)] = "Cam-P" if (si + di) % 2 == 0 else "Cam-S"
                for cam in ("Cam-P", "Cam-S"):
                    vdir = tmp / "dataset" / sj / d / "Videos" / cam
                    vdir.mkdir(parents=True, exist_ok=True)
                    sdir = tmp / "skeletons_raw" / sj
                    sdir.mkdir(parents=True, exist_ok=True)
                    for t in range(1, 11):
                        stem = f"{sj}_{d}_{cam}_T{t:02d}"
                        (vdir / f"{stem}.mp4").write_bytes(b"")          # empty placeholder clip
                        np.save(sdir / f"{stem}.npy",
                                synth_skeleton(int(rng.integers(100, 160)), si, rng, sign[d]))
        print("created 3 subjects x 2 directions x 2 cameras x 10 synthetic clips")

        print("\n--- camera selection logic ---", flush=True)
        side, front = select_camera.clip_view_metrics(synth_view("side", 30, rng)), \
            select_camera.clip_view_metrics(synth_view("front", 30, rng))
        assert side["side_score"] > 0.8 and front["side_score"] < 0.2, (side, front)
        pick = select_camera.decide({"Cam-P": select_camera.camera_metrics([side]),
                                     "Cam-S": select_camera.camera_metrics([front])})
        assert pick[0] == "Cam-P" and pick[3] == "auto_selected", pick
        unclear = select_camera.decide({"Cam-P": select_camera.camera_metrics([front]),
                                        "Cam-S": select_camera.camera_metrics([front])})
        assert unclear[0] == "" and unclear[3] == "needs_review", unclear

        def fake_sample(model, clip, n):                  # stands in for the pose model
            info = select_camera.parse_clip(clip)
            ambiguous = (info["subject"], info["direction"]) == ("S03", "PERP")
            kind = "side" if info["camera"] == side_cam[(info["subject"], info["direction"])] and not ambiguous \
                else "front"
            return synth_view(kind, n, rng), None

        rows = select_camera.inspect_all(None, False, sample_fn=fake_sample)
        for key, r in rows.items():
            if key == ("S03", "PERP"):
                assert r["review_status"] == "needs_review" and r["selected_camera"] == "", r
            else:
                assert r["selected_camera"] == side_cam[key] and r["review_status"] == "auto_selected", (key, r)
        print("5 groups selected correctly, the ambiguous group was flagged needs_review")

        step("verify_dataset (names only)", ["verify_dataset.py", "--quick"], env)
        step("make_split (S03 PERP still unreviewed)", ["make_split.py"], env)
        with open(tmp / "manifests" / "dataset_split.csv", newline="") as f:
            split_rows = list(csv.DictReader(f))
        assert len(split_rows) == 50, len(split_rows)         # 5 groups x 10 clips
        assert {r["subject"] + r["direction"] for r in split_rows}.isdisjoint({"S03PERP"})
        select_camera.set_manual("S03", "PERP", side_cam[("S03", "PERP")], "selftest")
        step("make_split (after the manual decision)", ["make_split.py"], env)
        with open(tmp / "manifests" / "dataset_split.csv", newline="") as f:
            split_rows = list(csv.DictReader(f))
        counts = {s: sum(r["split"] == s for r in split_rows) for s in ("train", "val", "test")}
        assert counts == {"train": 36, "val": 12, "test": 12}, counts
        assert len({r["clip_id"] for r in split_rows}) == 60
        assert all(r["selected_camera"] == side_cam[(r["subject"], r["direction"])] for r in split_rows)
        step("make_split --verify", ["make_split.py", "--verify"], env)
        again = [r["split"] for r in split_rows]
        step("make_split (same seed again)", ["make_split.py"], env)
        with open(tmp / "manifests" / "dataset_split.csv", newline="") as f:
            assert [r["split"] for r in csv.DictReader(f)] == again, "split is not reproducible"

        step("preprocess", ["preprocess.py"], env)
        proc = sorted((tmp / "processed").glob("*/*.npy"))
        assert len(proc) == 120 and np.load(proc[0]).shape[1:] == (17, 3)
        assert np.isfinite(np.load(proc[0])).all(), "NaN in processed skeleton"

        step("train (100 iters, validation every 50)", ["train.py", "--iters", "100", "--batch_size", "4",
                                                       "--num_workers", "0", "--save_every", "50"], env)
        ck_dir = tmp / "checkpoints"
        assert (ck_dir / "best.pt").exists() and (ck_dir / "last.pt").exists()
        assert (tmp / "results" / "training" / "val_log.csv").exists()
        assert (tmp / "results" / "training" / "training_curves.png").exists()
        step("train resume (to 120 iters)", ["train.py", "--resume", "--iters", "120",
                                             "--batch_size", "4", "--num_workers", "0"], env)
        ck = torch.load(ck_dir / "last.pt", map_location="cpu", weights_only=False)
        assert ck["iter"] == 120, f"resume failed, iter={ck['iter']}"
        best = torch.load(ck_dir / "best.pt", map_location="cpu", weights_only=False)
        assert best["best"]["iteration"] == best["iter"], "best.pt is not the best-validation checkpoint"
        step("test (best checkpoint)", ["test.py"], env)

        rec = tmp / "results" / "recognition"
        for f in ("features_gallery.npz", "features_probe.npz", "distance_matrix.csv", "predictions.csv",
                  "confusion_matrix.csv", "confusion_matrix.png", "classification_report.csv",
                  "accuracy.json", "report.txt"):
            assert (rec / f).exists(), f"missing result file {f}"
        feats = np.load(rec / "features_probe.npz")["features"]
        assert feats.shape == (12, config.FEATURE_DIM) and np.isfinite(feats).all()
        acc = json.loads((rec / "accuracy.json").read_text())
        assert acc["probe_videos"] == 12 and acc["gallery_videos"] == 36
        for key in ("accuracy_pct", "macro_precision_pct", "macro_recall_pct", "macro_f1_pct"):
            assert 0 <= acc[key] <= 100, key
        step("predict (one skeleton)", ["predict.py", str(tmp / "skeletons_raw" / "S02" / "S02_LINE_Cam-P_T01.npy")], env)
        print(f"\nsynthetic accuracy: {acc['accuracy_pct']}%  macro-F1: {acc['macro_f1_pct']}% "
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
