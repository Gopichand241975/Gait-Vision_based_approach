#!/usr/bin/env python3
"""
build_clip_index.py - one table of every clip in the dataset, for training.

Usage:
    python build_clip_index.py "D:\\...\\Dataset_publishment"

Reads only the cut_manifest.csv files that cut_clips.py wrote into the Cam-S / Cam-P folders, and the
session folders around them. Nothing is changed. It writes two files in the dataset root:

  clip_index.csv     one row per clip: subject, path, camera, trial, status, clip path (relative to
                     the root), the position of the trial inside the clip, the CSI window of the trial
                     on the board clock (csi_t_start_us, csi_t_end_us) and the session's raw CSI file.
  clip_coverage.csv  one row per session and camera: how many clips exist, which trial numbers, and
                     whether the camera is missing.

RULE FOR TRAINING: the model must only ever see the trial clips (the files inside the Cam-S / Cam-P
folders), never the whole-session videos that sit one level above them. Read clips ONLY through
clip_index.csv, and for CSI use only the rows between csi_t_start_us and csi_t_end_us of each clip.
This script checks that every clip it lists is a trial clip inside a camera folder, and stops if not.

Optional:  --export "D:\\CSI_GAIT\\Training_clips"  makes a separate folder that contains ONLY the trial clips
(hard links when it is on the same drive, otherwise copies) plus its own clip_index.csv, so no code
can reach a whole-session video from there.
"""
import argparse
import csv
import os
import re
import shutil
from pathlib import Path

CAM_DIR = re.compile(r"^cam[-_ ]?[a-z]$", re.I)
CAMS = ("Cam-P", "Cam-S")


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def norm_cam(name):
    n = name.replace("_", "-").replace(" ", "-")
    return n[:3].capitalize() + n[3:].upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--export", help="copy/link ONLY the trial clips into this new folder")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    sessions = {}                                   # session dir -> {cam: [rows]}
    for t in root.rglob("*_trials.csv"):
        if not any(CAM_DIR.match(x) for x in t.parts):
            sessions.setdefault(t.parent, {})
    index, missing_files = [], 0
    for mf in sorted(root.rglob("*.cut_manifest.csv")):
        camdir = mf.parent
        if not CAM_DIR.match(camdir.name):
            continue
        session = camdir.parent.parent              # ...\S01\LINE\videos\Cam-S -> ...\S01\LINE
        cam = norm_cam(camdir.name)
        raw = next(iter(sorted(session.glob("*_raw.csv"))), None)
        for r in read_csv(mf):
            clip = camdir / r["clip"]
            exists = clip.exists()
            missing_files += (not exists)
            row = {
                "subject": session.parent.name, "path": session.name, "camera": cam,
                "trial": int(r["trial"]), "status": r["csi_status"],
                "clip": str(clip.relative_to(root)), "clip_exists": "yes" if exists else "NO",
                "source_video": r["source_video"], "clip_len_s": r["clip_len_s"],
                "trial_start_in_clip_s": r["trial_start_in_clip_s"],
                "trial_end_in_clip_s": r["trial_end_in_clip_s"],
                "csi_t_start_us": r["csi_t_start_us"], "csi_t_end_us": r["csi_t_end_us"],
                "raw_csv": str(raw.relative_to(root)) if raw else "",
            }
            index.append(row)
            sessions.setdefault(session, {}).setdefault(cam, []).append(row)

    # guard: every listed file must be a trial clip inside a camera folder (never a session video)
    bad = [r["clip"] for r in index
           if not (CAM_DIR.match(Path(r["clip"]).parent.name) and re.search(r"_T\d+\.mp4$", r["clip"]))]
    if bad:
        raise SystemExit("STOP: these entries are not trial clips inside a Cam-S/Cam-P folder:\n  "
                         + "\n  ".join(bad[:10]))
    index.sort(key=lambda r: (r["subject"], r["path"], r["camera"], r["trial"]))
    cols = ["subject", "path", "camera", "trial", "status", "clip", "clip_exists", "source_video",
            "clip_len_s", "trial_start_in_clip_s", "trial_end_in_clip_s",
            "csi_t_start_us", "csi_t_end_us", "raw_csv"]
    with open(root / "clip_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(index)

    cov, both, one, none = [], 0, [], []
    for s in sorted(sessions):
        have = {c: sessions[s].get(c, []) for c in CAMS}
        for c in CAMS:
            trials = sorted({r["trial"] for r in have[c]})
            cov.append({"subject": s.parent.name, "path": s.name, "camera": c, "clips": len(have[c]),
                        "trials": " ".join(str(t) for t in trials),
                        "missing_camera": "" if have[c] else "YES"})
        n = sum(1 for c in CAMS if have[c])
        label = f"{s.parent.name}\\{s.name}"
        if n == 2:
            both += 1
        elif n == 1:
            one.append(f"{label} (only {[c for c in CAMS if have[c]][0]})")
        else:
            none.append(label)
    with open(root / "clip_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "path", "camera", "clips", "trials", "missing_camera"])
        w.writeheader()
        w.writerows(cov)

    if a.export:
        out = Path(a.export).resolve()
        if root in out.parents or out == root:
            raise SystemExit("The export folder must be outside the dataset folder.")
        n_link = n_copy = 0
        exp_rows = []
        for r in index:
            if r["clip_exists"] != "yes":
                continue
            src = root / r["clip"]
            rel = Path(r["subject"]) / r["path"] / r["camera"] / src.name
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                    n_link += 1
                except OSError:
                    shutil.copy2(src, dst)
                    n_copy += 1
            row = dict(r)
            row["clip"] = str(rel)
            row["raw_csv"] = str(root / r["raw_csv"]) if r["raw_csv"] else ""
            exp_rows.append(row)
        with open(out / "clip_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(exp_rows)
        print(f"exported {len(exp_rows)} trial clips to {out} ({n_link} linked, {n_copy} copied); "
              f"it contains no whole-session video")

    print(f"{len(index)} clips in {sum(1 for r in cov if r['clips'])} camera recordings, "
          f"{len(sessions)} sessions")
    print(f"sessions with both cameras: {both}; with one camera: {len(one)}; with none: {len(none)}")
    for x in one:
        print(f"   one camera only: {x}")
    for x in none:
        print(f"   no clips at all: {x}")
    if missing_files:
        print(f"WARNING: {missing_files} clip file(s) named in a manifest are missing on disk")
    odd = [r for r in cov if r["clips"] and r["clips"] != 10]
    for r in odd:
        print(f"   {r['subject']}\\{r['path']} {r['camera']}: {r['clips']} clips (not 10)")
    print(f"written: {root / 'clip_index.csv'}\n         {root / 'clip_coverage.csv'}")


if __name__ == "__main__":
    main()
