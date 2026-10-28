"""Step 0: check the dataset BEFORE using it. Nothing is modified or deleted.

Finds every video under dataset/, checks the names, counts and that each clip opens, and compares
what it finds with the expected layout in config.py (7 subjects x 4 directions x 2 cameras x
10 clips = 560 clips). Differences are REPORTED, never corrected.

Writes  manifests/dataset_inventory.csv   one row per clip (readable?, frames, fps, size, issues)
        manifests/verification_report.txt

    python verify_dataset.py              # full check (opens every clip)
    python verify_dataset.py --quick      # names and counts only, does not open the clips
"""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict

import config
from pose_extract import list_videos, parse_clip


def probe_video(path):
    """(readable, frames, fps, width, height) of a clip, using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, 0, 0.0, 0, 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        return bool(ok), frames, fps, w, h
    finally:
        cap.release()


def fingerprint(path, nbytes=1 << 20):
    """Size + hash of the first MB: finds files that are exact copies of each other."""
    with open(path, "rb") as f:
        return path.stat().st_size, hashlib.md5(f.read(nbytes)).hexdigest()


def inventory(quick=False):
    rows = []
    for v in list_videos():
        info = parse_clip(v)
        issues = []
        row = {"clip_id": v.stem, "subject": "", "direction": "", "camera": "", "trial": "",
               "path": v.relative_to(config.DATASET_DIR).as_posix(),
               "size_mb": round(v.stat().st_size / 1e6, 1), "readable": "", "frames": "",
               "fps": "", "width": "", "height": "", "duration_s": ""}
        if info is None:
            issues.append("name does not match subject_direction_camera_trial")
        else:
            row.update({k: info[k] for k in ("subject", "direction", "camera", "trial")})
            if info["subject"] not in config.EXPECTED_SUBJECTS:
                issues.append(f"unexpected subject {info['subject']}")
            if info["direction"] not in config.EXPECTED_DIRECTIONS:
                issues.append(f"unexpected direction {info['direction']}")
            if info["camera"] not in config.EXPECTED_CAMERAS:
                issues.append(f"unexpected camera {info['camera']}")
            parts = [p.lower() for p in v.relative_to(config.DATASET_DIR).parts]
            if (len(parts) >= 3 and (parts[0] != info["subject"].lower()
                                     or parts[1] != info["direction"].lower())):
                issues.append("file is in the wrong subject / direction folder")
            if info["camera"].lower() not in parts[:-1]:
                issues.append("file is not inside its camera folder")
        if v.stat().st_size == 0:
            issues.append("empty file")
        if not quick and v.stat().st_size > 0:
            ok, frames, fps, w, h = probe_video(v)
            row.update({"readable": "yes" if ok else "NO", "frames": frames, "fps": round(fps, 2),
                        "width": w, "height": h,
                        "duration_s": round(frames / fps, 1) if fps > 0 else ""})
            if not ok:
                issues.append("cannot be opened or read")
        row["issues"] = "; ".join(issues)
        rows.append(row)
    return rows


def analyse(rows):
    """Compare the inventory with the expected layout. Returns a list of report lines."""
    lines = []
    exp_clips = (len(config.EXPECTED_SUBJECTS) * len(config.EXPECTED_DIRECTIONS)
                 * len(config.EXPECTED_CAMERAS) * config.EXPECTED_CLIPS_PER_CAMERA)
    lines.append(f"Videos found: {len(rows)}   (expected {exp_clips})")
    named = [r for r in rows if r["subject"]]
    lines.append(f"Names that follow subject_direction_camera_trial: {len(named)} of {len(rows)}")
    for k, label in (("subject", "Subjects"), ("direction", "Directions"), ("camera", "Cameras")):
        c = Counter(r[k] for r in named)
        lines.append(f"{label} found: " + ", ".join(f"{a} ({n})" for a, n in sorted(c.items())))

    groups = defaultdict(int)
    for r in named:
        groups[(r["subject"], r["direction"], r["camera"])] += 1
    lines.append("")
    problems = []
    for s in config.EXPECTED_SUBJECTS:
        for d in config.EXPECTED_DIRECTIONS:
            for c in config.EXPECTED_CAMERAS:
                n = groups.get((s, d, c), 0)
                if n != config.EXPECTED_CLIPS_PER_CAMERA:
                    problems.append(f"  {s} {d} {c}: {n} clips (expected {config.EXPECTED_CLIPS_PER_CAMERA})")
    extra = [k for k in groups if k[0] not in config.EXPECTED_SUBJECTS
             or k[1] not in config.EXPECTED_DIRECTIONS or k[2] not in config.EXPECTED_CAMERAS]
    problems += [f"  {k[0]} {k[1]} {k[2]}: {groups[k]} clips (not in the expected layout)" for k in extra]
    lines.append("Subject / direction / camera folders with a different clip count than expected:"
                 if problems else "Every subject / direction / camera has the expected clip count.")
    lines += problems

    trial_gaps = []
    by_group = defaultdict(set)
    for r in named:
        by_group[(r["subject"], r["direction"], r["camera"])].add(r["trial"])
    wanted = {f"T{i:02d}" for i in range(1, config.EXPECTED_CLIPS_PER_CAMERA + 1)}
    for k, trials in sorted(by_group.items()):
        if trials != wanted and len(trials) == config.EXPECTED_CLIPS_PER_CAMERA:
            trial_gaps.append(f"  {k[0]} {k[1]} {k[2]}: trials are {sorted(trials)}")
    if trial_gaps:
        lines += ["", "Groups whose trial names are not T01..T10:"] + trial_gaps

    ids = Counter(r["clip_id"] for r in rows)
    dup_names = [a for a, n in ids.items() if n > 1]
    if dup_names:
        lines += ["", f"Duplicate clip names (same name in several folders): {dup_names[:5]}"]

    bad = [r for r in rows if r["issues"]]
    lines += ["", f"Clips with issues: {len(bad)}"]
    lines += [f"  {r['path']}: {r['issues']}" for r in bad[:40]]
    if len(bad) > 40:
        lines.append(f"  ... and {len(bad) - 40} more (see the issues column of the csv)")

    opened = [r for r in rows if r["readable"] != ""]
    if opened:
        fps = Counter(r["fps"] for r in opened if r["readable"] == "yes")
        res = Counter(f"{r['width']}x{r['height']}" for r in opened if r["readable"] == "yes")
        lines += ["", "Frame rates: " + ", ".join(f"{a} fps ({n})" for a, n in sorted(fps.items())),
                  "Resolutions: " + ", ".join(f"{a} ({n})" for a, n in sorted(res.items()))]
        dur = [r["duration_s"] for r in opened if r["duration_s"] != ""]
        if dur:
            lines.append(f"Clip length: shortest {min(dur)} s, longest {max(dur)} s")
        short = [r for r in opened if r["fps"] != "" and r["frames"] != ""
                 and 0 < int(r["frames"]) < config.SEQ_LEN]
        if short:
            lines.append(f"Clips shorter than one {config.SEQ_LEN}-frame window: {len(short)}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="names and counts only; do not open clips")
    ap.add_argument("--no_duplicates", action="store_true", help="skip the exact-copy check")
    a = ap.parse_args()
    config.ensure_dirs()
    if not list_videos():
        raise SystemExit(f"No videos found under {config.DATASET_DIR}")
    rows = inventory(a.quick)
    lines = analyse(rows)

    if not a.no_duplicates:
        seen = defaultdict(list)
        for v in list_videos():
            if v.stat().st_size > 0:
                seen[fingerprint(v)].append(v.relative_to(config.DATASET_DIR).as_posix())
        copies = [paths for paths in seen.values() if len(paths) > 1]
        lines += ["", f"Groups of identical files (same size and same first MB): {len(copies)}"]
        for paths in copies[:10]:
            lines.append("  " + "  ==  ".join(paths))
        if copies:
            lines.append("  Identical clips must stay in the same split: check them before trusting the test result.")

    cols = ["clip_id", "subject", "direction", "camera", "trial", "path", "size_mb", "readable",
            "frames", "fps", "width", "height", "duration_s", "issues"]
    with open(config.INVENTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    report = config.MANIFEST_DIR / "verification_report.txt"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {config.INVENTORY_CSV}\n       {report}")
    print("Next: python select_camera.py")


if __name__ == "__main__":
    main()
