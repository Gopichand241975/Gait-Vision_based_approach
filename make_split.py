"""Step 2: split the SELECTED clips into training / validation / testing (6 / 2 / 2 per group).

Group = one subject + one direction (e.g. S03 PERP). Only the clips of the camera chosen in
manifests/camera_selection.csv are used (status auto_selected or manual_confirmed). The clips of a
group are shuffled with a fixed seed (config.SPLIT_SEED), then: first 6 = train, next 2 = val, last 2 = test.
The result is manifests/dataset_split.csv; the same inputs always give the same file.

Counts that differ from the expected ones are REPORTED, never forced:
  - a group without a selected camera (needs_review) is left out;
  - a group that does not have 10 clips is split in the same 60/20/20 proportions and marked
    "nonstandard_count" in the group_status column (a group with fewer than 3 clips is left out).

    python make_split.py             # create dataset_split.csv, then verify it
    python make_split.py --verify    # only verify the existing dataset_split.csv
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict

import config
from pose_extract import list_videos, parse_clip

COLUMNS = ["clip_id", "original_filename", "relative_path", "subject", "direction", "selected_camera",
           "trial", "split", "recording_id", "group_clips", "group_status", "seed"]
USABLE = ("auto_selected", "manual_confirmed")
ORDER = {"train": 0, "val": 1, "test": 2}


def trial_number(trial):
    return int(trial[1:]) if trial[1:].isdigit() else 10 ** 6


def group_counts(n):
    """(train, val, test) clips for a group of n clips, or None if n is too small."""
    base = sum(config.SPLIT_CLIPS)
    if n == base:
        return tuple(config.SPLIT_CLIPS)
    if n < 3:
        return None
    n_val = max(1, round(n * config.SPLIT_CLIPS[1] / base))
    n_test = max(1, round(n * config.SPLIT_CLIPS[2] / base))
    return n - n_val - n_test, n_val, n_test


def read_selection():
    if not config.CAMERA_SELECTION_CSV.exists():
        raise SystemExit(f"{config.CAMERA_SELECTION_CSV} does not exist. Run python select_camera.py first.")
    with open(config.CAMERA_SELECTION_CSV, newline="") as f:
        return {(r["subject"], r["direction"]): r for r in csv.DictReader(f)}


def scan_clips():
    """{(subject, direction, camera): [(trial, path, info), ...]} from the file names only."""
    groups = defaultdict(list)
    for v in list_videos():
        info = parse_clip(v)
        if info:
            groups[(info["subject"], info["direction"], info["camera"])].append(info | {"path": v})
    return groups


def build_split():
    selection = read_selection()
    clips = scan_clips()
    rows, notes = [], []
    keys = sorted({(s, d) for s, d, _ in clips} | set(selection))
    for subject, direction in keys:
        sel = selection.get((subject, direction))
        label = f"{subject} {direction}"
        if sel is None:
            notes.append(f"{label}: has clips but is not in camera_selection.csv: left out")
            continue
        if sel["review_status"] not in USABLE or not sel["selected_camera"]:
            notes.append(f"{label}: camera not decided ({sel['review_status']}): left out")
            continue
        cam = sel["selected_camera"]
        items = sorted(clips.get((subject, direction, cam), []), key=lambda i: trial_number(i["trial"]))
        n = len(items)
        counts = group_counts(n)
        if counts is None:
            notes.append(f"{label} {cam}: only {n} clip(s) found: left out")
            continue
        status = "ok" if n == config.EXPECTED_CLIPS_PER_CAMERA and counts == tuple(config.SPLIT_CLIPS) \
            else f"nonstandard_count_{n}"
        if status != "ok":
            notes.append(f"{label} {cam}: {n} clips (expected {config.EXPECTED_CLIPS_PER_CAMERA}), "
                         f"split {counts[0]}/{counts[1]}/{counts[2]}")
        order = list(items)
        random.Random(f"{config.SPLIT_SEED}|{subject}|{direction}").shuffle(order)
        split_of = {}
        for i, it in enumerate(order):
            split_of[it["clip_id"]] = "train" if i < counts[0] else "val" if i < counts[0] + counts[1] else "test"
        for it in items:
            p = it["path"]
            rows.append({"clip_id": it["clip_id"], "original_filename": p.name,
                         "relative_path": p.relative_to(config.DATASET_DIR).as_posix(),
                         "subject": subject, "direction": direction, "selected_camera": cam,
                         "trial": it["trial"], "split": split_of[it["clip_id"]],
                         "recording_id": f"{subject}_{direction}_{cam}", "group_clips": n,
                         "group_status": status, "seed": config.SPLIT_SEED})
    rows.sort(key=lambda r: (r["subject"], r["direction"], ORDER[r["split"]], trial_number(r["trial"])))
    return rows, notes


def verify(rows, check_duplicates=True):
    """Returns a list of problems (empty = the split is valid)."""
    problems = []
    ids = Counter(r["clip_id"] for r in rows)
    problems += [f"clip {c} appears {n} times (must be in exactly one split)" for c, n in ids.items() if n > 1]
    bad_split = [r["clip_id"] for r in rows if r["split"] not in ORDER]
    problems += [f"clip {c} has an invalid split value" for c in bad_split]
    selection = read_selection()
    per_group = defaultdict(Counter)
    for r in rows:
        info = parse_clip(r["clip_id"])
        if info is None or (info["subject"], info["direction"], info["camera"]) != \
                (r["subject"], r["direction"], r["selected_camera"]):
            problems.append(f"{r['clip_id']}: subject / direction / camera in the table do not match the file name")
        sel = selection.get((r["subject"], r["direction"]))
        if sel is None or sel["selected_camera"] != r["selected_camera"] or sel["review_status"] not in USABLE:
            problems.append(f"{r['clip_id']}: camera differs from camera_selection.csv")
        if not (config.DATASET_DIR / r["relative_path"]).exists():
            problems.append(f"{r['clip_id']}: file {r['relative_path']} does not exist")
        if r["subject"] not in config.EXPECTED_SUBJECTS:
            problems.append(f"{r['clip_id']}: unexpected subject {r['subject']}")
        per_group[(r["subject"], r["direction"])][r["split"]] += 1
    for key, c in sorted(per_group.items()):
        want = group_counts(sum(c.values()))
        got = (c["train"], c["val"], c["test"])
        if want != got or 0 in got:
            problems.append(f"{key[0]} {key[1]}: split is {got[0]}/{got[1]}/{got[2]}, expected {want}")
    if check_duplicates:
        from verify_dataset import fingerprint
        where = defaultdict(set)
        for r in rows:
            p = config.DATASET_DIR / r["relative_path"]
            if p.exists() and p.stat().st_size > 0:
                where[fingerprint(p)].add(r["split"])
        n_leak = sum(len(s) > 1 for s in where.values())
        if n_leak:
            problems.append(f"{n_leak} identical file(s) are in different splits (data leakage)")
    return problems


def summarise(rows, notes):
    lines = []
    sizes = Counter(r["split"] for r in rows)
    groups = {(r["subject"], r["direction"]) for r in rows}
    lines.append(f"Groups (subject + direction) in the split: {len(groups)} of "
                 f"{len(config.EXPECTED_SUBJECTS) * len(config.EXPECTED_DIRECTIONS)} expected")
    lines.append(f"Selected clips: {len(rows)}   train {sizes['train']}   val {sizes['val']}   test {sizes['test']}")
    full = len(config.EXPECTED_SUBJECTS) * len(config.EXPECTED_DIRECTIONS)
    exp = [full * c for c in config.SPLIT_CLIPS]
    lines.append(f"Expected when every group is complete: {sum(exp)} clips = "
                 f"train {exp[0]}, val {exp[1]}, test {exp[2]}")
    if (sizes["train"], sizes["val"], sizes["test"]) != tuple(exp):
        lines.append("DIFFERENT from the expected counts. Reasons:")
        lines += [f"  {n}" for n in notes] or ["  (see the group_status column of the csv)"]
    elif notes:
        lines += ["Notes:"] + [f"  {n}" for n in notes]
    per_subject = defaultdict(Counter)
    for r in rows:
        per_subject[r["subject"]][r["split"]] += 1
    lines.append("")
    lines.append("subject   train  val  test")
    for s in sorted(per_subject):
        c = per_subject[s]
        lines.append(f"{s:<9} {c['train']:>5} {c['val']:>4} {c['test']:>5}")
    lines += ["", "Note: the clips of one group come from the same camera and the same session. Training, "
              "validation and test clips therefore share the recording conditions (same day, same "
              "background); expect the test result to be higher than on a completely new session."]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only verify the existing dataset_split.csv")
    ap.add_argument("--no_duplicates", action="store_true", help="skip the identical-file check")
    a = ap.parse_args()
    config.ensure_dirs()
    if a.verify:
        if not config.DATASET_SPLIT_CSV.exists():
            raise SystemExit("No dataset_split.csv yet. Run python make_split.py")
        with open(config.DATASET_SPLIT_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        notes = []
    else:
        rows, notes = build_split()
        if not rows:
            raise SystemExit("No clips could be assigned. " + "; ".join(notes[:3]))
        if config.DATASET_SPLIT_CSV.exists():
            backup = config.DATASET_SPLIT_CSV.with_name("dataset_split.previous.csv")
            shutil.copy2(config.DATASET_SPLIT_CSV, backup)
            print(f"Existing {config.DATASET_SPLIT_CSV.name} copied to {backup.name}")
        with open(config.DATASET_SPLIT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {config.DATASET_SPLIT_CSV}")
    print("\n".join(summarise(rows, notes)))
    problems = verify(rows, not a.no_duplicates)
    if problems:
        print(f"\nVERIFICATION FAILED ({len(problems)} problem(s)):")
        print("\n".join(f"  {p}" for p in problems[:30]))
        raise SystemExit(1)
    print("\nVerification passed: every selected clip is in exactly one split, the camera matches "
          "camera_selection.csv, and each group has the expected train / val / test counts.")
    print("Next: python pose_extract.py --all")


if __name__ == "__main__":
    main()
