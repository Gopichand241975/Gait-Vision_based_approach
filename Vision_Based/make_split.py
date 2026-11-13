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
USABLE = ("auto_selected",)
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


def read_trial_sessions(subject, direction):
    """Map trial label -> recording session id, using the subject/direction trials CSV if present.
    The trials files are the reliable source of session/boot metadata; filenames alone are not enough.
    """
    trials_csv = sorted((config.DATASET_DIR / subject / direction).glob("*_trials.csv"))
    if not trials_csv:
        return {}
    session_map = {}
    with open(trials_csv[0], newline="") as f:
        for row in csv.DictReader(f):
            trial = row.get("trial")
            if trial is None:
                continue
            try:
                trial_key = f"T{int(float(str(trial).strip())):02d}"
            except ValueError:
                continue
            session_id = row.get("boot_id") or row.get("session_id") or row.get("recording_id") or row.get("session")
            if session_id:
                session_map[trial_key] = str(session_id)
    return session_map


def build_split():
    selection = read_selection()
    clips = scan_clips()
    rows, notes = [], []

    keys = sorted({(s, d) for s, d, _ in clips} | set(selection))
    for subject, direction in keys:
        sel = selection.get((subject, direction))
        label = f"{subject} {direction}"

        if not sel:
            continue
        if sel["review_status"] not in USABLE:
            notes.append(f"{label}: review_status={sel['review_status']} -> left out")
            continue
        if not sel["selected_camera"]:
            notes.append(f"{label}: no selected camera -> left out")
            continue

        cam = sel["selected_camera"]
        items = sorted(clips.get((subject, direction, cam), []),
                       key=lambda i: (i.get("session", ""), trial_number(i["trial"])))

        if len(items) < 3:
            notes.append(f"{label} {cam}: only {len(items)} clips -> left out")
            continue

        trial_sessions = read_trial_sessions(subject, direction)
        if any(trial_sessions.get(it["trial"]) is None for it in items):
            notes.append(f"{label} {cam}: some trials have no recording-session metadata -> left out to avoid leakage")
            continue
        session_groups = defaultdict(list)
        for it in items:
            session_groups[str(trial_sessions[it["trial"]])].append(it)

        # For an honest benchmark, each split must be built from different recording sessions.
        # If the group only has one recording session, it cannot be assigned to train/val/test
        # without leakage, so we leave the group out.
        if len(session_groups) < 3:
            notes.append(f"{label} {cam}: only {len(session_groups)} distinct recording session(s) -> left out to avoid leakage")
            continue

        sessions = sorted(session_groups)
        if len(sessions) < 3:
            notes.append(f"{label} {cam}: not enough distinct recording sessions for a valid split -> left out")
            continue

        rng = random.Random(f"{config.SPLIT_SEED}|{subject}|{direction}")
        rng.shuffle(sessions)
        n_total = len(sessions)
        n_train = max(1, int(round(n_total * 0.6)))
        n_val = max(1, int(round(n_total * 0.2)))
        n_test = max(1, n_total - n_train - n_val)
        if n_train + n_val + n_test > n_total:
            n_test = n_total - n_train - n_val
        if n_test < 1:
            notes.append(f"{label} {cam}: session split became degenerate -> left out")
            continue

        train_sessions = set(sessions[:n_train])
        val_sessions = set(sessions[n_train:n_train + n_val])
        test_sessions = set(sessions[n_train + n_val:])

        split_of = {}
        for it in items:
            sid = trial_sessions[it["trial"]]
            if sid in train_sessions:
                split_of[it["clip_id"]] = "train"
            elif sid in val_sessions:
                split_of[it["clip_id"]] = "val"
            elif sid in test_sessions:
                split_of[it["clip_id"]] = "test"
            else:
                notes.append(f"{label} {cam}: session {sid} did not fall into train/val/test -> left out")
                split_of.clear()
                break

        if len(split_of) != len(items):
            notes.append(f"{label} {cam}: incomplete session assignment -> left out")
            continue

        # if the same session is split across train/val/test, reject that whole group
        split_sessions = defaultdict(set)
        for it in items:
            sid = trial_sessions[it["trial"]]
            split_sessions[sid].add(split_of[it["clip_id"]])
        if any(len(v) > 1 for v in split_sessions.values()):
            notes.append(f"{label} {cam}: same recording session appears in multiple splits -> left out")
            continue

        status = "ok" if len(items) == config.EXPECTED_CLIPS_PER_CAMERA else f"nonstandard_count_{len(items)}"

        for it in items:
            p = it["path"]
            rows.append({
                "clip_id": it["clip_id"],
                "original_filename": p.name,
                "relative_path": p.relative_to(config.DATASET_DIR).as_posix(),
                "subject": subject,
                "direction": direction,
                "selected_camera": cam,
                "trial": it["trial"],
                "split": split_of[it["clip_id"]],
                "recording_id": f"{subject}_{direction}_{cam}",
                "group_clips": len(items),
                "group_status": status,
                "seed": config.SPLIT_SEED
            })

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



# ----------------------------------------------------------------------------- direction split (honest protocol)
USABLE_DIRECTION = ("auto_selected", "manual_confirmed")


def direction_split_of(direction):
    """train / val / test for a walking direction, from config (TRAIN_DIRECTIONS, VAL_DIRECTION, TEST_DIRECTION)."""
    if direction == config.TEST_DIRECTION:
        return "test"
    if config.VAL_DIRECTION and direction == config.VAL_DIRECTION:
        return "val"
    if config.TRAIN_DIRECTIONS and direction not in config.TRAIN_DIRECTIONS:
        return None
    return "train"


def build_direction_split():
    """Every clip of the selected camera of every usable group (auto_selected AND manual_confirmed).
    The split follows the WALKING DIRECTION, so no direction is shared between train / val / test."""
    selection = read_selection()
    clips = scan_clips()
    rows, notes = [], []
    for (subject, direction), sel in sorted(selection.items()):
        label = f"{subject} {direction}"
        if sel["review_status"] not in USABLE_DIRECTION or not sel["selected_camera"]:
            notes.append(f"{label}: review_status={sel['review_status']}, no camera decided -> left out "
                         f"(decide with: python select_camera.py --set {subject} {direction} Cam-P|Cam-S --note \"...\")")
            continue
        split = direction_split_of(direction)
        if split is None:
            notes.append(f"{label}: direction not in TRAIN/VAL/TEST_DIRECTION -> left out")
            continue
        cam = sel["selected_camera"]
        items = sorted(clips.get((subject, direction, cam), []), key=lambda i: trial_number(i["trial"]))
        if not items:
            notes.append(f"{label}: no clips of {cam} found under dataset/ -> left out")
            continue
        status = "ok" if len(items) == config.EXPECTED_CLIPS_PER_CAMERA else f"nonstandard_count_{len(items)}"
        for it in items:
            p = it["path"]
            rows.append({"clip_id": it["clip_id"], "original_filename": p.name,
                         "relative_path": p.relative_to(config.DATASET_DIR).as_posix(),
                         "subject": subject, "direction": direction, "selected_camera": cam,
                         "trial": it["trial"], "split": split,
                         "recording_id": f"{subject}_{direction}_{cam}", "group_clips": len(items),
                         "group_status": status, "seed": "direction_holdout"})
    rows.sort(key=lambda r: (ORDER[r["split"]], r["subject"], r["direction"], trial_number(r["trial"])))
    return rows, notes


def verify_direction_split(rows):
    problems = []
    ids = Counter(r["clip_id"] for r in rows)
    problems += [f"clip {c} appears {n} times" for c, n in ids.items() if n > 1]
    selection = read_selection()
    dirs_of = defaultdict(set)
    for r in rows:
        dirs_of[r["direction"]].add(r["split"])
        sel = selection.get((r["subject"], r["direction"]))
        if sel is None or sel["selected_camera"] != r["selected_camera"]:
            problems.append(f"{r['clip_id']}: camera differs from camera_selection.csv")
        if not (config.DATASET_DIR / r["relative_path"]).exists():
            problems.append(f"{r['clip_id']}: file {r['relative_path']} does not exist")
        info = parse_clip(r["clip_id"])
        if info is None or (info["subject"], info["direction"], info["camera"]) != \
                (r["subject"], r["direction"], r["selected_camera"]):
            problems.append(f"{r['clip_id']}: table does not match the file name")
    problems += [f"direction {d} is in several splits {sorted(s)} (leakage)" for d, s in dirs_of.items() if len(s) > 1]
    try:
        from verify_dataset import fingerprint
        where = defaultdict(set)
        for r in rows:
            p = config.DATASET_DIR / r["relative_path"]
            if p.exists() and p.stat().st_size > 0:
                where[fingerprint(p)].add(r["split"])
        n_leak = sum(len(s) > 1 for s in where.values())
        if n_leak:
            problems.append(f"{n_leak} identical file(s) are in different splits (data leakage)")
    except Exception as exc:                                    # e.g. OpenCV missing: the check is optional
        print(f"[info] identical-file check skipped ({exc})")
    return problems


def summarise_direction_split(rows, notes):
    sizes = Counter(r["split"] for r in rows)
    expected = len(config.EXPECTED_SUBJECTS) * len(config.EXPECTED_DIRECTIONS) * config.EXPECTED_CLIPS_PER_CAMERA
    lines = [f"Clips in the split: {len(rows)} of {expected} expected   "
             f"train {sizes['train']}   val {sizes['val']}   test {sizes['test']}",
             f"train directions: {sorted({r['direction'] for r in rows if r['split'] == 'train'})}",
             f"val direction:    {sorted({r['direction'] for r in rows if r['split'] == 'val'})}",
             f"test direction:   {sorted({r['direction'] for r in rows if r['split'] == 'test'})}"]
    if len(rows) != expected:
        lines.append("DIFFERENT from the expected count. Reasons:")
        lines += [f"  {n}" for n in notes] or ["  (see group_status column)"]
    per_subject = defaultdict(Counter)
    for r in rows:
        per_subject[r["subject"]][r["split"]] += 1
    lines += ["", "subject   train  val  test"]
    lines += [f"{s:<9} {c['train']:>5} {c['val']:>4} {c['test']:>5}" for s, c in sorted(per_subject.items())]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only verify the existing dataset_split.csv")
    ap.add_argument("--no_duplicates", action="store_true", help="skip the identical-file check")
    ap.add_argument("--direction_split", action="store_true",
                    help="HONEST protocol: all clips of every selected camera (auto_selected and manual_confirmed), "
                         "split by walking direction (config TRAIN/VAL/TEST_DIRECTION). Use this one.")
    a = ap.parse_args()
    config.ensure_dirs()
    if a.direction_split:
        rows, notes = build_direction_split()
        if not rows:
            raise SystemExit("No clips could be assigned. " + "; ".join(notes[:3]))
        if config.DATASET_SPLIT_CSV.exists():
            shutil.copy2(config.DATASET_SPLIT_CSV, config.DATASET_SPLIT_CSV.with_name("dataset_split.previous.csv"))
        with open(config.DATASET_SPLIT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {config.DATASET_SPLIT_CSV}")
        print("\n".join(summarise_direction_split(rows, notes)))
        problems = verify_direction_split(rows)
        if problems:
            print(f"\nVERIFICATION FAILED ({len(problems)} problem(s)):")
            print("\n".join(f"  {p}" for p in problems[:30]))
            raise SystemExit(1)
        print("\nVerification passed: every clip is in exactly one split, no walking direction is shared between "
              "train / val / test, and the camera matches camera_selection.csv.")
        print("Next: python pose_extract.py --all")
        return
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
