"""Step 1: choose the camera that shows the SIDE view of the walk, for every subject + direction.

For each subject + direction, a few clips of each camera (Cam-P, Cam-S) are inspected with the
pose model. A side view shows the person crossing the image: the shoulders overlap (small shoulder
width / torso length) and the hips travel far sideways. Each camera gets a side-view score (0-1).
The better camera is selected ONLY if it clearly wins; otherwise the group is flagged
"needs_review" and no camera is selected, so nothing is guessed.

Writes  manifests/camera_selection.csv   subject, direction, selected_camera, reason, review_status, ...
        manifests/camera_review/<subject>_<direction>.jpg   Cam-P (left) and Cam-S (right) with scores

review_status:  auto_selected   chosen by the scores
                needs_review    unclear: look at the picture, then decide with --set
                manual_confirmed  decided by you; never overwritten unless you use --overwrite

    python select_camera.py                                  # inspect all groups
    python select_camera.py --set S03 PERP Cam-S --note "side view"   # record YOUR decision
    python select_camera.py --list                           # show the table
The original videos are only read. They are never changed or deleted.
"""
import argparse
import csv
import shutil
from collections import defaultdict

import numpy as np

import config
from pose_extract import list_videos, parse_clip, pick_person

LEFT_SH, RIGHT_SH, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12
COLUMNS = ["subject", "direction", "selected_camera", "reason", "review_status", "suggested_camera",
           "side_score_Cam-P", "side_score_Cam-S", "shoulder_ratio_Cam-P", "shoulder_ratio_Cam-S",
           "travel_Cam-P", "travel_Cam-S", "clips_checked_Cam-P", "clips_checked_Cam-S"]


# ----------------------------------------------------------------------------- measurements
def clip_view_metrics(kps):
    """kps: list of (17, 3) keypoint arrays (x, y, confidence) from ONE clip.
    Returns dict(valid, shoulder_ratio, travel, side_score) or None if too few usable frames."""
    ratios, hip_x, torso = [], [], []
    for kp in kps:
        if min(kp[j, 2] for j in (LEFT_SH, RIGHT_SH, LEFT_HIP, RIGHT_HIP)) < config.CAM_MIN_KP_CONF:
            continue
        sh_mid = (kp[LEFT_SH, :2] + kp[RIGHT_SH, :2]) / 2
        hip_mid = (kp[LEFT_HIP, :2] + kp[RIGHT_HIP, :2]) / 2
        t = float(np.linalg.norm(sh_mid - hip_mid))
        if t < 1e-3:
            continue
        torso.append(t)
        ratios.append(abs(float(kp[LEFT_SH, 0] - kp[RIGHT_SH, 0])) / t)
        hip_x.append(float(hip_mid[0]))
    if len(torso) < config.CAM_MIN_VALID_FRAMES:
        return None
    ratio = float(np.median(ratios))
    travel = (max(hip_x) - min(hip_x)) / float(np.median(torso))
    return {"valid": len(torso), "shoulder_ratio": ratio, "travel": travel,
            "side_score": side_score(ratio, travel)}


def side_score(shoulder_ratio, travel):
    """0 = front/back view, 1 = clean side view. Mean of two simple cues."""
    s1 = 1.0 - min(shoulder_ratio / config.CAM_SHOULDER_FRONTAL, 1.0)
    s2 = min(travel / config.CAM_TRAVEL_FULL, 1.0)
    return 0.5 * s1 + 0.5 * s2


def camera_metrics(clip_metrics):
    """Median over the inspected clips of one camera (None entries are skipped)."""
    ok = [m for m in clip_metrics if m]
    if not ok:
        return None
    return {k: float(np.median([m[k] for m in ok])) for k in ("shoulder_ratio", "travel", "side_score")} \
        | {"clips_ok": len(ok), "clips_checked": len(clip_metrics)}


def decide(metrics):
    """metrics: {camera: camera_metrics or None}. Returns (selected, suggested, reason, status)."""
    cams = config.EXPECTED_CAMERAS
    missing = [c for c in cams if metrics.get(c) is None]
    if missing:
        have = [c for c in cams if metrics.get(c)]
        suggested = have[0] if len(have) == 1 else ""
        return "", suggested, ("person not detected in enough frames, or no clips, for " +
                               ", ".join(missing)), "needs_review"
    score = {c: metrics[c]["side_score"] for c in cams}
    best = max(cams, key=lambda c: score[c])
    other = [c for c in cams if c != best][0]
    margin = score[best] - score[other]
    numbers = f"side scores {cams[0]} {score[cams[0]]:.2f}, {cams[1]} {score[cams[1]]:.2f}"
    if score[best] < config.CAM_MIN_SIDE_SCORE:
        return "", best, f"neither camera clearly shows a side view ({numbers})", "needs_review"
    if margin < config.CAM_MIN_MARGIN:
        return "", best, f"the two cameras look similar ({numbers})", "needs_review"
    return best, best, f"{best} is the clearer side view ({numbers})", "auto_selected"


# ----------------------------------------------------------------------------- video sampling
def sample_clip(model, video, n_frames):
    """Run the pose model on n evenly spaced frames. Returns (kps list, (frame, kp) middle example)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in np.linspace(0, total - 1, n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()
    if not frames:
        return [], None
    results = model(frames, conf=config.POSE_CONF, verbose=False)
    kps, example = [], None
    for frame, res in zip(frames, results):
        kp = pick_person(res)
        if kp is not None:
            kps.append(kp)
            if example is None and len(kps) >= max(1, len(frames) // 2):
                example = (frame, kp)
    return kps, example


def pick_clips(clips, n):
    """n clips spread over the trials (first, middle, last ...)."""
    clips = sorted(clips, key=lambda p: p.stem)
    if len(clips) <= n:
        return clips
    return [clips[i] for i in np.linspace(0, len(clips) - 1, n).astype(int)]


def make_review_sheet(subject, direction, examples, metrics):
    """One picture: Cam-P on the left, Cam-S on the right, skeleton + score written on each."""
    import cv2
    from check_pose import draw
    tiles = []
    for cam in config.EXPECTED_CAMERAS:
        ex = examples.get(cam)
        if ex is None:
            tile = np.zeros((360, 480, 3), np.uint8)
        else:
            tile = draw(ex[0].copy(), ex[1])
            h = 360
            tile = cv2.resize(tile, (int(tile.shape[1] * h / tile.shape[0]), h))
        m = metrics.get(cam)
        text = f"{cam}  side score " + (f"{m['side_score']:.2f}" if m else "n/a")
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(tile, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(tile)
    h = min(t.shape[0] for t in tiles)
    sheet = cv2.hconcat([t[:h] for t in tiles])
    config.CAMERA_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(config.CAMERA_REVIEW_DIR / f"{subject}_{direction}.jpg"), sheet)


# ----------------------------------------------------------------------------- table
def read_table():
    if not config.CAMERA_SELECTION_CSV.exists():
        return {}
    with open(config.CAMERA_SELECTION_CSV, newline="") as f:
        return {(r["subject"], r["direction"]): r for r in csv.DictReader(f)}


def write_table(rows):
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CAMERA_SELECTION_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow({c: rows[key].get(c, "") for c in COLUMNS})


def find_groups():
    """{(subject, direction): {camera: [clip paths]}} for every video named subject_direction_camera_trial."""
    groups = defaultdict(lambda: defaultdict(list))
    for v in list_videos():
        info = parse_clip(v)
        if info:
            groups[(info["subject"], info["direction"])][info["camera"]].append(v)
    return groups


def inspect_all(model, overwrite, sample_fn=sample_clip):
    groups = find_groups()
    if not groups:
        raise SystemExit(f"No clips named subject_direction_camera_trial under {config.DATASET_DIR}")
    old = read_table()
    if old:
        backup = config.CAMERA_SELECTION_CSV.with_name("camera_selection.previous.csv")
        shutil.copy2(config.CAMERA_SELECTION_CSV, backup)
        print(f"Existing {config.CAMERA_SELECTION_CSV.name} copied to {backup.name}")
    rows, kept = {}, 0
    for key in sorted(groups):
        subject, direction = key
        if key in old and old[key]["review_status"] == "manual_confirmed" and not overwrite:
            rows[key] = old[key]
            kept += 1
            continue
        cam_metrics, examples = {}, {}
        for cam in config.EXPECTED_CAMERAS:
            clips = pick_clips(groups[key].get(cam, []), config.CAM_SAMPLE_CLIPS)
            per_clip = []
            for clip in clips:
                kps, example = sample_fn(model, clip, config.CAM_FRAMES_PER_CLIP)
                per_clip.append(clip_view_metrics(kps))
                if example is not None and cam not in examples:
                    examples[cam] = example
            cam_metrics[cam] = camera_metrics(per_clip) if clips else None
        selected, suggested, reason, status = decide(cam_metrics)
        row = {"subject": subject, "direction": direction, "selected_camera": selected,
               "reason": reason, "review_status": status, "suggested_camera": suggested}
        for cam in config.EXPECTED_CAMERAS:
            m = cam_metrics[cam]
            row[f"side_score_{cam}"] = f"{m['side_score']:.3f}" if m else ""
            row[f"shoulder_ratio_{cam}"] = f"{m['shoulder_ratio']:.3f}" if m else ""
            row[f"travel_{cam}"] = f"{m['travel']:.2f}" if m else ""
            row[f"clips_checked_{cam}"] = f"{m['clips_ok']}/{m['clips_checked']}" if m else "0"
        rows[key] = row
        make_review_sheet(subject, direction, examples, cam_metrics)
        print(f"{subject} {direction:<15} {status:<14} {selected or '-':<6} {reason}", flush=True)
    write_table(rows)
    if kept:
        print(f"{kept} group(s) kept as manual_confirmed (use --overwrite to inspect them again)")
    return rows


def set_manual(subject, direction, camera, note):
    if camera not in config.EXPECTED_CAMERAS:
        raise SystemExit(f"camera must be one of {config.EXPECTED_CAMERAS}")
    rows = read_table()
    key = (subject, direction)
    if key not in rows:
        raise SystemExit(f"{subject} {direction} is not in {config.CAMERA_SELECTION_CSV.name}. "
                         f"Run python select_camera.py first.")
    shutil.copy2(config.CAMERA_SELECTION_CSV,
                 config.CAMERA_SELECTION_CSV.with_name("camera_selection.previous.csv"))
    rows[key].update({"selected_camera": camera, "review_status": "manual_confirmed",
                      "reason": f"decided manually: {note}" if note else "decided manually"})
    write_table(rows)
    print(f"{subject} {direction}: {camera} recorded as manual_confirmed")


def summary(rows):
    n = len(rows)
    by = defaultdict(list)
    for r in rows.values():
        by[r["review_status"]].append(r)
    print(f"\n{n} subject + direction groups: " + ", ".join(f"{len(v)} {k}" for k, v in sorted(by.items())))
    exp = len(config.EXPECTED_SUBJECTS) * len(config.EXPECTED_DIRECTIONS)
    if n != exp:
        print(f"[note] {exp} groups were expected, {n} found")
    todo = by.get("needs_review", [])
    if todo:
        print(f"\n{len(todo)} group(s) need YOUR decision. Open the picture, then record it, e.g.:")
        for r in todo:
            print(f"  {r['subject']} {r['direction']:<15} {r['reason']}")
            print(f"      picture: {config.CAMERA_REVIEW_DIR / (r['subject'] + '_' + r['direction'] + '.jpg')}")
        r = todo[0]
        print(f"  python select_camera.py --set {r['subject']} {r['direction']} "
              f"{r['suggested_camera'] or 'Cam-P'} --note \"side view\"")
        print("make_split.py leaves out every group that is not auto_selected or manual_confirmed.")
    else:
        print("All groups have a camera. Next: python make_split.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=3, metavar=("SUBJECT", "DIRECTION", "CAMERA"),
                    help="record your own decision for one group")
    ap.add_argument("--note", default="", help="reason written with --set")
    ap.add_argument("--list", action="store_true", help="show the saved table")
    ap.add_argument("--overwrite", action="store_true", help="also re-inspect manual_confirmed groups")
    a = ap.parse_args()
    config.ensure_dirs()
    if a.set:
        set_manual(*a.set, a.note)
        return
    if a.list:
        rows = read_table()
        if not rows:
            raise SystemExit("No camera_selection.csv yet. Run python select_camera.py")
        for r in rows.values():
            print(f"{r['subject']} {r['direction']:<15} {r['review_status']:<17} "
                  f"{r['selected_camera'] or '-':<6} {r['reason']}")
        summary(rows)
        return
    from pose_extract import load_model
    rows = inspect_all(load_model(), a.overwrite)
    print(f"\nSaved: {config.CAMERA_SELECTION_CSV}\nReview pictures: {config.CAMERA_REVIEW_DIR}")
    summary(rows)


if __name__ == "__main__":
    main()
