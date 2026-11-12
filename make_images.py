"""All pictures in ONE separate folder: images/

For every subject and every walking direction (selected camera only):

    images/<subject>/<direction>/frames/           the raw video frames of the walking person
    images/<subject>/<direction>/pose_extraction/  the same frames with the extracted skeleton drawn on
    images/<subject>/<direction>/skeleton/         the skeleton alone (stick figure on white)
    images/<subject>/<direction>/montage_<clip>.jpg    3 rows (frame / pose extraction / skeleton) x N frames,
                                                       cropped around the person: one gait cycle at a glance
    images/<subject>/<subject>_directions.jpg      the 3 rows for every walking direction side by side
    images/pose_extraction_overview.jpg            subjects (rows) x directions (columns), pose extraction
    images/README.txt

Needs: videos in dataset/, skeletons_raw/ (python pose_extract.py --all), manifests/dataset_split.csv.

    python make_images.py                      # trial T01 of every subject + direction, 6 frames each
    python make_images.py --frames 8 --step 5  # more frames, closer together
    python make_images.py --trials 1,2,3       # more trials
    python make_images.py --subjects S01,S02   # only some subjects
"""
import argparse
import csv

import cv2
import numpy as np

import config
from pose_extract import list_videos, manifest_videos, out_path, parse_clip

OUT = config.ROOT / "images"
DIR_ORDER = ["LINE", "PERP", "CROSS-RX-LEFT", "CROSS-RX-RIGHT"]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT = {2, 4, 6, 8, 10, 12, 14, 16}
BLUE, RED, GREEN = (255, 90, 0), (0, 0, 255), (0, 170, 0)           # BGR
CONF_MIN = 0.2


def side(j):
    return "L" if j in LEFT else ("R" if j in RIGHT else "C")


def draw_skeleton(img, xy, conf, thick=2, radius=3):
    """Draw 17-joint skeleton in place. xy (17, 2) in the pixel coordinates of img. Blue = left, red = right."""
    for a, b in config.SKELETON_EDGES:
        if conf[a] > CONF_MIN and conf[b] > CONF_MIN:
            sa, sb = side(a), side(b)
            color = BLUE if sa == sb == "L" else (RED if sa == sb == "R" else GREEN)
            cv2.line(img, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), color, thick, cv2.LINE_AA)
    for j in range(len(xy)):
        if conf[j] > CONF_MIN:
            cv2.circle(img, (int(xy[j][0]), int(xy[j][1])), radius, (30, 30, 30), -1, cv2.LINE_AA)
    return img


def person_window(sk_frames, margin=1.35):
    """Square window (cx, cy, side) in video pixels that contains the person in all given skeleton frames."""
    pts = np.concatenate([f[f[:, 2] > CONF_MIN, :2] for f in sk_frames if (f[:, 2] > CONF_MIN).any()] or
                         [np.zeros((1, 2))])
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1, 60.0) * margin


def crop_window(img, win, out, fill=(255, 255, 255)):
    cx, cy, s = win
    k = out / s
    m = np.array([[k, 0, out / 2 - k * cx], [0, k, out / 2 - k * cy]], np.float32)
    return cv2.warpAffine(img, m, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill), m


def label(img, text, org=(6, 18), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def three_tiles(frame, kp, win, out):
    """(raw crop, pose-extraction crop, skeleton-only tile), all `out` x `out` pixels, same window."""
    raw, m = crop_window(frame, win, out, fill=(255, 255, 255))
    over = frame.copy()
    draw_skeleton(over, kp[:, :2], kp[:, 2], thick=max(2, frame.shape[1] // 400), radius=max(3, frame.shape[1] // 300))
    over, _ = crop_window(over, win, out)
    xy = kp[:, :2] @ m[:, :2].T + m[:, 2]
    stick = np.full((out, out, 3), 255, np.uint8)
    draw_skeleton(stick, xy, kp[:, 2], thick=3, radius=4)
    return raw, over, stick


def choose_frames(sk, n, step):
    present = np.where(sk[..., 2].mean(axis=1) > 0)[0]
    if len(present) == 0:
        return []
    mid = int(present[len(present) // 2])
    idx = mid + (np.arange(n) - (n - 1) / 2.0) * step
    idx = np.clip(np.round(idx).astype(int), present[0], present[-1])
    return [int(i) for i in dict.fromkeys(idx) if sk[i, :, 2].mean() > 0]


def read_frames(video, indices):
    cap = cv2.VideoCapture(str(video))
    frames = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames[i] = f
    cap.release()
    return frames


def trial_number(clip):
    info = parse_clip(clip)
    return int(info["trial"][1:]) if info else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="1", help="trial numbers, e.g. 1 or 1,2,3 (default 1)")
    ap.add_argument("--frames", type=int, default=6, help="frames per clip (default 6)")
    ap.add_argument("--step", type=int, default=6, help="frames between pictures (default 6 = 0.2 s)")
    ap.add_argument("--subjects", default="", help="e.g. S01,S02 (default all)")
    ap.add_argument("--max_width", type=int, default=960, help="width of the saved full frames")
    ap.add_argument("--tile", type=int, default=240, help="size of the crops in the montages")
    a = ap.parse_args()
    trials = {int(t) for t in a.trials.split(",") if t.strip()}
    subjects = {s.strip() for s in a.subjects.split(",") if s.strip()}

    videos = manifest_videos()
    if videos is None:
        videos = list_videos()
    todo = []
    for v in videos:
        info = parse_clip(v)
        if info and out_path(v).exists() and trial_number(v) in trials and (not subjects or info["subject"] in subjects):
            todo.append((info, v))
    if not todo:
        raise SystemExit("Nothing to draw: need videos in dataset/ AND skeletons_raw/ (python pose_extract.py --all).")
    todo.sort(key=lambda t: (t[0]["subject"], DIR_ORDER.index(t[0]["direction"]) if t[0]["direction"] in DIR_ORDER
                             else 99, t[0]["trial"]))

    by_subject, overview, index_rows = {}, {}, []
    for info, v in todo:
        s, d, clip = info["subject"], info["direction"], info["clip_id"]
        sk = np.load(out_path(v))
        picks = choose_frames(sk, a.frames, a.step)
        frames = read_frames(v, picks)
        picks = [i for i in picks if i in frames]
        if not picks:
            print(f"[warn] {clip}: no readable frame with a person")
            continue
        base = OUT / s / d
        for sub in ("frames", "pose_extraction", "skeleton"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        win = person_window([sk[i] for i in picks])
        cols = [[], [], []]
        for i in picks:
            f, kp = frames[i], sk[i]
            scale = min(1.0, a.max_width / f.shape[1])
            small = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else f.copy()
            kp_s = kp.copy()
            kp_s[:, :2] *= scale
            over = small.copy()
            draw_skeleton(over, kp_s[:, :2], kp_s[:, 2], thick=max(2, over.shape[1] // 400),
                          radius=max(3, over.shape[1] // 300))
            name = f"{clip}_f{i:05d}"
            cv2.imwrite(str(base / "frames" / f"{name}.jpg"), small)
            cv2.imwrite(str(base / "pose_extraction" / f"{name}.jpg"), over)
            raw, ov, stick = three_tiles(f, kp, win, a.tile)
            big_stick, _ = crop_window(np.full_like(f, 255), win, 2 * a.tile)
            _, m2 = crop_window(f, win, 2 * a.tile)
            draw_skeleton(big_stick, kp[:, :2] @ m2[:, :2].T + m2[:, 2], kp[:, 2], thick=5, radius=7)
            cv2.imwrite(str(base / "skeleton" / f"{name}.png"), big_stick)
            for row, tile in zip(cols, (raw, ov, stick)):
                row.append(label(tile.copy(), f"f{i}"))
            index_rows.append([s, d, clip, i, f"frames/{name}.jpg", f"pose_extraction/{name}.jpg",
                               f"skeleton/{name}.png"])
        montage = np.vstack([np.hstack(r) for r in cols])
        label(montage, "video frame", (6, a.tile - 8))
        label(montage, "pose extraction", (6, 2 * a.tile - 8))
        label(montage, "skeleton", (6, 3 * a.tile - 8))
        cv2.imwrite(str(base / f"montage_{clip}.jpg"), montage)
        # one column per direction: the middle picked frame, window around that frame's person
        mid = picks[len(picks) // 2]
        w1 = person_window([sk[mid]], margin=1.5)
        t3 = three_tiles(frames[mid], sk[mid], w1, a.tile)
        by_subject.setdefault(s, {}).setdefault(d, t3)
        overview.setdefault(s, {}).setdefault(d, t3[1])
        print(f"  {clip}: {len(picks)} frames -> {base}")

    for s, dirs in by_subject.items():
        order = [d for d in DIR_ORDER if d in dirs] + [d for d in dirs if d not in DIR_ORDER]
        cols = [np.vstack([label(t.copy(), d, (6, 18)) if k == 0 else t for k, t in enumerate(dirs[d])])
                for d in order]
        cv2.imwrite(str(OUT / s / f"{s}_directions.jpg"), np.hstack(cols))

    subs = sorted(overview)
    all_dirs = [d for d in DIR_ORDER if any(d in overview[s] for s in subs)]
    blank = np.full((a.tile, a.tile, 3), 235, np.uint8)
    rows = []
    for s in subs:
        row = [label(overview[s].get(d, blank).copy(), f"{s} {d}", (6, 18), 0.45) for d in all_dirs]
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(str(OUT / "pose_extraction_overview.jpg"), np.vstack(rows))
    with open(OUT / "images_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "direction", "clip", "frame", "raw_frame", "pose_extraction", "skeleton"])
        w.writerows(index_rows)
    (OUT / "README.txt").write_text(
        "images/<subject>/<direction>/frames/            raw video frames of the walking person\n"
        "images/<subject>/<direction>/pose_extraction/   the same frames with the extracted skeleton drawn on\n"
        "images/<subject>/<direction>/skeleton/          the skeleton alone (blue = left side, red = right side)\n"
        "images/<subject>/<direction>/montage_<clip>.jpg frame / pose extraction / skeleton, cropped, one walk\n"
        "images/<subject>/<subject>_directions.jpg       the same three rows for every walking direction\n"
        "images/pose_extraction_overview.jpg             all subjects x all directions\n"
        "images/images_index.csv                         list of every picture\n"
        "Pose model: HRNet-W32.  Frames are those of the selected camera listed in manifests/dataset_split.csv.\n")
    print(f"\nDone. Pictures are in {OUT}")


if __name__ == "__main__":
    main()
