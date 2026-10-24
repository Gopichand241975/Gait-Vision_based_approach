"""Step 1: videos -> 17-joint skeletons.

Reads   every .mp4 anywhere under dataset/ (e.g. dataset/S01/LINE/videos/S01_LINE_Cam-P.mp4)
Writes  skeletons_raw/<person_id>/<video>.npy   shape (T, 17, 3) = x, y (pixels), confidence
        results/skeleton/skeleton_summary.csv   one row per video

Works in batches: each run processes the next N unprocessed videos (default 5) and stops.
Run it again for the next batch. Finished videos are skipped automatically.
"""
import argparse
import csv

import cv2
import numpy as np
from tqdm import tqdm

import config


def load_model():
    from ultralytics import YOLO
    return YOLO(config.POSE_MODEL)


def list_videos():
    """Every video anywhere under dataset/. Person id: see person_of()."""
    return sorted(p for p in config.DATASET_DIR.rglob("*")
                  if p.is_file() and p.suffix.lower() in config.VIDEO_EXTS)


def person_of(video):
    """Person id of a video.
    dataset/S01/LINE/videos/S01_LINE_Cam-P.mp4  -> S01  (folder is called 'videos': use the file name)
    dataset/cam1/P01/P01_STR_cam1.mp4           -> P01  (folder name)"""
    if video.parent.name.lower() == "videos":
        return video.stem.split("_")[0]
    return video.parent.name


def out_path(video):
    return config.SKELETON_RAW_DIR / person_of(video) / (video.stem + ".npy")


def failed_path(video):
    return config.SKELETON_RAW_DIR / person_of(video) / (video.stem + ".failed")


def pending_videos(retry=False):
    todo = []
    for v in list_videos():
        if out_path(v).exists():
            continue
        if failed_path(v).exists() and not retry:
            continue
        todo.append(v)
    return todo


def pick_person(result):
    """(17, 3) array for the largest detected person, or None."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    xy = result.keypoints.xy[i].cpu().numpy()
    if result.keypoints.conf is None:
        conf = np.ones((config.NUM_JOINTS, 1), dtype=np.float32)
    else:
        conf = result.keypoints.conf[i].cpu().numpy().reshape(-1, 1)
    return np.concatenate([xy, conf], axis=1).astype(np.float32)


def fill_missing(seq, valid):
    """Interpolate frames without a detected person. Filled frames get confidence 0."""
    idx_valid = np.where(valid)[0]
    idx_all = np.arange(len(seq))
    flat = seq.reshape(len(seq), -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(idx_all, idx_valid, flat[idx_valid, c])
    out = out.reshape(seq.shape)
    out[~valid, :, 2] = 0.0
    return out


def extract_video(model, video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] cannot open {video_path}")
        return None
    frames, valid = [], []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and len(frames) >= max_frames):
            break
        res = model(frame, conf=config.POSE_CONF, verbose=False)[0]
        kp = pick_person(res)
        if kp is None:
            frames.append(np.zeros((config.NUM_JOINTS, config.NUM_CHANNELS), np.float32))
            valid.append(False)
        else:
            frames.append(kp)
            valid.append(True)
    cap.release()
    valid = np.array(valid)
    if len(frames) == 0 or valid.sum() < config.MIN_VALID_FRAMES:
        print(f"  [warn] {video_path.name}: only {int(valid.sum())} frames with a person, skipped")
        return None
    return fill_missing(np.stack(frames), valid)


def write_summary():
    config.RESULTS_SKELETON_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_SKELETON_DIR / "skeleton_summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "video", "frames", "first_person_frame", "last_person_frame",
                    "frames_with_person_pct", "mean_confidence"])
        for p in sorted(config.SKELETON_RAW_DIR.glob("*/*.npy")):
            a = np.load(p)
            present = np.where(a[..., 2].mean(axis=1) > 0)[0]
            first = int(present[0]) if len(present) else -1
            last = int(present[-1]) if len(present) else -1
            w.writerow([p.parent.name, p.stem, a.shape[0], first, last,
                        round(100 * len(present) / len(a), 1),
                        round(float(a[present][..., 2].mean()) if len(present) else 0.0, 3)])
    return path


def run(batch_size=None, max_frames=None, retry=False, model=None):
    """Process the next batch. Returns the number of videos still pending afterwards."""
    config.ensure_dirs()
    todo = pending_videos(retry)
    if not todo:
        write_summary()
        return 0
    batch = todo[:batch_size] if batch_size else todo
    model = model or load_model()
    for vp in tqdm(batch, desc="pose extraction"):
        arr = extract_video(model, vp, max_frames)
        o = out_path(vp)
        o.parent.mkdir(parents=True, exist_ok=True)
        if arr is None:
            failed_path(vp).write_text("too few detected frames or unreadable video\n")
            continue
        np.save(o, arr)
        print(f"  {person_of(vp)}/{vp.name} -> {arr.shape}")
    write_summary()
    return len(pending_videos(retry))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=config.BATCH_VIDEOS,
                    help="videos per run (default %(default)s)")
    ap.add_argument("--all", action="store_true", help="process every pending video")
    ap.add_argument("--max_frames", type=int, default=None, help="only first N frames (testing)")
    ap.add_argument("--retry", action="store_true", help="retry videos that failed before")
    a = ap.parse_args()
    if not list_videos():
        raise SystemExit(f"No videos found. Put them under {config.DATASET_DIR}")
    left = run(None if a.all else a.batch_size, a.max_frames, a.retry)
    print(f"Done. Skeleton summary: {config.RESULTS_SKELETON_DIR / 'skeleton_summary.csv'}")
    if left:
        print(f"{left} video(s) still pending. Run the same command again for the next batch.")
    else:
        print("All videos processed. Next: python check_pose.py, then python preprocess.py")


if __name__ == "__main__":
    main()