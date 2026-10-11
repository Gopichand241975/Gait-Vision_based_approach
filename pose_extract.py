"""Extract 17 COCO keypoints per frame from every video in dataset/<person_id>/.

Output: skeletons_raw/<person_id>/<video_name>.npy with shape (T, 17, 3)
        columns = x (pixels), y (pixels), confidence
"""
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import config

def load_model():
    from ultralytics import YOLO
    return YOLO(config.POSE_MODEL)  # downloads weights on first run

def pick_person(result):
    """Return a (17, 3) array for the largest detected person, or None."""
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(areas))
    xy = result.keypoints.xy[i].cpu().numpy()            # (17, 2)
    if result.keypoints.conf is None:
        conf = np.ones((config.NUM_JOINTS, 1), dtype=np.float32)
    else:
        conf = result.keypoints.conf[i].cpu().numpy().reshape(-1, 1)
    return np.concatenate([xy, conf], axis=1).astype(np.float32)

def fill_missing(seq, valid):
    """Linearly interpolate frames where no person was found. Filled frames get conf=0."""
    t = len(seq)
    idx_valid = np.where(valid)[0]
    idx_all = np.arange(t)
    flat = seq.reshape(t, -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(idx_all, idx_valid, flat[idx_valid, c])
    out = out.reshape(seq.shape)
    out[~valid, :, 2] = 0.0
    return out

def extract_video(model, video_path, max_frames=None):
    """Return (T, 17, 3) array, or None if too few frames had a detected person."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] cannot open {video_path}")
        return None
    frames, valid = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and len(frames) >= max_frames:
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

    if len(frames) == 0:
        return None
    seq = np.stack(frames)
    valid = np.array(valid)
    if valid.sum() < config.MIN_VALID_FRAMES:
        print(f"  [warn] {video_path.name}: only {valid.sum()} valid frames, skipped")
        return None
    return fill_missing(seq, valid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_frames", type=int, default=None,
                        help="only read the first N frames of each video (fast testing)")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="only process the first N videos (fast testing)")
    parser.add_argument("--overwrite", action="store_true", help="redo existing outputs")
    args = parser.parse_args()

    config.ensure_dirs()
    videos = sorted(p for p in config.DATASET_DIR.glob("*/*")
                    if p.suffix.lower() in config.VIDEO_EXTS)
    if not videos:
        raise SystemExit(f"No videos found. Expected {config.DATASET_DIR}/<person_id>/*.mp4")
    if args.max_videos:
        videos = videos[:args.max_videos]

    model = load_model()
    done = skipped = 0
    for vp in tqdm(videos, desc="pose extraction"):
        out_path = config.SKELETON_RAW_DIR / vp.parent.name / (vp.stem + ".npy")
        if out_path.exists() and not args.overwrite:
            continue
        arr = extract_video(model, vp, args.max_frames)
        if arr is None:
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, arr)
        done += 1
        print(f"  {vp.parent.name}/{vp.name} -> {arr.shape}")
    print(f"Done. saved={done} skipped={skipped} output={config.SKELETON_RAW_DIR}")


if __name__ == "__main__":
    main()

