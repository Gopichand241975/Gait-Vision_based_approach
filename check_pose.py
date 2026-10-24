"""Step 2: draw the extracted skeleton on video frames to check it sits on the body.

Writes results/skeleton/pose_check/<person>_<video>_<n>.jpg
"""
import argparse

import cv2
import numpy as np

import config
from pose_extract import list_videos, out_path, person_of


def draw(frame, kp):
    for a, b in config.SKELETON_EDGES:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]),
                     (0, 255, 0), 2)
    for x, y, c in kp:
        if c > 0:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--frames", type=int, default=3, help="images per video")
    a = ap.parse_args()
    out_dir = config.RESULTS_SKELETON_DIR / "pose_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    shown = 0
    for v in list_videos():
        if shown >= a.max_videos:
            break
        if not out_path(v).exists():
            continue
        sk = np.load(out_path(v))
        present = np.where(sk[..., 2].mean(axis=1) > 0)[0]     # frames where a person was found
        if len(present) == 0:
            print(f"[warn] {v.name}: no person detected in any frame")
            continue
        picks = present[np.linspace(0, len(present) - 1, a.frames).astype(int)]
        cap = cv2.VideoCapture(str(v))
        for k, idx in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imwrite(str(out_dir / f"{person_of(v)}_{v.stem}_{k}_frame{int(idx)}.jpg"),
                        draw(frame, sk[int(idx)]))
            n_saved += 1
        cap.release()
        shown += 1
    print(f"Saved {n_saved} images in {out_dir}. Open them and check the joints are on the body.")


if __name__ == "__main__":
    main()