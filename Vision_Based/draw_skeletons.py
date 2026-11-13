"""Skeleton pictures drawn from processed/ (no videos, no GPU, no torch needed).

    python draw_skeletons.py

Writes results/skeleton/:
    skeleton_grid.png          one frame per subject (rows) x walking direction (columns)
    gait_strip_<clip>.png      10 frames of one clip, 6 frames apart, to show the walking motion
    skeleton_clips.csv       clips, frames, mean joint confidence per clip
Note: these are the NORMALISED skeletons (hip-centred, torso-scaled), not an overlay on the video.
For an overlay on the real video frames you need the original videos (check_pose.py).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
OUT = ROOT / "results" / "skeleton"
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}
DIRS = ["LINE", "PERP", "CROSS-RX-LEFT", "CROSS-RX-RIGHT"]


def draw(ax, frame, title=None):
    xy, conf = frame[:, :2], frame[:, 2]
    for a, b in EDGES:
        if conf[a] > 0.1 and conf[b] > 0.1:
            c = "tab:blue" if (a in LEFT and b in LEFT) else ("tab:red" if (a not in LEFT and b not in LEFT and a > 0) else "tab:gray")
            ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], "-", color=c, lw=2)
    ok = conf > 0.1
    ax.scatter(xy[ok, 0], xy[ok, 1], s=10, color="black", zorder=3)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(2.8, -2.2)          # image coordinates: y grows downwards
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main():
    files = sorted(PROCESSED.glob("*/*.npy"))
    if not files:
        raise SystemExit(f"No processed skeletons in {PROCESSED}")
    OUT.mkdir(parents=True, exist_ok=True)
    first = {}
    rows = []
    for p in files:
        subj, direction = p.parent.name, p.stem.split("_")[1]
        first.setdefault((subj, direction), p)
        s = np.load(p)
        rows.append([p.stem, subj, direction, len(s), round(float(s[..., 2].mean()), 3)])
    with open(OUT / "skeleton_clips.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "subject", "direction", "frames", "mean_joint_confidence"])
        w.writerows(rows)

    subjects = sorted({k[0] for k in first})
    dirs = [d for d in DIRS if any(k[1] == d for k in first)]
    fig, axes = plt.subplots(len(subjects), len(dirs), figsize=(2.2 * len(dirs), 2.6 * len(subjects)), squeeze=False)
    for i, s in enumerate(subjects):
        for j, d in enumerate(dirs):
            ax = axes[i][j]
            p = first.get((s, d))
            if p is None:
                ax.axis("off")
                continue
            seq = np.load(p)
            lo, hi = len(seq) // 3, 2 * len(seq) // 3          # best-detected frame in the middle third
            k = lo + int(np.argmax(seq[lo:hi, :, 2].mean(axis=1)))
            draw(ax, seq[k], s)
    fig.suptitle("Normalised 17-joint skeletons (blue = left side, red = right side)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for j, d in enumerate(dirs):
        pos = axes[0][j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.972, d, ha="center", fontsize=9, fontweight="bold")
    fig.savefig(OUT / "skeleton_grid.png", dpi=130)
    plt.close(fig)

    clip = first.get((subjects[0], "LINE")) or next(iter(first.values()))
    seq = np.load(clip)
    start = len(seq) // 3
    fig, axes = plt.subplots(1, 10, figsize=(20, 2.8))
    for k, ax in enumerate(axes):
        draw(ax, seq[start + 6 * k], f"frame {start + 6 * k}")
    fig.suptitle(f"Walking motion, {clip.stem}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"gait_strip_{clip.stem}.png", dpi=130)
    plt.close(fig)
    print(f"Saved in {OUT}:\n  skeleton_grid.png\n  gait_strip_{clip.stem}.png\n  skeleton_clips.csv")


if __name__ == "__main__":
    main()
