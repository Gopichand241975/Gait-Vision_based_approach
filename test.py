"""Step 5: gait recognition (gallery / probe matching) with the trained model.

Writes into results/recognition/:
    features_gallery.npz, features_probe.npz   gait feature vectors
    distance_matrix.csv                        Euclidean distance probe x gallery
    predictions.csv                            actual vs predicted identity for every probe
    confusion_matrix.csv / .png                actual person (rows) vs predicted person (columns)
    accuracy.json, report.txt                  Rank-1 / Rank-5 / Rank-10, per-person and per-direction accuracy
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import direction_of, load_record, to_features, window_of, window_starts
from model import GaitResGCN


@torch.no_grad()
def video_feature(model, seq, device):
    starts = window_starts(len(seq), config.SEQ_LEN, config.SEQ_STRIDE)
    x = np.stack([to_features(window_of(seq, s, config.SEQ_LEN)) for s in starts])
    feat = model(torch.from_numpy(x).to(device)).mean(0)
    return F.normalize(feat, dim=0).cpu().numpy()


def extract(model, records, device):
    return np.stack([video_feature(model, load_record(r), device) for r in records])


def rank_hits(dist, probe_persons, gallery_persons, k):
    hits = 0
    for i in range(len(dist)):
        order, seen = np.argsort(dist[i]), []
        for j in order:
            if gallery_persons[j] not in seen:
                seen.append(gallery_persons[j])
            if len(seen) == k:
                break
        hits += probe_persons[i] in seen
    return hits / len(dist)


def save_confusion(out, actual, predicted, people):
    idx = {p: i for i, p in enumerate(people)}
    cm = np.zeros((len(people), len(people)), dtype=int)
    for a, p in zip(actual, predicted):
        cm[idx[a], idx[p]] += 1
    with open(out / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual \\ predicted"] + people)
        for p, row in zip(people, cm):
            w.writerow([p] + list(row))
    size = max(4, 0.6 * len(people) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(people)))
    ax.set_yticks(range(len(people)))
    ax.set_xticklabels(people, rotation=90, fontsize=8)
    ax.set_yticklabels(people, fontsize=8)
    ax.set_xlabel("Predicted person")
    ax.set_ylabel("Actual person")
    ax.set_title("Confusion matrix (probe videos)")
    if len(people) <= 15:
        for i in range(len(people)):
            for j in range(len(people)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.CHECKPOINT_DIR / "last.pt"))
    a = ap.parse_args()
    device = config.get_device()
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    split = ckpt["split"]
    gallery, probe = split["gallery"], split["probe"]
    if not gallery or not probe:
        raise SystemExit("Split has no gallery or no probe videos (need >= 2 videos per person).")

    model = GaitResGCN(num_classes=len(split["train_persons"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    fg, fp = extract(model, gallery, device), extract(model, probe, device)
    dist = torch.cdist(torch.from_numpy(fp), torch.from_numpy(fg)).numpy()   # Euclidean
    gp, qp = [r["person"] for r in gallery], [r["person"] for r in probe]
    nearest = dist.argmin(1)
    pred = [gp[j] for j in nearest]
    correct = [p == q for p, q in zip(pred, qp)]

    out = config.RESULTS_RECOG_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "features_gallery.npz", features=fg,
             persons=np.array(gp), videos=np.array([r["video"] for r in gallery]))
    np.savez(out / "features_probe.npz", features=fp,
             persons=np.array(qp), videos=np.array([r["video"] for r in probe]))
    gal_names = [f"{r['person']}/{r['video']}" for r in gallery]
    with open(out / "distance_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe"] + gal_names)
        for r, row in zip(probe, dist):
            w.writerow([f"{r['person']}/{r['video']}"] + [f"{d:.4f}" for d in row])
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_video", "actual", "predicted", "nearest_gallery_video", "distance", "result"])
        for i, r in enumerate(probe):
            w.writerow([f"{r['person']}/{r['video']}", qp[i], pred[i], gal_names[nearest[i]],
                        f"{dist[i, nearest[i]]:.4f}", "Correct" if correct[i] else "Wrong"])

    n_people = len(set(gp))
    ranks = {k: rank_hits(dist, qp, gp, k) for k in (1, 5, 10) if k == 1 or k < n_people}
    per_person = {p: round(float(np.mean([c for c, q in zip(correct, qp) if q == p])), 4)
                  for p in sorted(set(qp))}

    per_direction = {}
    for d in sorted({direction_of(r["video"]) for r in probe}):
        flags = [c for c, r in zip(correct, probe) if direction_of(r["video"]) == d]
        per_direction[d] = round(float(np.mean(flags)), 4)

    save_confusion(out, qp, pred, sorted(set(gp) | set(qp)))

    acc = {"split_mode": split["mode"], "checkpoint_iteration": ckpt["iter"],
           "gallery_videos": len(gallery), "probe_videos": len(probe), "gallery_people": n_people}
    for k, v in ranks.items():
        acc[f"rank{k}_accuracy_pct"] = round(100 * v, 2)
    acc["per_person_rank1"] = per_person
    acc["per_direction_rank1"] = per_direction
    (out / "accuracy.json").write_text(json.dumps(acc, indent=2))
    report = ["Gait recognition report (17-joint skeletons, Euclidean distance)",
              f"Split mode: {split['mode']}   checkpoint iteration: {ckpt['iter']}",
              f"Gallery videos: {len(gallery)}   Probe videos: {len(probe)}   People: {n_people}"]
    report += [f"Rank-{k} accuracy: {100 * v:.2f}%" for k, v in ranks.items()]
    report += [f"(Rank-1 = {sum(correct)}/{len(correct)} probe videos correct)", "",
               "Per person (Rank-1):"]
    report += [f"  {p}: {100 * v:.1f}%" for p, v in per_person.items()]
    report += ["", "Per walking direction (Rank-1):"]
    report += [f"  {d}: {100 * v:.1f}%" for d, v in per_direction.items()]
    (out / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nFiles saved in {out}")


if __name__ == "__main__":
    main()