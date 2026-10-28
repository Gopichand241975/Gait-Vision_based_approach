"""Step 5: FINAL TEST of the checkpoint chosen on the validation clips.

Run this only after training and all decisions are finished: it evaluates the TEST clips.
The checkpoint is checkpoints/best.pt (best validation accuracy) if it exists, otherwise last.pt.
Each test clip is matched to the nearest TRAINING clip (Euclidean distance on the gait features).

Writes into results/recognition/:
    features_gallery.npz, features_probe.npz   gait feature vectors (gallery = training clips)
    distance_matrix.csv                        Euclidean distance test clip x training clip
    predictions.csv                            actual vs predicted subject for every test clip
    confusion_matrix.csv / .png                actual subject (rows) vs predicted subject (columns)
    classification_report.csv                  precision, recall, F1, support per subject + averages
    accuracy.json, report.txt                  accuracy, precision, recall, F1, Rank-k, per subject / direction
"""
import argparse
import csv
import json

import numpy as np
import torch

import config
from dataset import direction_of
from evaluate import (classification_metrics, rank_hits, recognise, save_classification_report,
                      save_confusion)
from model import GaitResGCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint to test (default: checkpoints/best.pt, else last.pt)")
    a = ap.parse_args()
    device = config.get_device()
    if a.ckpt:
        path = a.ckpt
    else:
        best = config.CHECKPOINT_DIR / config.BEST_CKPT
        path = str(best if best.exists() else config.CHECKPOINT_DIR / "last.pt")
        if not best.exists():
            print(f"[warn] {best.name} not found (no validation clips were used): testing last.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    split = ckpt["split"]
    gallery, probe = split["gallery"], split["probe"]
    if not gallery or not probe:
        raise SystemExit("Split has no gallery (training) clips or no test clips.")
    print(f"checkpoint: {path} (iteration {ckpt['iter']})")

    model = GaitResGCN(num_classes=len(split["train_persons"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    r = recognise(model, gallery, probe, device)
    dist, nearest = r["dist"], r["nearest"]
    gp, qp, pred = r["gallery_persons"], r["actual"], r["predicted"]
    correct = [p == q for p, q in zip(pred, qp)]
    metrics = classification_metrics(qp, pred, sorted(set(gp) | set(qp)))

    out = config.RESULTS_RECOG_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "features_gallery.npz", features=r["fg"], persons=np.array(gp),
             videos=np.array([x["video"] for x in gallery]), checkpoint_iteration=ckpt["iter"])
    np.savez(out / "features_probe.npz", features=r["fp"], persons=np.array(qp),
             videos=np.array([x["video"] for x in probe]), checkpoint_iteration=ckpt["iter"])
    gal_names = [f"{x['person']}/{x['video']}" for x in gallery]
    with open(out / "distance_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe"] + gal_names)
        for x, row in zip(probe, dist):
            w.writerow([f"{x['person']}/{x['video']}"] + [f"{d:.4f}" for d in row])
    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe_video", "actual", "predicted", "nearest_gallery_video", "distance", "result"])
        for i, x in enumerate(probe):
            w.writerow([f"{x['person']}/{x['video']}", qp[i], pred[i], gal_names[nearest[i]],
                        f"{dist[i, nearest[i]]:.4f}", "Correct" if correct[i] else "Wrong"])

    n_people = len(set(gp))
    ranks = {k: rank_hits(dist, qp, gp, k) for k in (1, 5, 10) if k == 1 or k < n_people}
    per_person = {p: round(float(np.mean([c for c, q in zip(correct, qp) if q == p])), 4)
                  for p in sorted(set(qp))}
    per_direction = {}
    for d in sorted({direction_of(x["video"]) for x in probe}):
        flags = [c for c, x in zip(correct, probe) if direction_of(x["video"]) == d]
        per_direction[d] = round(float(np.mean(flags)), 4)

    save_confusion(out, metrics, "Confusion matrix (test clips)")
    save_classification_report(out, metrics)

    acc = {"split_mode": split["mode"], "checkpoint": str(path), "checkpoint_iteration": ckpt["iter"],
           "gallery_videos": len(gallery), "probe_videos": len(probe), "gallery_people": n_people,
           "accuracy_pct": round(100 * metrics["accuracy"], 2)}
    for k, v in ranks.items():
        acc[f"rank{k}_accuracy_pct"] = round(100 * v, 2)
    for key in ("macro_precision", "macro_recall", "macro_f1", "weighted_precision",
                "weighted_recall", "weighted_f1"):
        acc[f"{key}_pct"] = round(100 * metrics[key], 2)
    acc["per_person_rank1"] = per_person
    acc["per_direction_rank1"] = per_direction
    acc["per_class"] = {l: {k: (round(v, 4) if k != "support" else v) for k, v in m.items()}
                        for l, m in metrics["per_class"].items()}
    (out / "accuracy.json").write_text(json.dumps(acc, indent=2))

    report = ["Gait recognition report (17-joint skeletons, nearest training clip, Euclidean distance)",
              f"Split mode: {split['mode']}   checkpoint: {path} (iteration {ckpt['iter']})",
              f"Training (gallery) clips: {len(gallery)}   Test clips: {len(probe)}   Subjects: {n_people}", "",
              f"Accuracy:            {100 * metrics['accuracy']:.2f}%   "
              f"({sum(correct)}/{len(correct)} test clips correct)",
              f"Macro precision:     {100 * metrics['macro_precision']:.2f}%",
              f"Macro recall:        {100 * metrics['macro_recall']:.2f}%",
              f"Macro F1-score:      {100 * metrics['macro_f1']:.2f}%",
              f"Weighted F1-score:   {100 * metrics['weighted_f1']:.2f}%"]
    report += [f"Rank-{k} accuracy:    {100 * v:.2f}%" for k, v in ranks.items() if k > 1]
    report += ["", "Per subject (precision / recall / F1 / test clips):"]
    report += [f"  {l}: {100 * m['precision']:.1f}% / {100 * m['recall']:.1f}% / {100 * m['f1']:.1f}% / {m['support']}"
               for l, m in metrics["per_class"].items()]
    report += ["", "Per walking direction (accuracy):"]
    report += [f"  {d}: {100 * v:.1f}%" for d, v in per_direction.items()]
    if len(probe) < 100:
        report += ["", f"Only {len(probe)} test clips: one clip changes the accuracy by "
                       f"{100 / len(probe):.1f} points. Treat small differences with care."]
    (out / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nFiles saved in {out}")


if __name__ == "__main__":
    main()
