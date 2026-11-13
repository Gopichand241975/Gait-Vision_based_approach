"""Train / validation / test accuracy side by side, plus graphs and CSV files.

    python all_results.py                       # uses checkpoints/best.pt (else last.pt)
    python all_results.py --ckpt checkpoints/last.pt

Run it AFTER train.py. It uses the split stored in the checkpoint, so train / val / test are exactly the
clips the model was trained and tested with. Writes results/all/ :
    accuracy_train_val_test.csv / .txt    the table (clip-level and window-level, per split)
    accuracy_train_val_test.png           bar chart
    confusion_<split>.csv / .png          one confusion matrix per split
    predictions_<split>.csv               actual vs predicted for every clip

How each number is measured (all use the SAME trained model, no augmentation):
  clip accuracy   every clip -> one gait feature -> identity of the NEAREST TRAINING clip.
                  For TRAIN clips the clip itself is left out (otherwise it matches itself = 100%).
  window accuracy every 60-frame window is classified on its own by the model's classifier head.
READ THIS: train clips share their walking direction with the other training clips of the same person, so
the train score is expected to be high. Val and test are walking directions the model never trained on,
so they are the honest numbers.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import load_record, to_features, window_of, window_starts
from evaluate import classification_metrics, extract, save_confusion
from model import GaitResGCN


@torch.no_grad()
def head_window_accuracy(model, records, person_to_idx, device):
    model.eval()
    ok = total = 0
    for r in records:
        seq = load_record(r)
        starts = window_starts(len(seq), config.SEQ_LEN, config.SEQ_LEN)
        x = np.stack([to_features(window_of(seq, s, config.SEQ_LEN)) for s in starts])
        logits = model.classify(model(torch.from_numpy(x).to(device)))
        ok += int((logits.argmax(1).cpu() == person_to_idx[r["person"]]).sum())
        total += len(starts)
    return (ok / total if total else float("nan")), total


def clip_predictions(model, gallery, probe, device, leave_one_out):
    fg = extract(model, gallery, device)
    fp = fg if leave_one_out else extract(model, probe, device)
    dist = torch.cdist(torch.from_numpy(fp), torch.from_numpy(fg)).numpy()
    if leave_one_out:
        np.fill_diagonal(dist, np.inf)           # a training clip may not match itself
    gp = [r["person"] for r in gallery]
    nearest = dist.argmin(1)
    return [gp[j] for j in nearest], [gallery[j]["video"] for j in nearest], dist[np.arange(len(probe)), nearest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    device = config.get_device()
    best = config.CHECKPOINT_DIR / config.BEST_CKPT
    path = a.ckpt or str(best if best.exists() else config.CHECKPOINT_DIR / "last.pt")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    split = ckpt["split"]
    persons = split["train_persons"]
    p2i = {p: i for i, p in enumerate(persons)}
    model = GaitResGCN(num_classes=len(persons)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    gallery = split["gallery"]
    sets = [("train", split["train"], True), ("val", split["val"], False), ("test", split["probe"], False)]
    print(f"checkpoint {path} (iteration {ckpt['iter']}), mode {split['mode']}")

    out = config.RESULTS_DIR / "all"
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for name, recs, loo in sets:
        if not recs:
            print(f"[skip] no {name} clips in this split")
            continue
        pred, near, dist = clip_predictions(model, gallery, recs, device, loo)
        actual = [r["person"] for r in recs]
        m = classification_metrics(actual, pred, sorted(set(actual) | set(pred)))
        w_acc, n_w = head_window_accuracy(model, recs, p2i, device)
        dirs = sorted({r["video"].split("_")[1] for r in recs})
        table.append({"split": name, "directions": "+".join(dirs), "clips": len(recs), "windows": n_w,
                      "clip_accuracy": round(m["accuracy"], 4), "clip_macro_f1": round(m["macro_f1"], 4),
                      "window_accuracy": round(w_acc, 4)})
        sub = out / name
        sub.mkdir(exist_ok=True)
        save_confusion(sub, m, f"Confusion matrix ({name} clips)")
        for f_ in ("confusion_matrix.csv", "confusion_matrix.png"):
            (sub / f_).replace(out / f_.replace("confusion_matrix", f"confusion_{name}"))
        sub.rmdir()
        with open(out / f"predictions_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "actual", "predicted", "nearest_training_clip", "distance", "result"])
            for r, p, nb, d in zip(recs, pred, near, dist):
                w.writerow([r["video"], r["person"], p, nb, f"{d:.4f}", "Correct" if p == r["person"] else "Wrong"])

    with open(out / "accuracy_train_val_test.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    lines = [f"Checkpoint: {path} (iteration {ckpt['iter']})   split mode: {split['mode']}", "",
             f"{'split':<6} {'directions':<28} {'clips':>5} {'clip acc':>9} {'macro-F1':>9} {'window acc':>11}"]
    lines += [f"{r['split']:<6} {r['directions']:<28} {r['clips']:>5} {100 * r['clip_accuracy']:>8.1f}% "
              f"{100 * r['clip_macro_f1']:>8.1f}% {100 * r['window_accuracy']:>10.1f}%" for r in table]
    lines += ["", "train clip accuracy leaves the clip itself out; train directions are SEEN in training,",
              "so a high train score is expected. val / test directions are unseen: those are the honest scores.",
              f"chance level with {len(persons)} subjects: {100 / len(persons):.1f}%"]
    (out / "accuracy_train_val_test.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    ax.bar(x - 0.18, [100 * r["clip_accuracy"] for r in table], 0.36, label="clip accuracy", color="tab:blue")
    ax.bar(x + 0.18, [100 * r["window_accuracy"] for r in table], 0.36, label="window accuracy", color="tab:orange")
    ax.axhline(100 / len(persons), color="gray", ls=":", label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['split']}\n({r['directions']})" for r in table], fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Train / validation / test accuracy")
    ax.legend(fontsize=8)
    for i, r in enumerate(table):
        ax.text(i - 0.18, 100 * r["clip_accuracy"] + 1, f"{100 * r['clip_accuracy']:.0f}", ha="center", fontsize=8)
        ax.text(i + 0.18, 100 * r["window_accuracy"] + 1, f"{100 * r['window_accuracy']:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_train_val_test.png", dpi=150)
    plt.close(fig)
    print(f"\nFiles saved in {out}")


if __name__ == "__main__":
    main()
