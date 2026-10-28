"""Shared evaluation code used by train.py (validation) and test.py (final test).

Recognition = gallery matching: every clip becomes one gait feature (mean of its 60-frame windows);
a probe clip gets the identity of the NEAREST gallery clip (Euclidean distance). The gallery is
always the TRAINING clips. Metrics are computed from the predicted vs the actual subject.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import load_record, to_features, window_of, window_starts


@torch.no_grad()
def video_feature(model, seq, device):
    starts = window_starts(len(seq), config.SEQ_LEN, config.SEQ_STRIDE)
    x = np.stack([to_features(window_of(seq, s, config.SEQ_LEN)) for s in starts])
    feat = model(torch.from_numpy(x).to(device)).mean(0)
    return F.normalize(feat, dim=0).cpu().numpy()


def extract(model, records, device):
    """One feature vector per record. The model is put in eval mode and restored afterwards."""
    was_training = model.training
    model.eval()
    try:
        return np.stack([video_feature(model, load_record(r), device) for r in records])
    finally:
        model.train(was_training)


def recognise(model, gallery, probe, device):
    """Nearest-gallery recognition. Returns dict with features, distances and predictions."""
    fg, fp = extract(model, gallery, device), extract(model, probe, device)
    dist = torch.cdist(torch.from_numpy(fp), torch.from_numpy(fg)).numpy()
    gp = [r["person"] for r in gallery]
    nearest = dist.argmin(1)
    return {"fg": fg, "fp": fp, "dist": dist, "nearest": nearest, "gallery_persons": gp,
            "predicted": [gp[j] for j in nearest], "actual": [r["person"] for r in probe]}


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


def classification_metrics(actual, predicted, labels=None):
    """Accuracy, precision, recall, F1 (per class, macro and weighted) and the confusion matrix.
    Macro = plain mean over the classes that occur in `actual`; weighted = mean weighted by support."""
    labels = sorted(set(actual) | set(predicted)) if labels is None else list(labels)
    idx = {p: i for i, p in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, p in zip(actual, predicted):
        cm[idx[a], idx[p]] += 1
    tp = np.diag(cm).astype(float)
    pred_n, support = cm.sum(0).astype(float), cm.sum(1).astype(float)
    precision = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    w = support / support.sum() if support.sum() else support
    out = {"accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
           "macro_precision": float(precision[present].mean()), "macro_recall": float(recall[present].mean()),
           "macro_f1": float(f1[present].mean()), "weighted_precision": float((precision * w).sum()),
           "weighted_recall": float((recall * w).sum()), "weighted_f1": float((f1 * w).sum()),
           "clips": int(cm.sum()), "labels": labels, "confusion_matrix": cm,
           "per_class": {l: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1": float(f1[i]), "support": int(support[i])} for l, i in idx.items()}}
    return out


def save_classification_report(out, metrics):
    with open(out / "classification_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "precision", "recall", "f1", "support"])
        for l, m in metrics["per_class"].items():
            w.writerow([l, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"]])
        w.writerow(["macro avg", f"{metrics['macro_precision']:.4f}", f"{metrics['macro_recall']:.4f}",
                    f"{metrics['macro_f1']:.4f}", metrics["clips"]])
        w.writerow(["weighted avg", f"{metrics['weighted_precision']:.4f}", f"{metrics['weighted_recall']:.4f}",
                    f"{metrics['weighted_f1']:.4f}", metrics["clips"]])
        w.writerow(["accuracy", "", "", f"{metrics['accuracy']:.4f}", metrics["clips"]])


def save_confusion(out, metrics, title="Confusion matrix (test clips)"):
    cm, people = metrics["confusion_matrix"], metrics["labels"]
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
    ax.set_xlabel("Predicted subject")
    ax.set_ylabel("Actual subject")
    ax.set_title(title)
    if len(people) <= 15:
        for i in range(len(people)):
            for j in range(len(people)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
