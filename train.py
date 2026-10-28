"""Step 4: train the gait model on the TRAINING clips and watch the VALIDATION clips.

Every --save_every iterations the validation clips are recognised against the training clips
(nearest gallery clip). The checkpoint with the best validation accuracy is kept as
checkpoints/best.pt; test.py uses it. The test clips are never touched here.

Writes  checkpoints/last.pt, best.pt, iter_XXXXXX.pt
        results/training/train_log.csv, val_log.csv, training_curves.png,
        best_checkpoint.json, split.json
Use --resume to continue from checkpoints/last.pt (the best checkpoint so far is remembered).
"""
import argparse
import csv
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import TrainDataset, list_records, make_split
from evaluate import classification_metrics, recognise
from model import GaitResGCN


def save_ckpt(path, model, opt, sched, it, split, best):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "iter": it, "split": split, "best": best}, tmp)
    tmp.replace(path)


def read_csv_columns(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def plot_curves(out_dir, best):
    """training_curves.png: loss + accuracy on the training windows, and validation accuracy / F1."""
    train = read_csv_columns(out_dir / "train_log.csv")
    val = read_csv_columns(out_dir / "val_log.csv") if (out_dir / "val_log.csv").exists() else {}
    if not train:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(train["iteration"], train["loss"], color="tab:red")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("Training loss")
    ax[1].plot(train["iteration"], train["train_accuracy"], label="training windows", color="tab:blue")
    if val:
        ax[1].plot(val["iteration"], val["val_accuracy"], "o-", label="validation clips", color="tab:green")
        ax[1].plot(val["iteration"], val["val_macro_f1"], "s--", label="validation macro F1", color="tab:orange")
    if best:
        ax[1].axvline(best["iteration"], color="gray", linestyle=":", label=f"best checkpoint ({best['iteration']})")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("accuracy / F1")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Accuracy")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=config.TOTAL_ITERS, help="total iterations")
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--save_every", type=int, default=config.SAVE_EVERY)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="a few iterations only")
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.batch_size, a.num_workers = config.SMOKE_ITERS, config.SMOKE_BATCH_SIZE, 0

    config.ensure_dirs()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    device = config.get_device()
    print("device:", device)

    ckpt, best = None, None
    last = config.CHECKPOINT_DIR / "last.pt"
    best_path = config.CHECKPOINT_DIR / config.BEST_CKPT
    if a.resume:
        if not last.exists():
            raise SystemExit("--resume given but checkpoints/last.pt does not exist")
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        split = ckpt["split"]
        best = ckpt.get("best")
    else:
        records = list_records()
        if not records:
            raise SystemExit(f"No processed skeletons in {config.PROCESSED_DIR}. Run preprocess.py first.")
        split = make_split(records)
    for w in split["warnings"]:
        print("[warn]", w)
    persons = split["train_persons"]
    if len(persons) < 2:
        raise SystemExit("Need at least 2 different people to train.")
    person_to_idx = {p: i for i, p in enumerate(persons)}
    do_val = bool(split["val"]) and bool(split["gallery"])
    print(f"split={split['mode']} train_clips={len(split['train'])} val_clips={len(split['val'])} "
          f"test_clips={len(split['probe'])} subjects={len(persons)}")
    if not do_val:
        print("[warn] no validation clips in this split: no best checkpoint is selected, last.pt is used")
    (config.RESULTS_TRAIN_DIR / "split.json").write_text(json.dumps(split, indent=2))

    ds = TrainDataset(split["train"], person_to_idx)
    bs = min(a.batch_size, len(ds))
    if bs < 2:
        raise SystemExit("Batch size must be at least 2.")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) >= bs,
                        num_workers=a.num_workers, pin_memory=device.type == "cuda")

    model = GaitResGCN(num_classes=len(persons)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.iters, 1),
                                                       eta_min=config.LR * 0.01)
    it = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        it = ckpt["iter"]
        print(f"resumed from iteration {it}")

    log_path = config.RESULTS_TRAIN_DIR / "train_log.csv"
    new_log = not (a.resume and log_path.exists())
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iteration", "loss", "train_accuracy", "lr"])

    val_path = config.RESULTS_TRAIN_DIR / "val_log.csv"
    new_val = not (a.resume and val_path.exists())
    val_f = open(val_path, "w" if new_val else "a", newline="")
    val_log = csv.writer(val_f)
    if new_val:
        val_log.writerow(["iteration", "val_accuracy", "val_macro_f1", "val_macro_precision", "val_macro_recall"])

    def validate_and_select(it):
        """Evaluate the validation clips; keep the checkpoint if it is the best so far."""
        nonlocal best
        if not do_val:
            return
        m = classification_metrics(*(lambda r: (r["actual"], r["predicted"]))(
            recognise(model, split["gallery"], split["val"], device)))
        val_log.writerow([it, round(m["accuracy"], 4), round(m["macro_f1"], 4),
                          round(m["macro_precision"], 4), round(m["macro_recall"], 4)])
        val_f.flush()
        print(f"validation @ {it}: accuracy {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}", flush=True)
        if best is None or m["accuracy"] >= best["val_accuracy"]:
            best = {"val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "iteration": it}
            save_ckpt(best_path, model, opt, sched, it, split, best)
            print(f"  new best checkpoint (iteration {it}) -> {best_path.name}", flush=True)

    model.train()
    run_loss = run_acc = n_run = 0
    while it < a.iters:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feat = model(x)
            loss = F.cross_entropy(model.classify(feat), y)
            if not torch.isfinite(loss):
                raise SystemExit(f"Loss is {loss.item()} at iteration {it}. Stopping.")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            run_loss += loss.item()
            run_acc += (model.classify(feat).argmax(1) == y).float().mean().item()
            n_run += 1
            if it % config.LOG_EVERY == 0 or it == a.iters:
                row = [it, round(run_loss / n_run, 5), round(run_acc / n_run, 4),
                       f"{sched.get_last_lr()[0]:.6f}"]
                log.writerow(row)
                log_f.flush()
                print(f"iter {row[0]:>6}  loss {row[1]:.4f}  acc {row[2]:.3f}  lr {row[3]}")
                run_loss = run_acc = n_run = 0
            if it % a.save_every == 0 and it < a.iters:
                validate_and_select(it)
                model.train()
                save_ckpt(config.CHECKPOINT_DIR / f"iter_{it:06d}.pt", model, opt, sched, it, split, best)
                save_ckpt(last, model, opt, sched, it, split, best)
            if it >= a.iters:
                break
    log_f.close()
    validate_and_select(it)
    val_f.close()
    save_ckpt(config.CHECKPOINT_DIR / f"iter_{it:06d}.pt", model, opt, sched, it, split, best)
    save_ckpt(last, model, opt, sched, it, split, best)
    if best:
        (config.RESULTS_TRAIN_DIR / "best_checkpoint.json").write_text(json.dumps(best, indent=2))
    plot_curves(config.RESULTS_TRAIN_DIR, best)
    print(f"Training finished at iteration {it}. Last checkpoint: {last}")
    if best:
        print(f"Best validation accuracy {100 * best['val_accuracy']:.1f}% at iteration "
              f"{best['iteration']}: {best_path}  (test.py uses this one)")


if __name__ == "__main__":
    main()
