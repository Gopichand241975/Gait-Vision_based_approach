"""Step 4: train the gait model.

Writes  checkpoints/last.pt and checkpoints/iter_XXXXXX.pt
        results/training/train_log.csv, results/training/split.json
Use --resume to continue from checkpoints/last.pt.
"""
import argparse
import csv
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import TrainDataset, list_records, make_split
from model import GaitResGCN


def save_ckpt(path, model, opt, sched, it, split):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "iter": it, "split": split}, tmp)
    tmp.replace(path)


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

    ckpt = None
    last = config.CHECKPOINT_DIR / "last.pt"
    if a.resume:
        if not last.exists():
            raise SystemExit("--resume given but checkpoints/last.pt does not exist")
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        split = ckpt["split"]
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
    print(f"split={split['mode']} train_videos={len(split['train'])} "
          f"gallery={len(split['gallery'])} probe={len(split['probe'])} people={len(persons)}")
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
                save_ckpt(config.CHECKPOINT_DIR / f"iter_{it:06d}.pt", model, opt, sched, it, split)
                save_ckpt(last, model, opt, sched, it, split)
            if it >= a.iters:
                break
    log_f.close()
    save_ckpt(config.CHECKPOINT_DIR / f"iter_{it:06d}.pt", model, opt, sched, it, split)
    save_ckpt(last, model, opt, sched, it, split)
    print(f"Training finished at iteration {it}. Checkpoint: {last}")


if __name__ == "__main__":
    main()