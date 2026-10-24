"""Last step: TOTAL accuracy of the vision-based approach over several test runs.

One test run = train.py + test.py with one walking direction held out (TEST_DIRECTION).
Doing it once per direction tests every video exactly once. This file collects the runs.

    python total_accuracy.py --save     # after test.py: keep this run in results/runs/<name>/
    python total_accuracy.py            # read every saved run and compute the total accuracy
    python total_accuracy.py --list     # show the saved runs only

Reads    results/runs/<name>/predictions.csv   (+ accuracy.json)   one folder per saved run
Writes   results/total/total_report.txt, total_accuracy.json, total_predictions.csv,
         total_confusion_matrix.csv / .png

Total accuracy = correct test videos over ALL saved runs / test videos over ALL saved runs x 100.
"""
import argparse
import csv
import json
import math
import shutil

import numpy as np

import config

RUNS_DIR = config.RESULTS_DIR / "runs"
TOTAL_DIR = config.RESULTS_DIR / "total"


def parse_video(probe_video):
    """'S01/S01_CROSS-RX-RIGHT_Cam-P' -> (direction, camera).
    'P01/P01_STR_cam1' -> ('STR', 'cam1').  Names without 3 parts -> ('all', 'all')."""
    parts = probe_video.split("/")[-1].split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "all", "all"


def wilson(k, n, z=1.96):
    """95% confidence interval of an accuracy k/n (n is small, so it is wide)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def saved_runs():
    if not RUNS_DIR.exists():
        return []
    return sorted(d for d in RUNS_DIR.iterdir() if (d / "predictions.csv").exists())


def run_name_of(rows, acc):
    """Name of the run = the held-out direction (cross_direction) or the split mode."""
    directions = sorted({parse_video(r["probe_video"])[0] for r in rows})
    if acc.get("split_mode") == "cross_direction" and len(directions) == 1:
        return directions[0]
    return acc.get("split_mode", "run")


def save_run(name=None):
    rec = config.RESULTS_RECOG_DIR
    pred, acc_path = rec / "predictions.csv", rec / "accuracy.json"
    if not pred.exists() or not acc_path.exists():
        raise SystemExit(f"No test result in {rec}. Run  python test.py  first.")
    rows = read_csv(pred)
    acc = json.loads(acc_path.read_text())
    name = name or run_name_of(rows, acc)
    dest = RUNS_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in rec.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    for extra in (config.RESULTS_TRAIN_DIR / "split.json", config.RESULTS_TRAIN_DIR / "train_log.csv"):
        if extra.exists():
            shutil.copy2(extra, dest / extra.name)
    ok = sum(r["result"] == "Correct" for r in rows)
    print(f"Saved run '{name}': {ok}/{len(rows)} test videos correct ({100 * ok / len(rows):.2f}%)")
    print(f"Folder: {dest}")
    runs = [d.name for d in saved_runs()]
    print(f"Runs saved so far ({len(runs)}): {', '.join(runs)}")


def save_confusion(out, actual, predicted, people):
    idx = {p: i for i, p in enumerate(people)}
    cm = np.zeros((len(people), len(people)), dtype=int)
    for a, p in zip(actual, predicted):
        cm[idx[a], idx[p]] += 1
    with open(out / "total_confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual \\ predicted"] + people)
        for p, row in zip(people, cm):
            w.writerow([p] + list(row))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    size = max(4, 0.6 * len(people) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(people)))
    ax.set_yticks(range(len(people)))
    ax.set_xticklabels(people, rotation=90, fontsize=8)
    ax.set_yticklabels(people, fontsize=8)
    ax.set_xlabel("Predicted person")
    ax.set_ylabel("Actual person")
    ax.set_title("Total confusion matrix (all saved runs)")
    if len(people) <= 15:
        for i in range(len(people)):
            for j in range(len(people)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out / "total_confusion_matrix.png", dpi=150)
    plt.close(fig)


def pct(k, n):
    return 100.0 * k / n if n else 0.0


def total():
    runs = saved_runs()
    if not runs:
        raise SystemExit(f"No saved runs in {RUNS_DIR}. After each  python test.py  run  "
                         f"python total_accuracy.py --save")
    rows, per_run, rank5_k, rank5_ok = [], [], 0, True
    for d in runs:
        r = read_csv(d / "predictions.csv")
        for x in r:
            direction, camera = parse_video(x["probe_video"])
            rows.append({"run": d.name, "probe_video": x["probe_video"], "actual": x["actual"],
                         "predicted": x["predicted"], "direction": direction, "camera": camera,
                         "correct": x["result"] == "Correct"})
        ok = sum(x["result"] == "Correct" for x in r)
        per_run.append((d.name, ok, len(r)))
        acc_file = d / "accuracy.json"
        a = json.loads(acc_file.read_text()) if acc_file.exists() else {}
        if "rank5_accuracy_pct" in a:
            rank5_k += round(a["rank5_accuracy_pct"] * len(r) / 100)
        else:
            rank5_ok = False

    warnings = []
    seen = {}
    for x in rows:
        seen.setdefault(x["probe_video"], set()).add(x["run"])
    dup = sorted(v for v, s in seen.items() if len(s) > 1)
    if dup:
        warnings.append(f"{len(dup)} test video(s) appear in more than one run and are counted "
                        f"twice, e.g. {dup[0]}. Check that every run held out a different direction.")

    n = len(rows)
    k = sum(x["correct"] for x in rows)
    lo, hi = wilson(k, n)
    mean_runs = float(np.mean([pct(o, m) for _, o, m in per_run]))

    def group(key):
        out = {}
        for x in rows:
            c = out.setdefault(x[key], [0, 0])
            c[0] += x["correct"]
            c[1] += 1
        return {g: (c[0], c[1]) for g, c in sorted(out.items())}

    by_person, by_camera, by_direction = group("actual"), group("camera"), group("direction")

    lines = ["TOTAL accuracy of the vision-based gait recognition (all saved runs)",
             f"Runs included: {len(runs)}   Test videos in total: {n}", "",
             "Per run (Rank-1):"]
    lines += [f"  {name:<16} {o:>3}/{m:<3} = {pct(o, m):6.2f}%" for name, o, m in per_run]
    lines += ["", f"TOTAL Rank-1 accuracy = {k}/{n} = {pct(k, n):.2f}%",
              f"95% confidence interval: {100 * lo:.1f}% to {100 * hi:.1f}%",
              f"Mean of the run accuracies: {mean_runs:.2f}%"]
    if rank5_ok and n:
        lines.append(f"TOTAL Rank-5 accuracy = {rank5_k}/{n} = {pct(rank5_k, n):.2f}%")
    lines += ["", "Per person (Rank-1):"]
    lines += [f"  {p}: {o}/{m} = {pct(o, m):.1f}%" for p, (o, m) in by_person.items()]
    lines += ["", "Per camera (Rank-1):"]
    lines += [f"  {c}: {o}/{m} = {pct(o, m):.1f}%" for c, (o, m) in by_camera.items()]
    lines += ["", "Per walking direction (Rank-1):"]
    lines += [f"  {g}: {o}/{m} = {pct(o, m):.1f}%" for g, (o, m) in by_direction.items()]
    if warnings:
        lines += ["", "WARNINGS:"] + [f"  {w}" for w in warnings]

    TOTAL_DIR.mkdir(parents=True, exist_ok=True)
    (TOTAL_DIR / "total_report.txt").write_text("\n".join(lines) + "\n")
    with open(TOTAL_DIR / "total_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "probe_video", "actual", "predicted", "direction", "camera", "result"])
        for x in rows:
            w.writerow([x["run"], x["probe_video"], x["actual"], x["predicted"],
                        x["direction"], x["camera"], "Correct" if x["correct"] else "Wrong"])
    summary = {"runs": [{"run": a, "correct": o, "videos": m, "rank1_pct": round(pct(o, m), 2)}
                        for a, o, m in per_run],
               "total_videos": n, "total_correct": k,
               "total_rank1_accuracy_pct": round(pct(k, n), 2),
               "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
               "mean_of_run_accuracies_pct": round(mean_runs, 2),
               "per_person_rank1_pct": {p: round(pct(o, m), 1) for p, (o, m) in by_person.items()},
               "per_camera_rank1_pct": {c: round(pct(o, m), 1) for c, (o, m) in by_camera.items()},
               "per_direction_rank1_pct": {g: round(pct(o, m), 1) for g, (o, m) in by_direction.items()},
               "warnings": warnings}
    if rank5_ok and n:
        summary["total_rank5_accuracy_pct"] = round(pct(rank5_k, n), 2)
    (TOTAL_DIR / "total_accuracy.json").write_text(json.dumps(summary, indent=2))
    people = sorted({x["actual"] for x in rows} | {x["predicted"] for x in rows})
    save_confusion(TOTAL_DIR, [x["actual"] for x in rows], [x["predicted"] for x in rows], people)
    print("\n".join(lines))
    print(f"\nFiles saved in {TOTAL_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="keep the current results\\recognition as one saved run")
    ap.add_argument("--name", default=None, help="name for --save (default: the held-out direction)")
    ap.add_argument("--list", action="store_true", help="only list the saved runs")
    a = ap.parse_args()
    if a.save:
        save_run(a.name)
    elif a.list:
        runs = saved_runs()
        print(f"{len(runs)} saved run(s) in {RUNS_DIR}: {', '.join(d.name for d in runs) or '-'}")
    else:
        total()


if __name__ == "__main__":
    main()