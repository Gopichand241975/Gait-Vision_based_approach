#!/usr/bin/env python3
"""
csi_loader.py - cut every trial's CSI out of the raw files, clean it, and save one small file per trial.

Usage (Anaconda Prompt, csi_env active):
    pip install scipy matplotlib
    python csi_loader.py "D:\\CSI_GAIT\\Training_clips\\clip_index.csv" --out D:\\CSI_GAIT\\csi_cache_v2 --plot
    (--plot also saves a picture of the first trial so you can look at it)

What it does for each trial (found through clip_index.csv, never by guessing file names):
  1. takes ONLY the raw rows with the trial's boot_id and esp_us between csi_t_start_us and csi_t_end_us,
     plus 1 s extra on each side so the filters have room to settle (the extra is trimmed away again);
  2. drops broken rows (not exactly 128 numbers) and the 12 unusable subcarriers
     (10 always-zero ones, subcarrier 0 = big outlier, subcarrier 1 = constant) -> 52 subcarriers left;
  3. amplitude per subcarrier = sqrt(imaginary^2 + real^2);
  4. resamples onto an even 100 Hz time grid (linear interpolation across the small packet gaps);
  5. removes spikes (Hampel filter, default 5 sigma), then low-pass filters (default 25 Hz)
     - never across two trials;
  6. trims the 1 s margins and saves  <out>\\<subject>_<path>_T<trial>.npz
     with csi (T, 52), t_us (T) = board time of every sample, and the QC numbers.

Nothing is normalised here. Normalisation is done later, when the training windows are made.
The raw files are only read, never changed. Each trial is saved once (both cameras share the same CSI).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

csv.field_size_limit(1 << 24)

FS = 100.0                                   # output sample rate (Hz)
N_SUB = 64
NULL_SUB = [2, 3, 4, 5, 32, 59, 60, 61, 62, 63]      # always zero (found by inspect_csi.py on S01 LINE)
# subcarrier 0 is a big outlier (amplitude ~105 vs ~16); subcarrier 1 is a flat line at 4.0 (seen in the
# first trial picture), so it carries no information and would break per-subcarrier normalisation.
DROP_SUB = sorted(set(NULL_SUB + [0, 1]))
KEEP_SUB = [i for i in range(N_SUB) if i not in DROP_SUB]


def parse_raw(path):
    """Read one session's raw csv -> (esp_us, boot_id, amplitude[N, 64], number_of_bad_rows)."""
    t, boot, amp, bad = [], [], [], 0
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd)
        col = {h: i for i, h in enumerate(header)}
        for need in ("esp_us", "boot_id", "CSI_DATA"):
            if need not in col:
                sys.exit(f"Column '{need}' not found in {path}. Columns are: {header}")
        ie, ib, ic = col["esp_us"], col["boot_id"], col["CSI_DATA"]
        for row in rd:
            if len(row) <= ic:
                bad += 1
                continue
            try:
                tt = int(row[ie])
            except ValueError:
                bad += 1
                continue
            nums = re.findall(r"-?\d+", row[ic])
            if len(nums) != 2 * N_SUB:
                bad += 1
                continue
            v = np.array(nums, dtype=np.float32).reshape(-1, 2)
            amp.append(np.hypot(v[:, 0], v[:, 1]))
            t.append(tt)
            boot.append(row[ib])
    return np.array(t, np.int64), np.array(boot), np.array(amp, np.float32), bad


def read_trials(raw_path):
    """trials.csv of the session -> {trial_number: row}. It gives the boot_id of every trial."""
    tf = next(iter(sorted(raw_path.parent.glob("*_trials.csv"))), None)
    if tf is None:
        return {}
    with open(tf, newline="", encoding="utf-8-sig") as f:
        return {int(r["trial"]): r for r in csv.DictReader(f)}


def hampel(x, k=11, n_sigma=5.0):
    """Replace single-sample spikes by the local median. x is (T, S); k = window length in samples.
    n_sigma = 0 switches the filter off."""
    if not n_sigma:
        return x, 0
    med = median_filter(x, size=(k, 1), mode="nearest")
    dev = np.abs(x - med)
    mad = median_filter(dev, size=(k, 1), mode="nearest")
    thr = n_sigma * 1.4826 * mad
    bad = (dev > thr) & (thr > 0)
    out = np.where(bad, med, x)
    return out, int(bad.sum())


def process_trial(t, amp, start_us, end_us, margin_s=1.0, lowpass_hz=25.0, highpass_hz=None,
                  hampel_k=11, hampel_sigma=5.0):
    """t, amp: rows of ONE boot. Returns (csi[T, 52], t_us[T], info) or (None, None, reason)."""
    m_us = int(margin_s * 1e6)
    sel = (t >= start_us - m_us) & (t <= end_us + m_us)
    if sel.sum() < int(0.5 * FS * (end_us - start_us) / 1e6):
        return None, None, "too few rows in the window"
    ts, a = t[sel], amp[sel][:, KEEP_SUB]
    order = np.argsort(ts, kind="stable")
    ts, a = ts[order], a[order]
    ts, first = np.unique(ts, return_index=True)             # no repeated timestamps
    a = a[first]
    inner = (ts >= start_us) & (ts <= end_us)
    gaps = np.diff(ts[inner]) / 1e3 if inner.sum() > 1 else np.array([np.nan])
    info = {"rows_in_window": int(inner.sum()), "max_gap_ms": float(np.nanmax(gaps)),
            "gaps_over_50ms": int((gaps > 50).sum())}

    grid = np.arange(start_us - m_us, end_us + m_us + 1, 1e6 / FS)          # even 100 Hz grid, in us
    x = np.empty((len(grid), a.shape[1]), np.float32)
    for j in range(a.shape[1]):
        x[:, j] = np.interp(grid, ts, a[:, j])
    x, n_spikes = hampel(x, hampel_k, hampel_sigma)
    info["spikes_removed"] = n_spikes
    info["spikes_pct"] = 100.0 * n_spikes / x.size
    sos = butter(4, lowpass_hz, btype="low", fs=FS, output="sos")
    x = sosfiltfilt(sos, x, axis=0)
    if highpass_hz:
        x = sosfiltfilt(butter(2, highpass_hz, btype="high", fs=FS, output="sos"), x, axis=0)
    n_m = int(round(margin_s * FS))
    x = x[n_m: len(x) - n_m]
    t_out = np.round(grid[n_m: len(grid) - n_m]).astype(np.int64)
    info["flat_subcarriers"] = int((x.std(axis=0) < 0.05).sum())     # near-constant channels left in
    return x.astype(np.float32), t_out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_index", help="clip_index.csv (use the one inside Training_clips)")
    ap.add_argument("--out", default=r"D:\CSI_GAIT\csi_cache_v2")
    ap.add_argument("--lowpass", type=float, default=25.0, help="low-pass cutoff in Hz (default 25)")
    ap.add_argument("--highpass", type=float, default=0.0, help="optional drift removal in Hz (0 = off)")
    ap.add_argument("--hampel-sigma", type=float, default=5.0,
                    help="spike filter strictness; higher = removes fewer samples (default 5, 0 = off)")
    ap.add_argument("--hampel-window", type=int, default=11, help="spike filter window in samples (default 11)")
    ap.add_argument("--plot", action="store_true", help="save a picture of the first trial")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(a.clip_index, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    trials = {}                                   # (subject, path, trial) -> row ; one entry per trial
    for r in rows:
        trials.setdefault((r["subject"], r["path"], int(r["trial"])), r)
    by_raw = {}
    for key, r in trials.items():
        by_raw.setdefault(r["raw_csv"], []).append((key, r))
    print(f"{len(rows)} clips -> {len(trials)} distinct trials in {len(by_raw)} sessions")

    summary, problems, plotted = [], [], False
    for raw_csv, items in sorted(by_raw.items()):
        raw_path = Path(raw_csv)
        label = f"{items[0][0][0]}\\{items[0][0][1]}"
        if not raw_path.is_file():
            problems.append(f"{label}: raw file not found: {raw_path}")
            continue
        t, boot, amp, bad = parse_raw(raw_path)
        tr = read_trials(raw_path)
        print(f"{label}: {len(t)} usable rows ({bad} broken rows skipped)")
        for (subj, path, trial), r in sorted(items, key=lambda kv: kv[0][2]):
            start, end = int(r["csi_t_start_us"]), int(r["csi_t_end_us"])
            boot_id = tr.get(trial, {}).get("boot_id")
            if boot_id is None:                   # fall back to the boot that holds this time window
                cand = [b for b in np.unique(boot) if ((t[boot == b] >= start) & (t[boot == b] <= end)).any()]
                boot_id = cand[0] if cand else None
            if boot_id is None:
                problems.append(f"{label} T{trial:02d}: no boot_id")
                continue
            m = boot == str(boot_id)
            x, t_us, info = process_trial(t[m], amp[m], start, end, 1.0, a.lowpass, a.highpass or None,
                                          a.hampel_window, a.hampel_sigma)
            if x is None:
                problems.append(f"{label} T{trial:02d}: {info}")
                continue
            fn = out / f"{subj}_{path}_T{trial:02d}.npz"
            np.savez_compressed(fn, csi=x, t_us=t_us, kept_subcarriers=np.array(KEEP_SUB), fs=FS,
                                subject=subj, path=path, trial=trial, boot_id=str(boot_id),
                                lowpass_hz=a.lowpass, **{k: v for k, v in info.items()})
            summary.append({"subject": subj, "path": path, "trial": trial, "file": fn.name, "samples": len(x),
                            "rows_in_window": info["rows_in_window"], "max_gap_ms": f"{info['max_gap_ms']:.0f}",
                            "gaps_over_50ms": info["gaps_over_50ms"], "spikes_removed": info["spikes_removed"],
                            "spikes_pct": f"{info['spikes_pct']:.2f}", "flat_subcarriers": info["flat_subcarriers"]})
            if info["max_gap_ms"] > 100 or info["rows_in_window"] < 2800:
                problems.append(f"{label} T{trial:02d}: rows {info['rows_in_window']}, max gap {info['max_gap_ms']:.0f} ms")
            if info["flat_subcarriers"]:
                problems.append(f"{label} T{trial:02d}: {info['flat_subcarriers']} constant subcarrier(s) still in the data")
            if a.plot and not plotted:
                plotted = True
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
                    tt = (t_us - t_us[0]) / 1e6
                    ax[0].imshow(x.T, aspect="auto", origin="lower", extent=[tt[0], tt[-1], 0, x.shape[1]])
                    ax[0].set_ylabel("subcarrier (kept)")
                    ax[0].set_title(f"{subj} {path} trial {trial}: filtered CSI amplitude")
                    ax[1].plot(tt, x[:, ::8])
                    ax[1].set_xlabel("seconds from trial start")
                    ax[1].set_ylabel("amplitude")
                    fig.tight_layout()
                    fig.savefig(out / "first_trial.png", dpi=110)
                    print(f"  picture saved: {out / 'first_trial.png'}")
                except ImportError:
                    print("  (no picture: run  pip install matplotlib  and add --plot again)")

    if summary:
        with open(out / "csi_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        pct = np.array([float(s["spikes_pct"]) for s in summary])
        print(f"\nspike filter replaced {np.median(pct):.2f}% of samples per trial (median), "
              f"{pct.max():.2f}% at most")
    print(f"saved {len(summary)} trials to {out}  (index: csi_index.csv)")
    print(f"kept {len(KEEP_SUB)} subcarriers; dropped {DROP_SUB}")
    if problems:
        print(f"{len(problems)} thing(s) to look at:")
        for p in problems:
            print("   " + p)
    else:
        print("no problems found")


if __name__ == "__main__":
    main()
