#!/usr/bin/env python3
"""
inspect_csi.py - look inside ONE session's raw CSI file before building any training code.

Usage:
    python inspect_csi.py "D:\\...\\Dataset_publishment\\S01\\LINE\\S01_LINE_raw.csv"

Reads only (nothing is changed). It reports, in plain text:
  * the columns, number of rows, boot_id values and the time span (esp_us);
  * the packet rate and the largest gaps;
  * how many numbers the CSI_DATA column holds per row, and which subcarriers look empty (always zero),
    assuming the numbers are pairs (imaginary, real) as in the ESP32-CSI-Tool;
  * for every trial in the session's trials.csv: how many rows fall inside the trial window, the rate
    and the largest gap, next to what trials.csv itself says (a cross-check of the time lookup).
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

import numpy as np

csv.field_size_limit(1 << 24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    a = ap.parse_args()
    raw = Path(a.raw).resolve()
    stem = raw.name[: -len("_raw.csv")] if raw.name.endswith("_raw.csv") else raw.stem

    t_us, boot, lens, seqs, markers = [], [], [], [], []
    amp_sum = None
    zero_cnt = None
    sample_row = None
    n_ok = 0
    with open(raw, newline="", encoding="utf-8-sig", errors="replace") as f:
        rd = csv.reader(f)
        header = next(rd)
        col = {h: i for i, h in enumerate(header)}
        for need in ("esp_us", "boot_id", "CSI_DATA"):
            if need not in col:
                sys.exit(f"Column '{need}' not found. Columns are: {header}")
        for row in rd:
            if len(row) <= col["CSI_DATA"]:
                continue
            try:
                t = int(row[col["esp_us"]])
                b = row[col["boot_id"]]
                nums = [int(x) for x in re.findall(r"-?\d+", row[col["CSI_DATA"]])]
            except ValueError:
                continue
            t_us.append(t)
            boot.append(b)
            lens.append(len(nums))
            if "seq" in col:
                seqs.append(row[col["seq"]])
            if "marker" in col:
                markers.append(row[col["marker"]])
            if len(nums) >= 2 and len(nums) % 2 == 0:
                v = np.array(nums, dtype=np.float32).reshape(-1, 2)
                amp = np.hypot(v[:, 0], v[:, 1])
                if amp_sum is None or len(amp) != len(amp_sum):
                    if amp_sum is None:
                        amp_sum, zero_cnt = np.zeros(len(amp)), np.zeros(len(amp))
                    else:
                        continue
                amp_sum += amp
                zero_cnt += (amp == 0)
                n_ok += 1
            if sample_row is None:
                sample_row = row

    n = len(t_us)
    print(f"file: {raw.name}")
    print(f"columns ({len(header)}): {', '.join(header)}")
    print(f"rows read: {n}")
    if not n:
        sys.exit("No rows could be read.")
    t = np.array(t_us, dtype=np.int64)
    print(f"boot_id values: {sorted(set(boot))}  (a board reboot starts a new boot_id and resets esp_us)")
    for b in sorted(set(boot)):
        idx = [i for i, x in enumerate(boot) if x == b]
        tb = t[idx]
        dur = (tb.max() - tb.min()) / 1e6
        gaps = np.diff(tb) / 1e3
        print(f"  boot {b}: {len(idx)} rows, {dur:.1f} s, rate {len(idx) / max(dur, 1e-9):.1f}/s, "
              f"median step {np.median(gaps):.1f} ms, largest gap {gaps.max():.0f} ms")
    c = {}
    for x in lens:
        c[x] = c.get(x, 0) + 1
    print(f"numbers in CSI_DATA per row: {dict(sorted(c.items(), key=lambda kv: -kv[1])[:5])}")
    if amp_sum is not None and n_ok:
        mean = amp_sum / n_ok
        empty = np.where(zero_cnt / n_ok > 0.99)[0]
        print(f"as (imaginary, real) pairs: {len(mean)} subcarriers; "
              f"{len(empty)} always zero (positions {empty.tolist()[:40]}{' ...' if len(empty) > 40 else ''})")
        print(f"usable subcarriers: {len(mean) - len(empty)}; "
              f"mean amplitude min/median/max: {mean[mean > 0].min():.1f} / {np.median(mean[mean > 0]):.1f} / {mean.max():.1f}")
    if markers:
        m = {}
        for x in markers:
            m[x] = m.get(x, 0) + 1
        print(f"marker column values (rows recorded while a beep plays are marked): {dict(list(m.items())[:6])}")
    print("first row (cut to 300 characters): " + ",".join(sample_row)[:300])

    tf = next(iter(sorted(raw.parent.glob(f"{stem}_trials.csv")) + sorted(raw.parent.glob("*_trials.csv"))), None)
    if tf is None:
        print("\nno trials.csv next to the raw file; skipping the per-trial check")
        return
    print(f"\nper-trial check against {tf.name}:")
    print("trial status   rows_in_window  rate/s  max_gap_ms | trials.csv: rows rate_per_s max_gap_ms")
    with open(tf, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                s, e = int(r["t_start_us"]), int(r["t_end_us"])
            except (ValueError, KeyError):
                print(f"{r.get('trial', '?'):>5} {r.get('status', ''):<12} (no window)")
                continue
            sel = [i for i in range(n) if boot[i] == r.get("boot_id", boot[0]) and s <= t_us[i] <= e]
            ts = t[sel]
            gap = (np.diff(ts).max() / 1e3) if len(ts) > 1 else float("nan")
            rate = len(sel) / ((e - s) / 1e6)
            print(f"{r['trial']:>5} {r['status']:<12} {len(sel):>10} {rate:>10.1f} {gap:>10.0f} | "
                  f"{r.get('rows', '')} {r.get('rate_per_s', '')} {r.get('max_gap_ms', '')}")


if __name__ == "__main__":
    main()
