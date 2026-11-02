# Vision-based gait recognition

Identify **which of 7 subjects is walking** from video only. No Wi-Fi, CSI or other sensors.

The dataset has 7 subjects (`S01`-`S07`), 4 walking directions (`LINE`, `PERP`, `CROSS-RX-LEFT`,
`CROSS-RX-RIGHT`), 2 cameras (`Cam-P`, `Cam-S`) and 10 clips per camera = **560 clips**.
For each subject + direction one camera (the side view) is selected, which leaves **280 clips**,
split 6 / 2 / 2 per group into **168 train / 56 validation / 56 test** clips.
`verify_dataset.py` checks these numbers on your data; they are expectations, not assumptions.

dataset.py` | checks names, folders, co





