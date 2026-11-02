# Vision-based gait recognition

Identify **which of 7 subjects is walking** from video only. No Wi-Fi, CSI or other sensors.

The dataset has 7 subjects (`S01`-`S07`), 4 walking directions (`LINE`, `PERP`, `CROSS-RX-LEFT`,
`CROSS-RX-RIGHT`), 2 cameras (`Cam-P`, `Cam-S`) and 10 clips per camera = **560 clips**.
For each subject + direction one camera (the side view) is selected, which leaves **280 clips**,
split 6 / 2 / 2 per group into **168 train / 56 validation / 56 test** clips.
`verify_dataset.py` checks these numbers on your data; they are expectations, not assumptions.

dataset.py` | checks names, folders, counts, that every clip opens, duplicates | `manifests/dataset_inventory.csv`, `verification_report.txt` |
| 2 | `python select_camera.py` | picks the side-view camera per subject + direction, flags unclear ones | `manifests/camera_selection.csv`, `camera_review/*.jpg` |
| 2b | `python select_camera.py --set S03 PERP Cam-S --note "..."` | records your own decision | `camera_selection.csv` |
| 3 | `python make_split.py` | 6 / 2 / 2 split with a fixed seed, then verifies it | `manifests/dataset_split.csv` |
| 4 | `python pose_extract.py --all` | skeletons for the selected clips only | `skeletons_raw/` |
| 5 | `python check_pose.py` | pictures to check the skeletons | `results/skeleton/pose_check/` |
| 6 | `python preprocess.py` | normalised skeletons | `processed/` |
| 7 | `python train.py` | trains, validates every 200 iterations, keeps the best checkpoint | `checkpoints/`, `results/training/` |
| 8 | `python test.py` | final test of `best.pt` on the test clips | `results/recognition/` |
| 9 | `python predict.py clip.mp4` | names the subject in a new clip | prints the result |

`python run_all.py` runs steps 1 to 8 in one go. It stops when a camera needs a manual decision.







