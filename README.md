# Vision-based gait recognition

Identify **which of 7 subjects is walking** from video only. No Wi-Fi, CSI or other sensors.

The dataset has 7 subjects (`S01`-`S07`), 4 walking directions (`LINE`, `PERP`, `CROSS-RX-LEFT`,
`CROSS-RX-RIGHT`), 2 cameras (`Cam-P`, `Cam-S`) and 10 clips per camera = **560 clips**.
For each subject + direction one camera (the side view) is selected, which leaves **280 clips**,
split 6 / 2 / 2 per group into **168 train / 56 validation / 56 test** clips.
`verify_dataset.py` checks these numbers on your data; they are expectations, not assumptions.

The exact commands are in `GUIDE.txt`. This file explains what the project does.

## How it works

1. **Pose** (`pose_extract.py`): YOLOv8-pose gives 17 body joints per frame (x, y, confidence).
2. **Normalise** (`preprocess.py`): each clip is smoothed, centred on the hips and scaled by its own
   torso length. Nothing is fitted on the whole dataset, so nothing can leak from validation or test.
3. **Windows** (`dataset.py`): the model sees 60-frame windows with 7 values per joint
   (x, y, confidence, velocity, bone direction). Clips shorter than 60 frames are repeated to fill a window.
4. **Model** (`model.py`): a graph convolution network (ResGCN style) turns a window into a 128-number
   gait feature. It is trained to tell the 7 subjects apart (`train.py`, with flip / scale / noise
   augmentation from `augment.py`).
5. **Recognition** (`evaluate.py`, `test.py`): each clip becomes one feature (mean over its windows).
   A validation or test clip gets the subject of the **nearest training clip** (Euclidean distance).

## Pipeline order

| # | Command | What it does | Writes |
|---|---|---|---|
| 0 | `python selftest.py --pose` | tests the whole code on fake data | nothing (temp folder) |
| 1 | `python verify_dataset.py` | checks names, folders, counts, that every clip opens, duplicates | `manifests/dataset_inventory.csv`, `verification_report.txt` |
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

## Camera selection

A side view shows the person crossing the image: the shoulders overlap and the hips travel far sideways.
`select_camera.py` inspects a few clips of each camera (`CAM_SAMPLE_CLIPS` clips, `CAM_FRAMES_PER_CLIP`
frames each), measures the shoulder width relative to the torso and the sideways hip travel, and gives each
camera a score from 0 to 1.

- One camera is selected **only** if its score reaches `CAM_MIN_SIDE_SCORE` and beats the other camera by
  `CAM_MIN_MARGIN`. Otherwise the group is `needs_review` and no camera is selected.
- Statuses: `auto_selected`, `needs_review`, `manual_confirmed` (set by you, never overwritten
  unless you use `--overwrite`).
- The thresholds in `config.py` are starting values that were not calibrated on your videos. Look at
  `manifests/camera_review/<subject>_<direction>.jpg` (Cam-P left, Cam-S right) before trusting them.
- `make_split.py` uses only `auto_selected` and `manual_confirmed` groups, and always uses all clips of
  the selected camera. The other camera's clips are ignored, never deleted or changed.

## Split rules

- Group = one subject + one direction. Within a group the clips are shuffled with `SPLIT_SEED` (42), then
  6 train, 2 validation, 2 test.
- `dataset_split.csv` columns: `clip_id`, `original_filename`, `relative_path`, `subject`, `direction`,
  `selected_camera`, `trial`, `split`, `recording_id`, `group_clips`, `group_status`, `seed`.
- Differences from the expected counts are reported, never forced: undecided groups are left out, groups
  without 10 clips are split in the same 60/20/20 proportions and marked `nonstandard_count_<n>`, groups
  with fewer than 3 clips are left out.
- The check at the end of `make_split.py` confirms that every clip is in exactly one split, that the camera
  matches `camera_selection.csv`, that the subject and direction match the file name, that each group has the
  expected counts, and that no identical files sit in different splits.

## Training, validation and testing

| Set | Used for |
|---|---|
| Train (6 clips per group) | updating the model; also the gallery that other clips are compared with |
| Validation (2 per group) | choosing the best checkpoint during training (`best.pt`), nothing else |
| Test (2 per group) | one final evaluation by `test.py`, after all decisions are made |

`train.py` never reads the test clips. `test.py` uses `best.pt`, or `last.pt` if there is no validation set.
Each checkpoint stores the split it was trained with, so `test.py` and `predict.py` always use that split,
not a newer `dataset_split.csv`. After a new split, train again.

## Metrics

`test.py` reports accuracy, macro and weighted precision / recall / F1, per-subject precision / recall / F1,
per-direction accuracy, Rank-5 accuracy (when there are more than 5 subjects) and a confusion matrix
(`results/recognition/report.txt`, `accuracy.json`, `classification_report.csv`, `confusion_matrix.png`).
Macro averages are the plain mean over the subjects that occur in the test clips.
`train.py` writes `results/training/val_log.csv` and `training_curves.png`.

## Settings you may change (`config.py`)

| Setting | Meaning | Default |
|---|---|---|
| `SPLIT_SEED`, `SPLIT_CLIPS` | split seed and train / val / test clips per group | 42, (6, 2, 2) |
| `CAM_*` | camera selection thresholds and how many clips / frames are inspected | see file |
| `TOTAL_ITERS`, `BATCH_SIZE`, `LR` | training length, batch size, learning rate | 3000, 16, 1e-3 |
| `SAVE_EVERY` | how often a checkpoint is saved and validation is run | 200 |
| `SEQ_LEN`, `SEQ_STRIDE` | window length and test-time step in frames | 60, 30 |
| `POSE_MODEL`, `POSE_CONF` | pose model and detection confidence | yolov8m-pose.pt, 0.25 |

The older split modes (`temporal`, `cross_direction`, `per_direction`, `videos`, `subjects`) are still in
the code but have no validation set. `total_accuracy.py` belonged to the cross-direction experiments and
is not part of the current pipeline.

## Limitations

- The 10 clips of a group come from the same camera and probably the same recording session, so train,
  validation and test clips share recording conditions. The test accuracy will probably be higher than on a
  new session. The check for identical files cannot find overlapping frames between clips.
- 56 test clips is small: one clip changes the accuracy by about 1.8 points.
- Camera selection is a heuristic. Treat `auto_selected` as a suggestion until you have looked at the pictures.
- The largest detected person in each frame is used, so other people in view can disturb the skeleton.
