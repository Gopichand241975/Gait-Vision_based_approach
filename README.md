# Vision-Based Gait Recognition (17-joint skeleton)

Pipeline: video (.mp4) -> YOLOv8-pose (17 COCO joints per frame) -> skeleton sequence
(x, y, confidence) -> ResGCN gait model -> gait feature vector -> gallery/probe matching
(Euclidean distance) -> identity -> Rank-1 / Rank-5 / Rank-10 + confusion matrix.

Deviations from the Week-1 diagram (state these in your report):
- YOLOv8-pose is the pose estimator instead of HRNet (both output the 17 COCO keypoints).
- The recognition model is a skeleton-graph ResGCN (GaitGraph style), not GaitSet/GaitPart
  (those two work on silhouette images, not skeletons).

## 1. Where to put the data
```
dataset/                    <- YOUR VIDEOS GO HERE (you create the subfolders)
  person1/walk1.mp4         one folder per person; folder name = identity (label)
  person1/walk2.mp4         >= 3 videos per person recommended
  person2/walk1.mp4
```
File names must be unique inside a person's folder. .mp4 .avi .mov .mkv are accepted.

## 2. How the same dataset is used for training AND testing (config.SPLIT_MODE)
- "videos" (default): all people are used in both. For each person the videos are sorted by
  file name; the first GALLERY_PER_PERSON (2) videos are TRAINING + GALLERY (reference). The
  remaining videos are PROBES (testing, never seen in training).
- "subjects": TRAIN_RATIO (60%) of the people train; the other people are the test set
  (their first 2 videos = gallery, the rest = probes). Needs many people.
The exact split used is saved in results/training/split.json.

## 3. Where results are saved
| Folder / file | Created by | Meaning |
|---|---|---|
| skeletons_raw/<person>/<video>.npy | pose_extract.py | (T frames, 17 joints, 3 = x, y pixels, confidence). A `.failed` file = video skipped |
| processed/<person>/<video>.npy | preprocess.py | same shape, normalised (smoothed, hip-centred, torso-scaled) |
| checkpoints/last.pt, iter_XXXXXX.pt | train.py | trained weights (+ split + iteration) |
| results/skeleton/skeleton_summary.csv | pose_extract.py | per video: frames, % frames with a person, mean confidence |
| results/skeleton/pose_check/*.jpg | check_pose.py | frames with skeleton drawn, to verify joints |
| results/training/train_log.csv | train.py | iteration, loss, training accuracy, learning rate |
| results/training/split.json | train.py | which videos are train / gallery / probe |
| results/recognition/features_*.npz | test.py | 128-number gait feature per video |
| results/recognition/distance_matrix.csv | test.py | Euclidean distance probe x gallery |
| results/recognition/predictions.csv | test.py | actual vs predicted person, Correct/Wrong |
| results/recognition/confusion_matrix.csv/.png | test.py | actual (rows) vs predicted (columns) |
| results/recognition/accuracy.json, report.txt | test.py | Rank-1/5/10 (Rank-k only if more than k people), per-person accuracy |

## 4. What each file does (what to edit)
| File | Purpose | Edit when |
|---|---|---|
| config.py | ALL settings and paths | batch size, iterations, split, model size, pose model |
| pose_extract.py | videos -> 17-joint skeletons, 5 videos per run | rarely |
| check_pose.py | draws skeletons on frames for checking | rarely |
| preprocess.py | normalises skeletons | change normalisation / smoothing |
| augment.py | flip, scale, rotation, noise, joint dropout | change augmentation |
| dataset.py | split logic, 60-frame windows, 7 input channels | change split / input features |
| model.py | ResGCN -> 128-number gait feature | change network |
| train.py | trains the model | change training loop |
| test.py | gallery/probe matching, ranks, confusion matrix | change metrics |
| run_all.py | runs all steps in order | rarely |
| selftest.py | smoke test, no videos needed | rarely |
| requirements.txt | libraries | add a library |

Most-changed settings in config.py: BATCH_SIZE (lower on "CUDA out of memory"), TOTAL_ITERS,
SAVE_EVERY, GALLERY_PER_PERSON, SPLIT_MODE, POSE_MODEL ("yolov8n-pose.pt" faster,
"yolov8m-pose.pt" more accurate), BATCH_VIDEOS (videos per pose run, default 5).

## 5. Run on Google Colab (runtime: GPU T4, NOT TPU)
```
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/gait_project
!pip install -r requirements.txt
!python selftest.py --pose        # must print SELFTEST PASSED
!python pose_extract.py           # 5 videos per run; run again for the next 5
!python check_pose.py             # look at results/skeleton/pose_check/*.jpg
!python preprocess.py
!python train.py --iters 300 --batch_size 8      # quick trial
!python test.py
```
Everything in one go: `!python run_all.py`. All videos at once: `!python pose_extract.py --all`.

## 6. Handover to the person who trains (GPU machine)
- Install PyTorch for the GPU (pytorch.org), then `pip install -r requirements.txt`.
- Run `python selftest.py --pose`; start training only after SELFTEST PASSED.
- Full training: `python train.py`, then `python test.py`.
- Interrupted? `python train.py --resume --iters <same total as before>`.
- If videos are added or changed: delete checkpoints/ and results/training/, then retrain.

## 7. What was tested
selftest.py (synthetic skeletons: preprocess, train, resume, test, all result files) on CPU.
NOT tested: real YOLOv8 download and real videos (run `selftest.py --pose` in Colab), CUDA,
long training, CASIA-B.