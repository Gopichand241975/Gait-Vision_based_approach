# Vision-Based Gait Recognition: run guide (GPU machine)

Pipeline: video -> YOLOv8-pose (17 COCO joints) -> skeleton sequence -> ResGCN -> gait feature
-> gallery/probe matching (Euclidean distance) -> Rank-1 accuracy + confusion matrix.

## What you need
- Windows PC with an NVIDIA GPU and an up-to-date NVIDIA driver
- Python 3.10, 3.11 or 3.12 (3.12 recommended; avoid 3.13 and newer)
- Internet on the first run (downloads the pose model, about 50 MB)
- About 40 GB free disk (the 28 videos are about 35 GB)

## 1. Set up the folder
Unzip `GaitCode.zip` into a folder, for example `C:\Gait\VisionBased`. Copy the `dataset` folder
into it so the layout is exactly:
```
VisionBased\
  config.py  pose_extract.py  check_pose.py  preprocess.py  augment.py
  dataset.py  model.py  train.py  test.py  run_all.py  selftest.py  requirements.txt
  dataset\cam1\P01\P01_STR_cam1.mp4 ... P07\P07_CR2_cam1.mp4
```
Open PowerShell in that folder and check:
```
cd C:\Gait\VisionBased
(Get-ChildItem dataset -Recurse -Filter *.mp4).Count
```
It must print **28**. Do not rename the videos or compress them. The person is the folder name
(P01) and the direction is the second part of the file name (STR, PRP, CR1, CR2).

## 2. Create the environment
```
py -3.12 -m venv gait
gait\Scripts\activate
```
`(gait)` must appear at the start of the line. If PowerShell blocks it, run this once, then activate again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Install PyTorch with GPU support
Check the GPU and driver:
```
nvidia-smi
```
Open https://pytorch.org/get-started/locally/ , choose Windows, Pip, Python, and the CUDA
version that nvidia-smi supports, then copy the command it shows. Example for CUDA 12.1:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Check it:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It must print `True` and the GPU name. If it prints `False`, the wrong (CPU) build is installed;
run `pip uninstall -y torch torchvision` and install again with the command from the selector.

## 4. Install the other libraries
```
pip install -r requirements.txt
python -c "import ultralytics, cv2, matplotlib; print('OK')"
```

## 5. Test the code (must pass before anything else)
```
python selftest.py --pose
```
The first line must say `device: cuda` and the last line must say `SELFTEST PASSED`.

## 6. Extract the skeletons (slow: many hours for 28 full videos)
```
python pose_extract.py --all
```
- Do not close the window or press Ctrl+C. If it stops, run the same command again; finished
  videos are skipped.
- Output: `skeletons_raw\P01\P01_STR_cam1.npy` etc. Each file is (frames, 17 joints, x/y/confidence).

## 7. Check the skeletons (do not skip)
```
python check_pose.py
Get-Content results\skeleton\skeleton_summary.csv
```
- Open the images in `results\skeleton\pose_check\`. The green skeleton must sit on the walking person.
- In the csv, `mean_confidence` should be above about 0.4 and `frames_with_person_pct` should be high.
  `first_person_frame` shows when the person walks in. Frames before and after are removed in step 8.
- If the skeletons are wrong, stop and report it. Do not train.

## 8. Normalise the skeletons
```
python preprocess.py
```
Output: `processed\...`

## 9. Train
Quick trial first, to confirm the loss falls and the accuracy rises:
```
python train.py --iters 300 --batch_size 16
```
Then the full run. Remove the trial output first:
```
Remove-Item checkpoints, results\training, results\recognition -Recurse -Force -ErrorAction SilentlyContinue
python train.py
```
- Full training is 3000 iterations. Checkpoints are saved every 200 iterations in `checkpoints\`.
- If training stops partway: `python train.py --resume --iters 3000`
- If you see "CUDA out of memory": `python train.py --batch_size 8`
- The training log is `results\training\train_log.csv`.

## 10. Test (re-identification accuracy)
```
python test.py
Get-Content results\recognition\report.txt
```
Results are in `results\recognition\`:
- `report.txt`, `accuracy.json`: Rank-1 (and Rank-5 when there are more than 5 people), per person, per direction
- `predictions.csv`: actual vs predicted person for every test segment
- `confusion_matrix.png` / `.csv`: who is confused with whom
- `distance_matrix.csv`, `features_*.npz`: distances and gait feature vectors

## 11. Second, harder test: unseen walking direction
In `config.py` change these two lines:
```
SPLIT_MODE = "cross_direction"
TEST_DIRECTION = "CR2"
```
Then:
```
Remove-Item checkpoints, results\training, results\recognition -Recurse -Force
python train.py
python test.py
```
This trains on STR, PRP and CR1 and recognises each person in CR2, a direction never seen in
training. Expect lower accuracy. Report both numbers. (Skeletons from steps 6 and 8 are reused.)

## What the default split means
`SPLIT_MODE = "temporal"`: the first 60% of every video is used for training and as the gallery.
The last 40% is the test. This assumes each video contains several walking passes.

## Send back
The whole `results` folder (especially `results\recognition\report.txt` and
`confusion_matrix.png`) and `checkpoints\last.pt`.

## Files
| File | Purpose |
|---|---|
| config.py | all settings and folder paths (edit here) |
| pose_extract.py | videos -> 17-joint skeletons |
| check_pose.py | draws skeletons on frames for checking |
| preprocess.py | trims empty frames, smooths, normalises |
| augment.py | training augmentation |
| dataset.py | train/gallery/probe split, windows, input features |
| model.py | ResGCN network |
| train.py | training |
| test.py | gallery/probe matching, accuracy, confusion matrix |
| run_all.py | runs all steps in order |
| selftest.py | smoke test with synthetic data |