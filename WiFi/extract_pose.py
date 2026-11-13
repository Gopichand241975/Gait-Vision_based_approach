"""
extract_pose.py - single-person 2D pose extraction for gait analysis.

Pipeline per frame:  torchvision Faster R-CNN (person)  ->  HRNet-W48 (17 COCO joints)

Usage examples (Anaconda Prompt, csi_env active):
    python extract_pose.py --input D:\\CSI_GAIT\\videos\\walk01.mp4 --save-video --max-frames 100
    python extract_pose.py --input D:\\CSI_GAIT\\videos --out D:\\CSI_GAIT\\pose_out

Output, one file per video:  <out>\\<video_name>_pose.npz  containing
    keypoints   (T, 17, 2)  x, y in original-frame pixels (NaN where no person found)
    scores      (T, 17)     HRNet heatmap confidence per joint
    boxes       (T, 4)      person box x1, y1, x2, y2 (NaN where no person found)
    box_scores  (T,)        detector confidence
    detected    (T,)        bool, True where a person was found
    frame_idx   (T,)        frame numbers
    time_s      (T,)        frame_idx / fps  (use this to sync with CSI timestamps)
    fps, width, height, video
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

REPO = r"D:\CSI_GAIT\tools\simple-HRNet"
DEFAULT_WEIGHTS = os.path.join(REPO, "weights", "pose_hrnet_w48_384x288.pth")
sys.path.insert(0, REPO)
from models_.hrnet import HRNet  # noqa: E402

IN_W, IN_H = 288, 384          # HRNet input (width, height)
HM_W, HM_H = 72, 96            # heatmap size
STRIDE = IN_W / HM_W           # 4
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv")

COCO_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
              "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
              "l_knee", "r_knee", "l_ankle", "r_ankle"]
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def load_detector(name, device):
    from torchvision.models import detection as det
    if name == "mobilenet":
        model = det.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        model = det.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    return model.to(device).eval()


def load_hrnet(weights, device):
    model = HRNet(c=48, nof_joints=17)
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def detect_people(detector, frame_rgb, device, thresh):
    t = torch.from_numpy(frame_rgb).to(device).permute(2, 0, 1).float() / 255.0
    out = detector([t])[0]
    keep = (out["labels"] == 1) & (out["scores"] >= thresh)   # label 1 = person (COCO)
    return out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy()


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def choose_box(boxes, scores, prev_box):
    """Follow the same person: best overlap with last box, else biggest confident person."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if prev_box is not None:
        ious = np.array([box_iou(prev_box, b) for b in boxes])
        if ious.max() > 0.1:
            return int(ious.argmax())
    return int((areas * scores).argmax())


@torch.no_grad()
def estimate_pose(model, frame_rgb, box, device):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = x2 - x1, y2 - y1
    aspect = IN_W / IN_H
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    w *= 1.25
    h *= 1.25
    s = IN_W / w
    M = np.array([[s, 0, IN_W / 2 - s * cx],
                  [0, s, IN_H / 2 - s * cy]], dtype=np.float32)
    crop = cv2.warpAffine(frame_rgb, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    heat = model(x)[0].cpu().numpy()          # (17, 96, 72)

    flat = heat.reshape(17, -1)
    idx = flat.argmax(1)
    conf = flat.max(1)
    py = (idx // HM_W).astype(np.float32)
    px = (idx % HM_W).astype(np.float32)
    for j in range(17):                        # quarter-pixel refinement
        ix, iy = int(px[j]), int(py[j])
        if 0 < ix < HM_W - 1:
            px[j] += 0.25 * np.sign(heat[j, iy, ix + 1] - heat[j, iy, ix - 1])
        if 0 < iy < HM_H - 1:
            py[j] += 0.25 * np.sign(heat[j, iy + 1, ix] - heat[j, iy - 1, ix])
    xy_crop = np.stack([px, py], axis=1) * STRIDE
    xy = (xy_crop - np.array([IN_W / 2, IN_H / 2])) / s + np.array([cx, cy])
    return xy.astype(np.float32), conf.astype(np.float32)


def draw_overlay(frame_bgr, box, kp, sc, min_conf=0.3):
    if box is not None:
        cv2.rectangle(frame_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    for a, b in SKELETON:
        if sc[a] > min_conf and sc[b] > min_conf:
            cv2.line(frame_bgr, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), (255, 200, 0), 2)
    for j in range(17):
        if sc[j] > min_conf:
            cv2.circle(frame_bgr, (int(kp[j][0]), int(kp[j][1])), 4, (0, 0, 255), -1)
    return frame_bgr


def process_video(path, out_dir, detector, hrnet, device, args):
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  Cannot open:", path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames) if total > 0 else args.max_frames
    print("%s: %dx%d, %.2f fps, ~%d frames" % (name, width, height, fps, total))

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(os.path.join(out_dir, name + "_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    kps, scs, bxs, bsc, det_ok = [], [], [], [], []
    prev_box = None
    t0 = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores = detect_people(detector, rgb, device, args.det_thresh)
        if len(boxes) == 0:
            kps.append(np.full((17, 2), np.nan, np.float32))
            scs.append(np.zeros(17, np.float32))
            bxs.append(np.full(4, np.nan, np.float32))
            bsc.append(np.nan)
            det_ok.append(False)
            if writer is not None:
                writer.write(frame)
        else:
            i = choose_box(boxes, scores, prev_box)
            box = boxes[i]
            prev_box = box
            kp, sc = estimate_pose(hrnet, rgb, box, device)
            kps.append(kp)
            scs.append(sc)
            bxs.append(box.astype(np.float32))
            bsc.append(float(scores[i]))
            det_ok.append(True)
            if writer is not None:
                writer.write(draw_overlay(frame, box, kp, sc))
        n += 1
        if n % 100 == 0:
            rate = n / (time.time() - t0)
            print("  frame %d / %d  (%.1f fps)" % (n, total, rate))
    cap.release()
    if writer is not None:
        writer.release()
    if n == 0:
        print("  No frames read.")
        return

    out_path = os.path.join(out_dir, name + "_pose.npz")
    frame_idx = np.arange(n)
    np.savez_compressed(
        out_path,
        keypoints=np.stack(kps), scores=np.stack(scs), boxes=np.stack(bxs),
        box_scores=np.array(bsc, dtype=np.float32), detected=np.array(det_ok),
        frame_idx=frame_idx, time_s=(frame_idx / fps).astype(np.float64),
        fps=fps, width=width, height=height, video=os.path.basename(path),
        joint_names=np.array(COCO_NAMES),
    )
    found = int(np.sum(det_ok))
    print("  Done: %d frames, person found in %d (%.1f%%). Saved %s" %
          (n, found, 100.0 * found / n, out_path))


def main():
    ap = argparse.ArgumentParser(description="Extract 2D pose (HRNet) from gait videos")
    ap.add_argument("--input", required=True, help="video file or folder of videos")
    ap.add_argument("--out", default=r"D:\CSI_GAIT\pose_out", help="output folder")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--detector", choices=["resnet50", "mobilenet"], default="resnet50",
                    help="person detector (mobilenet is faster, resnet50 more accurate)")
    ap.add_argument("--det-thresh", type=float, default=0.7)
    ap.add_argument("--save-video", action="store_true", help="also write an overlay mp4 to check results")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (for quick tests)")
    ap.add_argument("--recursive", action="store_true",
                    help="search all sub-folders of --input; output mirrors the folder structure")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip videos whose _pose.npz already exists (to resume a long run)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # jobs = list of (video_path, output_folder)
    jobs = []
    if os.path.isdir(args.input):
        if args.recursive:
            for root, _dirs, files in os.walk(args.input):
                rel = os.path.relpath(root, args.input)
                out_dir = args.out if rel == "." else os.path.join(args.out, rel)
                for f in sorted(files):
                    if f.lower().endswith(VIDEO_EXT):
                        jobs.append((os.path.join(root, f), out_dir))
        else:
            for f in sorted(os.listdir(args.input)):
                if f.lower().endswith(VIDEO_EXT):
                    jobs.append((os.path.join(args.input, f), args.out))
    else:
        jobs.append((args.input, args.out))
    jobs.sort()
    if not jobs:
        print("No videos found in", args.input, "(extensions:", ", ".join(VIDEO_EXT) + ")")
        return
    print("Videos to process:", len(jobs), "| device:", device)

    detector = load_detector(args.detector, device)
    hrnet = load_hrnet(args.weights, device)
    for i, (v, out_dir) in enumerate(jobs, 1):
        name = os.path.splitext(os.path.basename(v))[0]
        if args.skip_existing and os.path.isfile(os.path.join(out_dir, name + "_pose.npz")):
            print("[%d/%d] skip (already done): %s" % (i, len(jobs), v))
            continue
        os.makedirs(out_dir, exist_ok=True)
        print("[%d/%d] %s" % (i, len(jobs), v))
        try:
            process_video(v, out_dir, detector, hrnet, device, args)
        except Exception as e:  # keep going if one video is bad
            print("  FAILED on %s: %r" % (v, e))


if __name__ == "__main__":
    main()
