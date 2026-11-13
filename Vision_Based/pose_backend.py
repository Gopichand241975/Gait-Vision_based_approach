"""HRNet pose estimator. The rest of the project only calls:

    est = load_estimator()                 # HRNet-W32 (the only pose model)
    est.reset()                            # at the start of every video
    kp = est.estimate(frame_bgr)           # (17, 3) = x, y (pixels), confidence      or None if no person

HRNet backend = person detector (torchvision Faster R-CNN) + HRNet-W32 keypoints inside the box.
To be fast the person box is followed from frame to frame (box of the previous skeleton) and the
detector only runs every config.HRNET_REDETECT_EVERY frames or when the skeleton is lost.

Check everything BEFORE the long extraction (takes under a minute):

    python pose_backend.py --selftest
    python pose_backend.py --selftest --image some_frame.jpg
"""
import argparse
import time

import numpy as np

import config


def load_estimator():
    """The pose estimator. HRNet-W32 is the only pose model of this project."""
    return HRNetPose()


# ----------------------------------------------------------------------------- HRNet
def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class HRNetPose:
    def __init__(self):
        import torch
        from hrnet_pose import load_hrnet
        self.torch = torch
        self.device = config.get_device()
        self.net, self.weights_path = load_hrnet(self.device)
        self.fp16 = bool(config.HRNET_FP16 and self.device.type == "cuda")
        self.det = self._load_detector()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.name = f"HRNet-W32 256x192 ({self.weights_path.name}) + Faster R-CNN person detector"
        self.reset()

    def _load_detector(self):
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
            det = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                          min_size=config.HRNET_DET_MIN_SIZE,
                                          max_size=config.HRNET_DET_MAX_SIZE,
                                          box_score_thresh=0.3)
        except Exception as exc:
            raise SystemExit(
                f"Could not load the torchvision person detector ({exc}).\n"
                f"It needs internet once (about 160 MB, cached in ~/.cache/torch). "
                f"Install torchvision with the same CUDA/CPU build as torch.")
        return det.to(self.device).eval()

    def reset(self):
        self.prev_box = None
        self.since = 0

    # ---- person box
    def _detect(self, frame, prefer=None):
        torch = self.torch
        rgb = np.ascontiguousarray(frame[..., ::-1])
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self.det([t])[0]
        boxes = out["boxes"].float().cpu().numpy()
        keep = (out["labels"].cpu().numpy() == 1) & (out["scores"].float().cpu().numpy() >= config.HRNET_DET_SCORE)
        boxes = boxes[keep]
        if len(boxes) == 0:
            return None
        if prefer is not None:
            ious = [_iou(b, prefer) for b in boxes]
            j = int(np.argmax(ious))
            if ious[j] > 0.2:
                return boxes[j]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[int(np.argmax(areas))]                    # the largest person

    # ---- keypoints inside a box
    def _pose(self, frame, box):
        from hrnet_pose import crop_person, decode_heatmaps, normalise_crop
        torch = self.torch
        img, geom = crop_person(frame, box)
        x = torch.from_numpy(normalise_crop(img)).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            hm = self.net(x)
        return decode_heatmaps(hm[0].float().cpu().numpy(), geom)

    @staticmethod
    def _ok(kp):
        return int((kp[:, 2] > config.HRNET_JOINT_CONF).sum()) >= config.HRNET_MIN_JOINTS

    @staticmethod
    def _box_from_keypoints(kp, shape):
        good = kp[:, 2] > config.HRNET_JOINT_CONF
        x1, y1 = kp[good, 0].min(), kp[good, 1].min()
        x2, y2 = kp[good, 0].max(), kp[good, 1].max()
        w, h = max(x2 - x1, 40.0), max(y2 - y1, 80.0)
        H, W = shape[:2]
        return np.array([max(0.0, x1 - 0.10 * w), max(0.0, y1 - 0.15 * h),
                         min(W - 1.0, x2 + 0.10 * w), min(H - 1.0, y2 + 0.15 * h)], np.float32)

    def estimate(self, frame, track=True):
        need_detect = (not track) or self.prev_box is None or self.since >= config.HRNET_REDETECT_EVERY
        kp = None
        if not need_detect:
            kp = self._pose(frame, self.prev_box)
            self.since += 1
            if not self._ok(kp):                                # box lost the person: detect again
                kp = None
        if kp is None:
            box = self._detect(frame, self.prev_box if track else None)
            self.since = 0
            if box is None:
                self.prev_box = None
                return None
            kp = self._pose(frame, box)
            if not self._ok(kp):
                self.prev_box = None
                return None
        self.prev_box = self._box_from_keypoints(kp, frame.shape)
        return kp.astype(np.float32)


# ----------------------------------------------------------------------------- self-test
def _first_video():
    import csv
    from pose_extract import list_videos
    vids = list_videos()
    if config.DATASET_SPLIT_CSV.exists():
        with open(config.DATASET_SPLIT_CSV, newline="") as f:
            wanted = [r["clip_id"] for r in csv.DictReader(f)]
        vids = [v for v in vids if v.stem in set(wanted)] or vids
    return vids[0] if vids else None


def selftest(image=None):
    import cv2
    from hrnet_pose import selftest_hrnet_math
    err = selftest_hrnet_math()
    print(f"[1/5] crop / heatmap geometry OK (worst error {err:.1f} px)")
    t0 = time.time()
    est = load_estimator()
    print(f"[2/5] weights + detector loaded in {time.time() - t0:.1f} s on {est.device}: {est.name}")
    n_par = sum(p.numel() for p in est.net.parameters()) / 1e6
    note = "OK" if 28.3 <= n_par <= 28.7 else "UNEXPECTED (HRNet-W32 should have about 28.5 M)"
    print(f"[3/5] HRNet parameters: {n_par:.2f} M  {note}")
    x = est.torch.zeros(1, 3, config.HRNET_INPUT_H, config.HRNET_INPUT_W, device=est.device)
    with est.torch.no_grad():
        shape = tuple(est.net(x).shape)
    assert shape == (1, 17, config.HRNET_INPUT_H // 4, config.HRNET_INPUT_W // 4), shape
    print(f"[4/5] forward pass OK, heatmap shape {shape}")

    frames = []
    if image:
        f = cv2.imread(str(image))
        if f is None:
            raise SystemExit(f"cannot read {image}")
        frames = [f]
    else:
        v = _first_video()
        if v is None:
            print("[5/5] no video under dataset/ and no --image given: real-frame test skipped")
            return
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2 - 15))
        for _ in range(31):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        print(f"      test video: {v.name}")
    est.reset()
    kp0 = est.estimate(frames[0], track=False)
    if kp0 is None:
        print("[5/5] NO PERSON FOUND in the test frame. Try --image with a frame that shows a person.")
        return
    t1 = time.time()
    n = 0
    for f in frames[1:]:
        est.estimate(f)
        n += 1
    fps = n / (time.time() - t1) if n else float("nan")
    from check_pose import draw
    out = config.RESULTS_SKELETON_DIR / "hrnet_selftest.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw(frames[0].copy(), kp0))
    print(f"[5/5] real frame OK: {int((kp0[:, 2] > config.HRNET_JOINT_CONF).sum())}/17 joints found, "
          f"mean confidence {kp0[:, 2].mean():.2f}")
    if n:
        print(f"      speed with box tracking: {fps:.1f} frames/s  "
              f"(a 960-frame clip takes about {960 / fps:.0f} s, all 280 clips about {280 * 960 / fps / 3600:.1f} h)")
    print(f"      picture saved: {out}   <- open it and check the joints sit on the body")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--image", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.image)
    else:
        ap.print_help()
