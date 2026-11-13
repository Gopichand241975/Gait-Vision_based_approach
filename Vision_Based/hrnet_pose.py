"""HRNet-W32 (256x192, COCO 17 keypoints) top-down pose network.

This is the architecture of the official implementation (Sun et al., CVPR 2019, "Deep High-Resolution
Representation Learning for Human Pose Estimation"), written here so that no OpenMMLab / mmcv install
is needed. Pre-trained COCO weights are loaded from a normal PyTorch checkpoint:

    weights/hrnet_w32_coco_256x192.pth      (downloaded automatically, see config.HRNET_WEIGHTS_URL)
    or the official file  pose_hrnet_w32_256x192.pth  placed in weights/ (config.HRNET_WEIGHTS_OFFICIAL)

HRNet is a TOP-DOWN model: it needs a person box first (pose_backend.py does that), crops the person to
256 x 192 pixels and predicts 17 heatmaps of 64 x 48. The numpy helpers below (crop, decode) do not need
torch and are unit-tested in selftest_hrnet_math() at the bottom.
"""
import numpy as np

import config

NUM_JOINTS = 17
INPUT_W, INPUT_H = config.HRNET_INPUT_W, config.HRNET_INPUT_H        # 192, 256
HEATMAP_W, HEATMAP_H = INPUT_W // 4, INPUT_H // 4                    # 48, 64
BOX_PADDING = 1.25                      # the crop is 25% larger than the person box (as in the official code)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------------------------------------------------------- numpy helpers (no torch)
def box_to_crop(box):
    """(x1, y1, x2, y2) person box -> crop centre and size (cx, cy, crop_w, crop_h) with the network's
    3:4 aspect ratio (192:256), enlarged by BOX_PADDING."""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    aspect = INPUT_W / INPUT_H
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, w * BOX_PADDING, h * BOX_PADDING


def crop_person(frame_bgr, box):
    """Crop + resize the person to (INPUT_H, INPUT_W) RGB float32 in [0,1]; also return the crop geometry."""
    import cv2
    cx, cy, cw, ch = box_to_crop(box)
    sx, sy = INPUT_W / cw, INPUT_H / ch
    m = np.array([[sx, 0.0, INPUT_W / 2.0 - sx * cx],
                  [0.0, sy, INPUT_H / 2.0 - sy * cy]], np.float32)
    img = cv2.warpAffine(frame_bgr, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return img[..., ::-1].astype(np.float32) / 255.0, (cx, cy, cw, ch)


def normalise_crop(img_rgb01):
    """(H, W, 3) RGB [0,1] -> (3, H, W) ImageNet-normalised float32."""
    return np.ascontiguousarray(((img_rgb01 - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)


def decode_heatmaps(hm, geom):
    """hm: (17, 64, 48) heatmaps of ONE person; geom: (cx, cy, crop_w, crop_h).
    Returns (17, 3) = x, y in ORIGINAL frame pixels and confidence (heatmap peak value).
    Peak = argmax, shifted a quarter pixel towards the higher neighbour (official post-processing)."""
    k, hh, ww = hm.shape
    flat = hm.reshape(k, -1)
    idx = flat.argmax(1)
    conf = flat.max(1)
    px = (idx % ww).astype(np.float32)
    py = (idx // ww).astype(np.float32)
    for j in range(k):
        x, y = int(px[j]), int(py[j])
        if 1 < x < ww - 1 and 1 < y < hh - 1:
            px[j] += 0.25 * np.sign(hm[j, y, x + 1] - hm[j, y, x - 1])
            py[j] += 0.25 * np.sign(hm[j, y + 1, x] - hm[j, y - 1, x])
    cx, cy, cw, ch = geom
    out = np.zeros((k, 3), np.float32)
    out[:, 0] = cx + (px - ww / 2.0) * (cw / ww)
    out[:, 1] = cy + (py - hh / 2.0) * (ch / hh)
    out[:, 2] = conf
    return out


# ----------------------------------------------------------------------------- the network
def build_hrnet_w32():
    """Returns an nn.Module: (N, 3, 256, 192) -> (N, 17, 64, 48) heatmaps. Imports torch only when called."""
    import torch.nn as nn

    bn_momentum = 0.1

    def conv3x3(i, o, stride=1):
        return nn.Conv2d(i, o, 3, stride, 1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                residual = self.downsample(x)
            return self.relu(out + residual)

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
            self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4, momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            if self.downsample is not None:
                residual = self.downsample(x)
            return self.relu(out + residual)

    class HighResolutionModule(nn.Module):
        def __init__(self, num_branches, num_blocks, num_inchannels, num_channels, multi_scale_output=True):
            super().__init__()
            self.num_branches = num_branches
            self.num_inchannels = list(num_inchannels)
            self.multi_scale_output = multi_scale_output
            self.branches = nn.ModuleList(
                self._make_one_branch(b, num_blocks, num_channels) for b in range(num_branches))
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU(True)

        def _make_one_branch(self, b, num_blocks, num_channels):
            layers = [BasicBlock(self.num_inchannels[b], num_channels[b])]
            self.num_inchannels[b] = num_channels[b]
            for _ in range(1, num_blocks[b]):
                layers.append(BasicBlock(self.num_inchannels[b], num_channels[b]))
            return nn.Sequential(*layers)

        def _make_fuse_layers(self):
            n, inch = self.num_branches, self.num_inchannels
            fuse_layers = []
            for i in range(n if self.multi_scale_output else 1):
                fuse = []
                for j in range(n):
                    if j > i:
                        fuse.append(nn.Sequential(
                            nn.Conv2d(inch[j], inch[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(inch[i], momentum=bn_momentum),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest")))
                    elif j == i:
                        fuse.append(None)
                    else:
                        convs = []
                        for k in range(i - j):
                            if k == i - j - 1:
                                convs.append(nn.Sequential(
                                    nn.Conv2d(inch[j], inch[i], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(inch[i], momentum=bn_momentum)))
                            else:
                                convs.append(nn.Sequential(
                                    nn.Conv2d(inch[j], inch[j], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(inch[j], momentum=bn_momentum),
                                    nn.ReLU(True)))
                        fuse.append(nn.Sequential(*convs))
                fuse_layers.append(nn.ModuleList(fuse))
            return nn.ModuleList(fuse_layers)

        def forward(self, x):
            x = [self.branches[i](x[i]) for i in range(self.num_branches)]
            out = []
            for i, fuse in enumerate(self.fuse_layers):
                y = x[0] if i == 0 else fuse[0](x[0])
                for j in range(1, self.num_branches):
                    y = y + (x[j] if i == j else fuse[j](x[j]))
                out.append(self.relu(y))
            return out

    class PoseHighResolutionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
            self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(64, 4)

            self.transition1 = self._make_transition([256], [32, 64])
            self.stage2, pre = self._make_stage(1, 2, [4, 4], [32, 64], [32, 64])
            self.transition2 = self._make_transition(pre, [32, 64, 128])
            self.stage3, pre = self._make_stage(4, 3, [4, 4, 4], [32, 64, 128], [32, 64, 128])
            self.transition3 = self._make_transition(pre, [32, 64, 128, 256])
            self.stage4, pre = self._make_stage(3, 4, [4, 4, 4, 4], [32, 64, 128, 256], [32, 64, 128, 256],
                                                multi_scale_output=False)
            self.final_layer = nn.Conv2d(pre[0], NUM_JOINTS, 1, 1, 0)

        def _make_layer(self, planes, blocks):
            downsample = None
            if self.inplanes != planes * Bottleneck.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, 1, bias=False),
                    nn.BatchNorm2d(planes * Bottleneck.expansion, momentum=bn_momentum))
            layers = [Bottleneck(self.inplanes, planes, 1, downsample)]
            self.inplanes = planes * Bottleneck.expansion
            layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
            return nn.Sequential(*layers)

        @staticmethod
        def _make_transition(pre, cur):
            layers = []
            for i in range(len(cur)):
                if i < len(pre):
                    if cur[i] != pre[i]:
                        layers.append(nn.Sequential(
                            nn.Conv2d(pre[i], cur[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(cur[i], momentum=bn_momentum),
                            nn.ReLU(True)))
                    else:
                        layers.append(None)
                else:
                    convs = []
                    for j in range(i + 1 - len(pre)):
                        inch = pre[-1]
                        outch = cur[i] if j == i - len(pre) else inch
                        convs.append(nn.Sequential(
                            nn.Conv2d(inch, outch, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outch, momentum=bn_momentum),
                            nn.ReLU(True)))
                    layers.append(nn.Sequential(*convs))
            return nn.ModuleList(layers)

        @staticmethod
        def _make_stage(num_modules, num_branches, num_blocks, num_channels, num_inchannels,
                        multi_scale_output=True):
            modules = []
            for i in range(num_modules):
                multi = not (not multi_scale_output and i == num_modules - 1)
                m = HighResolutionModule(num_branches, num_blocks, num_inchannels, num_channels, multi)
                modules.append(m)
                num_inchannels = m.num_inchannels
            return nn.Sequential(*modules), num_inchannels

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.layer1(x)
            xs = [(t(x) if t is not None else x) for t in self.transition1]
            ys = self.stage2(xs)
            xs = [(t(ys[-1]) if t is not None else ys[i]) for i, t in enumerate(self.transition2)]
            ys = self.stage3(xs)
            xs = [(t(ys[-1]) if t is not None else ys[i]) for i, t in enumerate(self.transition3)]
            ys = self.stage4(xs)
            return self.final_layer(ys[0])

    return PoseHighResolutionNet()


# ----------------------------------------------------------------------------- weights
def _download(url, dest):
    import shutil
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[info] downloading HRNet weights (about 110 MB) from\n       {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def find_weights(download=True):
    """Path of the HRNet checkpoint: official file or OpenMMLab file in weights/, downloaded if missing."""
    for p in (config.HRNET_WEIGHTS, config.HRNET_WEIGHTS_OFFICIAL):
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    if not download:
        return None
    try:
        _download(config.HRNET_WEIGHTS_URL, config.HRNET_WEIGHTS)
    except Exception as exc:                                    # no internet, blocked, ...
        raise SystemExit(
            f"Could not download the HRNet weights ({exc}).\n"
            f"Download this file in a browser and save it as {config.HRNET_WEIGHTS}:\n  {config.HRNET_WEIGHTS_URL}\n"
            f"(or put the official pose_hrnet_w32_256x192.pth in {config.HRNET_WEIGHTS.parent}).")
    return config.HRNET_WEIGHTS


def load_hrnet(device, weights=None):
    """Build HRNet-W32, load COCO weights (OpenMMLab or official format), return the model in eval mode."""
    import torch
    net = build_hrnet_w32()
    path = weights or find_weights()
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    clean = {}
    for k, v in sd.items():
        for prefix in ("module.", "backbone.", "keypoint_head."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        clean[k] = v
    missing, unexpected = net.load_state_dict(clean, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing:
        raise SystemExit(
            f"HRNet weights {path.name} do not match the network: {len(missing)} missing keys "
            f"(first: {missing[:3]}), {len(unexpected)} unexpected (first: {list(unexpected)[:3]}). "
            f"Use pose_hrnet_w32_256x192.pth from the official HRNet model zoo (see weights/README.txt).")
    if unexpected:
        print(f"[info] ignored {len(unexpected)} unused checkpoint keys (e.g. {list(unexpected)[:2]})")
    return net.to(device).eval(), path


# ----------------------------------------------------------------------------- tests (no torch needed)
def selftest_hrnet_math():
    """Checks the crop / decode geometry with a synthetic person: a known joint position must come back
    at the same pixel after crop -> 'perfect heatmap' -> decode."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    box = (800.0, 200.0, 1000.0, 800.0)                     # 200 x 600 px person
    img, geom = crop_person(frame, box)
    assert img.shape == (INPUT_H, INPUT_W, 3), img.shape
    cx, cy, cw, ch = geom
    assert abs(cw / ch - INPUT_W / INPUT_H) < 1e-6
    worst = 0.0
    for (x, y) in [(900.0, 250.0), (850.0, 500.0), (960.0, 790.0), (900.0, 400.0)]:
        # position of this pixel inside the 64 x 48 heatmap (inverse of decode_heatmaps)
        hx = (x - cx) / (cw / HEATMAP_W) + HEATMAP_W / 2.0
        hy = (y - cy) / (ch / HEATMAP_H) + HEATMAP_H / 2.0
        hm = np.zeros((NUM_JOINTS, HEATMAP_H, HEATMAP_W), np.float32)
        ix, iy = int(round(hx)), int(round(hy))
        hm[0, iy, ix] = 1.0
        hm[0, iy, ix + 1] = 0.4 if hx > ix else 0.0           # make the quarter-pixel shift happen
        hm[0, iy, ix - 1] = 0.4 if hx < ix else 0.0
        out = decode_heatmaps(hm, geom)
        worst = max(worst, abs(out[0, 0] - x), abs(out[0, 1] - y))
        assert out[0, 2] == 1.0
    cell = cw / HEATMAP_W
    assert worst <= cell, f"decode error {worst:.1f}px is larger than one heatmap cell ({cell:.1f}px)"
    return worst


if __name__ == "__main__":
    err = selftest_hrnet_math()
    print(f"crop/decode geometry OK (worst error {err:.2f} px, heatmap cell = "
          f"{box_to_crop((800, 200, 1000, 800))[2] / HEATMAP_W:.1f} px)")
