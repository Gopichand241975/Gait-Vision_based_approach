# Vision-Based Gait Recognition Reinforced by WiFi Sensing

## Project Overview

This project focuses on human gait recognition by combining camera-based vision and WiFi sensing approaches.

The vision-based approach extracts human body keypoints from video, while the WiFi-based approach uses Channel State Information (CSI) to estimate human body movement.

The proposed system aims to use WiFi sensing to support vision-based gait recognition, especially under poor lighting and occlusion conditions.

## Project Structure

```text
Gait-Vision_based_approach/
│
├── Vision_Based/
│   ├── manifests/
│   ├── weights/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dataset.py
│   ├── model.py
│   ├── pose_extract.py
│   ├── requirements.txt
│   └── vision_based_gait_report.pdf
│
├── WiFi/
│   ├── build_clip_index.py
│   ├── csi_loader.py
│   ├── extract_pose.py
│   ├── inspect_csi.py
│   ├── train2.py
│   └── README.md
│
└── README.md
```

## 1. Vision-Based Approach

- Uses camera images and videos for human pose estimation.
- Extracts body joint keypoints.
- Processes skeleton sequences for gait recognition.
- Trains and evaluates the vision-based gait recognition model.

**Technologies:** Python, OpenCV, PyTorch, HRNet and Deep Learning.

## 2. WiFi-Based Approach

- Uses WiFi Channel State Information (CSI).
- Processes CSI signals to understand human movement.
- Estimates body joint positions.
- Trains a WiFi-based model for pose estimation.

**Technologies:** Python, NumPy, PyTorch and WiFi CSI.

## 3. Vision-WiFi Fusion Approach

The proposed fusion approach combines vision-based gait recognition with WiFi-based pose estimation. Vision remains the primary modality, while WiFi acts as a supporting modality to improve the reliability of body joint estimation.

The main idea is to use vision whenever the detected body joints are reliable and utilize WiFi estimates when certain joints are affected by occlusion, poor lighting or other visual limitations.

### How Fusion Works

**Step 1: Vision-Based Pose Estimation**

- The camera captures a person's walking sequence.
- HRNet extracts human body keypoints, such as shoulders, elbows, hips, knees and ankles.
- Each detected joint is associated with a confidence score that indicates its detection reliability.

**Step 2: WiFi-Based Pose Estimation**

- WiFi CSI signals are collected during the same walking activity.
- CSI processing extracts information related to human movement.
- The WiFi-based model estimates corresponding body joint positions.

**Step 3: Confidence-Based Joint Selection**

The system compares the reliability of the vision and WiFi estimates for individual body joints.

- If a joint is clearly detected by the camera with high confidence, the vision-based joint position is retained.
- If a joint is occluded or has low visual confidence, the corresponding WiFi-estimated joint position is used to replace or improve that joint.
- If both modalities provide reliable estimates, their information can be combined using confidence-based weighting.
- If both estimates are unreliable, the joint is marked as uncertain rather than using an unreliable position.

**Example:**

| Body Joint | Vision Reliability | WiFi Reliability | Selected Information |
|---|---|---|---|
| Left Shoulder | High | Medium | Vision |
| Right Elbow | Low | High | WiFi |
| Left Knee | Low | High | WiFi |
| Right Ankle | High | Medium | Vision |
| Hip | Medium | High | Weighted Fusion |

**Step 4: Fused Skeleton Generation**

- Reliable visual joints are preserved.
- Occluded or unreliable visual joints are replaced or refined using WiFi-based estimates.
- The selected joint positions are combined to generate an improved skeleton sequence.
- This fused skeleton provides more complete body movement information for gait recognition.

**Step 5: Gait Recognition**

- The fused skeleton sequence is passed to the gait recognition model.
- The model extracts gait-related features from the improved skeleton sequence.
- The objective is to improve recognition accuracy, particularly in challenging occlusion and poor-lighting conditions.

### Fusion Formula

Let:

- \(V_j\) = Vision-estimated position of joint \(j\).
- \(W_j\) = WiFi-estimated position of joint \(j\).
- \(C_j^V\) = Vision confidence for joint \(j\).
- \(C_j^W\) = WiFi reliability for joint \(j\).

The proposed confidence-based fusion can be represented as:

\[
F_j =
\begin{cases}
V_j, & C_j^V \geq T_V \\
W_j, & C_j^V < T_V \text{ and } C_j^W \geq T_W \\
\alpha V_j+(1-\alpha)W_j, & \text{Both reliable}
\end{cases}
\]

Where:

- \(F_j\) represents the final fused joint position.
- \(T_V\) and \(T_W\) are reliability thresholds.
- \(\alpha\) controls the contribution of vision and WiFi estimates.

The fusion is performed at the individual joint level instead of replacing the entire visual skeleton.

Before fusion, vision and WiFi joint estimates must be temporally synchronized and mapped into a compatible coordinate system.

## Project Objective

To explore how camera-based vision and WiFi sensing can work together to improve gait recognition, particularly in challenging environments.

The proposed multimodal approach aims to use reliable visual information as the primary source and WiFi-based joint estimates to compensate for unreliable visual observations, improving robustness under occlusion and poor lighting.

## Current Status

- Vision-based gait recognition pipeline implemented and evaluated.
- WiFi CSI processing and model training implemented and evaluated.
- Multimodal fusion of vision and WiFi is planned as future work.

## Future Work

- Integrate vision and WiFi-based pose information.
- Develop confidence-based joint-level fusion.
- Use WiFi-estimated joint positions to replace or refine occluded visual joints.
- Synchronize camera and WiFi measurements.
- Improve joint estimation under occlusion and poor lighting.
- Evaluate the system with more participants and different environments.
- Compare vision-only, WiFi-only and fused-model performance to measure improvements in gait recognition accuracy.

## Repository

This repository contains the separate implementation of the Vision-Based and WiFi-Based approaches.
