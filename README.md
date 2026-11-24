Vision-Based Gait Recognition Reinforced by WiFi Sensing

Project Overview

This project focuses on human gait recognition by combining camera-based vision and WiFi sensing approaches.

The vision-based approach extracts human body keypoints from video, while the WiFi-based approach uses Channel State Information (CSI) to estimate human body movement.

The proposed system follows a Vision-Primary, WiFi-Assisted Fusion Approach, where vision is the primary source for gait recognition and WiFi sensing acts as a supporting modality.

The main objective is to improve gait recognition reliability in challenging environments, particularly under poor lighting, partial occlusion, and situations where certain body joints are not clearly visible to the camera.

Project Structure

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

1. Vision-Based Approach

Uses camera images and videos for human pose estimation.

Extracts human body joint keypoints using pose estimation techniques.

Processes skeleton sequences for gait recognition.

Uses confidence scores to identify the reliability of detected joints.

Trains and evaluates the vision-based gait recognition model.

Technologies: Python, OpenCV, PyTorch, HRNet and Deep Learning.

2. WiFi-Based Approach

Uses WiFi Channel State Information (CSI) to capture changes in wireless signals caused by human movement.

Processes CSI signals to understand human movement patterns.

Estimates human body pose and joint positions.

Trains and evaluates a WiFi-based model for pose estimation.

Provides complementary joint information when visual observations are unreliable.

Technologies: Python, NumPy, PyTorch and WiFi CSI.

3. Proposed Vision-WiFi Fusion Approach

The proposed fusion system uses vision as the primary modality and WiFi sensing as a supporting modality.

Instead of treating both modalities equally, the system prioritizes visual keypoints whenever they are reliable and uses WiFi-estimated information to support or correct unreliable visual observations.

How Fusion Works

Step 1: Vision-Based Pose Estimation

The camera captures a person's walking sequence.

A pose estimation model extracts body keypoints, such as shoulders, elbows, hips, knees and ankles.

Each detected joint is assigned a confidence score indicating how reliably it was detected.

Step 2: WiFi-Based Pose Estimation

WiFi CSI signals are collected during the same walking sequence.

The CSI processing pipeline extracts movement-related information.

The WiFi-based model estimates corresponding body joint positions.

Step 3: Confidence-Based Joint Fusion

The system compares the confidence of vision-based joint estimations with the reliability of WiFi-based joint estimations.

If a visual joint has high confidence, the original vision-based joint is retained.

If a visual joint has low confidence due to occlusion or poor lighting, the corresponding WiFi-estimated joint can be used to support or replace it.

If both modalities provide reliable estimates, their information can be combined using confidence-based weighting.

If neither modality provides reliable information, the joint is marked as uncertain rather than forcing an unreliable replacement.

For example:

Body Joint

Vision Confidence

WiFi Reliability

Fusion Decision

Left Shoulder

High

Medium

Use Vision

Right Elbow

Low

High

Use WiFi

Left Knee

Low

High

Use WiFi

Right Ankle

High

Medium

Use Vision

Hip

Medium

High

Weighted Fusion

Step 4: Fused Skeleton Generation

The selected and combined joint positions form an improved skeleton sequence.

The fused skeleton preserves reliable visual information while using WiFi information to compensate for missing or unreliable visual joints.

The resulting skeleton sequence is passed to the gait recognition model.

Step 5: Gait Recognition

The fused skeleton sequence is processed by the gait recognition model.

The system extracts gait-related features from the walking sequence.

The final objective is to improve recognition accuracy, particularly in occlusion and poor-lighting scenarios.

Fusion Logic

Let:

(V_j) = Vision-estimated position of joint (j).

(W_j) = WiFi-estimated position of joint (j).

(C_j^V) = Vision confidence for joint (j).

(C_j^W) = WiFi reliability for joint (j).

A confidence-based fusion strategy can be represented as:

[
F_j =
\begin{cases}
V_j, & C_j^V \geq T_V \
W_j, & C_j^V < T_V \text{ and } C_j^W \geq T_W \
\alpha V_j + (1-\alpha)W_j, & \text{Both reliable}
\end{cases}
]

Here:

(F_j) represents the fused joint position.

(T_V) and (T_W) are reliability thresholds.

(\alpha) controls the contribution of each modality during weighted fusion.

The fusion process is performed for individual joints rather than replacing the entire visual skeleton.

Note: This represents the proposed fusion strategy. The fusion module and its performance evaluation are planned future work.
