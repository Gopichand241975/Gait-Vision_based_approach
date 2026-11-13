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

## Project Objective

To explore how camera-based vision and WiFi sensing can work together to improve gait recognition, particularly in challenging environments.

## Current Status

- Vision-based gait recognition pipeline implemented and evaluated.
- WiFi CSI processing and model training implemented and evaluated.
- Multimodal fusion of vision and WiFi is planned as future work.

## Future Work

- Integrate vision and WiFi-based pose information.
- Improve joint estimation under occlusion and poor lighting.
- Evaluate the system with more participants and different environments.

## Repository

This repository contains the separate implementation of the Vision-Based and WiFi-Based approaches.
