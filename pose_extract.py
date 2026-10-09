"""Extract 17 COCO keypoints per frame from every video in dataset/<person_id>/.

Output: skeletons_raw/<person_id>/<video_name>.npy with shape (T, 17, 3)
        columns = x (pixels), y (pixels), confidence
"""
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import config