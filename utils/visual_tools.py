import os
import shutil

import cv2
from colorthief import ColorThief
import colorsys
import csv
import pandas as pd
import numpy as np

def parse_timestamp(ts):
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h = 0
        m, s = parts
    else:
        raise ValueError(f"Invalid timestamp format: {ts}")
    
    return h * 3600 + m * 60 + s

#----------------------------------------------
# Load episode and do EDA
#----------------------------------------------
def video_info(video_path):
    """
    Return basic video metadata.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else None

    vid.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration": duration
    }


#----------------------------------------------
# Extract frames from video
#----------------------------------------------
def extract_frames(video_file, episode_name, fps_to_save=25):
    """
    Extract frames and store them in:
    ../data/processed/video/{episode_name}-frames/

    Returns:
        frames_dir, num_saved, video_info_dict
    """
    if fps_to_save <= 0:
        raise ValueError("fps_to_save must be > 0")

    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")

    orig_fps = vid.get(cv2.CAP_PROP_FPS)
    frame_step = max(int(orig_fps // fps_to_save), 1)

    frames_dir = f"../data/processed/video/{episode_name}-frames"
    os.makedirs(frames_dir, exist_ok=True)

    fid = saved = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        if fid % frame_step == 0:
            cv2.imwrite(
                os.path.join(frames_dir, f"frame{saved}.jpg"),
                frame
            )
            saved += 1

        fid += 1

    vid.release() 

    info = video_info(video_file)

    print(f"[frames ok] saved={saved} frames in {frames_dir}")
    return frames_dir, saved, info
    

#----------------------------------------------
# Ssanity check for no of frames with gt dimensions
#----------------------------------------------

def sanity_check_frames_vs_gt(frame_dir, gt_path):
    """
    Check that number of extracted frames == number of GT rows in xlsx
    """
    frame_files = [
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(".jpg")
    ]
    n_frames = len(frame_files)

    gt_df = pd.read_excel(gt_path)
    n_gt = len(gt_df)-1

    if n_frames != n_gt:
        print(f"[WARNING] Frame count mismatch!")
        print(f"Extracted frames: {n_frames}, GT rows: {n_gt}")
    else:
        print(f"[OK] Frames match GT ({n_gt} rows)")



#---------------*******************--------------------
# VISUAL FEATURE EXTRACTION - fir SIM1
#---------------*******************--------------------
def rgb_to_hsv(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return int(h*179), int(s*255), int(v*255)

def feat_dominant_color(frame_path):
    rgb = ColorThief(frame_path).get_color(quality=1)
    h, s, v = rgb_to_hsv(rgb)
    return {"dom_H": h, "dom_S": s, "dom_V": v}

def dominant_color_fast(frame_info):
    img = frame_info["img"]
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    mean_bgr = small.mean(axis=(0, 1))
    mean_rgb = mean_bgr[::-1]
    h, s, v = rgb_to_hsv(mean_rgb)
    return {"dom_H": h, "dom_S": s, "dom_V": v}


def green_fraction(frame_info):
    hsv = frame_info["hsv"]
    mask = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
        (hsv[:, :, 1] > 50) &
        (hsv[:, :, 2] > 50)
    )
    return {"green_frac": mask.mean()}


def edge_mean(frame_info):
    edges = cv2.Canny(frame_info["gray"], 100, 200)
    return {"edge_mean": edges.mean()}


VISUAL_FEATURES = {
    "dominant_color": dominant_color_fast,
    "green_mask": green_fraction,
    "edge_magnitude": edge_mean,
}


def feat_eye_pattern(img_hsv):
    # restrict to green area
    green_mask = (
        (img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 85) &
        (img_hsv[:, :, 1] > 50)
    )

    v = img_hsv[:, :, 2]

    # white blobs inside green
    white = (v > 200) & green_mask
    white = white.astype("uint8") * 255

    contours, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    blob_count = len(contours)
    centers = []

    for c in contours:
        if cv2.contourArea(c) < 20:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centers.append((cx, cy))

    horiz_align = 0.0
    if len(centers) >= 2:
        ys = [c[1] for c in centers]
        horiz_align = 1 / (1 + np.std(ys))  # higher = better alignment

    return {
        "eye_blob_count": blob_count,
        "eye_horizontal_align": horiz_align,
    }

#---------------*******************--------------------
# VISUAL FEATURE EXTRACTION - for SIM2
#---------------*******************--------------------

def feat_optical_flow(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    return {
        "flow_mag_mean": mag.mean(),
        "flow_mag_std": mag.std(),
        "flow_horiz_ratio": (np.abs(flow[...,0]).mean() /
                             (np.abs(flow[...,1]).mean() + 1e-6))
    }

def extract_visual_features_for_frame(frame_path, feature_list):
    """
    Extract selected visual features for a single frame.
    feature_list: list of feature names (e.g. SIM1_VISUAL_FEATURES)
    Returns: dict with frame name + feature values
    """
    fname = os.path.basename(frame_path)

    img = cv2.imread(frame_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {frame_path}")

    frame_info = {
        "img": img,
        "hsv": cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
        "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    }

    out = {"frame": fname}

    for feat in feature_list:
        out.update(VISUAL_FEATURES[feat](frame_info))

    return out
