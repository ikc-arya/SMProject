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

# def feat_dominant_color(frame_path):
#     rgb = ColorThief(frame_path).get_color(quality=1)
#     h, s, v = rgb_to_hsv(rgb)
#     return {"dom_H": h, "dom_S": s, "dom_V": v}

def dominant_color(frame_info, dry_run=False):
    if dry_run:
        return {"dom_H": None, "dom_S": None, "dom_V": None}
        
    img = frame_info["img"]
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    mean_bgr = small.mean(axis=(0, 1))
    mean_rgb = mean_bgr[::-1]
    h, s, v = rgb_to_hsv(mean_rgb)
    return {"dom_H": h, "dom_S": s, "dom_V": v}


def green_fraction(frame_info, dry_run=False):
    if dry_run:
        return {"green_frac": None}
    hsv = frame_info["hsv"]
    mask = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
        (hsv[:, :, 1] > 50) &
        (hsv[:, :, 2] > 50)
    )
    return {"green_frac": mask.mean()}


def edge_mean(frame_info, dry_run=False):
    if dry_run:
        return {"edge_mean": None}
    edges = cv2.Canny(frame_info["gray"], 100, 200)
    return {"edge_mean": edges.mean()}


def frog_eye_pattern(frame_info, dry_run=False):
    if dry_run:
        return {
            "eye_blob_count": None,
            "eye_horizontal_align": None
        }

    hsv = frame_info["hsv"]

    # green region
    green_mask = (
        (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) &
        (hsv[:, :, 1] > 50)
    )

    # white regions inside green (eyes)
    v = hsv[:, :, 2]
    white = (v > 200) & green_mask
    white = white.astype("uint8") * 255

    contours, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []
    centers = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:
            continue
        valid.append(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

    # horizontal alignment score (Kermit eyes)
    horiz_align = 0.0
    if len(centers) >= 2:
        ys = [c[1] for c in centers]
        horiz_align = 1 / (1 + np.std(ys))

    return {
        "eye_blob_count": len(valid),
        "eye_horizontal_align": horiz_align
    }


def brown_rhythm(frame_info, dry_run=False, patch_size=16, max_lag=10):
    """
    Rhythm detection of brown patches in the frame.
    Returns mean autocorrelation across lags.
    """
    if dry_run:
        return {"brown_rhythm": None}

    hsv = frame_info["hsv"]

    # Brown mask
    mask = (
        (hsv[:, :, 0] >= 10) & (hsv[:, :, 0] <= 25) &
        (hsv[:, :, 1] > 50) &
        (hsv[:, :, 2] > 50)
    ).astype(np.uint8)

    h, w = mask.shape
    n_h = h // patch_size
    n_w = w // patch_size

    # Sum brown pixels per patch (row-wise)
    patch_sums = []
    for i in range(n_h):
        for j in range(n_w):
            patch = mask[i*patch_size:(i+1)*patch_size,
                         j*patch_size:(j+1)*patch_size]
            patch_sums.append(patch.sum())

    patch_sums = np.array(patch_sums)
    patch_sums = patch_sums - patch_sums.mean()

    # Autocorrelation for lags 1..max_lag
    acorrs = []
    for k in range(1, min(max_lag, len(patch_sums))):
        ac = np.mean(patch_sums[:-k] * patch_sums[k:])
        acorrs.append(ac)

    rhythm_score = np.mean(acorrs) if acorrs else 0.0
    return {"brown_rhythm": rhythm_score}

VISUAL_FEATURES = {
    "dominant_color": dominant_color,
    "green_mask": green_fraction,
    "edge_magnitude": edge_mean,
    "frog_eye": frog_eye_pattern,
    "brown_rhythm": brown_rhythm,
}

#-------*******---------
# Note fot @ibembem:
# Update the VISUAL_FEATURE dictionary after adding new mwthods.
# Do not change the extract_visual_features_for_frame() function.
#-------*******---------


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





# to execute all feature functions and create feature space

def extract_visual_features_for_frame(frame_path, feature_list):
    """
    Extract selected visual features for a single frame.
    Returns: dict {feature_name: value}
    """
    fname = os.path.basename(frame_path)
    out = {"frame": fname}

    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Missing frame: {frame_path}")

    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"Could not read frame: {frame_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_data = {"img": img, "hsv": hsv, "gray": gray}

    for feat_name in feature_list:
        if feat_name not in VISUAL_FEATURES:
            raise ValueError(f"Unknown feature: {feat_name}")
        feat_vals = VISUAL_FEATURES[feat_name](frame_data)
        out.update(feat_vals)

    return out
