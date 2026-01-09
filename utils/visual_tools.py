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
def video_info(video_file):
    ''' 
    do some EDA, extract basic frame features about the episode
    '''
    if not os.path.exists(video_file):
        raise FileNotFoundError(video_file)

    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        raise RuntimeError("Cannot open video")

    fps = vid.get(cv2.CAP_PROP_FPS)
    w   = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    h   = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n   = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    # fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))

    # ok = False
    # for _ in range(5):
    #     ret, frame = vid.read()
    #     if ret and frame is not None:
    #         ok = True
    #         break

    # if not ok:
    #     raise RuntimeError("Unable to decode frames")

    ret, frame = vid.read()
    if not ret or frame is None:
        raise RuntimeError("Cannot decode frames")

    vid.release()

    return {
        "fps": fps,
        "width": int(w),
        "height": int(h),
        "num_frames": int(n),
        # "fourcc": fourcc,
        "duration_sec": n / fps if fps else None,
    }

#----------------------------------------------
# Extract frames and their metadata:
# -> what are the indices of frames
# -> whare is the intro part etc.
#----------------------------------------------
def extract_frames(video_file, fps_to_save):
    """
    Save frames from video at fps_to_save.
    Returns: output frames dir and number of frames saved.
    """
    vid = cv2.VideoCapture(video_file)
    if not vid.isOpened():
        raise RuntimeError(f"Cannot open video: {video_file}")

    orig_fps = vid.get(cv2.CAP_PROP_FPS)
    frame_step = max(int(orig_fps // fps_to_save), 1)  # save every Nth frame

    base,_ = os.path.splitext(video_file)
    out_dir = base + "-frames"
    os.makedirs(out_dir, exist_ok=True)

    fid = saved = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        if fid % frame_step == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame{saved}.jpg"), frame)
            saved += 1

        fid += 1

    vid.release()
    print(f"[frames ok] saved={saved} frames in {out_dir}")
    return out_dir, saved

def extract_frames_meta(video_file, intro_timestamp=0, fps_to_save=25):
    """
    Returns: list of (saved_index, filename), frames_dir, video_info
    """
    intro_skip_sec = parse_timestamp(intro_timestamp) if isinstance(intro_timestamp, str) else intro_timestamp

    info = video_info(video_file)
    frames_dir = os.path.splitext(video_file)[0] + "-frames"
    os.makedirs(frames_dir, exist_ok=True)

    frames = []
    for f in os.listdir(frames_dir):
        if f.lower().endswith(".jpg"):
            num = int(f[5:-4])
            frames.append((num, f))

    frames.sort(key=lambda x: x[0])

    # skip intro frames if FPS provided
    if fps_to_save:
        discard_count = int(intro_skip_sec * fps_to_save)
        discard_count = min(discard_count, len(frames))
        frames = frames[discard_count:]

    return frames, frames_dir, info



#----------------------------------------------
# Split frames for modeling
#----------------------------------------------
def split_frames(frames, video_file, split_timestamp, fps_to_save):
    """
    Split frames into train/test based on split_timestamp.
    Uses the sorted frames list; everything before the split → train,
    everything after → test. Does not compute FPS-based indices.
    """
    split_sec = parse_timestamp(split_timestamp) if isinstance(split_timestamp, str) else split_timestamp

    # Compute the frame index corresponding to the split time
    split_index = int(split_sec * fps_to_save)
    split_index = min(split_index, len(frames))  # avoid overflow

    train_frames = frames[:split_index]
    test_frames  = frames[split_index:]

    episode_name = os.path.splitext(os.path.basename(video_file))[0]
    frames_dir = os.path.splitext(video_file)[0] + "-frames"

    train_out = f"../data/processed/video/{episode_name}/train/"
    test_out  = f"../data/processed/video/{episode_name}/test/"
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    for _, fname in train_frames:
        shutil.copy(os.path.join(frames_dir, fname), os.path.join(train_out, fname))
    for _, fname in test_frames:
        shutil.copy(os.path.join(frames_dir, fname), os.path.join(test_out, fname))

    print(f"[YoHoo! split done :>] saved frames: train={len(train_frames)}, test={len(test_frames)}")
    return train_frames, test_frames
    

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


def feat_green_mask(img_hsv):
    mask = (
        (img_hsv[:, :, 0] >= 35) & (img_hsv[:, :, 0] <= 85) &
        (img_hsv[:, :, 1] > 50) & (img_hsv[:, :, 2] > 50)
    )
    return {"green_frac": mask.sum() / mask.size}


def feat_edge_magnitude(img_gray):
    edges = cv2.Canny(img_gray, 100, 200)
    return {"edge_mean": edges.mean()}


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


# FEATURES DICTIONARY
VISUAL_FEATURES = {
    "dominant_color": lambda frame_data: feat_dominant_color(frame_data["frame_path"]),
    "green_mask":     lambda frame_data: feat_green_mask(frame_data["hsv"]),
    "edge_magnitude": lambda frame_data: feat_edge_magnitude(frame_data["gray"]),
    "eye_apttern": lambda frame_data: feat_eye_pattern(frame_data["hsv"])
}




def extract_visual_features_for_frame(frame_path, features):
    """
    Extract selected visual features for a single frame.

    features: list of feature names to compute
              e.g. ["dominant_color", "green_mask", "edge_magnitude"]

    Returns: dict {feature_name: value}
    """
    fname = os.path.basename(frame_path)

    img = cv2.imread(frame_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame_data = {
        "frame_path": frame_path,
        "hsv": hsv,
        "gray": gray,
    }

    out = {"frame": fname}

    for feat in features:
        if feat not in VISUAL_FEATURES:
            raise ValueError(f"Unknown feature: {feat}")
        out.update(VISUAL_FEATURES[feat](frame_data))

    return out
