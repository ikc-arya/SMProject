import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import time, datetime

from utils import visual_tools as visualTools


def format_gt_timestamp(ts):
    """
    Convert timestamp to 'm:s.ms' with 2 decimals for ms.
    Accepts float, datetime, or string like '0:00:00.040000' and returns '0:00.04'
    """
    # Convert to float seconds
    if isinstance(ts, str):
        seconds = visualTools.parse_timestamp(ts)
    elif isinstance(ts, (float, int, np.float64, np.int64)):
        seconds = float(ts)
    elif isinstance(ts, time):
        # Convert time object to seconds
        seconds = ts.hour*3600 + ts.minute*60 + ts.second + ts.microsecond/1e6
    elif isinstance(ts, datetime):
        seconds = ts.hour*3600 + ts.minute*60 + ts.second + ts.microsecond/1e6
    else:
        raise ValueError(f"Unsupported timestamp type: {type(ts)} -> {ts}")

    m = int(seconds // 60)
    s = seconds % 60
    s = round(s, 2)

    return f"{m:02d}:{s:05.2f}"

#----------------------------
# Merge all episode GTs
#----------------------------
def all_ep_gt(episodes_dict):
    """
    Merge GTs from all episodes defined in `episodes_dict`.
    :param episodes_dict: dict containing all episode info
    """
    dfs = []
    for ep_name, ep in episodes_dict.items():
        gt_path = ep["ground_truth_path"]
        df = pd.read_excel(gt_path)
        df = df.iloc[:, :10]  # first 10 columns only

        # fix timestamp column
        if "Timestamp" in df.columns:
            df["Timestamp"] = df["Timestamp"].apply(lambda x: format_gt_timestamp(x))

        dfs.append(df)

    all_ep_gt_df = pd.concat(dfs, ignore_index=True)

    # Save
    out_dir = "../data/processed/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_ep_gt.csv")
    all_ep_gt_df.to_csv(out_path, index=False)

    print(f"Consolidated GT: {all_ep_gt_df.shape[0]} rows, {all_ep_gt_df.shape[1]} columns")
    return all_ep_gt_df


#----------------------------
# FEATURE EXTRACTION
#----------------------------
def build_feature_space_df(
    feature_extractor_fn,
    feature_list,
    gt_df,
    characters,
    video_name_to_gt,
    frames_base_dir="../data/processed/video",
    out_path="../data/processed/feature_spaces/visual_sim1.csv"
):
    """
    Build feature-space DF for all episodes and all frames in GT.

    - Computes all requested features from scratch
    - Shows a tqdm progress bar per episode
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Initialize DF from GT
    feature_df = gt_df.copy()
    feature_df = feature_df[["Video", "Frame_number", "Timestamp"] + characters]
    feature_df["frame"] = "frame" + feature_df["Frame_number"].astype(str) + ".jpg"

    all_records = []

    for episode_name, video_id in video_name_to_gt.items():
        ep_gt = gt_df[gt_df["Video"] == video_id].copy()
        frames_dir = os.path.join(frames_base_dir, f"{episode_name}-frames")

        if not os.path.exists(frames_dir):
            print(f"[Warning] Missing frames dir: {frames_dir}")
            continue

        # tqdm progress bar per episode
        for _, row in tqdm(ep_gt.iterrows(), total=len(ep_gt), desc=f"{episode_name}", ncols=100):
            frame_file = f"frame{row['Frame_number']}.jpg"
            frame_path = os.path.join(frames_dir, frame_file)

            feats = feature_extractor_fn(frame_path, feature_list)

            record = {
                "Video": row["Video"],
                "Frame_number": row["Frame_number"],
                "Timestamp": row["Timestamp"],
                "frame": frame_file,
            }
            record.update(feats)

            # Append labels for characters
            for char in characters:
                record[char] = row[char]

            all_records.append(record)

    # Build final DataFrame
    merged_df = pd.DataFrame(all_records)

    # Save CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    print(f"\n[Feature space] saved {merged_df.shape} -> {out_path}")

    return merged_df

def build_feature_space_df_sequential(
    feature_extractor_fn,
    feature_list,
    gt_df,
    characters,
    video_name_to_gt,
    frames_base_dir="../data/processed/video",
    out_path="../data/processed/feature_spaces/visual_sim2.csv"
):
    """
    Sequential feature extraction (SIM2): passes prev_frame_path -> current_frame_path.
    Needed for motion (optical flow).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_records = []

    for episode_name, video_id in video_name_to_gt.items():
        ep_gt = gt_df[gt_df["Video"] == video_id].copy()
        frames_dir = os.path.join(frames_base_dir, f"{episode_name}-frames")

        if not os.path.exists(frames_dir):
            print(f"[Warning] Missing frames dir: {frames_dir}")
            continue

        prev_frame_path = None

        for _, row in tqdm(ep_gt.iterrows(), total=len(ep_gt), desc=f"{episode_name}", ncols=100):
            frame_file = f"frame{row['Frame_number']}.jpg"
            frame_path = os.path.join(frames_dir, frame_file)

            feats = feature_extractor_fn(prev_frame_path, frame_path, feature_list)

            record = {
                "Video": row["Video"],
                "Frame_number": row["Frame_number"],
                "Timestamp": row["Timestamp"],
                "frame": frame_file,
            }
            record.update(feats)

            # labels
            for char in characters:
                record[char] = row[char]

            all_records.append(record)

            prev_frame_path = frame_path

    merged_df = pd.DataFrame(all_records)
    merged_df.to_csv(out_path, index=False)
    print(f"\n[Feature space SIM2] saved {merged_df.shape} -> {out_path}")
    return merged_df


#-----------------------------------
# FEATURE SPACE TRAIN-TEST SPLIT
#-----------------------------------
def split_feature_space_df(feature_df, EPISODES, EPISODE_NAME_TO_VIDEO_ID):
    """
    Split feature-space DF into train/test using per-episode split timestamps.
    """

    train_parts = []
    test_parts = []

    for episode_name, ep in EPISODES.items():
        split_ts = ep["train_split_timestamp"]
        split_sec = visualTools.parse_timestamp(split_ts)

        video_id = EPISODE_NAME_TO_VIDEO_ID[episode_name]

        # Filter rows for this episode
        ep_df = feature_df[feature_df["Video"] == video_id].copy()

        if ep_df.empty:
            print(f"[WARN] No rows for {episode_name} (Video={video_id})")
            continue

        # Timestamp → seconds
        ep_df["_ts_sec"] = ep_df["Timestamp"].apply(
            visualTools.parse_timestamp
        )

        train_ep = ep_df[ep_df["_ts_sec"] <= split_sec]
        test_ep  = ep_df[ep_df["_ts_sec"] > split_sec]

        train_parts.append(train_ep)
        test_parts.append(test_ep)

        print(
            f"[split] {episode_name} | "
            f"Video={video_id} | "
            f"train={len(train_ep)}, test={len(test_ep)}"
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df  = pd.concat(test_parts, ignore_index=True)

    train_df.drop(columns=["_ts_sec"], inplace=True)
    test_df.drop(columns=["_ts_sec"], inplace=True)

    print(
        f"[FINAL SPLIT] train={train_df.shape}, test={test_df.shape}"
    )

    return train_df, test_df