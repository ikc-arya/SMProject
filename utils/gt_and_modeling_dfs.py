import pandas as pd
import os


def all_ep_gt():
    '''
    merge all gt from the episodes to one
    '''
    gt_files = [
        "../data/muppets-gt-2025wt/Ground_Truth_New_01.xlsx",
        "../data/muppets-gt-2025wt/Ground_Truth_New_03.xlsx",
        "../data/muppets-gt-2025wt/Ground_Truth_New_04.xlsx",
    ]
    dfs = []
    for file in gt_files:
        df = pd.read_excel(file)
        # Restrict to expected GT schema: all rows, first 10 columns only
        df = df.iloc[:, :10]
        dfs.append(df)
    all_ep_gt = pd.concat(dfs, ignore_index=True)
    out_dir = "../data/processed/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "all_ep_gt.csv")
    all_ep_gt.to_csv(out_path, index=False)
    print(f"Consolidated GT: {all_ep_gt.shape[0]} rows, {all_ep_gt.shape[1]} columns")
    return all_ep_gt


def build_feature_space(frames_dir, episode_name, feature_extractor_fn):
    """
    Generic feature-space builder.
    - frames_dir: directory with frames
    - episode_name: episode identifier
    - feature_extractor_fn: function(frame_path) -> dict
    Returns: pandas DataFrame of features
    """
    records = []

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".jpg")],
        key=lambda x: int(x[5:-4])
    )

    for fname in frame_files:
        frame_path = os.path.join(frames_dir, fname)
        feat = feature_extractor_fn(frame_path)
        records.append(feat)

    df = pd.DataFrame(records)

    out_dir = f"../data/processed/video/{episode_name}/feature_spaces/"
    os.makedirs(out_dir, exist_ok=True)

    set_name = os.path.basename(os.path.normpath(frames_dir))  # train / test
    out_path = os.path.join(out_dir, f"features_{set_name}.csv")
    df.to_csv(out_path, index=False)

    print(f"[Feature space] saved {df.shape} -> {out_path}")
    return df


def build_feature_df(frames_dir, episode_name, feature_fn):
    '''
    Docstring for build_feature_df
    
    :param frames_dir: path processed frames dir
    :param episode_name: 
    :param feature_fn: Description
    '''
    records = []

    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".jpg")],
        key=lambda x: int(x[5:-4])
    )

    for fname in frame_files:
        path = os.path.join(frames_dir, fname)
        feat = feature_fn(path)

        feat["Frame_number"] = int(fname[5:-4])
        feat["Video"] = episode_name
        records.append(feat)

    return pd.DataFrame(records)






