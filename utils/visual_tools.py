import os
import shutil

import cv2


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
    fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))

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
        "fourcc": fourcc,
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

def extract_frames_meta(video_file, intro_timestamp=0, fps_to_save=8):
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
    

#------------- doing for 'Kermit' only here ----------------#

#------------------------------------------------
# DOMIANANT COLOR
#------------------------------------------------
def rgb_to_hsv(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return int(h*179), int(s*255), int(v*255)

def dominant_color_feature(frames_dir, episode_name, frame_files=None):
    out_dir = f"../data/processed/video/{episode_name}/features/"
    os.makedirs(out_dir, exist_ok=True)

    set_name = os.path.basename(os.path.normpath(frames_dir))  # train/test
    csv_path = os.path.join(out_dir, f"dominant_color_{set_name}.csv")

    if frame_files is None:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
                             key=lambda x: int(x[5:-4]))
        frame_files = [os.path.join(frames_dir, f) for f in frame_files]
    else:
        frame_files = [f if os.path.isabs(f) else os.path.join(frames_dir, f) for f in frame_files]

    data = []
    for frame_path in frame_files:
        rgb = ColorThief(frame_path).get_color(quality=1)
        h, s, v = rgb_to_hsv(rgb)
        data.append({'frame': os.path.basename(frame_path), 'H': h, 'S': s, 'V': v})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['frame','H','S','V'])
        writer.writeheader()
        writer.writerows(data)

    print(f"[Dominant color] saved {len(data)} frames -> {csv_path}")
    return data, csv_path



#------------------------------------------------
# MASK EVERY OTHER COLOR EXCEPT KERMIT'S DISTINCT  GREEN
#------------------------------------------------
def green_mask_feature(frames_dir, episode_name, frame_files=None):
    out_dir = f"../data/processed/video/{episode_name}/features/"
    os.makedirs(out_dir, exist_ok=True)

    set_name = os.path.basename(os.path.normpath(frames_dir))
    csv_path = os.path.join(out_dir, f"green_mask_{set_name}.csv")

    if frame_files is None:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
                             key=lambda x: int(x[5:-4]))
        frame_files = [os.path.join(frames_dir, f) for f in frame_files]
    else:
        frame_files = [f if os.path.isabs(f) else os.path.join(frames_dir, f) for f in frame_files]

    data = []
    for frame_path in frame_files:
        img = cv2.imread(frame_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask green pixels (Hue ~35-85)
        mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
        green_fraction = mask.sum() / mask.size
        data.append({'frame': os.path.basename(frame_path), 'green_frac': green_fraction})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','green_frac'])
        writer.writeheader()
        writer.writerows(data)

    print(f"[Green mask] saved {len(data)} frames -> {csv_path}")
    return data, csv_path


#------------------------------------------------
# SEPARATE KERMIT FROM BACKGROUND
#------------------------------------------------
def edge_magnitude_feature(frames_dir, episode_name, frame_files=None):
    out_dir = f"../data/processed/video/{episode_name}/features/"
    os.makedirs(out_dir, exist_ok=True)

    set_name = os.path.basename(os.path.normpath(frames_dir))
    csv_path = os.path.join(out_dir, f"edge_magnitude_{set_name}.csv")

    if frame_files is None:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")],
                             key=lambda x: int(x[5:-4]))
        frame_files = [os.path.join(frames_dir, f) for f in frame_files]
    else:
        frame_files = [f if os.path.isabs(f) else os.path.join(frames_dir, f) for f in frame_files]

    data = []
    for frame_path in frame_files:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 100, 200)
        mean_edge = edges.mean()
        data.append({'frame': os.path.basename(frame_path), 'mean_edge': mean_edge})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','mean_edge'])
        writer.writeheader()
        writer.writerows(data)

    print(f"[Edge magnitude] saved {len(data)} frames -> {csv_path}")
    return data, csv_path