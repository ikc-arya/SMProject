# utils/audio_tools.py

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import shutil
import numpy as np
import pandas as pd
import librosa


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp_to_seconds(ts: str) -> float:
    """
    Expected formats:
      - "MM:SS"
      - "MM:SS.xx"
      - "HH:MM:SS"
      - "HH:MM:SS.xx"
    Examples: "00:00.04", "25:39.72", "19:30"
    """
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        mm, ss = parts
        return float(mm) * 60.0 + float(ss)
    elif len(parts) == 3:
        hh, mm, ss = parts
        return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
    else:
        raise ValueError(f"Unrecognized timestamp format: {ts}")


def seconds_to_timestamp(sec: float) -> str:
    # keeps 2 decimals to match GT style like 00:00.04
    if sec < 0:
        sec = 0.0
    mm = int(sec // 60)
    ss = sec - 60 * mm
    return f"{mm:02d}:{ss:05.2f}"  # e.g., 00:00.04, 25:39.72



def run_ffmpeg_extract_audio(video_path: str, out_wav_path: str, sr: int = 22050) -> None:
    ensure_dir(os.path.dirname(out_wav_path))

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg not found in PATH.\n"
            "Install it:\n"
            "  macOS (brew): brew install ffmpeg\n"
            "  conda: conda install -c conda-forge ffmpeg\n"
            "Then restart the Jupyter kernel."
        )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        out_wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDERR:\n{proc.stderr}"
        )

# -----------------------------
# Audio feature extraction
# -----------------------------

@dataclass
class AudioFrameConfig:
    sr: int = 22050
    fps: int = 25                 # MUST match visual pipeline (0.04s per frame)
    n_fft: int = 2048
    n_mfcc: int = 13
    fmin: float = None            # set in __post_init__
    fmax: float = None            # set in __post_init__

    def __post_init__(self):
        if self.fmin is None:
            self.fmin = librosa.note_to_hz("C2")
        if self.fmax is None:
            self.fmax = librosa.note_to_hz("C5")

    @property
    def hop_length(self) -> int:
        # one feature vector per video frame
        return int(self.sr / self.fps)


def extract_frame_level_features(
    wav_path: str,
    cfg: AudioFrameConfig,
) -> np.ndarray:
    """
    Returns array shape (n_frames, n_features)
    Features:
      - MFCC (13) + deltas (13) + delta2 (13) => 39
      - spectral centroid => 1
      - f0 (pyin) => 1
    Total = 41
    """
    y, _ = librosa.load(wav_path, sr=cfg.sr, mono=True)

    # MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, sr=cfg.sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )  # (13, T)

    mfcc_d1 = librosa.feature.delta(mfcc, order=1)
    mfcc_d2 = librosa.feature.delta(mfcc, order=2)
    mfcc_full = np.concatenate([mfcc, mfcc_d1, mfcc_d2], axis=0)  # (39, T)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )[0]  # (T,)

    # f0 via pyin
    f0, voiced_flag, _ = librosa.pyin(
        y=y,
        sr=cfg.sr,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        hop_length=cfg.hop_length,
    )  # f0: (T,)

    # Align lengths (just in case)
    T = min(mfcc_full.shape[1], centroid.shape[0], f0.shape[0])
    mfcc_full = mfcc_full[:, :T]
    centroid = centroid[:T]
    f0 = f0[:T]

    # stack -> (41, T) then transpose -> (T, 41)
    feat = np.vstack([mfcc_full, centroid[np.newaxis, :], f0[np.newaxis, :]]).T
    return feat


def build_audio_feature_space_df(
    EPISODES: Dict[str, Dict],
    EPISODE_NAME_TO_VIDEO_ID: Dict[str, int],
    gt_df: pd.DataFrame,
    character_cols: List[str],
    out_csv_path: str = "../data/processed/feature_spaces/audio_sim1.csv",
    cache_dir: str = "../data/raw/_audio_cache",
    cfg: Optional[AudioFrameConfig] = None,
) -> pd.DataFrame:
    """
    Builds frame-level feature space:
      [Video, Frame_number, Timestamp] + audio_features + [character labels]
    Uses GT for (Video, Frame_number, Timestamp, labels).
    Extracts audio from avi to cache wav per episode.
    """
    if cfg is None:
        cfg = AudioFrameConfig()

    ensure_dir(os.path.dirname(out_csv_path))
    ensure_dir(cache_dir)

    rows = []

    # Basic sanity
    required_gt_cols = {"Video", "Frame_number", "Timestamp", *character_cols}
    missing = required_gt_cols - set(gt_df.columns)
    if missing:
        raise ValueError(f"GT df missing columns: {missing}")

    for episode_name, ep in EPISODES.items():
        video_id = EPISODE_NAME_TO_VIDEO_ID[episode_name]
        video_path = ep["path"]

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Video not found: {video_path}\n"
                f"Put avi locally under data/raw/ (it is gitignored, so safe)."
            )

        wav_cache = os.path.join(cache_dir, f"{episode_name}.wav")
        run_ffmpeg_extract_audio(video_path=video_path, out_wav_path=wav_cache, sr=cfg.sr)

        feat = extract_frame_level_features(wav_cache, cfg=cfg)  # (T, 41)

        # Pull GT slice for this video_id
        gt_ep = gt_df[gt_df["Video"] == video_id].copy()
        gt_ep = gt_ep.sort_values("Frame_number")

        # Align by number of frames
        T = min(len(gt_ep), feat.shape[0])
        gt_ep = gt_ep.iloc[:T].copy()
        feat = feat[:T, :]

        # Build dataframe
        feat_cols = []
        # 41 features: mfcc_0..12, d1_0..12, d2_0..12, centroid, f0
        for i in range(13):
            feat_cols.append(f"mfcc_{i}")
        for i in range(13):
            feat_cols.append(f"mfcc_d1_{i}")
        for i in range(13):
            feat_cols.append(f"mfcc_d2_{i}")
        feat_cols.append("spectral_centroid")
        feat_cols.append("f0")

        audio_df = pd.DataFrame(feat, columns=feat_cols)
        audio_df.insert(0, "Timestamp", gt_ep["Timestamp"].values)
        audio_df.insert(0, "Frame_number", gt_ep["Frame_number"].values)
        audio_df.insert(0, "Video", gt_ep["Video"].values)

        # Attach labels
        for c in character_cols:
            audio_df[c] = gt_ep[c].astype(int).values

        rows.append(audio_df)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_csv_path, index=False)
    print(f"[audio feature space] saved {out.shape} -> {out_csv_path}")
    return out
