import cv2
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from utils import visual_tools as visualTools

def show_visual_features(frame_path):
    img = cv2.imread(frame_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame_info = {
        "hsv": hsv,
        "gray": gray
    }

    eye_feats = visualTools.frog_eye_pattern(frame_info)
    brown_feats = visualTools.brown_rhythm(frame_info)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_rgb)
    axs[0].set_title(
        f"Eyes: {eye_feats['eye_blob_count']} | "
        f"Align: {eye_feats['eye_horizontal_align']:.2f}"
    )

    axs[1].imshow(img_rgb)
    axs[1].set_title(
        f"Brown rhythm: {brown_feats['brown_rhythm']:.3f}"
    )

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_eye_blobs(frame_path):
    img = cv2.imread(frame_path)
    img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    frame_info = {
        "hsv": hsv,
        "gray": gray
    }

    feats = visualTools.frog_eye_pattern(frame_info)
    sclera_mask = visualTools._debug_last_sclera_mask
    contours = visualTools._debug_last_eye_contours

    for c in contours:
        cv2.drawContours(img_vis, [c], -1, (255, 0, 0), 2)

    plt.figure(figsize=(5, 5))
    plt.imshow(img_vis)
    plt.title(f"Detected eyes: {feats['eye_blob_count']}")
    plt.axis("off")
    plt.show()


def show_audio_window(wav_path, frame_idx, fps=25, window_sec=0.5):
    y, sr = librosa.load(wav_path, sr=None)

    center_t = frame_idx / fps
    start = max(0, center_t - window_sec)
    end = center_t + window_sec

    y_seg = y[int(start * sr): int(end * sr)]

    plt.figure(figsize=(8, 2))
    librosa.display.waveshow(y_seg, sr=sr)
    plt.title(f"Waveform around frame {frame_idx}")
    plt.tight_layout()
    plt.show()

def show_pitch_and_onset(wav_path, frame_idx, fps=25):
    y, sr = librosa.load(wav_path, sr=22050)

    f0, voiced, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr
    )

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    t = librosa.times_like(f0, sr=sr)
    frame_t = frame_idx / fps

    plt.figure(figsize=(10, 4))
    plt.plot(t, f0, label="Pitch (f0)")
    plt.plot(
        librosa.times_like(onset_env, sr=sr),
        onset_env * 50,
        label="Onset strength (scaled)",
        alpha=0.7
    )
    plt.axvline(frame_t, color="red", linestyle="--", label="Frame")

    plt.legend()
    plt.title("Pitch + Rhythm (Cook signature)")
    plt.tight_layout()
    plt.show()