"""Extract wrist trajectories from a baseball swing video using MediaPipe Pose."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

_POSE = mp.solutions.pose
_LEFT_WRIST = _POSE.PoseLandmark.LEFT_WRIST.value
_RIGHT_WRIST = _POSE.PoseLandmark.RIGHT_WRIST.value

_MAX_LONG_SIDE = 720


@dataclass
class SwingData:
    frames: pd.DataFrame
    fps: float
    width: int
    height: int
    swing_start: int
    swing_end: int
    snapshot_bgr: Optional[np.ndarray]

    @property
    def swing_frames(self) -> pd.DataFrame:
        df = self.frames
        return df[(df["frame_idx"] >= self.swing_start) & (df["frame_idx"] <= self.swing_end)]


def _resize_for_inference(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= _MAX_LONG_SIDE:
        return frame, 1.0
    scale = _MAX_LONG_SIDE / long_side
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale


def _detect_swing_range(speeds: np.ndarray, fps: float) -> tuple[int, int]:
    if len(speeds) == 0 or np.all(np.isnan(speeds)):
        return 0, max(len(speeds) - 1, 0)
    filled = np.nan_to_num(speeds, nan=0.0)
    peak = int(np.nanargmax(filled))
    threshold = max(filled.max() * 0.15, 1e-6)
    start = peak
    while start > 0 and filled[start - 1] > threshold:
        start -= 1
    end = peak
    while end < len(filled) - 1 and filled[end + 1] > threshold:
        end += 1
    # Pad by ~0.1s on each side for context
    pad = max(int(fps * 0.1), 1)
    return max(start - pad, 0), min(end + pad, len(filled) - 1)


def analyze_video(
    video_path: str,
    min_detection_confidence: float = 0.5,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> SwingData:
    """Run MediaPipe Pose over every frame and build a swing trajectory DataFrame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows: list[dict] = []
    mid_frame_idx = total_frames // 2 if total_frames else 0
    snapshot_bgr: Optional[np.ndarray] = None

    with _POSE.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx == mid_frame_idx or snapshot_bgr is None:
                snapshot_bgr = frame.copy()

            small, _scale = _resize_for_inference(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            row: dict = {
                "frame_idx": frame_idx,
                "t": frame_idx / fps,
                "left_wrist_x": np.nan,
                "left_wrist_y": np.nan,
                "right_wrist_x": np.nan,
                "right_wrist_y": np.nan,
                "wrist_mid_x": np.nan,
                "wrist_mid_y": np.nan,
                "wrist_mid_wx": np.nan,
                "wrist_mid_wy": np.nan,
                "wrist_mid_wz": np.nan,
            }

            lm = result.pose_landmarks
            if lm is not None:
                lw = lm.landmark[_LEFT_WRIST]
                rw = lm.landmark[_RIGHT_WRIST]
                row["left_wrist_x"] = lw.x * width
                row["left_wrist_y"] = lw.y * height
                row["right_wrist_x"] = rw.x * width
                row["right_wrist_y"] = rw.y * height
                row["wrist_mid_x"] = (row["left_wrist_x"] + row["right_wrist_x"]) / 2.0
                row["wrist_mid_y"] = (row["left_wrist_y"] + row["right_wrist_y"]) / 2.0

            wlm = result.pose_world_landmarks
            if wlm is not None:
                lw = wlm.landmark[_LEFT_WRIST]
                rw = wlm.landmark[_RIGHT_WRIST]
                row["wrist_mid_wx"] = (lw.x + rw.x) / 2.0
                row["wrist_mid_wy"] = (lw.y + rw.y) / 2.0
                row["wrist_mid_wz"] = (lw.z + rw.z) / 2.0

            rows.append(row)
            frame_idx += 1

            if progress_callback and total_frames:
                progress_callback(min(frame_idx / total_frames, 1.0))

    cap.release()

    frames = pd.DataFrame(rows)
    if frames.empty:
        raise RuntimeError("No frames decoded from video.")

    dx = frames["wrist_mid_x"].diff()
    dy = frames["wrist_mid_y"].diff()
    pixel_speed = np.sqrt(dx.pow(2) + dy.pow(2)) * fps
    frames["wrist_speed_px_s"] = pixel_speed

    wdx = frames["wrist_mid_wx"].diff()
    wdy = frames["wrist_mid_wy"].diff()
    wdz = frames["wrist_mid_wz"].diff()
    world_speed = np.sqrt(wdx.pow(2) + wdy.pow(2) + wdz.pow(2)) * fps
    frames["wrist_speed_m_s"] = world_speed

    start, end = _detect_swing_range(pixel_speed.to_numpy(), fps)

    return SwingData(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        swing_start=int(frames["frame_idx"].iloc[start]),
        swing_end=int(frames["frame_idx"].iloc[end]),
        snapshot_bgr=snapshot_bgr,
    )


def compute_metrics(data: SwingData) -> dict:
    """Compute summary metrics over the detected swing range."""
    df = data.swing_frames.dropna(subset=["wrist_mid_x", "wrist_mid_y"])
    if df.empty:
        return {
            "max_speed_m_s": float("nan"),
            "max_speed_px_s": float("nan"),
            "path_length_m": float("nan"),
            "duration_s": 0.0,
            "swing_plane_tilt_deg": float("nan"),
        }

    max_speed_m = float(np.nanmax(df["wrist_speed_m_s"])) if df["wrist_speed_m_s"].notna().any() else float("nan")
    max_speed_px = float(np.nanmax(df["wrist_speed_px_s"])) if df["wrist_speed_px_s"].notna().any() else float("nan")

    world = df[["wrist_mid_wx", "wrist_mid_wy", "wrist_mid_wz"]].dropna().to_numpy()
    if len(world) >= 2:
        diffs = np.diff(world, axis=0)
        path_length_m = float(np.sum(np.linalg.norm(diffs, axis=1)))
    else:
        path_length_m = float("nan")

    duration_s = float(df["t"].iloc[-1] - df["t"].iloc[0])

    tilt_deg = float("nan")
    if len(world) >= 3:
        centered = world - world.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        principal = vh[0]
        # Angle between principal axis and the horizontal plane (xz-plane)
        horizontal_component = np.linalg.norm([principal[0], principal[2]])
        tilt_deg = float(np.degrees(np.arctan2(abs(principal[1]), horizontal_component)))

    return {
        "max_speed_m_s": max_speed_m,
        "max_speed_px_s": max_speed_px,
        "path_length_m": path_length_m,
        "duration_s": duration_s,
        "swing_plane_tilt_deg": tilt_deg,
    }
