"""Detect a baseball bat's swing trajectory from a video and render a stacked PNG.

Pipeline (v2, direct bat detection):
    1. MediaPipe Pose -> both wrists per frame (used as a spatial prior).
    2. Motion mask = MOG2 background subtraction AND dilated frame-diff threshold.
    3. Contour extraction on the mask. Each contour's minAreaRect gives an
       aspect ratio; regions with aspect >= ASPECT_MIN are bat candidates.
    4. Candidate is accepted only if its long-axis line passes within
       WRIST_GATE_PX of the wrist midpoint.
    5. Bat tip = candidate endpoint farther from the wrist midpoint.
    6. Missing frames with gaps <= INTERP_MAX_GAP are linearly interpolated;
       the tip track is smoothed with a moving-average window.
    7. Confidence (match quality) modulates line thickness and alpha when
       the trajectory is rendered on a single stacked PNG.

The output is a single PNG showing the full swing trajectory overlaid on the
key frame (frame with peak wrist motion).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe.solutions.pose as _POSE  # noqa: N812 (submodule import ensures solutions is loaded)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

# ----------------------------------------------------------------------------
# Tunable parameters (see README's tuning table)
# ----------------------------------------------------------------------------
ASPECT_MIN = 3.0            # minimum aspect ratio for a contour to be a bat candidate
MIN_CONTOUR_AREA = 80       # px^2; drop tiny noise contours
MOG_HISTORY = 300
MOG_VAR_THRESHOLD = 32.0
DIFF_THRESHOLD = 25         # abs frame-diff threshold (0-255)
MORPH_KERNEL = 5
MORPH_ITERATIONS = 2
WRIST_GATE_PX = 45          # max distance from wrist midpoint to candidate axis
INTERP_MAX_GAP = 10         # frames; longer gaps are left as missing
SMOOTH_WINDOW = 5           # frames; moving-average window on tip coords
MIN_CONF = 0.15             # confidence floor for rendering
POSE_MIN_DETECTION = 0.5

_LEFT_WRIST = _POSE.PoseLandmark.LEFT_WRIST.value
_RIGHT_WRIST = _POSE.PoseLandmark.RIGHT_WRIST.value


@dataclass
class FrameDetection:
    frame_idx: int
    tip_x: float
    tip_y: float
    confidence: float
    wrist_mid_x: float
    wrist_mid_y: float
    wrist_speed_px: float


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-6:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _rect_endpoints(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a 4x2 minAreaRect box, return midpoints of the two short sides."""
    # Pair opposite corners by sorting edges by length.
    edges = [(i, (i + 1) % 4, np.linalg.norm(box[i] - box[(i + 1) % 4])) for i in range(4)]
    edges.sort(key=lambda e: e[2])
    short1, short2 = edges[0], edges[1]
    mid1 = (box[short1[0]] + box[short1[1]]) / 2.0
    mid2 = (box[short2[0]] + box[short2[1]]) / 2.0
    return mid1, mid2


def _extract_wrists(pose_result, width: int, height: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    lm = pose_result.pose_landmarks
    if lm is None:
        return None
    lw = lm.landmark[_LEFT_WRIST]
    rw = lm.landmark[_RIGHT_WRIST]
    left = np.array([lw.x * width, lw.y * height], dtype=np.float32)
    right = np.array([rw.x * width, rw.y * height], dtype=np.float32)
    return left, right


def _select_bat_candidate(
    mask: np.ndarray,
    wrist_mid: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Return (endpoint_a, endpoint_b, confidence) for the best bat candidate."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[tuple[np.ndarray, np.ndarray, float]] = None
    best_conf = 0.0

    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        if min(w, h) < 1.0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect < ASPECT_MIN:
            continue

        box = cv2.boxPoints(rect)
        ep_a, ep_b = _rect_endpoints(box)
        dist = _point_to_segment_distance(wrist_mid, ep_a, ep_b)
        if dist > WRIST_GATE_PX:
            continue

        gate_score = max(0.0, 1.0 - dist / WRIST_GATE_PX)
        aspect_score = min(aspect / 6.0, 1.0)
        area = cv2.contourArea(contour)
        area_score = min(area / 1500.0, 1.0)
        confidence = 0.5 * gate_score + 0.3 * aspect_score + 0.2 * area_score

        if confidence > best_conf:
            best = (ep_a, ep_b, confidence)
            best_conf = confidence

    return best


def analyze(video_path: Path) -> tuple[pd.DataFrame, float, int, int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mog = cv2.createBackgroundSubtractorMOG2(
        history=MOG_HISTORY,
        varThreshold=MOG_VAR_THRESHOLD,
        detectShadows=False,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))

    rows: list[FrameDetection] = []
    prev_gray: Optional[np.ndarray] = None
    prev_wrist_mid: Optional[np.ndarray] = None
    best_frame: Optional[np.ndarray] = None
    best_motion = -1.0

    pose = _POSE.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=POSE_MIN_DETECTION,
        min_tracking_confidence=0.5,
    )
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mog_mask = mog.apply(frame)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, diff_mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
                diff_mask = cv2.dilate(diff_mask, kernel, iterations=1)
                motion_mask = cv2.bitwise_and(mog_mask, diff_mask)
            else:
                motion_mask = np.zeros_like(mog_mask)

            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.dilate(motion_mask, kernel, iterations=MORPH_ITERATIONS)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            wrists = _extract_wrists(result, width, height)

            tip_x = tip_y = np.nan
            confidence = 0.0
            wrist_mid_x = wrist_mid_y = np.nan
            wrist_speed = 0.0

            if wrists is not None:
                left, right = wrists
                wrist_mid = (left + right) / 2.0
                wrist_mid_x, wrist_mid_y = float(wrist_mid[0]), float(wrist_mid[1])

                if prev_wrist_mid is not None:
                    wrist_speed = float(np.linalg.norm(wrist_mid - prev_wrist_mid) * fps)
                prev_wrist_mid = wrist_mid

                candidate = _select_bat_candidate(motion_mask, wrist_mid)
                if candidate is not None:
                    ep_a, ep_b, confidence = candidate
                    # Bat tip = endpoint farther from wrist midpoint.
                    da = np.linalg.norm(ep_a - wrist_mid)
                    db = np.linalg.norm(ep_b - wrist_mid)
                    tip = ep_a if da > db else ep_b
                    tip_x, tip_y = float(tip[0]), float(tip[1])

            if wrist_speed > best_motion:
                best_motion = wrist_speed
                best_frame = frame.copy()

            rows.append(
                FrameDetection(
                    frame_idx=frame_idx,
                    tip_x=tip_x,
                    tip_y=tip_y,
                    confidence=confidence,
                    wrist_mid_x=wrist_mid_x,
                    wrist_mid_y=wrist_mid_y,
                    wrist_speed_px=wrist_speed,
                )
            )

            prev_gray = gray
            frame_idx += 1
    finally:
        pose.close()
        cap.release()

    if not rows:
        raise RuntimeError("No frames decoded.")

    df = pd.DataFrame([r.__dict__ for r in rows])
    if best_frame is None:
        # Fallback: re-open and grab middle frame.
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        _, best_frame = cap.read()
        cap.release()

    return df, fps, width, height, best_frame


def _interpolate_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate NaN runs whose length <= max_gap; leave longer gaps NaN."""
    s = series.copy()
    is_na = s.isna()
    if not is_na.any():
        return s

    group_ids = (is_na != is_na.shift()).cumsum()
    fillable = pd.Series(False, index=s.index)
    for _, group in is_na.groupby(group_ids):
        if not group.iloc[0]:
            continue
        if len(group) > max_gap:
            continue
        # Skip gaps at the very edges (no anchor for interpolation).
        if group.index[0] == s.index[0] or group.index[-1] == s.index[-1]:
            continue
        fillable.loc[group.index] = True

    filled = s.interpolate(method="linear", limit_area="inside")
    result = s.copy()
    result.loc[fillable] = filled.loc[fillable]
    return result


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tip_x"] = _interpolate_gaps(out["tip_x"], INTERP_MAX_GAP)
    out["tip_y"] = _interpolate_gaps(out["tip_y"], INTERP_MAX_GAP)
    # Interpolated frames retain partial confidence.
    interp_mask = df["confidence"].eq(0) & out["tip_x"].notna()
    out.loc[interp_mask, "confidence"] = 0.25

    out["tip_x"] = out["tip_x"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    out["tip_y"] = out["tip_y"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    out["confidence"] = out["confidence"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    return out


def render_stack_png(
    df: pd.DataFrame,
    background_bgr: np.ndarray,
    output_path: Path,
    width: int,
    height: int,
) -> None:
    usable = df.dropna(subset=["tip_x", "tip_y"])
    usable = usable[usable["confidence"] >= MIN_CONF]

    fig, ax = plt.subplots(figsize=(10, 10 * height / max(width, 1)))
    bg_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)
    ax.imshow(bg_rgb, extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    if len(usable) >= 2:
        xs = usable["tip_x"].to_numpy()
        ys = usable["tip_y"].to_numpy()
        conf = usable["confidence"].to_numpy()
        ts = np.linspace(0.0, 1.0, len(usable))

        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_conf = (conf[:-1] + conf[1:]) / 2.0
        seg_time = (ts[:-1] + ts[1:]) / 2.0

        linewidths = 2.0 + 8.0 * np.clip(seg_conf, 0.0, 1.0)
        alphas = 0.35 + 0.6 * np.clip(seg_conf, 0.0, 1.0)

        cmap = plt.get_cmap("plasma")
        colors = cmap(seg_time)
        colors[:, 3] = alphas

        lc = LineCollection(
            segments,
            linewidths=linewidths,
            colors=colors,
            capstyle="round",
            joinstyle="round",
        )
        ax.add_collection(lc)

        ax.scatter(xs[0], ys[0], c="lime", s=80, edgecolors="black", zorder=5, label="start")
        ax.scatter(xs[-1], ys[-1], c="red", s=80, edgecolors="black", zorder=5, label="end")
        ax.legend(loc="upper right", framealpha=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize a baseball bat swing trajectory (v2).")
    parser.add_argument("input", type=Path, help="Input swing video (mp4/mov/...)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/trajectory.png"),
        help="Output PNG path (default: output/trajectory.png)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    print(f"Analyzing {args.input} ...", file=sys.stderr)
    df, fps, width, height, key_frame = analyze(args.input)
    total = len(df)
    raw_detected = int((df["confidence"] > 0).sum())
    print(
        f"  frames={total} fps={fps:.1f} raw_detections={raw_detected} "
        f"({raw_detected / total:.0%})",
        file=sys.stderr,
    )

    df = postprocess(df)
    rendered = int(df["confidence"].ge(MIN_CONF).sum())
    print(f"  rendered_points={rendered} ({rendered / total:.0%})", file=sys.stderr)

    render_stack_png(df, key_frame, args.output, width, height)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
