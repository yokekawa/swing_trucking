"""Interactive bat-tip tracker (v3): user marks the swing range and draws
a bounding box around the point to track (e.g. the bat tip), then an
OpenCV CSRT tracker follows that box through the swing. The trajectory
is rendered to a single stacked PNG.

CSRT is a discriminative-model tracker (DSST / Channel and Spatial
Reliability) that handles fast motion and partial occlusion better than
naive optical flow.

Usage:
    python src/analyze_swing_v3.py INPUT.mp4 -o output/trajectory.png

Workflow:
    1. A scrub window opens. Use the slider to find the moment the swing
       starts and press [s]. Scrub to the swing end and press [e].
       Press [Enter] to confirm. ([q] aborts.)
    2. A second window shows the start frame. Drag a rectangle around
       the point to track (e.g. the bat tip). Press [Enter] / [Space]
       to confirm the box, or [c] to cancel.
    3. Tracking runs and the trajectory PNG is saved. Pass --preview to
       watch the tracker box live while it processes.

Non-interactive mode (skip the GUI by supplying everything on the CLI):
    python src/analyze_swing_v3.py INPUT.mp4 -o out.png \
        --start-sec 1.2 --end-sec 2.4 \
        --seed-x 620 --seed-y 300 --seed-w 40 --seed-h 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
SMOOTH_WINDOW = 3            # rolling-mean window for the trajectory
MIN_CONF_RENDER = 0.1
MAX_JUMP_FACTOR = 4.0        # reject a tracker update that jumps > this x
                             #   the recent median step size (px / frame)


def _create_csrt_tracker():
    # opencv-contrib provides TrackerCSRT_create either at cv2 top-level
    # (older builds) or under cv2.legacy.
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "CSRT tracker not found. Install opencv-contrib-python "
        "(pip install opencv-contrib-python)."
    )


def _scrub_select_range(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Video reports zero frames.")

    state = {"idx": 0, "start": 0, "end": total - 1}
    window = "Select swing range  [s]=start  [e]=end  [Enter]=OK  [q]=quit"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    def on_track(val: int) -> None:
        state["idx"] = val

    cv2.createTrackbar("frame", window, 0, total - 1, on_track)

    print(
        "[scrub]  s = mark start frame   e = mark end frame   "
        "Enter = confirm   q = abort",
        file=sys.stderr,
    )

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, state["idx"])
        ok, frame = cap.read()
        if not ok:
            state["idx"] = max(state["idx"] - 1, 0)
            cv2.setTrackbarPos("frame", window, state["idx"])
            continue

        info = (
            f"frame {state['idx']}/{total - 1}   "
            f"start={state['start']}   end={state['end']}"
        )
        cv2.putText(
            frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow(window, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            state["start"] = state["idx"]
        elif key == ord("e"):
            state["end"] = state["idx"]
        elif key in (13, 10):
            if state["end"] > state["start"]:
                break
            print("end frame must be greater than start frame.", file=sys.stderr)
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("aborted by user.", file=sys.stderr)
            sys.exit(1)

    cap.release()
    cv2.destroyWindow(window)
    return state["start"], state["end"]


def _select_roi(video_path: Path, frame_idx: int) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}.")

    print(
        "[seed]  drag a rectangle around the bat tip. "
        "Enter/Space = confirm, c = cancel.",
        file=sys.stderr,
    )
    window = "Drag rectangle around the bat tip"
    roi = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        print("no ROI selected.", file=sys.stderr)
        sys.exit(1)
    return x, y, w, h


def _track_csrt(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    seed_bbox: tuple[int, int, int, int],
    preview: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracker = _create_csrt_tracker()
    rows: list[dict] = []
    seed_frame_bgr: Optional[np.ndarray] = None
    prev_center: Optional[tuple[float, float]] = None
    recent_steps: list[float] = []
    lost = False

    preview_window = "tracking preview (q=abort)"
    if preview:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 960, 540)

    for fi in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break

        if fi == start_frame:
            seed_frame_bgr = frame.copy()
            tracker.init(frame, seed_bbox)
            x, y, w, h = seed_bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            confidence = 1.0
            prev_center = (cx, cy)
        else:
            success, bbox = tracker.update(frame)
            if success and not lost:
                x, y, w, h = bbox
                cx, cy = x + w / 2.0, y + h / 2.0
                if prev_center is not None:
                    step = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
                    if recent_steps:
                        median_step = float(np.median(recent_steps[-10:]))
                        limit = max(20.0, median_step * MAX_JUMP_FACTOR)
                        if step > limit:
                            # Likely tracker snap; mark low confidence and freeze
                            confidence = 0.0
                            cx, cy = prev_center
                        else:
                            confidence = 1.0
                            recent_steps.append(step)
                    else:
                        confidence = 1.0
                        recent_steps.append(step)
                else:
                    confidence = 1.0
                prev_center = (cx, cy)
            else:
                lost = True
                confidence = 0.0
                if prev_center is None:
                    cx = cy = 0.0
                else:
                    cx, cy = prev_center

        rows.append(
            {
                "frame_idx": fi,
                "tip_x": float(cx),
                "tip_y": float(cy),
                "confidence": float(confidence),
            }
        )

        if preview:
            display = frame.copy()
            if confidence > 0:
                px, py = int(round(cx)), int(round(cy))
                colour = (0, 255, 0) if confidence >= 0.5 else (0, 165, 255)
                cv2.circle(display, (px, py), 10, colour, 2)
                cv2.drawMarker(display, (px, py), colour, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                display,
                f"frame {fi}  conf={confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(preview_window, display)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if preview:
        cv2.destroyWindow(preview_window)
    if not rows or seed_frame_bgr is None:
        raise RuntimeError("Tracking produced no frames.")
    return pd.DataFrame(rows), seed_frame_bgr


def _smooth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tip_x"] = out["tip_x"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    out["tip_y"] = out["tip_y"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    out["confidence"] = (
        out["confidence"].rolling(SMOOTH_WINDOW, min_periods=1, center=True).mean()
    )
    return out


def _render_png(
    df: pd.DataFrame,
    background_bgr: np.ndarray,
    output_path: Path,
) -> None:
    height, width = background_bgr.shape[:2]
    df = df[df["confidence"] >= MIN_CONF_RENDER]

    fig, ax = plt.subplots(figsize=(10, 10 * height / max(width, 1)))
    ax.imshow(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB), extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    if len(df) >= 2:
        xs = df["tip_x"].to_numpy()
        ys = df["tip_y"].to_numpy()
        conf = df["confidence"].to_numpy()
        ts = np.linspace(0.0, 1.0, len(df))

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
    parser = argparse.ArgumentParser(description="Interactive bat-tip swing tracker (v3, CSRT).")
    parser.add_argument("input", type=Path, help="Input swing video")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/trajectory.png"),
        help="Output PNG path (default: output/trajectory.png)",
    )
    parser.add_argument("--start-sec", type=float, help="Swing start time in seconds")
    parser.add_argument("--end-sec", type=float, help="Swing end time in seconds")
    parser.add_argument("--seed-x", type=int, help="Seed bbox top-left x (pixels)")
    parser.add_argument("--seed-y", type=int, help="Seed bbox top-left y (pixels)")
    parser.add_argument("--seed-w", type=int, help="Seed bbox width (pixels)")
    parser.add_argument("--seed-h", type=int, help="Seed bbox height (pixels)")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window while tracking",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if args.start_sec is not None and args.end_sec is not None:
        start_frame = int(round(args.start_sec * fps))
        end_frame = int(round(args.end_sec * fps))
    else:
        start_frame, end_frame = _scrub_select_range(args.input)

    print(f"swing range: frames {start_frame}-{end_frame} (fps={fps:.1f})", file=sys.stderr)

    seed_flags = [args.seed_x, args.seed_y, args.seed_w, args.seed_h]
    if all(v is not None for v in seed_flags):
        seed_bbox = (args.seed_x, args.seed_y, args.seed_w, args.seed_h)
    else:
        seed_bbox = _select_roi(args.input, start_frame)

    print(f"seed bbox: {seed_bbox}", file=sys.stderr)

    df, seed_frame = _track_csrt(args.input, start_frame, end_frame, seed_bbox, args.preview)
    total = len(df)
    high_conf = int(df["confidence"].ge(0.5).sum())
    print(
        f"tracked frames={total}  high_confidence={high_conf} ({high_conf / max(total, 1):.0%})",
        file=sys.stderr,
    )

    df = _smooth(df)
    _render_png(df, seed_frame, args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
