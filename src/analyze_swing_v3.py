"""Interactive bat-tip tracker (v3): user marks the swing range and the
initial bat-tip position, then Lucas-Kanade optical flow tracks that point
through the swing. The trajectory is rendered to a single stacked PNG.

Usage:
    python src/analyze_swing_v3.py INPUT.mp4 -o output/trajectory.png

Workflow:
    1. A scrub window opens. Use the slider to find the moment the swing
       starts and press [s]. Scrub to the swing end and press [e].
       Press [Enter] to confirm. ([q] aborts.)
    2. A second window shows the start frame. Click the bat tip (or any
       point you want to track). You can re-click to move the marker.
       Press [Enter] to confirm.
    3. Tracking runs and the trajectory PNG is saved.

Non-interactive mode (skip the GUI by supplying everything on the CLI):
    python src/analyze_swing_v3.py INPUT.mp4 -o out.png \
        --start-sec 1.2 --end-sec 2.4 --seed-x 640 --seed-y 320
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
# Tracking parameters
# ----------------------------------------------------------------------------
LK_WIN_SIZE = (31, 31)
LK_MAX_LEVEL = 4
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
FB_ERR_GOOD = 1.5    # forward-backward error <= this counts as full confidence
FB_ERR_BAD = 8.0     # >= this counts as zero confidence
SMOOTH_WINDOW = 3    # rolling-mean window for the trajectory
MIN_CONF_RENDER = 0.1


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
        elif key in (13, 10):  # Enter
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


def _click_select_seed(video_path: Path, frame_idx: int) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}.")

    point: list[Optional[tuple[int, int]]] = [None]
    window = "Click the point to track (e.g. bat tip)  [Enter]=OK  [q]=quit"

    def on_mouse(event: int, x: int, y: int, *_args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point[0] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    cv2.setMouseCallback(window, on_mouse)

    print(
        "[seed]  click the point on the bat to track. "
        "Click again to move it. Enter = confirm. q = abort.",
        file=sys.stderr,
    )

    while True:
        display = frame.copy()
        if point[0] is not None:
            x, y = point[0]
            cv2.circle(display, (x, y), 12, (0, 255, 0), 2)
            cv2.drawMarker(display, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and point[0] is not None:
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            print("aborted by user.", file=sys.stderr)
            sys.exit(1)

    cv2.destroyWindow(window)
    return point[0]


def _track_lk(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    seed_xy: tuple[int, int],
) -> tuple[pd.DataFrame, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows: list[dict] = []
    prev_gray: Optional[np.ndarray] = None
    pt = np.array([[[float(seed_xy[0]), float(seed_xy[1])]]], dtype=np.float32)
    seed_frame_bgr: Optional[np.ndarray] = None

    for fi in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if fi == start_frame:
            seed_frame_bgr = frame.copy()
            confidence = 1.0
        else:
            new_pt, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                pt,
                None,
                winSize=LK_WIN_SIZE,
                maxLevel=LK_MAX_LEVEL,
                criteria=LK_CRITERIA,
            )
            if status[0, 0] == 1:
                back_pt, status_b, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    prev_gray,
                    new_pt,
                    None,
                    winSize=LK_WIN_SIZE,
                    maxLevel=LK_MAX_LEVEL,
                    criteria=LK_CRITERIA,
                )
                if status_b[0, 0] == 1:
                    fb_err = float(np.linalg.norm(back_pt - pt))
                else:
                    fb_err = FB_ERR_BAD
                if fb_err <= FB_ERR_GOOD:
                    confidence = 1.0
                elif fb_err >= FB_ERR_BAD:
                    confidence = 0.0
                else:
                    confidence = 1.0 - (fb_err - FB_ERR_GOOD) / (FB_ERR_BAD - FB_ERR_GOOD)
                pt = new_pt
            else:
                confidence = 0.0

        rows.append(
            {
                "frame_idx": fi,
                "tip_x": float(pt[0, 0, 0]),
                "tip_y": float(pt[0, 0, 1]),
                "confidence": float(confidence),
            }
        )
        prev_gray = gray

    cap.release()
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
    parser = argparse.ArgumentParser(description="Interactive bat-tip swing tracker (v3).")
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
    parser.add_argument("--seed-x", type=int, help="Seed point x (pixels) on the start frame")
    parser.add_argument("--seed-y", type=int, help="Seed point y (pixels) on the start frame")
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

    if args.seed_x is not None and args.seed_y is not None:
        seed = (args.seed_x, args.seed_y)
    else:
        seed = _click_select_seed(args.input, start_frame)

    print(f"seed point: {seed}", file=sys.stderr)

    df, seed_frame = _track_lk(args.input, start_frame, end_frame, seed)
    total = len(df)
    high_conf = int(df["confidence"].ge(0.5).sum())
    print(
        f"tracked frames={total}  high_confidence={high_conf} ({high_conf / total:.0%})",
        file=sys.stderr,
    )

    df = _smooth(df)
    _render_png(df, seed_frame, args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
