"""Keyframe-based swing trajectory (v4).

The user scrubs through the swing, clicking the bat tip on as many
frames as they like. A cubic spline is fit through the clicks to
produce a smooth trajectory. The background of the output PNG is a
semi-transparent stack of the clicked frames, so the user can visually
verify that the spline follows the bat head at every keyframe.

Usage:
    python src/analyze_swing_v4.py INPUT.mp4 -o output/trajectory.png

Workflow:
    1. Scrub window: slider + [s]/[e] for swing range, [Enter] to confirm.
    2. Annotation window: slider to move within the swing range.
         - left-click : add or move the keyframe for the current frame
         - right-click / [d] : delete the keyframe for the current frame
         - [Enter] : finish (need >= 2 keyframes)
         - [q] : abort
    3. Output PNG = composite of clicked frames + spline trajectory +
       keyframe markers.
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
from scipy.interpolate import CubicSpline

# ----------------------------------------------------------------------------
# Rendering parameters
# ----------------------------------------------------------------------------
COMPOSITE_MODE = "mean"   # "mean" (translucent stack) or "max" (lighten blend)


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
    cv2.createTrackbar("frame", window, 0, total - 1, lambda v: state.update(idx=v))

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
        info = f"frame {state['idx']}/{total - 1}   start={state['start']}   end={state['end']}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
            sys.exit(1)

    cap.release()
    cv2.destroyWindow(window)
    return state["start"], state["end"]


def _annotate_keyframes(
    video_path: Path,
    start_frame: int,
    end_frame: int,
) -> dict[int, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    total_range = end_frame - start_frame
    state = {"idx": start_frame, "keyframes": {}}
    window = (
        "Annotate keyframes  [click]=add/move  [right-click/d]=delete  "
        "[Enter]=OK  [q]=quit"
    )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    cv2.createTrackbar(
        "frame", window, 0, total_range,
        lambda v: state.update(idx=start_frame + v),
    )

    def on_mouse(event: int, x: int, y: int, *_args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["keyframes"][state["idx"]] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["keyframes"].pop(state["idx"], None)

    cv2.setMouseCallback(window, on_mouse)

    print(
        f"[annotate]  {total_range + 1} frames in range. "
        "Scrub and left-click to add keyframes. Right-click or d to delete. "
        "Enter to finish (>= 2 keyframes).",
        file=sys.stderr,
    )

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, state["idx"])
        ok, frame = cap.read()
        if not ok:
            state["idx"] = max(state["idx"] - 1, start_frame)
            cv2.setTrackbarPos("frame", window, state["idx"] - start_frame)
            continue

        display = frame.copy()
        for kf_idx, (kx, ky) in state["keyframes"].items():
            if kf_idx == state["idx"]:
                cv2.circle(display, (kx, ky), 14, (0, 255, 255), 2)
                cv2.drawMarker(display, (kx, ky), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            else:
                cv2.circle(display, (kx, ky), 5, (0, 255, 0), 2)

        info = (
            f"frame {state['idx']}  range {start_frame}-{end_frame}  "
            f"keyframes={len(state['keyframes'])}"
        )
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            if len(state["keyframes"]) >= 2:
                break
            print("need at least 2 keyframes.", file=sys.stderr)
        elif key == ord("d"):
            state["keyframes"].pop(state["idx"], None)
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

    cap.release()
    cv2.destroyWindow(window)
    return state["keyframes"]


def _interpolate_trajectory(keyframes: dict[int, tuple[int, int]]) -> pd.DataFrame:
    items = sorted(keyframes.items())
    frames = np.array([f for f, _ in items], dtype=np.float64)
    xs = np.array([p[0] for _, p in items], dtype=np.float64)
    ys = np.array([p[1] for _, p in items], dtype=np.float64)

    dense = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype=np.float64)
    if len(frames) >= 3:
        sx = CubicSpline(frames, xs, bc_type="natural")
        sy = CubicSpline(frames, ys, bc_type="natural")
        dx = sx(dense)
        dy = sy(dense)
    else:
        dx = np.interp(dense, frames, xs)
        dy = np.interp(dense, frames, ys)
    return pd.DataFrame({"frame_idx": dense.astype(int), "tip_x": dx, "tip_y": dy})


def _collect_keyframe_images(video_path: Path, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    images: list[np.ndarray] = []
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok:
            images.append(frame)
    cap.release()
    if not images:
        raise RuntimeError("No keyframe images captured.")
    return images


def _composite_background(images: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(images, axis=0).astype(np.float32)
    if mode == "max":
        return np.max(stack, axis=0).astype(np.uint8)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def _render_png(
    trajectory: pd.DataFrame,
    keyframes: dict[int, tuple[int, int]],
    background_bgr: np.ndarray,
    output_path: Path,
) -> None:
    height, width = background_bgr.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * height / max(width, 1)))
    ax.imshow(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB), extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    xs = trajectory["tip_x"].to_numpy()
    ys = trajectory["tip_y"].to_numpy()
    if len(xs) >= 2:
        ts = np.linspace(0.0, 1.0, len(xs))
        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_time = (ts[:-1] + ts[1:]) / 2.0
        cmap = plt.get_cmap("plasma")
        colors = cmap(seg_time)
        colors[:, 3] = 0.9
        lc = LineCollection(
            segments, linewidths=4.5, colors=colors, capstyle="round", joinstyle="round"
        )
        ax.add_collection(lc)
        ax.scatter(xs[0], ys[0], c="lime", s=90, edgecolors="black", zorder=5, label="start")
        ax.scatter(xs[-1], ys[-1], c="red", s=90, edgecolors="black", zorder=5, label="end")

    for idx, (fi, (kx, ky)) in enumerate(sorted(keyframes.items())):
        ax.scatter(kx, ky, s=130, facecolors="none", edgecolors="white", linewidths=2.0, zorder=6)
        ax.annotate(
            f"{idx + 1}", (kx, ky), textcoords="offset points", xytext=(8, -8),
            color="white", fontsize=10, fontweight="bold",
        )

    ax.legend(loc="upper right", framealpha=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


def _parse_keyframe_string(raw: str) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [int(p) for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"bad keyframe triple: {chunk!r}")
        result[parts[0]] = (parts[1], parts[2])
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Keyframe-based swing trajectory (v4).")
    parser.add_argument("input", type=Path, help="Input swing video")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("output/trajectory.png"),
        help="Output PNG path (default: output/trajectory.png)",
    )
    parser.add_argument("--start-sec", type=float, help="Swing start time in seconds")
    parser.add_argument("--end-sec", type=float, help="Swing end time in seconds")
    parser.add_argument(
        "--keyframes", type=str,
        help='Non-interactive keyframes: "frame,x,y;frame,x,y;..."',
    )
    parser.add_argument(
        "--composite", choices=("mean", "max"), default=COMPOSITE_MODE,
        help="Background blend mode (default: mean)",
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

    if args.keyframes:
        keyframes = _parse_keyframe_string(args.keyframes)
    else:
        keyframes = _annotate_keyframes(args.input, start_frame, end_frame)

    if len(keyframes) < 2:
        print("need at least 2 keyframes.", file=sys.stderr)
        return 1

    sorted_frames = sorted(keyframes.keys())
    print(
        f"keyframes: {len(keyframes)}  "
        f"(frames {sorted_frames[0]}..{sorted_frames[-1]})",
        file=sys.stderr,
    )

    images = _collect_keyframe_images(args.input, sorted_frames)
    background = _composite_background(images, args.composite)
    trajectory = _interpolate_trajectory(keyframes)
    _render_png(trajectory, keyframes, background, args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
