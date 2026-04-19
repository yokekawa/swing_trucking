"""SAM2-based automatic bat-tip tracking (v5).

Workflow:
    1. User selects the swing range (scrub window).
    2. User clicks ONE point on the bat tip on the start frame.
    3. SAM2 video predictor segments and propagates that object through the
       whole swing. For each frame, the tracked point is the endpoint of the
       mask's long axis (PCA via minAreaRect) nearest the previous tip.
    4. Output PNG: mean-blended composite of every processed frame as
       background + smooth trajectory.

SAM2 ships its own Python package and model weights. They are NOT in
requirements.txt because torch + sam2 is a heavy install that isn't
required for v2-v4. Install with:

    pip install torch torchvision
    pip install "git+https://github.com/facebookresearch/sam2.git"

The tiny checkpoint (~40 MB) is downloaded automatically on first run to
models/sam2.1_hiera_tiny.pt. On CPU, inference is slow — expect several
seconds per frame. Use --device cuda if a GPU is available.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------
try:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "SAM2 dependencies missing. Install with:\n"
        '  pip install torch torchvision\n'
        '  pip install "git+https://github.com/facebookresearch/sam2.git"\n'
    )
    raise

# ---------------------------------------------------------------------------
# Checkpoint configuration
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINTS = {
    "tiny": {
        "file": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    },
    "small": {
        "file": "sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "file": "sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    },
}


def _ensure_checkpoint(model_size: str) -> tuple[Path, str]:
    info = CHECKPOINTS[model_size]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / info["file"]
    if not ckpt_path.exists():
        print(f"[sam2] downloading {info['file']} ...", file=sys.stderr)

        def _progress(blocks: int, block_size: int, total: int) -> None:
            if total > 0:
                pct = min(100, 100 * blocks * block_size // total)
                print(f"\r  {pct:3d}% ", end="", file=sys.stderr, flush=True)

        urllib.request.urlretrieve(info["url"], ckpt_path, reporthook=_progress)
        print("", file=sys.stderr)
    return ckpt_path, info["config"]


# ---------------------------------------------------------------------------
# Reused UI helpers (same as v3/v4)
# ---------------------------------------------------------------------------
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
        "[scrub]  s=start  e=end  Enter=confirm  q=abort", file=sys.stderr
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
            print("end must be after start.", file=sys.stderr)
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)
    cap.release()
    cv2.destroyWindow(window)
    return state["start"], state["end"]


def _click_select_point(video_path: Path, frame_idx: int) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}.")

    pt: list[Optional[tuple[int, int]]] = [None]
    window = "Click the bat tip  [Enter]=OK  [q]=quit"

    def on_mouse(event: int, x: int, y: int, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt[0] = (x, y)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    cv2.setMouseCallback(window, on_mouse)
    print("[seed]  click the bat tip, then Enter.", file=sys.stderr)

    while True:
        display = frame.copy()
        if pt[0] is not None:
            cv2.circle(display, pt[0], 12, (0, 255, 0), 2)
            cv2.drawMarker(display, pt[0], (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and pt[0] is not None:
            break
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(1)
    cv2.destroyWindow(window)
    return pt[0]


# ---------------------------------------------------------------------------
# Mask -> tip point
# ---------------------------------------------------------------------------
def _mask_to_tip(
    mask: np.ndarray,
    prev_tip: Optional[tuple[float, float]],
) -> Optional[tuple[float, float]]:
    mask = mask.astype(np.uint8)
    if mask.sum() < 25:
        return None
    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    edges = [(i, (i + 1) % 4, float(np.linalg.norm(box[i] - box[(i + 1) % 4]))) for i in range(4)]
    edges.sort(key=lambda e: e[2])
    mid1 = (box[edges[0][0]] + box[edges[0][1]]) / 2.0
    mid2 = (box[edges[1][0]] + box[edges[1][1]]) / 2.0
    if prev_tip is None:
        pick = mid1
    else:
        prev = np.array(prev_tip, dtype=np.float32)
        pick = mid1 if np.linalg.norm(mid1 - prev) <= np.linalg.norm(mid2 - prev) else mid2
    return float(pick[0]), float(pick[1])


# ---------------------------------------------------------------------------
# SAM2 tracking
# ---------------------------------------------------------------------------
def _extract_frames(video_path: Path, start: int, end: int, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    count = 0
    for _fi in range(start, end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{count:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1
    cap.release()
    return count


def _track_sam2(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    seed_xy: tuple[int, int],
    model_size: str,
    device: str,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    ckpt_path, config_name = _ensure_checkpoint(model_size)
    print(f"[sam2] loading model ({model_size}) on {device} ...", file=sys.stderr)
    predictor = build_sam2_video_predictor(config_name, str(ckpt_path), device=device)

    with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp:
        tmp_path = Path(tmp)
        n = _extract_frames(video_path, start_frame, end_frame, tmp_path)
        print(f"[sam2] extracted {n} frames; running propagation ...", file=sys.stderr)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else torch.autocast(device_type="cpu", dtype=torch.float32)
        )
        with torch.inference_mode(), autocast_ctx:
            state = predictor.init_state(video_path=str(tmp_path))
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=np.array([[seed_xy[0], seed_xy[1]]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

            rows: list[dict] = []
            frames_bgr: list[np.ndarray] = []
            prev_tip: Optional[tuple[float, float]] = (float(seed_xy[0]), float(seed_xy[1]))

            for out_idx, _obj_ids, out_logits in predictor.propagate_in_video(state):
                mask = (out_logits[0] > 0.0).detach().cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                tip = _mask_to_tip(mask, prev_tip)
                if tip is None:
                    tip = prev_tip if prev_tip is not None else (float(seed_xy[0]), float(seed_xy[1]))
                    confidence = 0.0
                else:
                    confidence = 1.0
                    prev_tip = tip

                rows.append({
                    "frame_idx": start_frame + out_idx,
                    "tip_x": tip[0],
                    "tip_y": tip[1],
                    "confidence": confidence,
                })
                frame = cv2.imread(str(tmp_path / f"{out_idx:05d}.jpg"))
                if frame is not None:
                    frames_bgr.append(frame)

                if out_idx % 10 == 0 or out_idx == n - 1:
                    print(f"  frame {out_idx + 1}/{n}", file=sys.stderr)

    return pd.DataFrame(rows), frames_bgr


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _composite(frames: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.float32)
    if mode == "max":
        return np.max(stack, axis=0).astype(np.uint8)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def _render_png(
    trajectory: pd.DataFrame,
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

    df = trajectory[trajectory["confidence"] > 0]
    if len(df) >= 2:
        xs = df["tip_x"].to_numpy()
        ys = df["tip_y"].to_numpy()
        ts = np.linspace(0.0, 1.0, len(df))
        points = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_time = (ts[:-1] + ts[1:]) / 2.0
        cmap = plt.get_cmap("plasma")
        colors = cmap(seg_time)
        colors[:, 3] = 0.9
        lc = LineCollection(segments, linewidths=4.5, colors=colors, capstyle="round", joinstyle="round")
        ax.add_collection(lc)
        ax.scatter(xs[0], ys[0], c="lime", s=90, edgecolors="black", zorder=5, label="start")
        ax.scatter(xs[-1], ys[-1], c="red", s=90, edgecolors="black", zorder=5, label="end")
        ax.legend(loc="upper right", framealpha=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SAM2-based bat-tip tracker (v5).")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/trajectory.png"))
    parser.add_argument("--start-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--seed-x", type=int)
    parser.add_argument("--seed-y", type=int)
    parser.add_argument("--model", choices=list(CHECKPOINTS.keys()), default="tiny",
                        help="SAM2 checkpoint size (default: tiny)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--composite", choices=("mean", "max"), default="mean")
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
    print(f"swing range: frames {start_frame}-{end_frame}  fps={fps:.1f}", file=sys.stderr)

    if args.seed_x is not None and args.seed_y is not None:
        seed = (args.seed_x, args.seed_y)
    else:
        seed = _click_select_point(args.input, start_frame)
    print(f"seed point: {seed}", file=sys.stderr)

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    trajectory, frames = _track_sam2(args.input, start_frame, end_frame, seed, args.model, device)

    total = len(trajectory)
    hits = int(trajectory["confidence"].ge(0.5).sum())
    print(f"tracked frames={total}  valid_masks={hits} ({hits / max(total, 1):.0%})", file=sys.stderr)

    background = _composite(frames, args.composite) if frames else np.zeros((100, 100, 3), np.uint8)
    _render_png(trajectory, background, args.output)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
