"""Streamlit UI for the baseball swing trajectory visualiser."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from swing_analyzer import analyze_video, compute_metrics
from visualizer import plot_speed_profile, plot_trajectory_2d, plot_trajectory_3d

st.set_page_config(page_title="Swing Trajectory Visualizer", layout="wide")

st.title("野球スイング軌道 可視化アプリ")
st.caption(
    "動画をアップロードすると、MediaPipe Pose で両手首の軌道を抽出し、"
    "2D / 3D でスイングの軌跡を可視化します。"
)

with st.sidebar:
    st.header("入力設定")
    uploaded = st.file_uploader("スイング動画 (mp4 / mov)", type=["mp4", "mov", "m4v", "avi"])
    sensitivity = st.slider(
        "検出感度 (min_detection_confidence)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )
    st.markdown(
        "**撮影ガイド**\n"
        "- 横から全身が入る構図\n"
        "- 60fps 以上推奨\n"
        "- 手元が明るく写っていること",
    )


@st.cache_data(show_spinner=False)
def _run_analysis(video_bytes: bytes, sensitivity: float):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)

    progress = st.progress(0.0, text="スイングを解析中...")

    def _update(p: float) -> None:
        progress.progress(p, text=f"スイングを解析中... {int(p * 100)}%")

    try:
        data = analyze_video(
            str(tmp_path),
            min_detection_confidence=sensitivity,
            progress_callback=_update,
        )
    finally:
        progress.empty()
        tmp_path.unlink(missing_ok=True)

    metrics = compute_metrics(data)
    return data, metrics


if uploaded is None:
    st.info("サイドバーから動画をアップロードしてください。")
    st.stop()

video_bytes = uploaded.getvalue()
st.video(video_bytes)

with st.spinner("解析を実行しています..."):
    data, metrics = _run_analysis(video_bytes, sensitivity)

st.success(
    f"解析完了: {len(data.frames)} フレーム / "
    f"{data.fps:.1f} fps / スイング区間 "
    f"{data.swing_start}-{data.swing_end}"
)

tab_2d, tab_3d, tab_metrics = st.tabs(["2D 軌道", "3D 軌道", "メトリクス"])

with tab_2d:
    st.pyplot(plot_trajectory_2d(data))
    st.caption("両手首の中点を pixel 座標で追跡した 2D 軌道です。色は時間の進行を表します。")

with tab_3d:
    st.plotly_chart(plot_trajectory_3d(data), use_container_width=True)
    st.caption(
        "MediaPipe world landmarks による 3D 軌道 (腰中心を原点とした相対座標)。"
        "Z 軸方向の精度は限定的である点にご留意ください。"
    )

with tab_metrics:
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "最大スイング速度",
        f"{metrics['max_speed_m_s']:.2f} m/s" if metrics["max_speed_m_s"] == metrics["max_speed_m_s"] else "n/a",
    )
    col2.metric("スイング所要時間", f"{metrics['duration_s']:.2f} s")
    col3.metric(
        "スイング面の傾き",
        f"{metrics['swing_plane_tilt_deg']:.1f}°"
        if metrics["swing_plane_tilt_deg"] == metrics["swing_plane_tilt_deg"]
        else "n/a",
    )

    col4, col5 = st.columns(2)
    col4.metric(
        "3D 軌道長",
        f"{metrics['path_length_m']:.2f} m"
        if metrics["path_length_m"] == metrics["path_length_m"]
        else "n/a",
    )
    col5.metric(
        "最大 pixel 速度",
        f"{metrics['max_speed_px_s']:.0f} px/s"
        if metrics["max_speed_px_s"] == metrics["max_speed_px_s"]
        else "n/a",
    )

    st.pyplot(plot_speed_profile(data))
