"""Streamlit interactive interface for Video Anomaly Detection Using Deep Features and Multi-Prior Fusion."""

import importlib
import inspect
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

for mod_name in [
    "vad.config",
    "vad.data.video_loader",
    "vad.models.videomae_encoder",
    "vad.models.dinov2_encoder",
    "vad.inference.predictor",
    "vad.inference.visualization",
]:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

from vad.device import resolve_device
from vad.inference.predictor import VideoAnomalyPredictor
from vad.inference.visualization import export_annotated_video, plot_anomaly_timeline


def render_app() -> None:
    st.set_page_config(
        page_title="Video Anomaly Detection — Multi-Prior Fusion",
        layout="wide",
    )

    st.title("Video Anomaly Detection Using Deep Features and Multi-Prior Fusion")
    st.markdown(
        "**Architecture:** VideoMAE Spatiotemporal Encoder + DINOv2 Appearance Encoder "
        "+ Dilated Residual TCN + Learned Multi-Prior Gated Fusion"
    )

    st.info(
        "**Ethical & Operational Disclaimer:** High anomaly scores represent statistical deviations "
        "in visual or temporal motion patterns learned by the model. They are **model predictions**, "
        "**not** proof of criminal activity, malicious intent, or dangerous behavior."
    )
    st.warning(
        "**Hardware Notice:** Feature extraction and sliding-window inference speed depend heavily "
        "on your hardware (CPU vs. GPU) and video duration. On laptops without an NVIDIA CUDA GPU, "
        "use a larger window stride (e.g., 16 frames) for faster processing."
    )

    with st.sidebar:
        st.header("Inference Configuration")
        device_choice = st.selectbox(
            "Execution Device",
            options=["auto", "cpu", "cuda", "directml", "mps"],
            index=0,
        )
        dev_info = resolve_device(device_choice)
        st.caption(f"Resolved Hardware: `{dev_info.device}` ({dev_info.device_name})")

        default_ckpt = (
            "checkpoints/best.pt"
            if (PROJECT_ROOT / "checkpoints/best.pt").exists()
            else "checkpoints/synthetic_demo/best.pt"
        )
        checkpoint_path_str = st.text_input(
            "Model Checkpoint Path (.pt)",
            value=default_ckpt,
        )

        use_synthetic_encoders = st.checkbox(
            "Synthetic Smoke-Test Mode (No 700MB Hugging Face Download)",
            value=not (PROJECT_ROOT / "checkpoints/best.pt").exists(),
            help="When enabled, uses deterministic frame statistics instead of downloading VideoMAE/DINOv2 weights.",
        )

        ingest_res_choice = st.selectbox(
            "Video Ingest Resolution",
            options=[
                "384p (Fast & Low Memory — Recommended)",
                "480p (Standard Surveillance)",
                "720p (High Res — High RAM/GPU)",
                "Native (Original Video Size)",
            ],
            index=0,
            help="Surveillance foundation models (VideoMAE & DINOv2) operate at 224x224. Downscaling HD video upon decoding avoids multi-gigabyte memory exhaustion and runs inference much faster on CPU.",
        )
        res_map = {
            "384p (Fast & Low Memory — Recommended)": 384,
            "480p (Standard Surveillance)": 480,
            "720p (High Res — High RAM/GPU)": 720,
            "Native (Original Video Size)": None,
        }
        max_dim = res_map[ingest_res_choice]

        window_length = st.slider("Temporal Window Length (frames)", 8, 32, 16, step=4)
        window_stride = st.slider(
            "Sliding Window Stride (frames)",
            2,
            32,
            16,
            step=2,
            help="Stride 16 processes twice as fast as stride 8 with negligible detection loss on CPU.",
        )
        smoothing_window = st.slider("Temporal Smoothing Window (clips)", 1, 9, 3, step=2)
        threshold = st.slider("Visualization Threshold", 0.05, 0.95, 0.50, step=0.05)
        generate_video_overlay = st.checkbox("Export Annotated Video Overlay (.mp4)", value=True)

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Surveillance Video (.mp4, .avi, .mov, .mkv)",
            type=["mp4", "avi", "mov", "mkv"],
        )
    with col_sample:
        sample_dirs = [PROJECT_ROOT / "demo_samples", PROJECT_ROOT / "data/raw"]
        sample_videos = []
        for s_dir in sample_dirs:
            if s_dir.exists():
                sample_videos.extend(sorted(s_dir.rglob("*.mp4")))
        sample_options = ["(None — use uploaded video)"] + [
            str(p.relative_to(PROJECT_ROOT)) for p in sample_videos
        ]
        selected_sample = st.selectbox("Or Select Demo Surveillance Video", options=sample_options)

    run_btn = st.button("Run Anomaly Detection Pipeline", type="primary")

    if not run_btn:
        return

    ckpt_path = Path(checkpoint_path_str)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    if not ckpt_path.exists():
        st.error(
            f"Checkpoint file not found at `{ckpt_path}`. "
            "Please train a checkpoint via `python scripts/train.py` or run "
            "`python scripts/generate_demo.py` to generate a synthetic smoke-test checkpoint."
        )
        return

    target_video_path: Path | None = None
    temp_file_path: Path | None = None

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_file_path = Path(tmp.name)
            target_video_path = temp_file_path
    elif selected_sample != "(None — use uploaded video)":
        target_video_path = PROJECT_ROOT / selected_sample
    else:
        st.error("Please upload a video file or select a workspace video before running inference.")
        return

    progress = st.progress(10, text="Loading AnomalyDetector checkpoint and encoders...")
    try:
        predictor = VideoAnomalyPredictor.from_checkpoint(
            checkpoint_path=ckpt_path,
            device_override=device_choice,
            use_synthetic_encoders=use_synthetic_encoders,
        )
        progress.progress(35, text="Decoding video frames and extracting clip features...")
        pred_kwargs = {
            "video_path": target_video_path,
            "window_length": window_length,
            "window_stride": window_stride,
            "threshold": threshold,
            "smoothing_window": smoothing_window,
        }
        sig = inspect.signature(predictor.predict_video_file)
        if "max_dimension" in sig.parameters:
            pred_kwargs["max_dimension"] = max_dim
        else:
            if hasattr(predictor.config.data, "max_dimension"):
                predictor.config.data.max_dimension = max_dim

        pred_result, decoded = predictor.predict_video_file(**pred_kwargs)
        progress.progress(85, text="Rendering anomaly timeline and tabular summaries...")
    except Exception as exc:
        progress.empty()
        st.error(
            f"Inference failed: {exc}\n\n"
            "If pretrained Hugging Face weights (`MCG-NJU/videomae-base` or `facebook/dinov2-base`) "
            "are not cached locally and internet access is disabled, either enable "
            "'Synthetic Smoke-Test Mode' in the sidebar or download the weights first."
        )
        return
    finally:
        if temp_file_path is not None and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                pass

    progress.progress(100, text="Inference complete.")

    if pred_result.is_synthetic_mode:
        st.warning(
            "**SYNTHETIC SMOKE-TEST MODE ACTIVE:** These scores were generated using synthetic "
            "encoders or a synthetic smoke-test checkpoint. They verify pipeline mechanics only "
            "and do not represent pretrained VideoMAE/DINOv2 benchmark inference."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Anomaly Score", f"{float(pred_result.smoothed_clip_scores.max()):.3f}")
    m2.metric("Mean Anomaly Score", f"{float(pred_result.smoothed_clip_scores.mean()):.3f}")
    m3.metric("Total Frames / Clips", f"{pred_result.total_frames} / {len(pred_result.windows)}")
    m4.metric("Throughput", f"{pred_result.effective_fps:.1f} FPS ({pred_result.inference_time_sec:.2f}s)")

    st.subheader("Anomaly Score Timeline & Multi-Prior Gate Weights")
    fig = plot_anomaly_timeline(pred_result)
    st.pyplot(fig)
    plt.close(fig)

    clip_df = pred_result.to_clip_dataframe()
    flagged_df = clip_df[clip_df["anomaly_score"] >= threshold].sort_values(
        by="anomaly_score", ascending=False
    )

    st.subheader(f"High-Scoring Temporal Windows (Score >= {threshold:.2f})")
    if flagged_df.empty:
        st.success(f"No temporal windows exceeded the threshold of {threshold:.2f}.")
    else:
        st.dataframe(flagged_df, use_container_width=True)

    with st.expander("View All Temporal Windows"):
        st.dataframe(clip_df, use_container_width=True)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Download Clip-Level Scores (CSV)",
            data=clip_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{pred_result.video_id}_clip_scores.csv",
            mime="text/csv",
        )
    with dl_col2:
        frame_df = pred_result.to_frame_dataframe()
        st.download_button(
            label="Download Frame-Level Scores (CSV)",
            data=frame_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{pred_result.video_id}_frame_scores.csv",
            mime="text/csv",
        )

    if generate_video_overlay:
        st.subheader("Annotated Video Output")
        out_vid_dir = PROJECT_ROOT / "outputs/streamlit"
        out_vid_path = out_vid_dir / f"{pred_result.video_id}_annotated.mp4"
        export_annotated_video(decoded.frames_rgb, pred_result, out_vid_path)
        with open(out_vid_path, "rb") as vf:
            vid_bytes = vf.read()
        st.download_button(
            label="Download Annotated MP4 Video",
            data=vid_bytes,
            file_name=out_vid_path.name,
            mime="video/mp4",
        )


if __name__ == "__main__":
    render_app()
