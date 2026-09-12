from pathlib import Path
from typing import Dict
import io
import math
import struct
import wave
import streamlit as st
import matplotlib.pyplot as plt

from detect import (
    run_image_folder,
    detect_single_image,
    detect_video,
    load_metrics,
    calculate_percentages,
    get_device_info
)
from suggestion import generate_safety_suggestions

# Configuration
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="PPE Detection Dashboard",
        layout="wide",
        page_icon="🦺"
    )


def render_sidebar():
    """Render sidebar navigation menu."""
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Go to:",
        ["📊 Batch Analysis", "🖼️ Test Single Image", "🎥 Video Analysis", "💡 AI Suggestions"]
    )


def render_metric_cards(metrics: Dict[str, int], color: str = "#4b8bff"):
    """Render metric cards in a grid layout."""
    cols = st.columns(min(len(metrics), 4))
    
    emoji_map = {
        "NO-Hardhat": "⛑️",
        "NO-Mask": "😷",
        "NO-Safety Vest": "🦺",
        "Hardhat": "⛑️",
        "Mask": "😷",
        "Safety Vest": "🦺",
        "Person": "👤",
        "Safety Cone": "🚧",
        "machinery": "⚙️",
        "vehicle": "🚗"
    }
    
    for idx, (label, count) in enumerate(metrics.items()):
        with cols[idx % len(cols)]:
            emoji = emoji_map.get(label, "📋")
            
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                    border-left: 4px solid {color};
                    border-radius: 12px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    <div style="font-size: 32px; margin-bottom: 8px;">{emoji}</div>
                    <div style="font-size: 28px; font-weight: 700; color: {color}; margin-bottom: 4px;">{count}</div>
                    <div style="font-size: 14px; color: var(--text-color); font-weight: 500;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_percentage_cards(percentages: Dict[str, float]):
    """Render percentage violation cards."""
    cols = st.columns(min(len(percentages), 3))
    
    label_map = {
        "no_hardhat_rate": "No Hardhat",
        "no_mask_rate": "No Mask",
        "no_safety_vest_rate": "No Safety Vest"
    }
    
    for idx, (key, value) in enumerate(percentages.items()):
        with cols[idx % len(cols)]:
            label = label_map.get(key, key.replace("_", " ").title())
            
            # Color based on severity
            if value >= 50:
                color = "#ff4b4b"
                severity = "Critical"
            elif value >= 25:
                color = "#ffa500"
                severity = "High"
            elif value >= 10:
                color = "#ffcc00"
                severity = "Medium"
            else:
                color = "#00cc66"
                severity = "Low"
            
            st.markdown(
                f"""
                <div style="
                    background: rgba(128, 128, 128, 0.05);
                    border: 2px solid {color};
                    border-radius: 16px;
                    padding: 24px;
                    margin: 10px 0;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    text-align: center;
                ">
                    <div style="font-size: 48px; font-weight: 800; color: {color}; margin-bottom: 8px;">{value}%</div>
                    <div style="font-size: 16px; color: var(--text-color); font-weight: 600; margin-bottom: 8px;">{label}</div>
                    <div style="
                        display: inline-block;
                        background: {color}20;
                        color: {color};
                        padding: 4px 12px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: 600;
                    ">{severity} Risk</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def display_metrics(metrics: Dict[str, int]):
    """Display metrics in organized categories."""
    st.subheader("Detection Metrics")
    
    # Categorize metrics
    ppe_compliant = {}
    ppe_violations = {}
    other_detections = {}
    
    for label, count in metrics.items():
        if label.startswith("NO-"):
            ppe_violations[label] = count
        elif label in ["Hardhat", "Mask", "Safety Vest"]:
            ppe_compliant[label] = count
        else:
            other_detections[label] = count
    
    # Display categories
    if ppe_violations:
        st.markdown("### ⚠️ PPE Violations")
        render_metric_cards(ppe_violations, color="#ff4b4b")
    
    if ppe_compliant:
        st.markdown("### ✅ PPE Compliance")
        render_metric_cards(ppe_compliant, color="#00cc66")
    
    if other_detections:
        st.markdown("### 🔍 Other Detections")
        render_metric_cards(other_detections, color="#4b8bff")
    
    # Display violation percentages
    percentages = calculate_percentages(metrics)
    if percentages:
        st.markdown("---")
        st.markdown("### 📊 Violation Rates")
        render_percentage_cards(percentages)


def display_chart(metrics: Dict[str, int]):
    """Display bar chart of metrics."""
    if not metrics:
        return
    
    st.markdown("---")
    st.subheader("📈 Visual Analytics")
    
    labels = list(metrics.keys())
    counts = list(metrics.values())
    
    # Color code bars
    colors = []
    for label in labels:
        if label.startswith("NO-"):
            colors.append('#ff4b4b')
        elif label in ["Hardhat", "Mask", "Safety Vest"]:
            colors.append('#00cc66')
        else:
            colors.append('#4b8bff')
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Style the chart
    ax.set_title("PPE Detection Metrics", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("PPE Categories", fontsize=13, fontweight='600')
    ax.set_ylabel("Count", fontsize=13, fontweight='600')
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_batch_analysis():
    """Render batch image analysis page."""
    st.header("📊 Batch Image Analysis")
    
    folder_path = st.text_input(
        "Folder path containing test images:",
        value="images/test_images"
    )
    
    if st.button("Run Batch Analysis", type="primary"):
        if not Path(folder_path).exists():
            st.error(f"Folder not found: {folder_path}")
            return
        
        with st.spinner("Running detection on all images..."):
            try:
                metrics = run_image_folder(folder_path)
                st.success("Analysis completed!")
                
                display_metrics(metrics)
                display_chart(metrics)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")


def generate_alarm_sound(frequency: int = 950, beep_ms: int = 220, gap_ms: int = 120, repeats: int = 3) -> bytes:
    """Generate a short WAV alarm beep in-memory (no external audio file needed)."""
    sample_rate = 44100
    amplitude = 26000  # keep under int16 max to avoid clipping

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for _ in range(repeats):
            # Tone
            n_samples = int(sample_rate * (beep_ms / 1000.0))
            for i in range(n_samples):
                t = i / sample_rate
                value = int(amplitude * math.sin(2 * math.pi * frequency * t))
                wav_file.writeframesraw(struct.pack("<h", value))
            # Silence gap
            n_silence = int(sample_rate * (gap_ms / 1000.0))
            for _ in range(n_silence):
                wav_file.writeframesraw(struct.pack("<h", 0))

    return buf.getvalue()


def render_video_analysis():
    """Render video upload + PPE detection page, with an alarm if violations are found."""
    st.header("🎥 Video PPE Analysis")

    uploaded_video = st.file_uploader(
        "Upload a video to analyze",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if not uploaded_video:
        st.info("👆 Upload a video to get started")
        return

    temp_video_path = TEMP_DIR / uploaded_video.name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.markdown("#### Original Video")
    st.video(str(temp_video_path))

    device_info = get_device_info()
    if device_info["has_gpu"]:
        st.caption(f"⚡ GPU detected ({device_info['device_name']}) — using high-quality defaults.")
        default_frame_skip, default_imgsz = 1, 640
    else:
        st.caption("🖥️ Running on CPU (e.g. Streamlit Cloud) — using speed-friendly defaults so it stays usable for every visitor.")
        default_frame_skip, default_imgsz = 3, 480

    col_a, col_b = st.columns(2)
    with col_a:
        unique_per_frame = st.checkbox(
            "Count each violation once per frame (recommended)",
            value=True,
            help="Off counts every single detected box across all frames, which inflates numbers for long clips."
        )
    with col_b:
        frame_skip = st.slider(
            "Analyze every Nth frame (speed vs accuracy)",
            min_value=1, max_value=10, value=default_frame_skip,
            help="1 = analyze every frame (most accurate, slowest). Higher values skip frames and reuse the last detection, which is much faster."
        )

    imgsz = st.select_slider(
        "Inference size (smaller = faster)",
        options=[320, 480, 640, 960, 1280],
        value=default_imgsz,
        help="Frames are resized to this size before detection. 640 is the model's typical training size."
    )

    if st.button("Run Video Detection", type="primary"):
        with st.spinner("Running detection on video... this can take a while depending on length."):
            try:
                output_path, metrics = detect_video(
                    str(temp_video_path),
                    output_name=f"annotated_{uploaded_video.name}",
                    unique_per_frame=unique_per_frame,
                    frame_skip=frame_skip,
                    imgsz=imgsz,
                    half=device_info["has_gpu"]
                )

                st.success("Video processing complete!")

                total_violations = sum(v for k, v in metrics.items() if k.startswith("NO-"))

                # 🚨 Alarm if violators were detected
                if total_violations > 0:
                    st.error(f"🚨 {total_violations} violation frame(s) detected! Alarm triggered.")
                    alarm_bytes = generate_alarm_sound()
                    st.audio(alarm_bytes, format="audio/wav", autoplay=True)
                else:
                    st.success("✅ No PPE violations detected in this video.")

                st.markdown("---")
                st.markdown("#### Annotated Video")
                st.video(output_path)

                display_metrics(metrics)
                display_chart(metrics)

            except Exception as e:
                st.error(f"Error during video detection: {str(e)}")



def render_single_image():
    """Render single image test page."""
    st.header("🖼️ Test a Single Image")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    
    if not uploaded_file:
        st.info("👆 Upload an image to get started")
        return
    
    # Save uploaded file temporarily
    temp_path = TEMP_DIR / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Image")
        st.image(str(temp_path), use_container_width=True)
    
    if st.button("Run Detection", type="primary"):
        with st.spinner("Detecting PPE..."):
            try:
                label_counts, annotated_path = detect_single_image(str(temp_path))
                
                st.success("Detection successful!")
                
                # Display annotated image
                with col2:
                    st.markdown("#### Detected Objects")
                    st.image(annotated_path, use_container_width=True)
                
                # Display detections
                st.markdown("---")
                st.markdown("### Detection Results")
                render_metric_cards(label_counts, color="#4b8bff")
                
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")


def render_ai_suggestions():
    """Render AI suggestions page."""
    st.header("💡 AI Safety Recommendations")
    
    metrics = load_metrics()
    
    if not metrics:
        st.warning("No metrics found! Run batch analysis first.")
        return
    
    # Display current status overview
    percentages = calculate_percentages(metrics)
    st.markdown("### Current Status Overview")
    cols = st.columns(3)
    
    total_detections = sum(metrics.values())
    total_violations = sum(v for k, v in metrics.items() if k.startswith("NO-"))
    violation_rate = (total_violations / total_detections * 100) if total_detections > 0 else 0
    
    with cols[0]:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #4b8bff15 0%, #4b8bff05 100%);
                border-left: 4px solid #4b8bff;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size: 36px; font-weight: 700; color: #4b8bff;">{total_detections}</div>
                <div style="font-size: 14px; color: var(--text-color); font-weight: 500;">Total Detections</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with cols[1]:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #ff4b4b15 0%, #ff4b4b05 100%);
                border-left: 4px solid #ff4b4b;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size: 36px; font-weight: 700; color: #ff4b4b;">{total_violations}</div>
                <div style="font-size: 14px; color: var(--text-color); font-weight: 500;">Total Violations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with cols[2]:
        color = "#ff4b4b" if violation_rate >= 25 else "#ffa500" if violation_rate >= 10 else "#00cc66"
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                border-left: 4px solid {color};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size: 36px; font-weight: 700; color: {color};">{violation_rate:.1f}%</div>
                <div style="font-size: 14px; color: var(--text-color); font-weight: 500;">Violation Rate</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Generate suggestions
    if st.button("Generate AI Recommendations", type="primary"):
        with st.spinner("Generating expert safety suggestions..."):
            try:
                suggestions = generate_safety_suggestions(metrics, percentages)
                
                st.success("Recommendations generated!")
                st.markdown("### 📋 Safety Recommendations")
                
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(128, 128, 128, 0.05);
                        border: 2px solid rgba(128, 128, 128, 0.2);
                        border-radius: 16px;
                        padding: 30px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        line-height: 1.8;
                        color: var(--text-color);
                    ">
                        <div style="white-space: pre-wrap;">{suggestions}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Error generating suggestions: {str(e)}")


def main():
    """Main entry point for the dashboard."""
    configure_page()
    
    st.title("🦺 PPE Detection Dashboard")
    st.write("Analyze PPE compliance using your pretrained YOLOv8 model.")
    
    # Sidebar navigation
    menu = render_sidebar()
    
    # Route to appropriate page
    if menu == "📊 Batch Analysis":
        render_batch_analysis()
    elif menu == "🖼️ Test Single Image":
        render_single_image()
    elif menu == "🎥 Video Analysis":
        render_video_analysis()
    elif menu == "💡 AI Suggestions":
        render_ai_suggestions()


if __name__ == "__main__":
    main()