import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple
from ultralytics import YOLO
import torch
import cv2
import imageio_ffmpeg

# Configuration
MODEL_PATH = "models/ppe.pt"
RESULTS_DIR = Path("results/detections")
METRICS_FILE = Path("results/metrics.json")

# Class labels
LABELS = {
    0: 'Hardhat',
    1: 'Mask',
    2: 'NO-Hardhat',
    3: 'NO-Mask',
    4: 'NO-Safety Vest',
    5: 'Person',
    6: 'Safety Cone',
    7: 'Safety Vest',
    8: 'machinery',
    9: 'vehicle'
}

# Image extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Initialize model
model = YOLO(MODEL_PATH)

# Create necessary directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_device_info() -> Dict[str, object]:
    """Detect whether a CUDA GPU is available right now on this machine.

    Used by the UI to pick sensible defaults automatically: fast/high-quality
    settings when a real GPU is present (e.g. running locally on your own
    machine), and speed-friendly settings when only a CPU is available
    (e.g. once deployed on Streamlit Community Cloud, which has no GPU).
    """
    has_gpu = torch.cuda.is_available()
    return {
        "has_gpu": has_gpu,
        "device_name": torch.cuda.get_device_name(0) if has_gpu else "CPU",
    }


def _make_browser_playable(raw_path: Path, final_path: Path) -> str:
    """Re-encode a cv2-written video into H.264 so it plays inline in
    browsers (st.video/Chrome only reliably support avc1/H.264 — cv2's
    default 'mp4v' fourcc plays fine in VLC/Windows Media Player but shows
    as a black, frozen player inside a web <video> tag).

    Falls back to the raw (cv2) file if ffmpeg re-encoding fails for any
    reason, so the pipeline never hard-crashes over a playback nicety.
    """
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(raw_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(final_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        raw_path.unlink(missing_ok=True)
        return str(final_path)
    except Exception as e:
        print(f"⚠️ Browser re-encode failed ({e}); serving the original file (may not preview in-browser).")
        return str(raw_path)


def detect_single_image(img_path: str) -> Tuple[Dict[str, int], str]:
    """Detect objects in a single image and save annotated result."""
    img_path = Path(img_path)
    save_path = RESULTS_DIR / img_path.name
    
    # Run detection
    results = model(str(img_path))[0]
    
    # Count detections by label
    label_counts = {}
    for box in results.boxes:
        cls = int(box.cls[0])
        label = LABELS.get(cls, "Unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Save annotated image
    annotated_img = results.plot()
    cv2.imwrite(str(save_path), annotated_img)
    
    return label_counts, str(save_path)


def run_image_folder(folder_path: str) -> Dict[str, int]:
    """Run detection on all images in a folder."""
    folder_path = Path(folder_path)
    metrics = {}
    
    # Process all images
    for img_path in folder_path.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        
        label_counts, _ = detect_single_image(str(img_path))
        
        # Accumulate counts
        for label, count in label_counts.items():
            metrics[label] = metrics.get(label, 0) + count
    
    # Save metrics
    save_metrics(metrics)
    return metrics


def detect_video(
    video_path: str,
    output_name: str = "output_video.mp4",
    save: bool = True,
    unique_per_frame: bool = True,
    frame_skip: int = 1,
    imgsz: int = None,
    device: str = None,
    half: bool = False
) -> Tuple[str, Dict[str, int]]:
    """Run detection on video, save annotated result, and tally PPE metrics.

    Args:
        video_path: Path to input video file
        output_name: Name for output video file
        save: If True, persist the aggregated metrics to METRICS_FILE
              (same file the batch/image flow uses) so the dashboard's
              AI Suggestions page can pick them up.
        unique_per_frame: If True (default), each label is counted at most
              once per frame instead of once per box, avoiding inflated
              counts on long clips.
        frame_skip: Run the model every Nth frame (default 1 = every frame).
              Skipped frames reuse the last detected boxes for drawing and
              counting. E.g. frame_skip=3 runs the model roughly 3x less
              often — a big speedup with only a small accuracy trade-off on
              typical (non-fast-motion) footage.
        imgsz: Resize frames to this size (e.g. 640) before running the
              model. Smaller = faster, at some cost to small-object
              accuracy. Leave as None to use the model's default size.
        device: Force a device, e.g. "cuda:0" or "cpu". Leave as None to
              let Ultralytics auto-pick (GPU if available).
        half: If True, run inference in FP16 (only effective on GPU) for
              an extra speed boost with negligible accuracy impact.

    Returns:
        Tuple of (output_video_path, metrics) where metrics is a
        Dict[str, int] with the same shape as detect_single_image /
        run_image_folder (e.g. {"NO-Hardhat": 12, "Hardhat": 40, ...}).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # cv2 writes with 'mp4v' which plays fine in VLC/Windows Media Player but
    # not inside a browser <video> tag — write to a raw temp file first, then
    # re-encode to H.264 for the final, browser-playable output.
    final_output_path = RESULTS_DIR / output_name
    raw_output_path = RESULTS_DIR / f"_raw_{output_name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_output_path), fourcc, fps, (width, height))
    
    print(f"   Total frames: {total_frames}")
    print(f"   Resolution: {width}x{height} @ {fps} FPS")
    if frame_skip > 1:
        print(f"   Speed mode: analyzing every {frame_skip} frame(s)")
    print()
    
    metrics: Dict[str, int] = {}
    predict_kwargs = {}
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    if device is not None:
        predict_kwargs["device"] = device
    if half:
        predict_kwargs["half"] = True
    
    try:
        frame_count = 0
        last_results = None  # reused on skipped frames
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            run_detection_this_frame = (
                frame_skip <= 1 or (frame_count - 1) % frame_skip == 0 or last_results is None
            )
            
            if run_detection_this_frame:
                results = model(frame, **predict_kwargs)[0]
                last_results = results
            else:
                results = last_results
            
            annotated_frame = results.plot(img=frame if not run_detection_this_frame else None)
            writer.write(annotated_frame)
            
            # Tally detections for this frame (skipped frames inherit the last tally too)
            if unique_per_frame:
                seen_labels_this_frame = set()
                for box in results.boxes:
                    cls = int(box.cls[0])
                    label = LABELS.get(cls, "Unknown")
                    seen_labels_this_frame.add(label)
                for label in seen_labels_this_frame:
                    metrics[label] = metrics.get(label, 0) + 1
            else:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    label = LABELS.get(cls, "Unknown")
                    metrics[label] = metrics.get(label, 0) + 1
            
            # Progress indicator
            progress = (frame_count / total_frames) * 100
            remaining = total_frames - frame_count
            print(f"\r   Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | Remaining: {remaining} frames", end="")
        
        print() 
        
    finally:
        cap.release()
        writer.release()
    
    output_path = _make_browser_playable(raw_output_path, final_output_path)
    
    if save:
        save_metrics(metrics)
    
    return output_path, metrics


def save_metrics(metrics: Dict[str, int]) -> None:
    """Save metrics to JSON file."""
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)


def load_metrics() -> Dict[str, int]:
    """Load metrics from JSON file."""
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def calculate_percentages(metrics: Dict[str, int]) -> Dict[str, float]:
    """Calculate PPE non-compliance percentages."""
    
    def calc_rate(non_compliant: int, compliant: int) -> float:
        """Calculate non-compliance percentage."""
        total = non_compliant + compliant
        return round((non_compliant / total) * 100, 2) if total > 0 else 0.0
    
    rates = {}
    
    # Hardhat compliance
    if "Hardhat" in metrics or "NO-Hardhat" in metrics:
        rates["no_hardhat_rate"] = calc_rate(
            metrics.get("NO-Hardhat", 0),
            metrics.get("Hardhat", 0)
        )
    
    # Mask compliance
    if "Mask" in metrics or "NO-Mask" in metrics:
        rates["no_mask_rate"] = calc_rate(
            metrics.get("NO-Mask", 0),
            metrics.get("Mask", 0)
        )
    
    # Safety vest compliance
    if "Safety Vest" in metrics or "NO-Safety Vest" in metrics:
        rates["no_safety_vest_rate"] = calc_rate(
            metrics.get("NO-Safety Vest", 0),
            metrics.get("Safety Vest", 0)
        )
    
    return rates