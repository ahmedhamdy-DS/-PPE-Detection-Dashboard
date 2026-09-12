# PPE Detection Dashboard

A real-time Personal Protective Equipment (PPE) detection system built with YOLOv8 and Streamlit. The application monitors workplace safety compliance by detecting hardhats, masks, and safety vests, and flagging violations as they happen.


## Features

- **Batch Image Analysis**: Process a folder of images and generate aggregated safety metrics
- **Single Image Detection**: Upload one image and view annotated detection results
- **Video Processing**: Run frame-by-frame PPE violation detection on video files
- **Live Webcam Detection**: Monitor PPE compliance in real time through a webcam feed
- **AI-Powered Suggestions**: Generate actionable safety recommendations based on detected violations
- **Interactive Dashboard**: A clean, user-friendly interface with visual analytics

## Detection Capabilities

The model detects the following classes:

**Compliant PPE**
- Hardhat
- Mask
- Safety Vest

**Violations**
- No Hardhat
- No Mask
- No Safety Vest

**Other Objects**
- Person
- Safety Cone
- Machinery
- Vehicle

## Project Structure

```
PPE_DASHBOARD/
├── images/
│   └── test_images/          # Sample images for batch processing
├── models/
│   └── ppe.pt                # YOLOv8 trained model
├── results/
│   └── detections/           # Output folder for processed images/videos
├── temp/                     # Temporary files for uploads
├── videos/
│   └── test_videos/          # Video files for processing
│       ├── ppe-1.mp4
│       ├── ppe-2.mp4
│       └── ppe-3.mp4
├── .env                       # Environment variables (API keys)
├── app.py                     # Main Streamlit dashboard
├── detect.py                  # Core detection functions
├── suggestion.py               # AI-powered safety suggestions
├── run_video.py                # Standalone video processing script
├── webcam.py                   # Live webcam detection
├── requirements.txt            # Python dependencies
└── README.md
```

## Dashboard Overview

### Batch Analysis
- Upload a folder of images
- View aggregated safety metrics
- Review violation rates and compliance statistics
- Export results to JSON

### Single Image Detection
- Upload an individual image
- View annotated detections instantly
- See detailed object counts

### AI Safety Recommendations
- Analyze compliance data
- Receive AI-generated safety suggestions
- Get actionable insights for improving workplace safety

## Configuration

### Model Settings

Edit `detect.py` to customize the model path and output directory:

```python
MODEL_PATH = "models/ppe.pt"
RESULTS_DIR = Path("results/detections")
```

### Class Labels

The system detects 10 classes:

```python
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
```

## Dependencies

- `ultralytics` — YOLOv8 implementation
- `opencv-python` — Image and video processing
- `streamlit` — Web dashboard framework
- `matplotlib` — Data visualization
- `openai` — AI-powered suggestions
- `python-dotenv` — Environment variable management

## Core Modules

### Detection (`detect.py`)
- `detect_single_image()` — Process a single image
- `run_image_folder()` — Batch process a folder of images
- `detect_video()` — Process video files
- `calculate_percentages()` — Calculate violation rates

### AI Suggestions (`suggestion.py`)
- `generate_safety_suggestions()` — Generate AI-based safety recommendations

### Webcam Detection (`webcam.py`)
- Real-time PPE monitoring
- Color-coded bounding boxes (green for compliant, red for violations)

## Tips

1. Use clear, well-lit images for best detection accuracy
2. Larger video files will take longer to process
3. Ensure good lighting when using webcam detection
4. Run batch analysis first to generate the metrics needed for AI suggestions

## Contact

For questions or support, please open an issue in the repository.
