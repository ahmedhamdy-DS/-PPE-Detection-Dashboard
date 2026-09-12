# 🦺 PPE Detection Dashboard

A real-time Personal Protective Equipment (PPE) detection system using YOLOv8 and Streamlit. This application helps monitor workplace safety compliance by detecting hardhats, masks, safety vests, and identifying violations.

## 🌟 Features

- **Batch Image Analysis**: Process multiple images from a folder and get comprehensive safety metrics
- **Single Image Detection**: Upload and analyze individual images with annotated results
- **Video Processing**: Detect PPE violations in video files with frame-by-frame analysis
- **Live Webcam Detection**: Real-time PPE monitoring through your webcam
- **AI-Powered Suggestions**: Get actionable safety recommendations based on detected violations
- **Interactive Dashboard**: Beautiful, user-friendly interface with visual analytics

## 🚀 Live Demo

Access the deployed application here: [PPE Detection Dashboard](https://ppe-detection-avaz-turgay.streamlit.app/)

## 📋 Detection Capabilities

The system can detect:
- ✅ **Compliant PPE**: Hardhat, Mask, Safety Vest
- ⚠️ **Violations**: NO-Hardhat, NO-Mask, NO-Safety Vest
- 🔍 **Other Objects**: Person, Safety Cone, Machinery, Vehicle

## 📁 Project Structure
```
PPE_DASHBOARD/
├── images/
│   └── test_images/          # Sample images for batch processing
├── models/
│   └── ppe.pt               # YOLOv8 trained model
├── results/
│   └── detections/          # Output folder for processed images/videos
├── temp/                    # Temporary files for uploads
├── videos/
│   └── test_videos/         # Video files for processing
│       ├── ppe-1.mp4
│       ├── ppe-2.mp4
│       └── ppe-3.mp4
├── .env                     # Environment variables (API keys)
├── app.py                   # Main Streamlit dashboard
├── detect.py                # Core detection functions
├── suggestion.py            # AI-powered safety suggestions
├── run_video.py            # Standalone video processing script
├── webcam.py               # Live webcam detection
├── requirements.txt         # Python dependencies
└── README.md
```

## 📊 Dashboard Features

### Batch Analysis
- Upload a folder of images
- Get aggregated safety metrics
- View violation rates and compliance statistics
- Export results to JSON

### Single Image Detection
- Upload individual images
- See annotated detections in real-time
- Get detailed object counts

### AI Safety Recommendations
- Analyze compliance data
- Receive AI-generated safety suggestions
- Get actionable insights for improving workplace safety

## 🔧 Configuration

### Model Settings
Edit `detect.py` to customize:
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

## 📦 Dependencies

- `ultralytics` - YOLOv8 implementation
- `opencv-python` - Image and video processing
- `streamlit` - Web dashboard framework
- `matplotlib` - Data visualization
- `openai` - AI-powered suggestions
- `python-dotenv` - Environment variable management

## 🎨 Features Breakdown

### Detection Functions (`detect.py`)
- `detect_single_image()` - Process single image
- `run_image_folder()` - Batch process images
- `detect_video()` - Process video files
- `calculate_percentages()` - Calculate violation rates

### AI Suggestions (`suggestion.py`)
- `generate_safety_suggestions()` - Get AI recommendations

### Webcam Detection (`webcam.py`)
- Real-time PPE monitoring
- Color-coded bounding boxes (Green: compliant, Red: violation)

## 💡 Tips

1. **For best results**: Use clear, well-lit images
2. **Video processing**: Larger videos may take time to process
3. **Webcam detection**: Ensure good lighting for accurate detection
4. **AI suggestions**: Run batch analysis first to generate comprehensive metrics

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using YOLOv8 and Streamlit**
