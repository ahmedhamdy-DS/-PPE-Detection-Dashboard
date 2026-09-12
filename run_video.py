from pathlib import Path
from detect import detect_video, calculate_percentages

# Configuration
VIDEO_PATH = "videos/test_videos/ppe-2.mp4"
OUTPUT_NAME = "output_video.mp4"


def process_video(video_path: str, output_name: str = "output_video.mp4"):

    video_path = Path(video_path)
    
    # Check if video exists
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Processing video: {video_path.name}")
    print(f"   Output will be saved as: {output_name}")
    
    try:
        output_path, metrics = detect_video(str(video_path), output_name)
        print(f"\n✅ Video processing complete!")
        print(f"   Result saved to: {output_path}")
        
        # Violation summary
        violations = {k: v for k, v in metrics.items() if k.startswith("NO-")}
        total_violations = sum(violations.values())
        
        print("\n📊 Detection Summary:")
        for label, count in sorted(metrics.items()):
            print(f"   {label}: {count}")
        
        print(f"\n🚨 Total violation frames: {total_violations}")
        for label, count in violations.items():
            print(f"   {label}: {count}")
        
        percentages = calculate_percentages(metrics)
        if percentages:
            print("\n📈 Violation Rates:")
            for key, value in percentages.items():
                print(f"   {key}: {value}%")
        
        print("\n💾 Metrics saved to results/metrics.json — the AI Suggestions page in app.py will now pick these up too.")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")


if __name__ == "__main__":
    process_video(VIDEO_PATH, OUTPUT_NAME)