import time
import threading
import cv2
from ultralytics import YOLO

try:
    import winsound
    ALARM_AVAILABLE = True
except ImportError:

    ALARM_AVAILABLE = False

# Configuration
MODEL_PATH = "models/ppe.pt"
ALARM_FREQUENCY = 1200     
ALARM_BEEP_MS = 250        


violation_active = False

# Load model
model = YOLO(MODEL_PATH)


def _alarm_worker():

    while True:
        if violation_active:
            if ALARM_AVAILABLE:
                winsound.Beep(ALARM_FREQUENCY, ALARM_BEEP_MS)
            else:
                print("\a", end="", flush=True)
                time.sleep(ALARM_BEEP_MS / 1000)
        else:
            time.sleep(0.05)  


def run_webcam():
    
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam not found!")
        return
    
    print("📸 Webcam running... Press 'q' to quit.")
    
    threading.Thread(target=_alarm_worker, daemon=True).start()
    
   
    try:
        cv2.namedWindow("PPE Live Detection")
        show_frame = True
    except:
        print("⚠️ Unable to open display window. Running without preview.")
        show_frame = False
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame.")
            break
        
        # Run detection
        results = model(frame, stream=True)
        
        violation_detected_this_frame = False
        
        # Draw detections on frame
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                is_violation = "NO" in label
                if is_violation:
                    violation_detected_this_frame = True
                
                # Green for compliant, Red for violations
                color = (0, 255, 0) if not is_violation else (0, 0, 255)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        

        global violation_active
        violation_active = violation_detected_this_frame
        
        if violation_detected_this_frame:
            cv2.putText(
                frame,
                "VIOLATION DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )
        
        # Display frame
        if show_frame:
            try:
                cv2.imshow("PPE Live Detection", frame)
            except:
                print("⚠️ Cannot display video window anymore.")
                show_frame = False
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()