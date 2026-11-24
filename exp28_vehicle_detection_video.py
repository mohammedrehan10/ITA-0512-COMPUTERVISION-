"""
EXPERIMENT 28
Question: Implement a vehicle detection algorithm using Open CV to detect and locate vehicles in each
frame of the video.
"""
import os
import sys
import cv2

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "assets", "real_traffic_cars.mp4")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(base_dir, "assets", "traffic_video.mp4")

    print(f"Opening traffic video: {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp28_vehicle_detection.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Background subtractor to detect moving vehicles
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    frame_count = 0
    max_frames = 120
    print("Processing frames and locating vehicles...")
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        # Remove noise & shadows
        _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_morph = cv2.morphologyEx(fg_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of moving vehicles
        contours, _ = cv2.findContours(fg_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vehicle_count = 0
        annotated = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 450: # Threshold for vehicle size
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 4.0:
                    vehicle_count += 1
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated, "Vehicle", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(annotated, f"Frame: {frame_count} | Vehicles Located: {vehicle_count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        out.write(annotated)

    cap.release()
    out.release()
    print(f"Experiment 28 completed. Processed {frame_count} frames.")
    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
