"""
EXPERIMENT 07
Question: Perform basic video processing operations on the captured video. Read captured video in
python and display the video, in slow motion and in fast motion.
"""
import os
import sys
import cv2

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "assets", "real_traffic_cars.mp4")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(base_dir, "assets", "sample_video.mp4")

    print(f"Opening video: {vid_path}")
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    slow_path = os.path.join(out_dir, "exp07_slow_motion.mp4")
    fast_path = os.path.join(out_dir, "exp07_fast_motion.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Slow motion: half fps
    slow_writer = cv2.VideoWriter(slow_path, fourcc, max(1.0, fps / 2.0), (width, height))
    # Fast motion: double fps
    fast_writer = cv2.VideoWriter(fast_path, fourcc, fps * 2.0, (width, height))

    print("Processing video frames for slow and fast motion...")
    count = 0
    max_frames = 150 # Process first 150 frames
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        
        s_frame = frame.copy()
        cv2.putText(s_frame, f"SLOW MOTION (0.5x) - Frame {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        slow_writer.write(s_frame)
        
        f_frame = frame.copy()
        cv2.putText(f_frame, f"FAST MOTION (2.0x) - Frame {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        fast_writer.write(f_frame)

    cap.release()
    slow_writer.release()
    fast_writer.release()

    print(f"Experiment 07 completed. Processed {count} frames.")
    print(f"[SAVED] {slow_path}")
    print(f"[SAVED] {fast_path}")

if __name__ == "__main__":
    main()
