"""
EXPERIMENT 39
Question: Write a Python function to play the given video in reverse mode in slow motion.
"""
import os
import sys
import cv2

def play_reverse_slow_motion(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    max_frames = 120
    print("Reading frames...")
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # 1. Reverse frames
    rev_frames = frames[::-1]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39_reverse_slow_motion.mp4")

    # 2. Slow motion speed (half fps)
    slow_fps = max(1.0, fps / 2.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, slow_fps, (width, height))

    for idx, f in enumerate(rev_frames):
        annotated = f.copy()
        cv2.putText(annotated, f"REVERSE SLOW-MO (0.5x) - Frame {idx + 1}/{len(rev_frames)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out.write(annotated)
    out.release()

    print(f"Experiment 39 completed. Saved reverse slow-motion video with {len(rev_frames)} frames.")
    print(f"[SAVED] {out_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "assets", "real_traffic_cars.mp4")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(base_dir, "assets", "sample_video.mp4")
    play_reverse_slow_motion(vid_path)

if __name__ == "__main__":
    main()
