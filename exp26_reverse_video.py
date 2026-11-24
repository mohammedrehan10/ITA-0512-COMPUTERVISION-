"""
EXPERIMENT 26
Question: Implement a function to reverse the frames of the video to create a video in reverse mode
using Open CV.
"""
import os
import sys
import cv2

def reverse_video(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames = []
    print("Reading frames into memory...")
    max_frames = 120 # Read up to 120 frames
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Read {len(frames)} frames. Reversing frames...")

    # Reverse frame sequence
    reversed_frames = frames[::-1]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp26_reversed_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for idx, f in enumerate(reversed_frames):
        annotated = f.copy()
        cv2.putText(annotated, f"REVERSE MODE - Frame {idx + 1}/{len(reversed_frames)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        out.write(annotated)
    out.release()

    print("Experiment 26 completed successfully.")
    print(f"[SAVED] {out_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "assets", "real_traffic_cars.mp4")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(base_dir, "assets", "sample_video.mp4")
    reverse_video(vid_path)

if __name__ == "__main__":
    main()
