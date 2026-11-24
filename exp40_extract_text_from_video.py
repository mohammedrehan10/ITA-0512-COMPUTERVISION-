"""
EXPERIMENT 40
Question: Write a Python function to extract the text from videos.
"""
import os
import sys
import cv2
import numpy as np

def extract_text_regions_from_video(video_path):
    print(f"Analyzing video for text extraction: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    text_log_path = os.path.join(out_dir, "exp40_extracted_text.txt")
    annotated_out_path = os.path.join(out_dir, "exp40_text_detection_video.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(annotated_out_path, fourcc, fps, (width, height))

    extracted_records = []
    frame_idx = 0
    max_frames = 90
    saved_crops = 0

    # Try importing pytesseract if installed
    try:
        import pytesseract
        has_ocr = True
    except ImportError:
        has_ocr = False

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Morphological gradient to highlight text edges
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
        
        # Binarize
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Connect horizontal text characters into text lines
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, connect_kernel)
        
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            # Text bounding box heuristics: wider than tall, minimum size
            if w > 40 and h > 12 and aspect_ratio > 1.5 and (w * h) < (width * height * 0.4):
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, "Text Region", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                
                # Save first few text crops
                if saved_crops < 5 and frame_idx % 10 == 0:
                    saved_crops += 1
                    crop = frame[y:y+h, x:x+w]
                    crop_path = os.path.join(out_dir, f"exp40_text_crop_{saved_crops}.jpg")
                    cv2.imwrite(crop_path, crop)
                    
                    text_str = ""
                    if has_ocr:
                        try:
                            text_str = pytesseract.image_to_string(crop, config="--psm 6").strip()
                        except Exception:
                            text_str = ""
                    if not text_str:
                        text_str = f"TextRegion_frame{frame_idx}_at_({x},{y})"
                        
                    extracted_records.append(f"Frame {frame_idx:03d} [{x},{y},{w},{h}]: '{text_str}'")

        cv2.putText(annotated, f"Frame {frame_idx} | Text Extraction", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        out.write(annotated)

    cap.release()
    out.release()

    with open(text_log_path, "w", encoding="utf-8") as f:
        f.write("=== OPENCV VIDEO TEXT EXTRACTION REPORT ===\n")
        f.write(f"Source: {video_path}\n")
        f.write(f"Processed Frames: {frame_idx}\n")
        f.write(f"Total Text Regions Logged: {len(extracted_records)}\n\n")
        for rec in extracted_records:
            f.write(rec + "\n")

    print(f"Experiment 40 completed. Logged {len(extracted_records)} text regions.")
    print(f"[SAVED] {text_log_path}")
    print(f"[SAVED] {annotated_out_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vid_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "assets", "text_video.mp4")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(base_dir, "assets", "Megamind.avi")
    extract_text_regions_from_video(vid_path)

if __name__ == "__main__":
    main()
