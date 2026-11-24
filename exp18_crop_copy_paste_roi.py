"""
EXPERIMENT 18
Question: Implement image cropping, copying and pasting to select a region of interest (ROI) from the
source image using Open CV.
"""
import os
import sys
import cv2
import numpy as np

def get_asset_path(filename, fallback="assets/sample_image.jpg"):
    """Returns path to asset, falling back if not found."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    primary = os.path.join(base_dir, "assets", filename)
    if os.path.exists(primary):
        return primary
    fb = os.path.join(base_dir, fallback)
    if os.path.exists(fb):
        return fb
    return filename

def display_and_save(window_name, image, output_filename=None):
    """Saves image to outputs/ and displays it unless running in headless mode."""
    if output_filename:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, output_filename)
        cv2.imwrite(out_path, image)
        print(f"[SAVED] {out_path}")
    
    if os.environ.get("HEADLESS") != "1":
        cv2.imshow(window_name, image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("messi5.jpg", "assets/lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    # In messi5.jpg, the ball is at approx y:[280:340], x:[330:390]
    h, w = original.shape[:2]
    # Select ROI (e.g. center region or object)
    y1, y2 = int(h * 0.5), int(h * 0.75)
    x1, x2 = int(w * 0.5), int(w * 0.75)
    
    # 1. Crop ROI
    roi = original[y1:y2, x1:x2].copy()
    
    # 2. Copy and Paste ROI to another location (e.g. top-left)
    result = original.copy()
    rh, rw = roi.shape[:2]
    target_y1, target_x1 = 30, 30
    result[target_y1:target_y1 + rh, target_x1:target_x1 + rw] = roi

    # Highlight source and pasted ROI
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(result, "Source ROI", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.rectangle(result, (target_x1, target_y1), (target_x1 + rw, target_y1 + rh), (0, 255, 0), 2)
    cv2.putText(result, "Pasted ROI", (target_x1, target_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print("Experiment 18 completed successfully.")
    display_and_save("Experiment 18 - Crop Copy Paste ROI", result, "exp18_roi_pasted_output.jpg")

if __name__ == "__main__":
    main()
