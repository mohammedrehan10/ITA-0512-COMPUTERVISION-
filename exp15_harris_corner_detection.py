"""
EXPERIMENT 15
Question: Perform basic Image Handling and processing operations on the image is to read an image
in python and detect the corners in the image using Harris Corner Detection function.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("building.jpg", "assets/home.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    
    # Harris Corner Detection: blockSize=2, ksize=3, k=0.04
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    # Dilate corner points for better visual marking
    dst = cv2.dilate(dst, None)
    
    marked = original.copy()
    # Threshold for an optimal value, mark corners in bright red
    marked[dst > 0.01 * dst.max()] = [0, 0, 255]

    comparison = np.hstack((original, marked))
    cv2.putText(comparison, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Harris Corners Detected", (original.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Experiment 15 completed successfully.")
    display_and_save("Experiment 15 - Harris Corner Detection", comparison, "exp15_harris_corners_output.jpg")

if __name__ == "__main__":
    main()
