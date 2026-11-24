"""
EXPERIMENT 14
Question: Perform Perspective Transformation on the given image using python and Open CV.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("building.jpg", "assets/lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    h, w = original.shape[:2]
    
    # 4 corner source points and target destination coordinates
    pts1 = np.float32([[int(0.1*w), int(0.2*h)], [int(0.9*w), int(0.1*h)], [int(0.2*w), int(0.85*h)], [int(0.85*w), int(0.9*h)]])
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Calculate 3x3 Perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed = cv2.warpPerspective(original, M, (w, h))

    comparison = np.hstack((original, transformed))
    cv2.putText(comparison, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Perspective Warp", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Experiment 14 completed successfully.")
    display_and_save("Experiment 14 - Perspective Transformation", comparison, "exp14_perspective_output.jpg")

if __name__ == "__main__":
    main()
