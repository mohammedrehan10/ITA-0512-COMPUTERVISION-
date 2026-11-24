"""
EXPERIMENT 13
Question: Perform Affine Transformation on the given image using python and Open CV.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    rows, cols = original.shape[:2]
    
    # 3 source points and their new mapped positions
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    
    # Calculate 2x3 Affine transformation matrix
    M = cv2.getAffineTransform(pts1, pts2)
    # Apply warpAffine
    transformed = cv2.warpAffine(original, M, (cols, rows))

    comparison = np.hstack((original, transformed))
    cv2.putText(comparison, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Affine Transformed", (cols + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Experiment 13 completed successfully.")
    display_and_save("Experiment 13 - Affine Transformation", comparison, "exp13_affine_output.jpg")

if __name__ == "__main__":
    main()
