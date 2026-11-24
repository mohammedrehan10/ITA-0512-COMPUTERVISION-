"""
EXPERIMENT 01
Question: Perform basic Image Handling and processing operations on the image is to read an image
in python and Convert an Image to Gray-scale.
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
    
    # 1. Read input image in BGR
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # 3. Create side-by-side comparison
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack((original, gray_bgr))
    
    # Add descriptive annotations
    cv2.putText(comparison, "Original (BGR)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Grayscale", (original.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Experiment 01 completed successfully.")
    display_and_save("Experiment 01 - Grayscale Conversion", comparison, "exp01_grayscale_output.jpg")

if __name__ == "__main__":
    main()
