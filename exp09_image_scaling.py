"""
EXPERIMENT 09
Question: Implement the image scaling techniques to resize the images to both bigger and small
sizes.
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

    h, w = original.shape[:2]
    # Scaling to bigger size (1.5x) using bicubic interpolation
    bigger = cv2.resize(original, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Scaling to smaller size (0.5x) using pixel area relation interpolation
    smaller = cv2.resize(original, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    print(f"Original shape: {original.shape}")
    print(f"Bigger shape (1.5x): {bigger.shape}")
    print(f"Smaller shape (0.5x): {smaller.shape}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "exp09_scaled_bigger.jpg"), bigger)
    cv2.imwrite(os.path.join(out_dir, "exp09_scaled_smaller.jpg"), smaller)

    # Visualization
    pad_smaller = np.zeros_like(original)
    pad_smaller[:smaller.shape[0], :smaller.shape[1]] = smaller
    comp = np.hstack((original, pad_smaller))
    cv2.putText(comp, f"Original ({w}x{h})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comp, f"Smaller 0.5x ({smaller.shape[1]}x{smaller.shape[0]})", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("Experiment 09 completed successfully.")
    display_and_save("Experiment 09 - Image Scaling", comp, "exp09_scaling_comparison.jpg")

if __name__ == "__main__":
    main()
