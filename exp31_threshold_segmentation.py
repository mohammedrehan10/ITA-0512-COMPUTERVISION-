"""
EXPERIMENT 31
Question: Implement a Segmentation algorithm using Open CV to segment the given input image based
on the given threshold values.
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

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    thresh_val = 127
    
    # 1. Binary Threshold
    _, t_bin = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    # 2. Binary Inverted
    _, t_inv = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    # 3. Truncate
    _, t_trunc = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_TRUNC)
    # 4. To Zero
    _, t_tozero = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_TOZERO)
    # 5. Otsu's Thresholding (Automatic threshold selection)
    otsu_val, t_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Helper function to label panels
    def label(im, text):
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape) == 2 else im.copy()
        cv2.putText(bgr, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return bgr

    r1 = np.hstack((label(original, "Original"), label(t_bin, f"Binary (th={thresh_val})"), label(t_inv, "Binary Inv")))
    r2 = np.hstack((label(t_trunc, "Truncate"), label(t_tozero, "ToZero"), label(t_otsu, f"Otsu (th={int(otsu_val)})")))
    grid = np.vstack((r1, r2))

    print(f"Experiment 31 completed. Otsu threshold calculated: {otsu_val}")
    display_and_save("Experiment 31 - Threshold Segmentation", grid, "exp31_threshold_segmentation_output.jpg")

if __name__ == "__main__":
    main()
