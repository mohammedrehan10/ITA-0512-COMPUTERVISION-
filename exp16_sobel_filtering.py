"""
EXPERIMENT 16
Question: Implement a Sobel algorithm using Open CV to filter the input image.
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

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Sobel X (horizontal gradient) & Sobel Y (vertical gradient)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    abs_x = cv2.convertScaleAbs(sobel_x)
    abs_y = cv2.convertScaleAbs(sobel_y)
    # Combined gradient
    sobel_combined = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    
    h, w = gray.shape
    top = np.hstack((original, cv2.cvtColor(abs_x, cv2.COLOR_GRAY2BGR)))
    bottom = np.hstack((cv2.cvtColor(abs_y, cv2.COLOR_GRAY2BGR), cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)))
    grid = np.vstack((top, bottom))
    
    cv2.putText(grid, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(grid, "Sobel X", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(grid, "Sobel Y", (20, h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(grid, "Sobel Combined", (w + 20, h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Experiment 16 completed successfully.")
    display_and_save("Experiment 16 - Sobel Filter", grid, "exp16_sobel_filtered_output.jpg")

if __name__ == "__main__":
    main()
