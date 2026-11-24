"""
EXPERIMENT 22
Question: Implement the Closing technique as a Morphological operation to dilate the foreground
regions based on Open CV.
Note: Closing is Dilation followed by Erosion (cv2.MORPH_CLOSE), used to close holes inside foreground objects.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("imageTextN.png", "assets/lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Morphological Closing
    closed = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)

    comparison = np.hstack((original, closed))
    cv2.putText(comparison, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Closing (Dilation + Erosion)", (original.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print("Experiment 22 completed successfully.")
    display_and_save("Experiment 22 - Morphological Closing", comparison, "exp22_closing_morphology_output.jpg")

if __name__ == "__main__":
    main()
