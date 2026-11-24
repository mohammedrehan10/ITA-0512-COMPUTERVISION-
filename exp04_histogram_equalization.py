"""
EXPERIMENT 04
Question: Implement histogram equalization on the given image and compare it with the original
image using Open CV.
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


def plot_histogram(gray_img, width, height=180, title="Histogram"):
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, height - 30, cv2.NORM_MINMAX)
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 240
    for i in range(1, 256):
        pt1 = (int((i - 1) * width / 256.0), height - 10 - int(hist[i - 1]))
        pt2 = (int(i * width / 256.0), height - 10 - int(hist[i]))
        cv2.line(canvas, pt1, pt2, (20, 20, 180), 2)
    cv2.putText(canvas, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return canvas

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    
    h, w = gray.shape
    hist_orig = plot_histogram(gray, w, 160, "Original Histogram")
    hist_eq = plot_histogram(equalized, w, 160, "Equalized Histogram")
    
    top = np.hstack((cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)))
    bottom = np.hstack((hist_orig, hist_eq))
    comparison = np.vstack((top, bottom))
    
    cv2.putText(comparison, "Original Grayscale", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "Equalized Image", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    print("Experiment 04 completed successfully.")
    display_and_save("Experiment 04 - Histogram Equalization", comparison, "exp04_histogram_equalization_output.jpg")

if __name__ == "__main__":
    main()
