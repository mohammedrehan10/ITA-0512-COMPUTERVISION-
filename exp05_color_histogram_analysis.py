"""
EXPERIMENT 05
Question: Write a Python function to analyze the histogram of the given input image based on color
levels using Open CV.
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


def analyze_color_histogram(img):
    """Calculates and plots histogram for Blue, Green, and Red channels."""
    h_canvas, w_canvas = 320, 512
    canvas = np.ones((h_canvas, w_canvas, 3), dtype=np.uint8) * 255
    
    channels = (('Blue Channel', (255, 0, 0), 0),
                ('Green Channel', (0, 180, 0), 1),
                ('Red Channel', (0, 0, 255), 2))
                
    for name, color, ch in channels:
        hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, h_canvas - 50, cv2.NORM_MINMAX)
        for i in range(1, 256):
            pt1 = (int((i - 1) * w_canvas / 256.0), h_canvas - 15 - int(hist[i - 1]))
            pt2 = (int(i * w_canvas / 256.0), h_canvas - 15 - int(hist[i]))
            cv2.line(canvas, pt1, pt2, color, 2)
            
    cv2.putText(canvas, "Color Histogram Analysis", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, "Blue", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(canvas, "Green", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)
    cv2.putText(canvas, "Red", (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return canvas

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    hist_img = analyze_color_histogram(original)
    hist_resized = cv2.resize(hist_img, (int(original.shape[1]), original.shape[0]))
    combined = np.hstack((original, hist_resized))
    
    print("Experiment 05 completed successfully.")
    display_and_save("Experiment 05 - Color Histogram Analysis", combined, "exp05_color_histogram_output.jpg")

if __name__ == "__main__":
    main()
