"""
EXPERIMENT 34
Question: Write a Python function to create a white image size entered by the user and then create a
shape of Circle using Open CV.
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


def draw_circle_image(width=500, height=500):
    # Create white canvas
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    center = (width // 2, height // 2)
    radius = min(width, height) // 3
    
    # Draw Circle (BGR: Crimson Red (40, 40, 210), thickness=3)
    cv2.circle(img, center, radius, (40, 40, 210), thickness=3)
    cv2.circle(img, center, 5, (0, 0, 0), -1) # Center point
    
    cv2.putText(img, f"Circle Center: {center}, Radius: {radius}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 210), 2)
    return img

def main():
    try:
        if sys.stdin.isatty():
            w_in = input("Enter image width (default 500): ").strip()
            h_in = input("Enter image height (default 500): ").strip()
            w = int(w_in) if w_in else 500
            h = int(h_in) if h_in else 500
        else:
            w, h = 500, 500
    except Exception:
        w, h = 500, 500

    img = draw_circle_image(w, h)
    print("Experiment 34 completed successfully.")
    display_and_save("Experiment 34 - Circle Shape", img, "exp34_circle_output.jpg")

if __name__ == "__main__":
    main()
