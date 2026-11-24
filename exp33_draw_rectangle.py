"""
EXPERIMENT 33
Question: Write a Python function to create a white image size entered by the user and then create a
shape of Rectangle using Open CV.
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


def draw_rectangle_image(width=500, height=500):
    # Create white canvas
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define rectangle coordinates
    pt1 = (int(width * 0.2), int(height * 0.2))
    pt2 = (int(width * 0.8), int(height * 0.8))
    
    # Draw Rectangle (BGR: Deep Blue (180, 50, 20), thickness=3)
    cv2.rectangle(img, pt1, pt2, (180, 50, 20), thickness=3)
    
    # Label
    cv2.putText(img, "OpenCV Rectangle", (pt1[0], pt1[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 50, 20), 2)
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

    img = draw_rectangle_image(w, h)
    print("Experiment 33 completed successfully.")
    display_and_save("Experiment 33 - Rectangle Shape", img, "exp33_rectangle_output.jpg")

if __name__ == "__main__":
    main()
