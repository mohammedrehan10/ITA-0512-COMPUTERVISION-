"""
EXPERIMENT 32
Question: Write a Python function to create a white image size entered by the user and then create 4
boxes of Black, Blue, Green and Red respectively on each corner of the image. The size of the
colored boxes should be 1/10th the size of the image. (HINT: the arrays of ones and zeros can be
in more than 2 dimensions).
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


def create_corner_boxes_image(width=500, height=500):
    # HINT: Multi-dimensional array of ones scaled to 255 for pure white
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Box dimensions: 1/10th of the image dimensions
    box_h = max(1, height // 10)
    box_w = max(1, width // 10)
    
    print(f"Image Size: {width}x{height} | Box Size: {box_w}x{box_h}")
    
    # 1. Top-Left: Black [0, 0, 0]
    img[0:box_h, 0:box_w] = [0, 0, 0]
    
    # 2. Top-Right: Blue in BGR is [255, 0, 0]
    img[0:box_h, width - box_w:width] = [255, 0, 0]
    
    # 3. Bottom-Left: Green in BGR is [0, 255, 0]
    img[height - box_h:height, 0:box_w] = [0, 255, 0]
    
    # 4. Bottom-Right: Red in BGR is [0, 0, 255]
    img[height - box_h:height, width - box_w:width] = [0, 0, 255]

    # Annotate labels next to corners
    cv2.putText(img, "Black", (10, box_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Blue", (width - box_w - 45, box_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(img, "Green", (10, height - box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    cv2.putText(img, "Red", (width - box_w - 40, height - box_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
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

    img = create_corner_boxes_image(w, h)
    print("Experiment 32 completed successfully.")
    display_and_save("Experiment 32 - Corner Boxes", img, "exp32_corner_boxes_output.jpg")

if __name__ == "__main__":
    main()
