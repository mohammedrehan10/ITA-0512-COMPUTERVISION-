"""
EXPERIMENT 36
Question: Write a Python function to subtract the background of the given input image based on color
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


def subtract_background_by_color(img):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # In portrait images (e.g. david1 or lena), background is light/neutral
    # Segment dominant background color range
    lower_bg = np.array([0, 0, 160])
    upper_bg = np.array([180, 60, 255])
    
    # Mask of the background
    bg_mask = cv2.inRange(hsv, lower_bg, upper_bg)
    # Foreground mask is inverted
    fg_mask = cv2.bitwise_not(bg_mask)
    
    # Refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask: background subtracted (set to black or transparent)
    fg_isolated = cv2.bitwise_and(img, img, mask=fg_mask)
    
    comp = np.hstack((img, fg_isolated))
    cv2.putText(comp, "Original", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comp, "Background Subtracted", (img.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return comp

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("real_face_portrait.jpg", "assets/lena.jpg")
    print(f"Reading image from: {img_path}")
    
    original = cv2.imread(img_path)
    if original is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    result = subtract_background_by_color(original)
    print("Experiment 36 completed successfully.")
    display_and_save("Experiment 36 - Background Subtraction", result, "exp36_background_subtracted_output.jpg")

if __name__ == "__main__":
    main()
