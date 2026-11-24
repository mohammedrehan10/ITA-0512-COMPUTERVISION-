"""
EXPERIMENT 25
Question: Recognize watch from the given image by general Object recognition using Open CV.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("real_watch.jpg", "assets/watch_sample.jpg")
    tmpl_path = sys.argv[2] if len(sys.argv) > 2 else get_asset_path("real_watch_template.jpg", "assets/watch_template.jpg")
    
    image = cv2.imread(img_path)
    template = cv2.imread(tmpl_path)
    
    if image is None or template is None:
        print("Error: Could not load watch image or template.")
        return

    th, tw = template.shape[:2]
    # Perform Template Matching using normalized cross-correlation
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = max_loc
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    
    result = image.copy()
    cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(result, f"Watch Recognized (Conf: {max_val:.2f})", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    print(f"Watch detected at {top_left} with match score {max_val:.4f}")
    display_and_save("Experiment 25 - Watch Recognition", result, "exp25_watch_recognition_output.jpg")

if __name__ == "__main__":
    main()
