"""
EXPERIMENT 35
Question: Write a Python function to create a text string entered by the user that must be appeared on
the given image using Open CV.
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


def put_text_on_image(image, text_str="OpenCV Computer Vision"):
    result = image.copy()
    h, w = result.shape[:2]
    
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.0
    thickness = 2
    
    # Calculate text dimensions to center it
    text_size = cv2.getTextSize(text_str, font, scale, thickness)[0]
    org = ((w - text_size[0]) // 2, (h + text_size[1]) // 2)
    
    # Background shadow/box
    pad = 12
    cv2.rectangle(result, (org[0] - pad, org[1] - text_size[1] - pad),
                  (org[0] + text_size[0] + pad, org[1] + pad), (0, 0, 0), -1)
    # Render text
    cv2.putText(result, text_str, org, font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
    return result

def main():
    img_path = get_asset_path("lena.jpg")
    image = cv2.imread(img_path)
    if image is None:
        image = np.ones((400, 600, 3), dtype=np.uint8) * 200

    try:
        if sys.stdin.isatty():
            user_text = input("Enter text string to display (press Enter for default): ").strip()
            if not user_text:
                user_text = "SIMATS Computer Vision Lab 2026"
        else:
            user_text = "SIMATS Computer Vision Lab 2026"
    except Exception:
        user_text = "SIMATS Computer Vision Lab 2026"

    result = put_text_on_image(image, user_text)
    print(f"Rendered text: '{user_text}'")
    display_and_save("Experiment 35 - Put Text on Image", result, "exp35_text_output.jpg")

if __name__ == "__main__":
    main()
