"""
EXPERIMENT 17
Question: Design and implement a water marking technique to insert the watermark into the original
effectively image using Open CV.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("lena.jpg")
    wm_path = sys.argv[2] if len(sys.argv) > 2 else get_asset_path("opencv-logo.png", "assets/watermark_logo.png")
    
    original = cv2.imread(img_path)
    watermark = cv2.imread(wm_path, cv2.IMREAD_UNCHANGED)
    
    if original is None or watermark is None:
        print("Error: Could not load source or watermark image.")
        return

    h_img, w_img = original.shape[:2]
    # Resize watermark to fit nicely in corner (e.g. 15% of image width)
    wm_w = int(w_img * 0.22)
    wm_h = int(wm_w * (watermark.shape[0] / watermark.shape[1]))
    resized_wm = cv2.resize(watermark, (wm_w, wm_h))

    # Place at bottom-right corner
    y_offset = h_img - wm_h - 20
    x_offset = w_img - wm_w - 20
    roi = original[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w]

    result = original.copy()
    if resized_wm.shape[2] == 4:
        # Alpha blending
        alpha = resized_wm[:, :, 3] / 255.0
        for c in range(3):
            result[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w, c] = (
                alpha * resized_wm[:, :, c] + (1 - alpha) * roi[:, :, c]
            )
    else:
        # AddWeighted overlay
        blended = cv2.addWeighted(roi, 0.4, resized_wm[:, :, :3], 0.6, 0)
        result[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w] = blended

    cv2.rectangle(result, (x_offset - 2, y_offset - 2), (x_offset + wm_w + 2, y_offset + wm_h + 2), (0, 255, 0), 2)
    cv2.putText(result, "Watermarked", (x_offset, y_offset - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print("Experiment 17 completed successfully.")
    display_and_save("Experiment 17 - Watermarking", result, "exp17_watermarked_output.jpg")

if __name__ == "__main__":
    main()
