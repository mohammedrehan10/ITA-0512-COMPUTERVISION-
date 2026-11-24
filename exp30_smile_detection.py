"""
EXPERIMENT 30
Question: Implement a Smile detection algorithm using Open CV to detect and locate human smile in
the images.
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
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("real_group_faces.png", "assets/real_face_portrait.jpg")
    print(f"Reading image from: {img_path}")
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    smile_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml"))

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    result = image.copy()
    total_smiles = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = result[y:y + h, x:x + w]
        
        # Detect smile inside lower region of face
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=15, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            total_smiles += 1
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Smile :)", (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(result, f"Smiles Detected: {total_smiles}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    print(f"Experiment 30: {total_smiles} smiles detected.")
    display_and_save("Experiment 30 - Smile Detection", result, "exp30_smile_detection_output.jpg")

if __name__ == "__main__":
    main()
