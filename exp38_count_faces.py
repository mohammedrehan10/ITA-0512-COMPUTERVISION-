"""
EXPERIMENT 38
Question: Write a Python function to Count the number of faces for the given input image using Open
CV.
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


def count_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    face_count = len(faces)
    
    annotated = image.copy()
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"Face #{idx + 1}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Top banner showing total count
    cv2.rectangle(annotated, (15, 15), (320, 65), (0, 0, 0), -1)
    cv2.putText(annotated, f"Total Faces Count: {face_count}", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
    return face_count, annotated

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else get_asset_path("real_group_faces.png", "assets/faces_sample.jpg")
    print(f"Reading image from: {img_path}")
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    count, result = count_faces(image)
    print(f"Successfully counted {count} face(s) in image.")
    display_and_save("Experiment 38 - Face Counter", result, "exp38_face_count_output.jpg")

if __name__ == "__main__":
    main()
