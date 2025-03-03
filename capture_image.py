#capture_faces.py
import cv2

cam = cv2.VideoCapture(0)  # 0 for default webcam
ret, frame = cam.read()

if ret:
    cv2.imwrite("test_image.jpg", frame)
    print("Image captured successfully!")
else:
    print("Failed to capture image.")

cam.release()
