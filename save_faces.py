import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
count = 0
user_name = input("Enter User Name (e.g., John): ")  # Names instead of numbers

os.makedirs(f"dataset/{user_name}", exist_ok=True)

while count < 10:  # 10 images per user for variety
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/{user_name}/{count}.jpg", face)
        count += 1

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {user_name}.")