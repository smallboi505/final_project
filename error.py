# import cv2
# import face_recognition

# image = face_recognition.load_image_file("dataset/1/0.jpg")
# face_locations = face_recognition.face_locations(image)
# print("Found {} face(s) in the image.".format(len(face_locations)))

import pickle
with open("encodings.pkl", "rb") as f:
    encodings = pickle.load(f)
print(encodings)