#train_faces.py

import os
import face_recognition
import pickle

dataset_path = "dataset"
encodings = {}

for user_name in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user_name)
    encodings[user_name] = []

    for image_name in os.listdir(user_path):
        image_path = os.path.join(user_path, image_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            encodings[user_name].append(encoding[0])
        else:
            print(f"No face detected in {image_name}")

with open("encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)

print("Training complete!")