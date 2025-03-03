import cv2
import face_recognition
import pickle

# Load trained encodings
try:
    with open("encodings.pkl", "rb") as f:
        encodings = pickle.load(f)
    
    # Flatten the encodings and match names
    face_names = []
    face_encodings = []
    for name, encoding_list in encodings.items():
        for encoding in encoding_list:
            face_names.append(name)
            face_encodings.append(encoding)
except (FileNotFoundError, EOFError):
    print("Error: encodings.pkl not found or corrupted. Train the model first.")
    exit()

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings_current = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings_current, face_locations):
        matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.6)
        print("Matches:", matches, type(matches))  # Keep this for now to verify
        name = "Unknown"

        if any(matches):  # Should work now
            match_indexes = [i for i, match in enumerate(matches) if match]
            face_distances = face_recognition.face_distance(face_encodings, face_encoding)
            best_match_index = min(match_indexes, key=lambda index: face_distances[index])
            name = face_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()