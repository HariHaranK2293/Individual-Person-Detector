import cv2
import os
import face_recognition
from sklearn.metrics import accuracy_score

def load_faces(dataset_dir):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  
            image_path = os.path.join(dataset_dir, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])  # Using os.path.splitext to get filename without extension
    return known_face_encodings, known_face_names

# Function to detect and recognize faces in an image
def detect_and_recognize_faces(frame, known_face_encodings, known_face_names):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Absent"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw text above the rectangle
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

    return frame

# Path to your dataset directory
dataset_dir = r"C:\Users\haris\OneDrive\Pictures\Camera Roll\samp"

# Load known faces and their names
known_face_encodings, known_face_names = load_faces(dataset_dir)

# Open the laptop camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and recognize faces
    frame_with_faces = detect_and_recognize_faces(frame, known_face_encodings, known_face_names)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame_with_faces)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
