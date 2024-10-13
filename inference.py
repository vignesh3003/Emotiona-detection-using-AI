import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained emotion detection model
model = load_model('/Users/vigneshraj/emotion_recognition/emotion_detection_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)  # Change this to 1 or 2 if necessary

if not cap.isOpened():
    print("Camera failed to initialize.")
else:
    print("Camera initialized successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Preprocess the frame for emotion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))  # Resize to the model's expected input size
    normalized_frame = resized_frame / 255.0  # Normalize the pixel values
    reshaped_frame = np.reshape(normalized_frame, (1, 48, 48, 1))  # Reshape for model input

    # Make predictions
    emotion_prediction = model.predict(reshaped_frame)
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]

    # Display the predicted emotion on the frame with larger and clearer text
    font_scale = 1.5  # Increase font size
    color = (0, 255, 0)  # Use green color for better contrast
    thickness = 3  # Increase text thickness
    cv2.putText(frame, emotion_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    # Show the webcam feed with emotion prediction
    cv2.imshow("Webcam - Emotion Detection", frame)

    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
