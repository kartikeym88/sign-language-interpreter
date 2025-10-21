import cv2
import numpy as np
import mediapipe as mp
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Simulate a pre-trained model (replace with your trained model)
# For demo, we'll create a dummy classifier
def create_dummy_model():
    # Dummy data: 21 landmarks * 3 (x, y, z) = 63 features per sample
    X = np.random.rand(100, 63)  # 100 samples, 63 features
    y = np.random.choice(['A', 'B', 'C'], 100)  # Random labels (A, B, C)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_scaled, y)
    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return model, scaler

# Load or create model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except:
    model, scaler = create_dummy_model()

# Function to extract hand landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks), image
    return None, image

# Main function for real-time detection
def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Extract landmarks and process
        landmarks, frame = extract_landmarks(frame)
        if landmarks is not None:
            # Scale the features
            landmarks_scaled = scaler.transform([landmarks])
            # Predict the sign
            prediction = model.predict(landmarks_scaled)[0]
            probability = model.predict_proba(landmarks_scaled)[0].max()
            # Display the prediction
            cv2.putText(frame, f'Sign: {prediction} ({probability:.2f})', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Sign Language Interpreter', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()