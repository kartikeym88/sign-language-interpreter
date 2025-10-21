import cv2
import mediapipe as mp

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image (so it acts like a mirror)
    img = cv2.flip(img, 1)

    # Convert image to RGB (MediaPipe needs RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # If hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions (thumb tip = 4, index tip = 8)
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Get pixel positions
            h, w, _ = img.shape
            thumb_y = int(thumb_tip.y * h)
            index_y = int(index_tip.y * h)

            # If thumb is above index finger → "Thumbs Up"
            if thumb_y < index_y:
                cv2.putText(img, "👍 Thumbs Up Detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show camera output
    cv2.imshow("Sign Language Interpreter", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
