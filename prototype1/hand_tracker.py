import cv2
import mediapipe as mp
import time

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Try to start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[INFO] Webcam opened:", cap.isOpened())

# Small delay before starting frame read
time.sleep(1)

# Timer to auto-close after 10 seconds (optional)
start_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        print("[ERROR] Failed to read frame from webcam")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Manual exit.")
        break

    if time.time() - start_time > 10:
        print("Auto-closing after 10 seconds.")
        break

cap.release()
cv2.destroyAllWindows()
