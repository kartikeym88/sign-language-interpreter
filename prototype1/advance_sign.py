import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Finger tip landmark IDs (thumb, index, middle, ring, pinky)
tip_ids = [4, 8, 12, 16, 20]

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def fingers_up(hand_landmarks):
    fingers = []

    # Get positions of landmarks
    lm = hand_landmarks.landmark

    # Thumb
    if lm[tip_ids[0]].x < lm[tip_ids[0] - 1].x:
        fingers.append(1)  # Thumb is open
    else:
        fingers.append(0)  # Closed

    # Fingers: Index to Pinky
    for id in range(1, 5):
        if lm[tip_ids[id]].y < lm[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def detect_sign(fingers):
    if fingers == [0, 1, 0, 0, 0]:
        return " Pointing Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return " Peace"
    elif fingers == [1, 0, 0, 0, 0]:
        return " Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return " Stop"
    else:
        return "🤔 Unknown Sign"

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    sign = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_status = fingers_up(hand_landmarks)
            sign = detect_sign(finger_status)

    # Display sign
    if sign:
        cv2.putText(img, f"Sign: {sign}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Show camera
    cv2.imshow("Advanced Sign Language Interpreter", img)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
