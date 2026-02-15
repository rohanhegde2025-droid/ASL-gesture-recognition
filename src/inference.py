import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import deque

# load model and label encoder
clf = pickle.load(open("models/asl_classifier.pkl", "rb"))
le = pickle.load(open("models/label_encoder.pkl", "rb"))

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# smoothing buffer - takes last 10 predictions and shows most frequent
# prevents flickering between letters
buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)
print("Camera started - press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # flips frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # draws hand landmarks on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # extracts and normalizes landmarks
            lm = hand_landmarks.landmark
            wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
            row = []
            for point in lm:
                row.extend([
                    point.x - wrist_x,
                    point.y - wrist_y,
                    point.z - wrist_z
                ])

            # predict
            pred_encoded = clf.predict([row])[0]
            pred_label = le.inverse_transform([pred_encoded])[0]
            buffer.append(pred_label)

            # gets most frequent prediction from buffer
            prediction = max(set(buffer), key=buffer.count)

    cv2.putText(frame, f"Sign: {prediction}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    if prediction:
        confidence = buffer.count(prediction) / len(buffer) * 100
        cv2.putText(frame, f"Confidence: {confidence:.0f}%", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("ASL Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):  
        break

cap.release()
cv2.destroyAllWindows()