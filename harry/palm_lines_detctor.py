# palm_lines_detector.py
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# -------------------------------
# Text-to-speech setup
# -------------------------------
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------------------------
# Mediapipe hands setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only detect one hand at a time
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# Palm line detection helper
# -------------------------------
def enhance_palm_lines(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Edge detection to highlight lines
    edges = cv2.Canny(enhanced, 40, 100)
    # Convert edges to 3 channel
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_color

# -------------------------------
# Main loop
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    text = "No hand detected"

    if result.multi_hand_landmarks and result.multi_handedness:
        hand_landmarks = result.multi_hand_landmarks[0]
        hand_type = result.multi_handedness[0].classification[0].label

        if hand_type != "Right":
            text = "Please show RIGHT hand"
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            speak(text)
        else:
            # Get bounding box for palm region
            h, w = frame.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            idxs = [0, 1, 2, 5, 9, 13, 17]
            xs = [pts[i][0] for i in idxs]
            ys = [pts[i][1] for i in idxs]
            x_min, x_max = max(0, min(xs)-30), min(w, max(xs)+30)
            y_min, y_max = max(0, min(ys)-30), min(h, max(ys)+30)
            palm_roi = frame[y_min:y_max, x_min:x_max].copy()

            # Enhance palm lines
            palm_lines = enhance_palm_lines(palm_roi)

            # Place processed ROI back to frame
            frame[y_min:y_max, x_min:x_max] = palm_lines

            text = "Right hand detected"
            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Show frame
    cv2.imshow("Palm Lines Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
