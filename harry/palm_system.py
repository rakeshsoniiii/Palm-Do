# palm_system.py
import cv2, numpy as np, mediapipe as mp, time
from palm_model import build_embedding_model

IMG_SIZE = 128
SIM_THRESHOLD = 0.65

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)

class PalmRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.embed_model = build_embedding_model()
        self.db = {}  # user_id -> embeddings

    def preprocess_roi(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        img = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        return img

    def crop_palm_roi(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        idxs = [0, 1, 2, 5, 9, 13, 17]
        xs = [pts[i][0] for i in idxs]
        ys = [pts[i][1] for i in idxs]
        x_min, x_max = max(0, min(xs) - 30), min(w, max(xs) + 30)
        y_min, y_max = max(0, min(ys) - 30), min(h, max(ys) + 30)
        return frame[y_min:y_max, x_min:x_max].copy()

    def enroll_user(self, user_id, num_samples=60):
        cap = cv2.VideoCapture(0)
        collected = []
        print(f"[ENROLL] Show RIGHT palm for {user_id}...")

        while len(collected) < num_samples:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                roi = self.crop_palm_roi(frame, lm)
                img = self.preprocess_roi(roi)
                if img is not None:
                    emb = self.embed_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                    collected.append(emb)
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Enroll Palm", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

        if len(collected) >= 5:
            arr = np.vstack(collected)
            self.db[user_id] = arr
            print(f"[ENROLL] Done for {user_id}, samples={len(arr)}")
        else:
            print("Not enough samples collected!")

    def recognize(self):
        if not self.db:
            print("No users enrolled!")
            return
        cap = cv2.VideoCapture(0)
        print("[RECOGNIZE] Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            text = "No hand"
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                roi = self.crop_palm_roi(frame, lm)
                img = self.preprocess_roi(roi)
                if img is not None:
                    emb = self.embed_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                    user, score = self.match_embedding(emb)
                    if score >= SIM_THRESHOLD:
                        text = f"Recognized: {user} ({score:.2f})"
                    else:
                        text = "Unknown"
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Palm Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

    def match_embedding(self, emb):
        best_user, best_score = None, -1
        for user, arr in self.db.items():
            sims = cosine_similarity(emb, arr)
            top = float(np.max(sims))
            if top > best_score:
                best_score = top
                best_user = user
        return best_user, best_score
