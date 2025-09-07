# palm_recognizer.py
import os
import time
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
import mediapipe as mp
import tensorflow as tf

# -----------------------
# Config / hyperparams
# -----------------------
IMG_SIZE = 128
NUM_ENROLL_SAMPLES = 60        # take many samples per user (50-100 recommended)
SIM_THRESHOLD = 0.62           # cosine similarity threshold (tune 0.55-0.75)
VOTE_WINDOW = 7                # number of frames to majority-vote across
LIVENESS_MOTION_THRESHOLD = 0.003  # wrist motion std threshold to accept (tune)
EMBEDDINGS_DB_FILE = "embeddings_db.npz"

# -----------------------
# Embedding model
# -----------------------
def build_embedding_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), embed_dim=128):
    # Use MobileNetV2 backbone (pretrained on ImageNet).
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
    )
    # Freeze backbone (fine-tune later if you have lots of user data)
    base.trainable = False

    inp = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)  # expects 0-255
    x = base(x, training=False)
    x = tf.keras.layers.Dense(embed_dim, use_bias=False)(x)
    x = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)  # L2 norm
    model = tf.keras.Model(inputs=inp, outputs=x, name="palm_embedder")
    return model

# -----------------------
# Utility: cosine similarity
# -----------------------
def cosine_similarity(a, b):
    # a: (d,), b: (n,d) or (d,)
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b, a)  # returns (n,) similarities

# -----------------------
# Palm Recognizer Class
# -----------------------
class PalmRecognizer:
    def __init__(self):
        # mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # embedding model
        try:
            self.embed_model = build_embedding_model()
        except Exception as e:
            print("Embedding model build failed:", e)
            raise

        # embeddings DB: dict user_id -> np.array of embeddings (N_samples x D)
        self.db = {}
        self.load_db()

        # recognition state
        self.running = False
        self.vote_queue = []   # list of best_user per frame
        self.score_queue = []  # corresponding best scores
        self.landmark_history = []  # history of wrist positions for liveness

    # -----------------------
    # preprocessing
    # -----------------------
    def preprocess_roi(self, roi_bgr):
        # roi_bgr: cropped BGR image (OpenCV)
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        img = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0  # normalize to [0,1]
        return img

    def crop_palm_roi(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        # Choose bounding box covering wrist and finger base joints for robust palm crop
        idxs = [0, 1, 2, 5, 9, 13, 17]  # wrist, thumb cmc, ... some bases
        xs = [pts[i][0] for i in idxs]
        ys = [pts[i][1] for i in idxs]
        x_min, x_max = max(0, min(xs) - 30), min(w, max(xs) + 30)
        y_min, y_max = max(0, min(ys) - 30), min(h, max(ys) + 30)
        roi = frame[y_min:y_max, x_min:x_max].copy()
        return roi

    # -----------------------
    # enrollment
    # -----------------------
    def enroll_user(self, user_id, num_samples=NUM_ENROLL_SAMPLES, show_window=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return False

        collected = []
        start = time.time()
        last_capture = 0.0
        print(f"[ENROLL] Show RIGHT palm for user '{user_id}'. Capturing {num_samples} samples...")

        while len(collected) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror for UX
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Visual guidance
            if results.multi_hand_landmarks and results.multi_handedness:
                # Check handedness predicted by mediapipe
                label = results.multi_handedness[0].classification[0].label
                if label != "Right":
                    cv2.putText(frame, "Use RIGHT hand", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Enroll - Press q to cancel", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                hand_landmarks = results.multi_hand_landmarks[0]
                roi = self.crop_palm_roi(frame, hand_landmarks)
                img = self.preprocess_roi(roi)
                if img is not None and time.time() - last_capture > 0.3:  # small throttle
                    emb = self.embed_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                    collected.append(emb)
                    last_capture = time.time()
                    print(f"Collected {len(collected)}/{num_samples}")

                # draw
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Collected: {len(collected)}/{num_samples}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                cv2.putText(frame, "Show your RIGHT palm clearly", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if show_window:
                cv2.imshow("Enroll - Press q to cancel", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        if len(collected) >= 5:
            arr = np.vstack(collected)  # (N, D)
            # store or append
            if user_id in self.db:
                self.db[user_id] = np.vstack([self.db[user_id], arr])
            else:
                self.db[user_id] = arr
            self.save_db()
            print(f"[ENROLL] Done. Total samples for {user_id}: {len(self.db[user_id])}")
            return True
        else:
            print("[ENROLL] Not enough samples collected.")
            return False

    # -----------------------
    # recognition
    # -----------------------
    def recognize(self, require_consecutive=VOTE_WINDOW, sim_threshold=SIM_THRESHOLD):
        """
        Run recognition loop. Requires `sim_threshold` and `require_consecutive` frame voting.
        """
        if not self.db:
            print("No enrolled users in DB.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        self.running = True
        self.vote_queue.clear()
        self.score_queue.clear()
        self.landmark_history.clear()
        print("[RECOGNIZE] Starting. Press 'q' to quit.")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            display_text = "No hand"
            if results.multi_hand_landmarks and results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
                lm = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

                # Require right hand
                if handedness != "Right":
                    display_text = "Please use RIGHT hand"
                else:
                    # Crop ROI, create embedding
                    roi = self.crop_palm_roi(frame, lm)
                    img = self.preprocess_roi(roi)
                    if img is not None:
                        emb = self.embed_model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
                        # Compare to DB
                        best_user, best_score = self.match_embedding(emb)
                        self.vote_queue.append(best_user if best_score >= 0 else None)
                        self.score_queue.append(best_score)
                        if len(self.vote_queue) > require_consecutive:
                            self.vote_queue.pop(0)
                            self.score_queue.pop(0)

                        # Liveness check: add wrist position history
                        wrist = lm.landmark[0]
                        self.landmark_history.append((wrist.x, wrist.y))
                        if len(self.landmark_history) > 10:
                            self.landmark_history.pop(0)

                        # Decide using voting + liveness
                        candidate, vote_score = self.vote_decision(sim_threshold)
                        if candidate is not None and vote_score >= sim_threshold and self.liveness_ok():
                            display_text = f"Recognized: {candidate} ({vote_score:.2f})"
                            # Trigger once and then small cooldown to avoid repeated payments:
                            cv2.putText(frame, display_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                            cv2.imshow("Palm Recognizer", frame)
                            # call payment (stub) in separate thread
                            threading.Thread(target=self.process_payment, args=(candidate,)).start()
                            # After success, small delay and continue
                            time.sleep(1.2)
                        else:
                            if best_user is None:
                                display_text = "No match"
                            else:
                                display_text = f"{best_user} ({best_score:.2f})"
                    else:
                        display_text = "Bad ROI"

            # show info
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("Palm Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        print("[RECOGNIZE] Stopped.")

    def match_embedding(self, emb):
        """Return best matching user and best similarity score"""
        best_user = None
        best_score = -1.0
        for user, arr in self.db.items():
            sims = cosine_similarity(emb, arr)  # (n,) similarities
            top = float(np.max(sims))
            if top > best_score:
                best_score = top
                best_user = user
        return best_user, best_score

    def vote_decision(self, sim_threshold=SIM_THRESHOLD):
        """
        Takes vote_queue and score_queue and decides final candidate.
        Approach: count occurrences of same user in vote_queue; pick that user
        and return average score among frames where it appeared.
        """
        if not self.vote_queue:
            return None, 0.0
        # Filter out None
        votes = [v for v in self.vote_queue if v is not None]
        if not votes:
            return None, 0.0
        # majority
        unique, counts = np.unique(votes, return_counts=True)
        idx = np.argmax(counts)
        candidate = unique[idx]
        # average score for frames where candidate appeared
        scores = [s for v, s in zip(self.vote_queue, self.score_queue) if v == candidate]
        avg_score = float(np.mean(scores)) if scores else 0.0
        return candidate, avg_score

    def liveness_ok(self):
        """Basic motion-based liveness: require small motion in wrist positions."""
        if len(self.landmark_history) < 6:
            return False
        arr = np.array(self.landmark_history)
        # compute std of wrist x and y
        s = float(np.mean(np.std(arr, axis=0)))
        # Accept if motion is above threshold
        return s >= LIVENESS_MOTION_THRESHOLD

    # -----------------------
    # DB save/load
    # -----------------------
    def save_db(self, path=EMBEDDINGS_DB_FILE):
        # save dict as npz: keys are user ids; arrays are embeddings
        np.savez_compressed(path, **{k: v for k, v in self.db.items()})
        print(f"[DB] Saved embeddings DB to {path}")

    def load_db(self, path=EMBEDDINGS_DB_FILE):
        if not os.path.exists(path):
            print("[DB] No embeddings DB found; start fresh.")
            return
        try:
            data = np.load(path, allow_pickle=True)
            for key in data.files:
                self.db[key] = data[key]
            print(f"[DB] Loaded embeddings DB with users: {list(self.db.keys())}")
        except Exception as e:
            print("Failed to load DB:", e)

    # -----------------------
    # Payment stub
    # -----------------------
    def process_payment(self, user_id):
        """
        Placeholder: integrate your payment logic here (UPI / Razorpay / backend API).
        This runs in its own thread to avoid blocking the recognition loop.
        """
        print(f"[PAYMENT] Triggering payment for {user_id} ... (stub)")
        # Example: call your backend API to create payment order and confirm.
        time.sleep(0.5)  # simulate network delay
        print(f"[PAYMENT] Payment for {user_id} completed (simulated).")

# -----------------------
# Simple Tkinter GUI
# -----------------------
class PalmApp:
    def __init__(self, root):
        self.root = root
        root.title("PalmPay - High Accuracy Recognizer")
        root.geometry("360x280")
        self.sys = PalmRecognizer()

        btn_enroll = tk.Button(root, text="Enroll User", width=25, command=self.enroll_ui)
        btn_recognize = tk.Button(root, text="Start Recognition", width=25, command=self.start_recognize_thread)
        btn_save = tk.Button(root, text="Save DB", width=25, command=self.sys.save_db)
        btn_load = tk.Button(root, text="Load DB", width=25, command=self.sys.load_db)
        btn_quit = tk.Button(root, text="Quit", width=25, fg="white", bg="firebrick", command=self.quit_app)

        btn_enroll.pack(pady=10)
        btn_recognize.pack(pady=10)
        btn_save.pack(pady=8)
        btn_load.pack(pady=8)
        btn_quit.pack(pady=12)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var, fg="blue").pack(pady=6)

    def enroll_ui(self):
        user_id = simpledialog.askstring("Enroll", "Enter unique user ID:", parent=self.root)
        if not user_id:
            return
        self.status_var.set(f"Enrolling {user_id} ...")
        threading.Thread(target=self._enroll_thread, args=(user_id,), daemon=True).start()

    def _enroll_thread(self, user_id):
        ok = self.sys.enroll_user(user_id, NUM_ENROLL_SAMPLES)
        if ok:
            self.status_var.set(f"Enrolled {user_id}")
        else:
            self.status_var.set("Enroll failed / aborted")

    def start_recognize_thread(self):
        if not self.sys.db:
            messagebox.showwarning("No Users", "No enrolled users found. Enroll first.")
            return
        self.sys.running = True
        self.status_var.set("Recognition running...")
        threading.Thread(target=self.sys.recognize, daemon=True).start()

    def quit_app(self):
        self.sys.running = False
        time.sleep(0.3)
        self.root.quit()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PalmApp(root)
    root.mainloop()
