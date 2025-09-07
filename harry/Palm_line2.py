# final_palm_app.py
import os, cv2, numpy as np, threading, tensorflow as tf
import mediapipe as mp
import tkinter as tk
from tkinter import simpledialog, messagebox
import pyttsx3

# -----------------------
# Config
# -----------------------
IMG_SIZE = 128
NUM_ENROLL_SAMPLES = 80
SIM_THRESHOLD = 0.65
EMBEDDINGS_DB_FILE = "embeddings_db.npz"

# -----------------------
# Text-to-speech
# -----------------------
tts = pyttsx3.init()
def speak(text):
    threading.Thread(target=lambda: tts.say(text) or tts.runAndWait()).start()

# -----------------------
# Embedding model
# -----------------------
def build_embedding_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), embed_dim=128):
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp*255.0)
    x = base(x, training=False)
    x = tf.keras.layers.Dense(embed_dim, use_bias=False)(x)
    x = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)
    return tf.keras.Model(inputs=inp, outputs=x)

# -----------------------
# Cosine similarity
# -----------------------
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a)+1e-10)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True)+1e-10)
    return np.dot(b, a)

# -----------------------
# Palm Recognizer
# -----------------------
class PalmRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.embed_model = build_embedding_model()
        self.db = {}
        self.load_db()

    def crop_palm_roi(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks.landmark]
        idxs = [0,1,2,5,9,13,17]
        xs = [pts[i][0] for i in idxs]
        ys = [pts[i][1] for i in idxs]
        x_min, x_max = max(0,min(xs)-30), min(w,max(xs)+30)
        y_min, y_max = max(0,min(ys)-30), min(h,max(ys)+30)
        return frame[y_min:y_max, x_min:x_max].copy(), x_min, y_min, y_max, x_max

    def preprocess_roi(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size==0: return None
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 40, 100)
        img = cv2.resize(edges, (IMG_SIZE, IMG_SIZE))
        img = np.stack([img]*3, axis=-1)/255.0
        return img

    def enroll_user(self, user_id, num_samples=NUM_ENROLL_SAMPLES):
        cap = cv2.VideoCapture(0)
        collected = []
        speak(f"Enrollment started for {user_id}. Show your right palm.")
        while len(collected)<num_samples:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            display_text = "Show RIGHT palm"

            if results.multi_hand_landmarks and results.multi_handedness:
                lm = results.multi_hand_landmarks[0]
                label = results.multi_handedness[0].classification[0].label
                if label!="Right":
                    display_text = "Please use RIGHT hand!"
                    speak(display_text)
                else:
                    roi, _, _, _, _ = self.crop_palm_roi(frame, lm)
                    img = self.preprocess_roi(roi)
                    if img is not None:
                        emb = self.embed_model.predict(np.expand_dims(img,axis=0), verbose=0)[0]
                        collected.append(emb)
                    display_text = f"Collected: {len(collected)}/{num_samples}"

            cv2.putText(frame, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
            cv2.imshow("Enroll Palm", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        if len(collected)>5:
            arr = np.vstack(collected)
            if user_id in self.db:
                self.db[user_id] = np.vstack([self.db[user_id], arr])
            else:
                self.db[user_id] = arr
            self.save_db()
            speak(f"Enrollment done for {user_id}")
        else:
            speak("Enrollment failed. Not enough samples.")

    def match_embedding(self, emb):
        best_user, best_score = None, -1
        for user, arr in self.db.items():
            sims = cosine_similarity(emb, arr)
            top = float(np.max(sims))
            if top>best_score:
                best_score = top
                best_user = user
        return best_user, best_score

    def recognize(self):
        if not self.db:
            speak("No enrolled users. Please enroll first.")
            return
        cap = cv2.VideoCapture(0)
        speak("Recognition started. Show your right palm.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            display_text = "No hand"

            if results.multi_hand_landmarks and results.multi_handedness:
                lm = results.multi_hand_landmarks[0]
                label = results.multi_handedness[0].classification[0].label
                if label!="Right":
                    display_text="Please use RIGHT hand!"
                    speak(display_text)
                else:
                    roi, x_min, y_min, y_max, x_max = self.crop_palm_roi(frame, lm)
                    # Palm lines
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    edges = cv2.Canny(enhanced, 40, 100)
                    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    frame[y_min:y_max, x_min:x_max] = edges_color

                    img = self.preprocess_roi(roi)
                    if img is not None:
                        emb = self.embed_model.predict(np.expand_dims(img,axis=0), verbose=0)[0]
                        user, score = self.match_embedding(emb)
                        if score>=SIM_THRESHOLD:
                            display_text=f"Recognized: {user} ({score:.2f})"
                        else:
                            display_text="No match"
            cv2.putText(frame, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
            cv2.imshow("Palm Recognizer", frame)
            if cv2.waitKey(1) & 0xFF==ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

    def save_db(self):
        np.savez_compressed(EMBEDDINGS_DB_FILE, **{k:v for k,v in self.db.items()})
        speak("Database saved.")

    def load_db(self):
        if os.path.exists(EMBEDDINGS_DB_FILE):
            data = np.load(EMBEDDINGS_DB_FILE, allow_pickle=True)
            for key in data.files:
                self.db[key] = data[key]
            speak("Database loaded.")

# -----------------------
# GUI
# -----------------------
class PalmApp:
    def __init__(self, root):
        self.root = root
        root.title("PalmPay - Real-time Palm Lines")
        root.geometry("360x280")
        self.sys = PalmRecognizer()

        tk.Button(root,text="Enroll User", width=25, command=self.enroll_ui).pack(pady=10)
        tk.Button(root,text="Start Recognition", width=25, command=self.start_recognize_thread).pack(pady=10)
        tk.Button(root,text="Save DB", width=25, command=self.sys.save_db).pack(pady=8)
        tk.Button(root,text="Load DB", width=25, command=self.sys.load_db).pack(pady=8)
        tk.Button(root,text="Quit", width=25, fg="white", bg="firebrick", command=self.quit_app).pack(pady=12)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root,textvariable=self.status_var, fg="blue").pack(pady=6)

    def enroll_ui(self):
        user_id = simpledialog.askstring("Enroll", "Enter unique user ID:", parent=self.root)
        if not user_id: return
        self.status_var.set(f"Enrolling {user_id}...")
        threading.Thread(target=lambda:self.sys.enroll_user(user_id),daemon=True).start()

    def start_recognize_thread(self):
        if not self.sys.db:
            messagebox.showwarning("No Users","No enrolled users found.")
            return
        threading.Thread(target=self.sys.recognize,daemon=True).start()
        self.status_var.set("Recognition running...")

    def quit_app(self):
        self.root.quit()

# -----------------------
# Run
# -----------------------
if __name__=="__main__":
    root=tk.Tk()
    app=PalmApp(root)
    root.mainloop()
