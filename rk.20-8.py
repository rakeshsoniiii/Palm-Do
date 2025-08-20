import os
import cv2
import time
import pickle
import threading
import numpy as np
from collections import defaultdict
from math import atan2, degrees, sqrt
import tkinter as tk
from tkinter import simpledialog, messagebox

# sklearn & joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import joblib

# mediapipe & tts
import mediapipe as mp
import pyttsx3

DATA_FILE = "palm_features.pkl"
MODEL_FILE = "palm_model.joblib"


def speak(text):
    """Voice assistant via pyttsx3 (non-blocking)"""
    print(text)  # Fallback to print instead of speech
    return
    # Disabled TTS due to issues
    def _s():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_s, daemon=True).start()


class PalmBiometricSystem:
    def __init__(self):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Data storage: label -> list of feature vectors
        self.features = []  # list of np arrays
        self.labels = []    # corresponding labels

        # Model
        self.model = None

        # Flags
        self.capture_running = False
        self.recognize_running = False

        # Load if available
        self.load_data()

    # -------------------------
    # Feature extraction
    # -------------------------
    def _landmark_to_np(self, landmarks):
        """Convert MediaPipe landmarks into Nx3 numpy array"""
        arr = np.zeros((21, 3), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark):
            arr[i, 0] = lm.x
            arr[i, 1] = lm.y
            arr[i, 2] = lm.z
        return arr

    def extract_features(self, landmarks):
        """
        Build a scale-invariant, rotation-robust feature vector:
         - normalized fingertip positions relative to wrist, scaled by palm size
         - pairwise fingertip distances
         - angles between fingers
         - palm width/height ratio
        Returns a 1D numpy array of fixed length.
        """
        pts = self._landmark_to_np(landmarks)  # shape (21,3)
        wrist = pts[0, :2]

        # Key indices
        tips = [4, 8, 12, 16, 20]    # thumb, index, middle, ring, pinky tips
        bases = [2, 5, 9, 13, 17]    # some base points for palms/fingers

        # Compute palm size as distance between wrist and middle_mcp (9) or bounding box diagonal
        mcp_middle = pts[9, :2]
        palm_size = np.linalg.norm(mcp_middle - wrist) + 1e-6

        # Normalized fingertip positions relative to wrist, scaled by palm_size
        pos_feats = []
        for idx in tips:
            rel = (pts[idx, :2] - wrist) / palm_size
            pos_feats.extend([rel[0], rel[1]])

        # Inter-fingertip pairwise distances (normalized)
        dist_feats = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(pts[tips[i], :2] - pts[tips[j], :2]) / palm_size
                dist_feats.append(d)

        # Angles between vector from wrist to each fingertip (in degrees)
        angle_feats = []
        for idx in tips:
            v = pts[idx, :2] - wrist
            ang = degrees(atan2(v[1], v[0])) / 180.0  # normalize by 180
            angle_feats.append(ang)

        # Palm aspect ratio: width (distance between index_base and pinky_base) / height (wrist to middle_tip)
        index_base = pts[5, :2]
        pinky_base = pts[17, :2]
        middle_tip = pts[12, :2]
        palm_width = np.linalg.norm(index_base - pinky_base)
        palm_height = np.abs(middle_tip[1] - wrist[1]) + 1e-6
        ratio = (palm_width / palm_height)

        feats = np.array(pos_feats + dist_feats + angle_feats + [ratio], dtype=np.float32)
        return feats

    # -------------------------
    # Scanning / Registering
    # -------------------------
    def scan_palm_for_user(self, user_id, num_scans=6):
        """
        Open camera, let user press 'c' to capture each valid scan (hand detected).
        Stores features and returns number of captures saved.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Cannot open camera")
            print("Cannot open camera")
            return 0

        speak(f"Starting capture for {user_id}. Please show your palm.")
        print(f"Starting capture for {user_id}. Press 'c' to capture. Need {num_scans} captures.")
        saved = 0
        start_time = time.time()

        while saved < num_scans and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Detected - Press 'c' to capture ({saved}/{num_scans})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"No hand detected - Adjust your palm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(f"Registering: {user_id}", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and results.multi_hand_landmarks:
                try:
                    feats = self.extract_features(results.multi_hand_landmarks[0])
                    self.features.append(feats)
                    self.labels.append(user_id)
                    saved += 1
                    speak(f"Captured {saved} for {user_id}")
                    print(f"Captured {saved}/{num_scans}")
                    cv2.imshow("Capture Saved", frame)
                    cv2.waitKey(300)
                    cv2.destroyWindow("Capture Saved")
                except Exception as e:
                    print("Feature extraction failed:", e)
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        duration = time.time() - start_time
        print(f"Finished capturing for {user_id}. Saved: {saved} captures in {duration:.1f}s")
        if saved > 0:
            speak(f"Saved {saved} captures for {user_id}")
        return saved

    # -------------------------
    # Training
    # -------------------------
    def train_model(self, force_grid=False):
        """Train SVC on stored features. Uses GridSearchCV for improved hyperparams."""
        if len(self.labels) < 2:
            speak("Need at least two users to train")
            print("Need at least two different users to train.")
            return False

        X = np.vstack(self.features)
        y = np.array(self.labels)

        # If dataset is small, do simple split and gridsearch with small CV
        param_grid = {
            'svc__C': [1, 10],
            'svc__gamma': ['scale', 0.1]
        }

        pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

        # Grid search
        try:
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X, y)
            best = grid.best_estimator_
            self.model = best
            # Evaluate with cross-val
            scores = cross_val_score(self.model, X, y, cv=3)
            mean_acc = float(scores.mean())
            print(f"Grid search best params: {grid.best_params_}; CV acc: {mean_acc:.3f}")
            speak(f"Model trained with accuracy {mean_acc:.2f}")
            return True
        except Exception as e:
            print("GridSearch/Training failed, trying fallback simple fit:", e)
            try:
                pipeline.fit(X, y)
                self.model = pipeline
                acc = pipeline.score(X, y)
                print(f"Fallback training accuracy (train set): {acc:.3f}")
                speak("Model trained")
                return True
            except Exception as e2:
                print("Training completely failed:", e2)
                speak("Training failed")
                return False

    # -------------------------
    # Real-time recognition
    # -------------------------
    def recognize_loop(self):
        """Real-time recognition loop using camera. Runs until 'q' pressed or flag turned off."""
        if self.model is None:
            speak("Model not trained yet")
            print("Model not trained. Please train first.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Cannot open camera")
            print("Cannot open camera")
            return

        speak("Starting recognition. Press q to quit.")
        print("Starting recognition. Press 'q' to quit.")
        recent_predictions = []
        while self.recognize_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

                try:
                    feats = self.extract_features(lm)
                    probs = self.model.predict_proba([feats])[0]
                    pred = self.model.classes_[np.argmax(probs)]
                    conf = np.max(probs)
                    recent_predictions.append((pred, conf))
                    # Keep last few predictions to stabilize
                    if len(recent_predictions) > 6:
                        recent_predictions.pop(0)
                    # Voting
                    votes = defaultdict(float)
                    for p, c in recent_predictions:
                        votes[p] += c
                    best = max(votes.items(), key=lambda x: x[1])
                    label, score_sum = best
                    # normalize score
                    score = score_sum / (len(recent_predictions) + 1e-6)
                    cv2.putText(frame, f"{label} ({score:.0%})", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    # Announce only if high confidence
                    if score > 0.7:
                        speak(f"Recognized {label} with confidence {int(score*100)} percent")
                except Exception as e:
                    print("Recognition feature extraction/prediction error:", e)

            cv2.imshow("Palm Recognition - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        speak("Stopping recognition")
        print("Recognition stopped.")

    # -------------------------
    # Save / Load
    # -------------------------
    def save_data(self, features_file=DATA_FILE, model_file=MODEL_FILE):
        try:
            with open(features_file, 'wb') as f:
                pickle.dump({'features': self.features, 'labels': self.labels}, f)
            if self.model is not None:
                joblib.dump(self.model, model_file)
            print(f"Saved features to {features_file} and model to {model_file}")
            speak("Data saved")
            return True
        except Exception as e:
            print("Save failed:", e)
            speak("Save failed")
            return False

    def delete_user(self, user_id):
        """Delete all data for a specific user"""
        if user_id not in self.labels:
            print(f"User {user_id} not found")
            speak(f"User {user_id} not found")
            return False
        
        # Get indices of all samples for this user
        indices = [i for i, label in enumerate(self.labels) if label != user_id]
        
        # Keep only samples that are not from this user
        self.features = [self.features[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Reset model since data changed
        self.model = None
        
        print(f"Deleted all data for user {user_id}")
        speak(f"Deleted all data for user {user_id}")
        return True

    def load_data(self, features_file=DATA_FILE, model_file=MODEL_FILE):
        loaded_any = False
        if os.path.exists(features_file):
            try:
                with open(features_file, 'rb') as f:
                    data = pickle.load(f)
                self.features = data.get('features', [])
                self.labels = data.get('labels', [])
                print(f"Loaded features: {len(self.features)} samples, {len(set(self.labels))} users")
                loaded_any = True
            except Exception as e:
                print("Failed to load features:", e)
        else:
            print("Features file not found. Starting fresh.")

        if os.path.exists(model_file):
            try:
                self.model = joblib.load(model_file)
                print("Loaded trained model.")
                loaded_any = True
            except Exception as e:
                print("Failed to load model:", e)
        else:
            print("Model file not found; model not loaded.")

        return loaded_any

# -------------------------
# Tkinter GUI wrapper
# -------------------------
class PalmApp:
    def __init__(self, root):
        self.root = root
        root.title("Palm Biometric System")
        root.geometry("420x320")
        self.sys = PalmBiometricSystem()

        # Buttons
        btn_register = tk.Button(root, text="Register New User", width=25, command=self.register_user)
        btn_delete = tk.Button(root, text="Delete User", width=25, command=self.delete_user)
        btn_train = tk.Button(root, text="Train Recognition Model", width=25, command=self.train_model)
        btn_recognize = tk.Button(root, text="Real-time Recognition", width=25, command=self.start_recognition_thread)
        btn_save = tk.Button(root, text="Save Data", width=25, command=self.save_data)
        btn_load = tk.Button(root, text="Load Data", width=25, command=self.load_data)
        btn_quit = tk.Button(root, text="Exit", width=25, fg="white", bg="firebrick", command=self.quit_app)

        # Layout
        btn_register.pack(pady=10)
        btn_delete.pack(pady=5)
        btn_train.pack(pady=5)
        btn_recognize.pack(pady=5)
        btn_save.pack(pady=5)
        btn_load.pack(pady=5)
        btn_quit.pack(pady=15)

        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var, fg="blue")
        self.status_label.pack(pady=5)

    def set_status(self, txt):
        self.status_var.set(txt)
        print(txt)

    # Register flow
    def register_user(self):
        user_id = simpledialog.askstring("User ID", "Enter user ID (unique):", parent=self.root)
        if not user_id:
            return
        num = simpledialog.askinteger("Captures", "Number of captures (default 6):", initialvalue=6, minvalue=3, maxvalue=25, parent=self.root)
        if not num:
            num = 6

        self.set_status(f"Registering {user_id}...")
        self.sys.capture_running = True

        def _do():
            saved = self.sys.scan_palm_for_user(user_id, num_scans=num)
            self.sys.capture_running = False
            self.set_status(f"Captured {saved} scans for {user_id}")
            speak(f"Captured {saved} scans for {user_id}")

        threading.Thread(target=_do, daemon=True).start()

    def train_model(self):
        self.set_status("Training model...")
        def _do():
            ok = self.sys.train_model()
            if ok:
                self.set_status("Model trained successfully")
            else:
                self.set_status("Model training failed")
        threading.Thread(target=_do, daemon=True).start()

    def start_recognition_thread(self):
        if self.sys.model is None:
            messagebox.showwarning("Model missing", "Please train or load a model first.")
            return
        if self.sys.recognize_running:
            messagebox.showinfo("Recognition", "Recognition already running.")
            return
        self.sys.recognize_running = True
        self.set_status("Recognition running...")
        threading.Thread(target=self.sys.recognize_loop, daemon=True).start()

    def save_data(self):
        ok = self.sys.save_data()
        if ok:
            self.set_status("Data saved")
        else:
            self.set_status("Save failed")

    def load_data(self):
        ok = self.sys.load_data()
        if ok:
            self.set_status("Data loaded")
        else:
            self.set_status("No saved data found")

    def delete_user(self):
        if not self.sys.labels:
            messagebox.showwarning("No Data", "No user data available to delete.")
            return
        
        # Show list of unique users
        unique_users = list(set(self.sys.labels))
        user_list = "\n".join(unique_users)
        
        user_id = simpledialog.askstring(
            "Delete User", 
            f"Enter user ID to delete.\nAvailable users:\n{user_list}",
            parent=self.root
        )
        
        if not user_id:
            return
            
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete all data for user {user_id}?"):
            if self.sys.delete_user(user_id):
                self.set_status(f"Deleted user {user_id}")
                # After deleting, save changes
                self.sys.save_data()
            else:
                self.set_status(f"User {user_id} not found")

    def quit_app(self):
        if self.sys.recognize_running:
            self.sys.recognize_running = False
            time.sleep(0.5)
        self.root.quit()


def main():
    root = tk.Tk()
    app = PalmApp(root)
    root.mainloop()


if __name__ == "__main__":
    # If packages missing, the import at top will already fail; handle at runtime.
    try:
        main()
    except Exception as e:
        print("Application error:", e)
        print("Ensure required packages are installed: mediapipe, opencv-python, scikit-learn, joblib, pyttsx3, numpy")
