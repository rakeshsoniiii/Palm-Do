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

# TensorFlow for embeddings and improved training
import tensorflow as tf
from tensorflow.keras import layers, models

DATA_FILE = "palm_features.pkl"
MODEL_FILE = "palm_model.joblib"
EMBEDDINGS_FILE = "palm_embeddings.npy"
IMAGES_FILE = "palm_images.npy"
LABELS_FILE = "palm_labels.npy"

# Global TTS engine to avoid initialization issues
tts_engine = None

def init_tts():
    """Initialize the TTS engine once"""
    global tts_engine
    if tts_engine is None:
        try:
            tts_engine = pyttsx3.init()
            # Set properties (optional)
            tts_engine.setProperty('rate', 150)  # Speed percent
            tts_engine.setProperty('volume', 0.9)  # Volume 0-1
        except Exception as e:
            print("TTS initialization error:", e)
            tts_engine = None
    return tts_engine

def speak(text):
    """Voice assistant via pyttsx3 (non-blocking)"""
    print(text)  # Always print for fallback
    
    def _speak():
        try:
            engine = init_tts()
            if engine:
                engine.say(text)
                engine.runAndWait()
            else:
                print("TTS engine not available")
        except Exception as e:
            print("TTS error:", e)
    
    # Run in a thread to avoid blocking
    threading.Thread(target=_speak, daemon=True).start()


class PalmBiometricSystem:
    def __init__(self):
        # MediaPipe Hands and Palm
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        
        # Palm lines detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Data storage
        self.features = []  # list of np arrays
        self.labels = []    # corresponding labels
        self.embeddings = []  # TensorFlow embeddings
        self.images = []     # Raw images for training
        self.embedding_model = None

        # Model
        self.model = None
        self.label_mapping = []

        # Flags
        self.capture_running = False
        self.recognize_running = False
        self.auto_training = True  # Enable auto-training by default

        # Load if available
        self.load_data()
        self.build_embedding_model()

    def build_embedding_model(self):
        """Build a simple CNN model for palm embeddings"""
        input_shape = (128, 128, 3)  # Resized image dimensions
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu')  # Embedding vector
        ])
        
        self.embedding_model = model

    def is_right_hand(self, landmarks):
        """Check if the detected hand is a right hand"""
        # Use wrist and thumb landmarks to determine handedness
        wrist = landmarks.landmark[0]
        thumb_cmc = landmarks.landmark[1]
        pinky_mcp = landmarks.landmark[17]
        
        # For right hand, thumb is on the left side of the hand
        # when palm is facing the camera
        if thumb_cmc.x < wrist.x and pinky_mcp.x > wrist.x:
            return True
        return False

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

    def extract_features(self, landmarks, image=None):
        """
        Build a scale-invariant, rotation-robust feature vector:
         - normalized fingertip positions relative to wrist, scaled by palm size
         - pairwise fingertip distances
         - angles between fingers
         - palm width/height ratio
         - palm line features (if image provided)
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

        # Palm line features (if image provided)
        line_feats = []
        if image is not None:
            try:
                # Extract palm region based on landmarks
                palm_region = self.extract_palm_region(image, landmarks)
                line_feats = self.extract_palm_line_features(palm_region)
            except Exception as e:
                print("Palm line feature extraction failed:", e)
                line_feats = [0] * 10  # Default features if extraction fails

        feats = np.array(pos_feats + dist_feats + angle_feats + [ratio] + line_feats, dtype=np.float32)
        return feats

    def extract_palm_region(self, image, landmarks):
        """Extract the palm region from the image based on landmarks"""
        h, w = image.shape[:2]
        pts = self._landmark_to_np(landmarks)
        
        # Get bounding box around palm (using wrist and finger bases)
        palm_points = pts[[0, 1, 5, 9, 13, 17], :2]  # wrist, thumb cmc, finger mcp joints
        palm_points = (palm_points * [w, h]).astype(np.int32)
        
        # Expand the bounding box a bit
        x_min = max(0, np.min(palm_points[:, 0]) - 20)
        y_min = max(0, np.min(palm_points[:, 1]) - 20)
        x_max = min(w, np.max(palm_points[:, 0]) + 20)
        y_max = min(h, np.max(palm_points[:, 1]) + 20)
        
        # Extract and return the palm region
        return image[y_min:y_max, x_min:x_max]

    def extract_palm_line_features(self, palm_region):
        """Extract features from palm lines using image processing"""
        if palm_region.size == 0:
            return [0] * 10
            
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(palm_region, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply edge detection
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        
        # Extract line features
        line_count = 0
        avg_length = 0
        avg_angle = 0
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            line_count = len(lines)
            lengths = []
            angles = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                lengths.append(length)
                
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                angles.append(angle)
                
                if -30 <= angle <= 30:
                    horizontal_lines += 1
                elif 60 <= angle <= 120 or -120 <= angle <= -60:
                    vertical_lines += 1
            
            if lengths:
                avg_length = np.mean(lengths)
            if angles:
                avg_angle = np.mean(angles)
        
        # Return a feature vector based on line characteristics
        return [
            line_count, avg_length, avg_angle, 
            horizontal_lines, vertical_lines,
            line_count/100 if line_count > 0 else 0,
            horizontal_lines/line_count if line_count > 0 else 0,
            vertical_lines/line_count if line_count > 0 else 0,
            np.std(lengths) if lengths else 0,
            np.std(angles) if angles else 0
        ]

    def create_embedding(self, image):
        """Create an embedding vector from the palm image using the CNN model"""
        if image is None or self.embedding_model is None:
            return np.zeros(32)  # Return zero vector if no image or model
        
        # Preprocess the image
        resized = cv2.resize(image, (128, 128))
        normalized = resized / 255.0
        expanded = np.expand_dims(normalized, axis=0)
        
        # Generate embedding
        embedding = self.embedding_model.predict(expanded, verbose=0)[0]
        return embedding

    # -------------------------
    # Scanning / Registering
    # -------------------------
    def scan_palm_for_user(self, user_id, num_scans=10):
        """
        Open camera, automatically capture images every 2 seconds if hand is detected.
        Stores features, embeddings, and images.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Cannot open camera")
            print("Cannot open camera")
            return 0

        speak(f"Starting capture for {user_id}. Please show your right palm.")
        print(f"Starting capture for {user_id}. Need {num_scans} captures.")
        saved = 0
        start_time = time.time()
        last_capture_time = 0
        wrong_hand_warning_time = 0

        while saved < num_scans and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # Check if it's a right hand
                if not self.is_right_hand(landmarks):
                    current_time = time.time()
                    # Only speak warning every 5 seconds to avoid spamming
                    if current_time - wrong_hand_warning_time > 5:
                        speak("Please use your right hand, not your left hand")
                        wrong_hand_warning_time = current_time
                    
                    cv2.putText(frame, "Please use your RIGHT hand", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(f"Registering: {user_id}", frame)
                    cv2.waitKey(1)
                    continue
                
                self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Auto-capture every 2 seconds if hand is detected
                current_time = time.time()
                if current_time - last_capture_time >= 2:
                    try:
                        feats = self.extract_features(landmarks, frame)
                        embedding = self.create_embedding(frame)
                        
                        self.features.append(feats)
                        self.embeddings.append(embedding)
                        self.images.append(frame)
                        self.labels.append(user_id)
                        
                        saved += 1
                        last_capture_time = current_time
                        
                        speak(f"Captured {saved} for {user_id}")
                        print(f"Captured {saved}/{num_scans}")
                        
                        # Show capture feedback
                        cv2.imshow("Capture Saved", frame)
                        cv2.waitKey(300)
                        cv2.destroyWindow("Capture Saved")
                        
                        # Auto-train if we have enough samples
                        if saved == num_scans and self.auto_training:
                            speak("Training model with captured data")
                            self.train_model()
                            
                    except Exception as e:
                        print("Feature extraction failed:", e)
                
                cv2.putText(frame, f"Detected - Auto-capturing ({saved}/{num_scans})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"No hand detected - Adjust your palm",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(f"Registering: {user_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        """Train model on stored features and embeddings using TensorFlow."""
        if len(self.labels) < 2:
            speak("Need at least two users to train")
            print("Need at least two different users to train.")
            return False

        # Prepare data
        X_features = np.vstack(self.features)
        X_embeddings = np.vstack(self.embeddings)
        
        # Combine traditional features with embeddings
        X_combined = np.hstack([X_features, X_embeddings])
        y = np.array(self.labels)
        
        # Convert labels to numerical indices
        unique_labels = list(set(self.labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_numeric = np.array([label_to_idx[label] for label in self.labels])
        
        # Build a simple neural network classifier
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_combined.shape[1],)),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(unique_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        try:
            history = model.fit(
                X_combined, y_numeric,
                epochs=50,
                batch_size=16,
                validation_split=0.2,
                verbose=0
            )
            
            # Store the trained model and label mapping
            self.model = model
            self.label_mapping = unique_labels
            
            # Evaluate the model
            val_acc = history.history['val_accuracy'][-1]
            print(f"Model trained with validation accuracy: {val_acc:.3f}")
            speak(f"Model trained successfully with accuracy {val_acc:.2f}")
            
            return True
        except Exception as e:
            print("Training failed:", e)
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
        wrong_hand_warning_time = 0
        
        while self.recognize_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                
                # Check if it's a right hand
                if not self.is_right_hand(lm):
                    current_time = time.time()
                    # Only speak warning every 5 seconds to avoid spamming
                    if current_time - wrong_hand_warning_time > 5:
                        speak("Please use your right hand, not your left hand")
                        wrong_hand_warning_time = current_time
                    
                    cv2.putText(frame, "Please use your RIGHT hand", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow("Palm Recognition - press q to quit", frame)
                    cv2.waitKey(1)
                    continue
                
                self.mp_drawing.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

                try:
                    # Extract features and embedding
                    feats = self.extract_features(lm, frame)
                    embedding = self.create_embedding(frame)
                    
                    # Combine features
                    combined = np.hstack([feats, embedding])
                    combined = np.expand_dims(combined, axis=0)
                    
                    # Predict
                    probs = self.model.predict(combined, verbose=0)[0]
                    pred_idx = np.argmax(probs)
                    pred = self.label_mapping[pred_idx]
                    conf = probs[pred_idx]
                    
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
    def save_data(self, features_file=DATA_FILE, model_file=MODEL_FILE, 
                 embeddings_file=EMBEDDINGS_FILE, images_file=IMAGES_FILE, labels_file=LABELS_FILE):
        try:
            # Save features and labels
            with open(features_file, 'wb') as f:
                pickle.dump({'features': self.features, 'labels': self.labels}, f)
            
            # Save embeddings
            np.save(embeddings_file, np.array(self.embeddings))
            
            # Save images (compressed)
            np.savez_compressed(images_file, *self.images)
            
            # Save labels
            np.save(labels_file, np.array(self.labels))
            
            # Save model if it exists
            if self.model is not None:
                self.model.save(model_file)
                
            print(f"Saved data to files")
            speak("Data saved successfully")
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
        self.embeddings = [self.embeddings[i] for i in indices]
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Reset model since data changed
        self.model = None
        
        print(f"Deleted all data for user {user_id}")
        speak(f"Deleted all data for user {user_id}")
        return True

    def load_data(self, features_file=DATA_FILE, model_file=MODEL_FILE, 
                 embeddings_file=EMBEDDINGS_FILE, images_file=IMAGES_FILE, labels_file=LABELS_FILE):
        loaded_any = False
        
        # Load features and labels
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

        # Load embeddings
        if os.path.exists(embeddings_file):
            try:
                self.embeddings = np.load(embeddings_file, allow_pickle=True).tolist()
                print(f"Loaded embeddings: {len(self.embeddings)}")
                loaded_any = True
            except Exception as e:
                print("Failed to load embeddings:", e)

        # Load images
        if os.path.exists(images_file + '.npz'):
            try:
                loaded_images = np.load(images_file + '.npz', allow_pickle=True)
                self.images = [loaded_images[f'arr_{i}'] for i in range(len(loaded_images.files))]
                print(f"Loaded images: {len(self.images)}")
                loaded_any = True
            except Exception as e:
                print("Failed to load images:", e)

        # Load model
        if os.path.exists(model_file):
            try:
                self.model = tf.keras.models.load_model(model_file)
                print("Loaded trained model.")
                loaded_any = True
                
                # Load label mapping from labels file
                if os.path.exists(labels_file):
                    all_labels = np.load(labels_file, allow_pickle=True)
                    self.label_mapping = list(set(all_labels))
                    speak("Model loaded successfully")
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
        self.sys.auto_training = True  # Enable auto-training

        # Initialize TTS
        init_tts()

        # Buttons
        btn_register = tk.Button(root, text="Register New User", width=25, command=self.register_user)
        btn_delete = tk.Button(root, text="Delete User", width=25, command=self.delete_user)
        btn_recognize = tk.Button(root, text="Real-time Recognition", width=25, command=self.start_recognition_thread)
        btn_save = tk.Button(root, text="Save Data", width=25, command=self.save_data)
        btn_load = tk.Button(root, text="Load Data", width=25, command=self.load_data)
        btn_quit = tk.Button(root, text="Exit", width=25, fg="white", bg="firebrick", command=self.quit_app)

        # Layout
        btn_register.pack(pady=10)
        btn_delete.pack(pady=5)
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
        
        self.set_status(f"Registering {user_id}...")
        self.sys.capture_running = True

        def _do():
            saved = self.sys.scan_palm_for_user(user_id, num_scans=10)
            self.sys.capture_running = False
            self.set_status(f"Captured {saved} scans for {user_id}")
            speak(f"Captured {saved} scans for {user_id}")

        threading.Thread(target=_do, daemon=True).start()

    def start_recognition_thread(self):
        if self.sys.model is None:
            messagebox.showwarning("Model missing", "Please register users first.")
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
        print("Ensure required packages are installed: mediapipe, opencv-python, scikit-learn, joblib, pyttsx3, numpy, tensorflow")
