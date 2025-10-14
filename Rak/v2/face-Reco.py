import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
import os
from pathlib import Path
import threading
import time
import pyttsx3
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize

class FaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("InsightFace Recognition System")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")
        
        # Initialize variables
        self.cap = None
        self.is_running = False
        self.is_capturing = False
        self.face_database = {}
        self.database_file = "face_database.pkl"
        self.captured_images = []
        self.capture_count = 0
        self.current_name = ""
        
        # Initialize InsightFace
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # Load existing database
        self.load_database()
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="Face Recognition System", 
            font=("Arial", 24, "bold"),
            bg="#2c3e50", 
            fg="white"
        )
        title_label.pack(pady=10)
        
        # Video frame
        self.video_frame = tk.Label(self.root, bg="black")
        self.video_frame.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root, 
            text="Status: Ready", 
            font=("Arial", 12),
            bg="#2c3e50", 
            fg="#ecf0f1"
        )
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, 
            length=400, 
            mode='determinate'
        )
        self.progress.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg="#2c3e50")
        button_frame.pack(pady=10)
        
        # Buttons
        self.start_btn = tk.Button(
            button_frame,
            text="Start Camera",
            command=self.start_camera,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            width=15,
            height=2
        )
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.register_btn = tk.Button(
            button_frame,
            text="Register Face",
            command=self.register_face,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.register_btn.grid(row=0, column=1, padx=5)
        
        self.recognize_btn = tk.Button(
            button_frame,
            text="Start Recognition",
            command=self.toggle_recognition,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.recognize_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="Stop Camera",
            command=self.stop_camera,
            font=("Arial", 12, "bold"),
            bg="#95a5a6",
            fg="white",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.stop_btn.grid(row=0, column=3, padx=5)
        
        # Registered faces label
        self.faces_label = tk.Label(
            self.root,
            text=f"Registered Faces: {len(self.face_database)}",
            font=("Arial", 11),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        self.faces_label.pack(pady=5)
        
    def speak(self, text):
        """Text-to-speech in separate thread"""
        def _speak():
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
        
    def load_database(self):
        """Load face database from file"""
        if os.path.exists(self.database_file):
            try:
                with open(self.database_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"Loaded {len(self.face_database)} faces from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.face_database = {}
        
    def save_database(self):
        """Save face database to file"""
        try:
            with open(self.database_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            print("Database saved successfully")
        except Exception as e:
            print(f"Error saving database: {e}")
            
    def start_camera(self):
        """Start camera feed"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.register_btn.config(state=tk.NORMAL)
        self.recognize_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.update_frame()
        self.speak("Camera started")
        
    def stop_camera(self):
        """Stop camera feed"""
        self.is_running = False
        self.recognizing = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state=tk.NORMAL)
        self.register_btn.config(state=tk.DISABLED)
        self.recognize_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.video_frame.config(image='')
        self.status_label.config(text="Status: Camera Stopped")
        self.speak("Camera stopped")
        
    def update_frame(self):
        """Update video frame"""
        if not self.is_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Detect faces
            faces = self.app.get(frame)
            
            for face in faces:
                # Draw bounding box
                bbox = face.bbox.astype(int)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # If recognizing, identify the face
                if hasattr(self, 'recognizing') and self.recognizing:
                    name = self.recognize_face(face.embedding)
                    cv2.putText(display_frame, name, (bbox[0], bbox[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(image=img)
            
            self.video_frame.config(image=photo)
            self.video_frame.image = photo
        
        self.root.after(10, self.update_frame)
        
    def register_face(self):
        """Register a new face"""
        name = simpledialog.askstring("Register Face", "Enter person's name:")
        if not name or name.strip() == "":
            return
        
        self.current_name = name.strip()
        self.captured_images = []
        self.capture_count = 0
        self.is_capturing = True
        
        self.status_label.config(text=f"Capturing images for {self.current_name}...")
        self.progress['value'] = 0
        self.speak(f"Registering {self.current_name}. Please look at the camera")
        
        # Disable buttons during capture
        self.register_btn.config(state=tk.DISABLED)
        self.recognize_btn.config(state=tk.DISABLED)
        
        # Start capture thread
        thread = threading.Thread(target=self.capture_images, daemon=True)
        thread.start()
        
    def capture_images(self):
        """Capture 40 images as fast as possible"""
        target_count = 40
        
        while self.capture_count < target_count and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                faces = self.app.get(frame)
                
                if len(faces) > 0:
                    # Get the largest face
                    largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    self.captured_images.append(largest_face.embedding)
                    self.capture_count += 1
                    
                    # Update progress
                    progress_val = (self.capture_count / target_count) * 100
                    self.progress['value'] = progress_val
                    self.status_label.config(
                        text=f"Captured {self.capture_count}/{target_count} images"
                    )
        
        self.is_capturing = False
        
        # Train with captured images
        if len(self.captured_images) >= 20:
            self.train_face()
        else:
            self.root.after(0, lambda: messagebox.showwarning(
                "Warning", 
                f"Only captured {len(self.captured_images)} images. Please try again."
            ))
            self.root.after(0, lambda: self.status_label.config(text="Status: Capture failed"))
        
        # Re-enable buttons
        self.root.after(0, lambda: self.register_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.recognize_btn.config(state=tk.NORMAL))
        
    def train_face(self):
        """Train face model with captured images"""
        if len(self.captured_images) == 0:
            return
        
        self.status_label.config(text=f"Training model for {self.current_name}...")
        
        # Average the embeddings for better accuracy
        embeddings_array = np.array(self.captured_images)
        avg_embedding = np.mean(embeddings_array, axis=0)
        avg_embedding = normalize(avg_embedding.reshape(1, -1))[0]
        
        # Store in database
        self.face_database[self.current_name] = {
            'embedding': avg_embedding,
            'samples': embeddings_array
        }
        
        # Save database
        self.save_database()
        
        # Update UI
        self.faces_label.config(text=f"Registered Faces: {len(self.face_database)}")
        self.status_label.config(text=f"Status: {self.current_name} registered successfully!")
        self.progress['value'] = 0
        
        self.speak(f"{self.current_name} registered successfully")
        messagebox.showinfo("Success", f"{self.current_name} has been registered!")
        
    def recognize_face(self, embedding):
        """Recognize face from embedding"""
        if len(self.face_database) == 0:
            return "Unknown"
        
        embedding = normalize(embedding.reshape(1, -1))[0]
        
        min_distance = float('inf')
        recognized_name = "Unknown"
        threshold = 0.6  # Cosine similarity threshold
        
        for name, data in self.face_database.items():
            stored_embedding = data['embedding']
            
            # Calculate cosine similarity
            similarity = np.dot(embedding, stored_embedding)
            distance = 1 - similarity
            
            if distance < min_distance:
                min_distance = distance
                recognized_name = name
        
        # If distance is too large, it's unknown
        if min_distance > threshold:
            recognized_name = "Unknown"
        
        return recognized_name
        
    def toggle_recognition(self):
        """Toggle face recognition mode"""
        if not hasattr(self, 'recognizing'):
            self.recognizing = False
            
        if len(self.face_database) == 0:
            messagebox.showwarning("Warning", "No faces registered yet!")
            return
        
        self.recognizing = not self.recognizing
        
        if self.recognizing:
            self.recognize_btn.config(text="Stop Recognition", bg="#e67e22")
            self.register_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Recognizing faces...")
            self.speak("Face recognition started")
            self.last_recognized = {}
            self.start_recognition_loop()
        else:
            self.recognize_btn.config(text="Start Recognition", bg="#e74c3c")
            self.register_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Recognition stopped")
            self.speak("Face recognition stopped")
            
    def start_recognition_loop(self):
        """Continuous recognition with announcements"""
        if not self.recognizing:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            faces = self.app.get(frame)
            
            current_time = time.time()
            
            for face in faces:
                name = self.recognize_face(face.embedding)
                
                # Announce only once every 5 seconds per person
                if name != "Unknown":
                    if name not in self.last_recognized or \
                       current_time - self.last_recognized[name] > 5.0:
                        self.speak(f"Hello {name}")
                        self.last_recognized[name] = current_time
        
        # Schedule next recognition
        self.root.after(500, self.start_recognition_loop)
        
    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionSystem(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()

    
