#!/usr/bin/env python3
"""
Smart Palm Pay System by Rudraksha
A unified biometric payment and digital locker system
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
import os
import sqlite3
import threading
import time
import pyttsx3
import json
import socket
import uuid
from datetime import datetime
import mediapipe as mp
from insightface.app import FaceAnalysis
from sklearn.preprocessing import normalize
import math

class SmartPalmPaySystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Palm Pay by Rudraksha")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        
        # Initialize components
        self.init_databases()
        self.init_face_recognition()
        self.init_palm_detection()
        self.init_gesture_recognition()
        self.init_tts()
        
        # State variables
        self.current_user = None
        self.current_mode = "face_recognition"  # face_recognition, palm_scan, payment, gesture_pin, digital_locker
        self.cap = None
        self.camera_running = False
        self.payment_amount = 0
        self.pin_buffer = []
        
        self.create_main_ui()
        
    def init_databases(self):
        """Initialize all required databases"""
        # Face database
        self.face_database = {}
        self.face_db_file = "face_database.pkl"
        self.load_face_database()
        
        # User profiles database
        self.conn = sqlite3.connect('smart_palm_pay.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                balance REAL DEFAULT 10000,
                created_at TEXT,
                last_active TEXT
            )
        ''')
        
        # Digital locker table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS digital_locker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                document_type TEXT,
                document_name TEXT,
                document_data BLOB,
                created_at TEXT,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        # Transactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                sender TEXT,
                receiver TEXT,
                amount REAL,
                timestamp TEXT,
                status TEXT
            )
        ''')
        
        # Gesture PIN table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gesture_pins (
                username TEXT PRIMARY KEY,
                pin TEXT,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        self.conn.commit()
        
    def init_face_recognition(self):
        """Initialize face recognition system"""
        try:
            self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"Face recognition initialization error: {e}")
            self.face_app = None
            
    def init_palm_detection(self):
        """Initialize palm detection system"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def init_gesture_recognition(self):
        """Initialize gesture recognition for PIN"""
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [2, 6, 10, 14, 18]
        self.finger_mcp = [1, 5, 9, 13, 17]
        self.stable_count = -1
        self.stable_frames = 0
        self.cooldown_frames = 0
        self.in_cooldown = False
        
    def init_tts(self):
        """Initialize text-to-speech"""
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
        except:
            self.tts_engine = None
            
    def speak(self, text):
        """Text-to-speech function"""
        if self.tts_engine:
            def _speak():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except:
                    pass
            threading.Thread(target=_speak, daemon=True).start()
        print(f"[SYSTEM]: {text}")
        
    def create_main_ui(self):
        """Create the main user interface"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🖐️ Smart Palm Pay by Rudraksha",
            font=("Arial", 28, "bold"),
            bg="#1a1a2e",
            fg="#00d4ff"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side - Camera feed
        left_frame = tk.Frame(main_frame, bg="#16213e", relief=tk.RIDGE, bd=3)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        camera_label = tk.Label(
            left_frame,
            text="📷 Camera Feed",
            font=("Arial", 16, "bold"),
            bg="#16213e",
            fg="#00d4ff"
        )
        camera_label.pack(pady=10)
        
        self.video_label = tk.Label(left_frame, bg="#0f3460")
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right side - Status and controls
        right_frame = tk.Frame(main_frame, bg="#16213e", relief=tk.RIDGE, bd=3)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.config(width=400)
        
        # Status display
        self.status_label = tk.Label(
            right_frame,
            text="Welcome to Smart Palm Pay!\nPlace your face in front of camera for recognition",
            font=("Arial", 14),
            bg="#16213e",
            fg="#ffffff",
            wraplength=350,
            justify=tk.LEFT
        )
        self.status_label.pack(pady=20, padx=10)
        
        # User info display
        self.user_info_label = tk.Label(
            right_frame,
            text="User: Not logged in",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#ffcc00"
        )
        self.user_info_label.pack(pady=10)
        
        # Balance display
        self.balance_label = tk.Label(
            right_frame,
            text="Balance: ₹0.00",
            font=("Arial", 12, "bold"),
            bg="#16213e",
            fg="#00ff00"
        )
        self.balance_label.pack(pady=5)
        
        # PIN display (for gesture input)
        self.pin_display = tk.Label(
            right_frame,
            text="PIN: _ _ _ _ _",
            font=("Arial", 14, "bold"),
            bg="#16213e",
            fg="#ff6b6b"
        )
        self.pin_display.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(right_frame, bg="#16213e")
        button_frame.pack(pady=20)
        
        self.start_btn = tk.Button(
            button_frame,
            text="Start System",
            command=self.start_system,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            width=15,
            height=2
        )
        self.start_btn.pack(pady=5)
        
        self.register_btn = tk.Button(
            button_frame,
            text="Register New User",
            command=self.register_new_user,
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            width=15,
            height=2
        )
        self.register_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="Stop System",
            command=self.stop_system,
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.stop_btn.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(right_frame, length=300, mode='determinate')
        self.progress.pack(pady=10)
        
    def load_face_database(self):
        """Load face database from file"""
        if os.path.exists(self.face_db_file):
            try:
                with open(self.face_db_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                print(f"Loaded {len(self.face_database)} faces from database")
            except Exception as e:
                print(f"Error loading face database: {e}")
                self.face_database = {}
                
    def save_face_database(self):
        """Save face database to file"""
        try:
            with open(self.face_db_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            print("Face database saved successfully")
        except Exception as e:
            print(f"Error saving face database: {e}")
            
    def start_system(self):
        """Start the camera and system"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
            
        self.camera_running = True
        self.current_mode = "face_recognition"
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.speak("Smart Palm Pay system started. Please show your face for recognition")
        self.update_status("Face Recognition Mode\nPlace your face in front of camera")
        
        self.update_frame()
        
    def stop_system(self):
        """Stop the camera and system"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.video_label.config(image='')
        self.current_user = None
        self.current_mode = "face_recognition"
        self.update_status("System stopped")
        self.user_info_label.config(text="User: Not logged in")
        self.balance_label.config(text="Balance: ₹0.00")
        
        self.speak("System stopped")
        
    def update_status(self, text):
        """Update status display"""
        self.status_label.config(text=text)
        
    def update_frame(self):
        """Main frame update loop"""
        if not self.camera_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            # Process based on current mode
            if self.current_mode == "face_recognition":
                self.process_face_recognition(frame)
            elif self.current_mode == "palm_scan":
                self.process_palm_scan(frame)
            elif self.current_mode == "gesture_pin":
                self.process_gesture_pin(frame)
                
            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480))
            photo = ImageTk.PhotoImage(image=img)
            
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
        self.root.after(10, self.update_frame)
        
    def process_face_recognition(self, frame):
        """Process face recognition"""
        if not self.face_app:
            cv2.putText(frame, "Face recognition not available", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
            
        faces = self.face_app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Recognize face
            name = self.recognize_face(face.embedding)
            cv2.putText(frame, name, (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            if name != "Unknown":
                self.current_user = name
                self.load_user_data()
                self.switch_to_palm_scan()
                
    def recognize_face(self, embedding):
        """Recognize face from embedding"""
        if len(self.face_database) == 0:
            return "Unknown"
            
        embedding = normalize(embedding.reshape(1, -1))[0]
        
        min_distance = float('inf')
        recognized_name = "Unknown"
        threshold = 0.6
        
        for name, data in self.face_database.items():
            stored_embedding = data['embedding']
            similarity = np.dot(embedding, stored_embedding)
            distance = 1 - similarity
            
            if distance < min_distance:
                min_distance = distance
                recognized_name = name
                
        if min_distance > threshold:
            recognized_name = "Unknown"
            
        return recognized_name
        
    def load_user_data(self):
        """Load user data from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (self.current_user,))
        user_data = cursor.fetchone()
        
        if user_data:
            balance = user_data[1]
            self.user_info_label.config(text=f"User: {self.current_user}")
            self.balance_label.config(text=f"Balance: ₹{balance:,.2f}")
        else:
            # Create new user
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(
                "INSERT INTO users (username, balance, created_at, last_active) VALUES (?, ?, ?, ?)",
                (self.current_user, 10000, now, now)
            )
            self.conn.commit()
            self.user_info_label.config(text=f"User: {self.current_user}")
            self.balance_label.config(text="Balance: ₹10,000.00")
            
    def switch_to_palm_scan(self):
        """Switch to palm scanning mode"""
        self.current_mode = "palm_scan"
        self.speak(f"Welcome {self.current_user}. Please show your palm for scanning")
        self.update_status(f"Welcome {self.current_user}!\nShow RIGHT palm for Payment\nShow LEFT palm for Digital Locker")
        
    def process_palm_scan(self, frame):
        """Process palm scanning to determine left or right hand"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        cv2.putText(frame, "Show your palm", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "RIGHT palm = Payment", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "LEFT palm = Digital Locker", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_type = results.multi_handedness[0].classification[0].label
            confidence = results.multi_handedness[0].classification[0].score
            
            # Only proceed if confidence is high enough
            if confidence > 0.8:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                cv2.putText(frame, f"{hand_type} Hand Detected ({confidence:.2f})", (20, 170),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add stability check - only trigger after consistent detection
                if not hasattr(self, 'palm_detection_count'):
                    self.palm_detection_count = 0
                    self.last_detected_hand = None
                
                if self.last_detected_hand == hand_type:
                    self.palm_detection_count += 1
                else:
                    self.palm_detection_count = 1
                    self.last_detected_hand = hand_type
                
                # Require 30 consistent frames (about 1 second) before triggering
                if self.palm_detection_count >= 30:
                    if hand_type == "Right":
                        self.switch_to_payment()
                    elif hand_type == "Left":
                        self.switch_to_digital_locker()
                    
                    # Reset counter
                    self.palm_detection_count = 0
                
                # Show countdown
                remaining = 30 - self.palm_detection_count
                cv2.putText(frame, f"Hold steady: {remaining}", (20, 210),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
    def switch_to_payment(self):
        """Switch to payment mode"""
        self.speak("Right hand detected. Opening payment system")
        self.update_status("Payment Mode\nEnter amount to transfer")
        
        # Get payment amount
        amount = simpledialog.askfloat(
            "Payment Amount",
            "Enter amount to pay (₹):",
            minvalue=1,
            maxvalue=50000
        )
        
        if amount:
            self.payment_amount = amount
            self.current_mode = "gesture_pin"
            self.pin_buffer = []
            self.speak(f"Payment amount set to {amount} rupees. Please enter your PIN using finger gestures")
            self.update_status(f"Payment: ₹{amount:,.2f}\nEnter 5-digit PIN using finger gestures")
            self.update_pin_display()
        else:
            self.switch_to_palm_scan()
            
    def switch_to_digital_locker(self):
        """Switch to digital locker mode"""
        self.speak("Left hand detected. Opening digital locker")
        self.show_digital_locker()
        # Add delay to prevent repeated opening
        time.sleep(2)
        
    def show_digital_locker(self):
        """Show digital locker interface"""
        locker_window = tk.Toplevel(self.root)
        locker_window.title(f"Digital Locker - {self.current_user}")
        locker_window.geometry("800x600")
        locker_window.configure(bg="#2c3e50")
        
        # Title
        title = tk.Label(
            locker_window,
            text=f"🗄️ Digital Locker - {self.current_user}",
            font=("Arial", 20, "bold"),
            bg="#2c3e50",
            fg="#00d4ff"
        )
        title.pack(pady=20)
        
        # Buttons frame
        btn_frame = tk.Frame(locker_window, bg="#2c3e50")
        btn_frame.pack(pady=20)
        
        upload_btn = tk.Button(
            btn_frame,
            text="📤 Upload Document",
            command=lambda: self.upload_document(locker_window),
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            width=20,
            height=2
        )
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        view_btn = tk.Button(
            btn_frame,
            text="👁️ View Documents",
            command=lambda: self.view_documents(locker_window),
            font=("Arial", 12, "bold"),
            bg="#3498db",
            fg="white",
            width=20,
            height=2
        )
        view_btn.pack(side=tk.LEFT, padx=10)
        
        # Documents list
        list_frame = tk.Frame(locker_window, bg="#34495e", relief=tk.RIDGE, bd=2)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(
            list_frame,
            text="📋 Your Documents",
            font=("Arial", 14, "bold"),
            bg="#34495e",
            fg="#ffffff"
        ).pack(pady=10)
        
        # Load and display documents
        self.load_documents_list(list_frame)
        
        # Back to palm scan after closing window
        def on_locker_close():
            locker_window.destroy()
            self.switch_to_palm_scan()
        
        locker_window.protocol("WM_DELETE_WINDOW", on_locker_close)
        
    def upload_document(self, parent_window):
        """Upload document to digital locker"""
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("PDF files", "*.pdf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            doc_type = simpledialog.askstring(
                "Document Type",
                "Enter document type (e.g., Aadhaar, PAN, License):"
            )
            
            if doc_type:
                try:
                    with open(file_path, 'rb') as f:
                        doc_data = f.read()
                    
                    doc_name = os.path.basename(file_path)
                    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    cursor = self.conn.cursor()
                    cursor.execute(
                        "INSERT INTO digital_locker (username, document_type, document_name, document_data, created_at) VALUES (?, ?, ?, ?, ?)",
                        (self.current_user, doc_type, doc_name, doc_data, now)
                    )
                    self.conn.commit()
                    
                    messagebox.showinfo("Success", f"{doc_type} uploaded successfully!")
                    self.speak(f"{doc_type} document uploaded successfully")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to upload document: {e}")
                    
    def view_documents(self, parent_window):
        """View documents in digital locker"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT document_type, document_name, created_at FROM digital_locker WHERE username = ?",
            (self.current_user,)
        )
        documents = cursor.fetchall()
        
        if not documents:
            messagebox.showinfo("Info", "No documents found in your locker")
            return
            
        # Create documents view window
        docs_window = tk.Toplevel(parent_window)
        docs_window.title("Documents List")
        docs_window.geometry("600x400")
        
        # Create treeview for documents
        tree = ttk.Treeview(docs_window, columns=('Type', 'Name', 'Date'), show='headings')
        tree.heading('Type', text='Document Type')
        tree.heading('Name', text='File Name')
        tree.heading('Date', text='Upload Date')
        
        for doc in documents:
            tree.insert('', tk.END, values=doc)
            
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def load_documents_list(self, parent_frame):
        """Load and display documents list"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT document_type, COUNT(*) FROM digital_locker WHERE username = ? GROUP BY document_type",
            (self.current_user,)
        )
        doc_counts = cursor.fetchall()
        
        if doc_counts:
            for doc_type, count in doc_counts:
                doc_label = tk.Label(
                    parent_frame,
                    text=f"📄 {doc_type}: {count} file(s)",
                    font=("Arial", 12),
                    bg="#34495e",
                    fg="#ecf0f1"
                )
                doc_label.pack(anchor=tk.W, padx=20, pady=5)
        else:
            no_docs_label = tk.Label(
                parent_frame,
                text="No documents uploaded yet",
                font=("Arial", 12),
                bg="#34495e",
                fg="#95a5a6"
            )
            no_docs_label.pack(pady=20)
            
    def process_gesture_pin(self, frame):
        """Process gesture-based PIN input"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # Draw instructions
        cv2.putText(frame, f"Payment: Rs.{self.payment_amount:,.2f}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Show RIGHT hand for PIN", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_type = results.multi_handedness[0].classification[0].label
            
            if hand_type == "Right":
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Count fingers
                finger_count = self.count_fingers(hand_landmarks.landmark)
                
                # Handle cooldown
                if self.in_cooldown:
                    self.cooldown_frames += 1
                    if self.cooldown_frames >= 30:  # 1 second cooldown
                        self.in_cooldown = False
                        self.cooldown_frames = 0
                        self.stable_count = -1
                        self.stable_frames = 0
                        
                    # Draw cooldown overlay
                    remaining = 1 - (self.cooldown_frames / 30)
                    cv2.putText(frame, f"COOLDOWN: {remaining:.1f}s", (20, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Stability check
                    if finger_count == self.stable_count:
                        self.stable_frames += 1
                    else:
                        self.stable_count = finger_count
                        self.stable_frames = 1
                        
                    # If stable for 30 frames and we need more digits
                    if self.stable_frames == 30 and len(self.pin_buffer) < 5:
                        self.pin_buffer.append(finger_count)
                        self.update_pin_display()
                        self.speak(str(finger_count))
                        
                        # Start cooldown
                        self.in_cooldown = True
                        self.cooldown_frames = 0
                        
                        # Check if PIN is complete
                        if len(self.pin_buffer) == 5:
                            self.process_payment()
                            
                    # Draw finger count
                    cv2.putText(frame, f"Fingers: {finger_count}", (20, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                               
    def count_fingers(self, landmarks):
        """Count extended fingers"""
        fingers_up = []
        
        # Thumb
        if landmarks[4].x > landmarks[3].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
            
        # Other fingers
        for i in range(1, 5):
            if landmarks[self.finger_tips[i]].y < landmarks[self.finger_pips[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
                
        return sum(fingers_up)
        
    def update_pin_display(self):
        """Update PIN display"""
        pin_text = "PIN: " + " ".join([str(d) for d in self.pin_buffer])
        pin_text += " _" * (5 - len(self.pin_buffer))
        self.pin_display.config(text=pin_text)
        
    def process_payment(self):
        """Process the payment with entered PIN"""
        pin_str = ''.join(map(str, self.pin_buffer))
        
        # Verify PIN (for demo, we'll use a simple check)
        cursor = self.conn.cursor()
        cursor.execute("SELECT pin FROM gesture_pins WHERE username = ?", (self.current_user,))
        stored_pin = cursor.fetchone()
        
        if stored_pin and stored_pin[0] == pin_str:
            # PIN correct, process payment
            self.execute_payment()
        elif not stored_pin:
            # First time, set PIN
            cursor.execute(
                "INSERT INTO gesture_pins (username, pin) VALUES (?, ?)",
                (self.current_user, pin_str)
            )
            self.conn.commit()
            self.speak("PIN set successfully. Payment processed")
            self.execute_payment()
        else:
            # Wrong PIN
            self.speak("Incorrect PIN. Payment failed")
            messagebox.showerror("Error", "Incorrect PIN! Payment failed.")
            self.switch_to_palm_scan()
            
    def execute_payment(self):
        """Execute the payment transaction"""
        # For demo, we'll simulate a payment
        cursor = self.conn.cursor()
        
        # Get current balance
        cursor.execute("SELECT balance FROM users WHERE username = ?", (self.current_user,))
        current_balance = cursor.fetchone()[0]
        
        if current_balance >= self.payment_amount:
            # Deduct amount
            new_balance = current_balance - self.payment_amount
            cursor.execute(
                "UPDATE users SET balance = ?, last_active = ? WHERE username = ?",
                (new_balance, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.current_user)
            )
            
            # Record transaction
            transaction_id = f"TXN{uuid.uuid4().hex[:12].upper()}"
            cursor.execute(
                "INSERT INTO transactions (transaction_id, sender, receiver, amount, timestamp, status) VALUES (?, ?, ?, ?, ?, ?)",
                (transaction_id, self.current_user, "Merchant", self.payment_amount, 
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "completed")
            )
            
            self.conn.commit()
            
            # Update UI
            self.balance_label.config(text=f"Balance: ₹{new_balance:,.2f}")
            
            # Success message
            self.speak(f"Payment successful. {self.payment_amount} rupees paid. Remaining balance {new_balance} rupees")
            messagebox.showinfo(
                "Payment Success",
                f"✅ Payment Successful!\n\nAmount: ₹{self.payment_amount:,.2f}\nTransaction ID: {transaction_id}\nRemaining Balance: ₹{new_balance:,.2f}"
            )
        else:
            self.speak("Insufficient balance. Payment failed")
            messagebox.showerror("Error", "Insufficient balance!")
            
        # Reset and go back to palm scan
        self.pin_buffer = []
        self.payment_amount = 0
        self.update_pin_display()
        self.switch_to_palm_scan()
        
    def register_new_user(self):
        """Register a new user with face recognition"""
        name = simpledialog.askstring("Register User", "Enter user name:")
        if not name or name.strip() == "":
            return
            
        name = name.strip()
        
        if name in self.face_database:
            messagebox.showerror("Error", "User already exists!")
            return
            
        # Start face capture for registration
        self.capture_face_for_registration(name)
        
    def capture_face_for_registration(self, name):
        """Capture face data for new user registration"""
        if not self.face_app:
            messagebox.showerror("Error", "Face recognition not available")
            return
            
        self.speak(f"Registering {name}. Please look at the camera")
        messagebox.showinfo("Registration", f"Registering {name}\nPlease look at the camera and press OK when ready")
        
        captured_images = []
        capture_count = 0
        target_count = 30
        
        # Capture frames
        while capture_count < target_count and self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                faces = self.face_app.get(frame)
                
                if len(faces) > 0:
                    largest_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    captured_images.append(largest_face.embedding)
                    capture_count += 1
                    
                    # Update progress
                    progress_val = (capture_count / target_count) * 100
                    self.progress['value'] = progress_val
                    
            time.sleep(0.1)  # Small delay
            
        if len(captured_images) >= 20:
            # Average embeddings
            embeddings_array = np.array(captured_images)
            avg_embedding = np.mean(embeddings_array, axis=0)
            avg_embedding = normalize(avg_embedding.reshape(1, -1))[0]
            
            # Store in database
            self.face_database[name] = {
                'embedding': avg_embedding,
                'samples': embeddings_array
            }
            
            self.save_face_database()
            
            # Create user in database
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO users (username, balance, created_at, last_active) VALUES (?, ?, ?, ?)",
                (name, 10000, now, now)
            )
            self.conn.commit()
            
            self.progress['value'] = 0
            self.speak(f"{name} registered successfully")
            messagebox.showinfo("Success", f"{name} has been registered successfully!")
        else:
            messagebox.showerror("Error", "Failed to capture enough face data. Please try again.")
            
        self.progress['value'] = 0
        
    def run(self):
        """Run the application"""
        self.speak("Smart Palm Pay system ready")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle application closing"""
        self.stop_system()
        if self.conn:
            self.conn.close()
        self.root.destroy()

if __name__ == "__main__":
    app = SmartPalmPaySystem()
    app.run()
