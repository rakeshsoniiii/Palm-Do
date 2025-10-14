import cv2
import mediapipe as mp
import pyttsx3
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sqlite3
import math
import threading
from PIL import Image, ImageTk

class HandGestureRecognizer:
    """Core hand recognition from the original code"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [2, 6, 10, 14, 18]
        self.finger_mcp = [1, 5, 9, 13, 17]
        
    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def is_thumb_open(self, landmarks):
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        thumb_cmc = landmarks[1]
        index_mcp = landmarks[5]
        wrist = landmarks[0]
        
        palm_center_x = (wrist.x + index_mcp.x) / 2
        palm_center_y = (wrist.y + index_mcp.y) / 2
        
        tip_to_palm = math.sqrt((thumb_tip.x - palm_center_x)**2 + 
                                (thumb_tip.y - palm_center_y)**2)
        cmc_to_palm = math.sqrt((thumb_cmc.x - palm_center_x)**2 + 
                                (thumb_cmc.y - palm_center_y)**2)
        
        distance_check = tip_to_palm > cmc_to_palm * 1.8
        
        tip_to_mcp = self.calculate_distance(thumb_tip, thumb_mcp)
        ip_to_mcp = self.calculate_distance(thumb_ip, thumb_mcp)
        extension_check = tip_to_mcp > ip_to_mcp * 1.3
        
        horizontal_check = thumb_tip.x < thumb_mcp.x - 0.05
        
        tip_to_index = self.calculate_distance(thumb_tip, index_mcp)
        ip_to_index = self.calculate_distance(thumb_ip, index_mcp)
        separation_check = tip_to_index > ip_to_index * 0.9
        
        checks_passed = sum([distance_check, extension_check, horizontal_check, separation_check])
        return checks_passed >= 3
    
    def is_finger_open(self, landmarks, finger_idx):
        tip = self.finger_tips[finger_idx]
        pip = self.finger_pips[finger_idx]
        mcp = self.finger_mcp[finger_idx]
        
        tip_pos = landmarks[tip]
        pip_pos = landmarks[pip]
        mcp_pos = landmarks[mcp]
        wrist = landmarks[0]
        
        if tip_pos.y > pip_pos.y:
            return False
        
        tip_to_wrist = self.calculate_distance(tip_pos, wrist)
        pip_to_wrist = self.calculate_distance(pip_pos, wrist)
        
        if tip_to_wrist <= pip_to_wrist:
            return False
        
        tip_to_mcp = self.calculate_distance(tip_pos, mcp_pos)
        tip_to_pip = self.calculate_distance(tip_pos, pip_pos)
        pip_to_mcp = self.calculate_distance(pip_pos, mcp_pos)
        
        straightness = tip_to_mcp / (tip_to_pip + pip_to_mcp)
        return straightness > 0.85
    
    def count_fingers(self, landmarks):
        fingers_up = []
        fingers_up.append(1 if self.is_thumb_open(landmarks) else 0)
        
        for i in range(1, 5):
            fingers_up.append(1 if self.is_finger_open(landmarks, i) else 0)
        
        return sum(fingers_up)
    
    def detect_gesture(self, frame):
        """Process frame and return finger count for right hand"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        fingers_status = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                if hand_label == "Right":
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                    )
                    
                    # Get individual finger status
                    fingers_up = []
                    fingers_up.append(1 if self.is_thumb_open(hand_landmarks.landmark) else 0)
                    
                    for i in range(1, 5):
                        fingers_up.append(1 if self.is_finger_open(hand_landmarks.landmark, i) else 0)
                    
                    finger_count = sum(fingers_up)
                    return finger_count, True, fingers_up
        
        return -1, False, []


class DatabaseManager:
    """Handle SQLite database operations"""
    def __init__(self):
        self.conn = sqlite3.connect('gesture_lock.db', check_same_thread=False)
        self.create_table()
    
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                pin TEXT NOT NULL
            )
        ''')
        self.conn.commit()
    
    def register_user(self, name, pin):
        try:
            cursor = self.conn.cursor()
            cursor.execute('INSERT INTO users (name, pin) VALUES (?, ?)', (name, pin))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def verify_user(self, name, pin):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE name = ? AND pin = ?', (name, pin))
        return cursor.fetchone() is not None
    
    def user_exists(self, name):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM users WHERE name = ?', (name,))
        return cursor.fetchone() is not None


class GestureLockApp:
    """Main application with Tkinter GUI"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gesture-Based PIN Lock System")
        self.root.geometry("1200x700")
        self.root.configure(bg='#1a1a2e')
        
        self.db = DatabaseManager()
        self.recognizer = HandGestureRecognizer()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 150)
        
        self.cap = None
        self.camera_running = False
        self.current_mode = None  # 'register', 'setpin', 'login'
        self.current_user = None
        self.pin_buffer = []
        self.stable_count = -1
        self.stable_frames = 0
        self.cooldown_frames = 0
        self.in_cooldown = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title = tk.Label(self.root, text="🔒 GESTURE LOCK SYSTEM", 
                        font=("Arial", 28, "bold"), bg='#1a1a2e', fg='#00d4ff')
        title.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side - Camera feed
        left_frame = tk.Frame(main_frame, bg='#16213e', relief=tk.RIDGE, bd=3)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        camera_label = tk.Label(left_frame, text="📷 Camera Feed", 
                               font=("Arial", 16, "bold"), bg='#16213e', fg='#00d4ff')
        camera_label.pack(pady=10)
        
        self.video_label = tk.Label(left_frame, bg='#0f3460')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Right side - Controls
        right_frame = tk.Frame(main_frame, bg='#16213e', relief=tk.RIDGE, bd=3)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.config(width=350)
        
        # Mode selection
        mode_label = tk.Label(right_frame, text="Select Mode", 
                             font=("Arial", 18, "bold"), bg='#16213e', fg='#ffffff')
        mode_label.pack(pady=15)
        
        btn_style = {'font': ('Arial', 14, 'bold'), 'width': 25, 'height': 2, 
                    'bg': '#00d4ff', 'fg': '#000000', 'cursor': 'hand2'}
        
        self.register_btn = tk.Button(right_frame, text="1️⃣ REGISTER USER", 
                                     command=self.start_registration, **btn_style)
        self.register_btn.pack(pady=10)
        
        self.setpin_btn = tk.Button(right_frame, text="2️⃣ SET PIN", 
                                   command=self.start_setpin, **btn_style)
        self.setpin_btn.pack(pady=10)
        
        self.login_btn = tk.Button(right_frame, text="3️⃣ LOGIN", 
                                  command=self.start_login, **btn_style)
        self.login_btn.pack(pady=10)
        
        # Status display
        status_frame = tk.Frame(right_frame, bg='#0f3460', relief=tk.SUNKEN, bd=2)
        status_frame.pack(pady=20, padx=10, fill=tk.BOTH, expand=True)
        
        tk.Label(status_frame, text="📊 Status", font=("Arial", 14, "bold"),
                bg='#0f3460', fg='#00d4ff').pack(pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready to start...", 
                                     font=("Arial", 12), bg='#0f3460', fg='#ffffff',
                                     wraplength=300, justify=tk.LEFT)
        self.status_label.pack(pady=10, padx=10)
        
        # PIN display
        self.pin_display = tk.Label(status_frame, text="PIN: _ _ _ _ _", 
                                   font=("Arial", 16, "bold"), bg='#0f3460', fg='#ffcc00')
        self.pin_display.pack(pady=10)
        
        # Gesture display
        self.gesture_label = tk.Label(status_frame, text="Gesture: -", 
                                     font=("Arial", 14), bg='#0f3460', fg='#00ff00')
        self.gesture_label.pack(pady=10)
    
    def speak(self, text):
        """Text-to-speech in separate thread"""
        def _speak():
            self.tts.say(text)
            self.tts.runAndWait()
        threading.Thread(target=_speak, daemon=True).start()
    
    def update_status(self, text):
        self.status_label.config(text=text)
    
    def update_pin_display(self):
        pin_text = "PIN: " + " ".join([str(d) for d in self.pin_buffer])
        pin_text += " _" * (5 - len(self.pin_buffer))
        self.pin_display.config(text=pin_text)
    
    def start_registration(self):
        name = simpledialog.askstring("Register", "Enter your name:", parent=self.root)
        if name:
            if self.db.user_exists(name):
                messagebox.showerror("Error", "User already exists!", parent=self.root)
                self.speak("User already exists")
            else:
                self.current_user = name
                self.current_mode = 'setpin'
                self.pin_buffer = []
                self.update_status(f"Welcome {name}!\nShow RIGHT HAND and make gestures (0-5)\nto set your 5-digit PIN")
                self.update_pin_display()
                self.speak(f"Welcome {name}. Please set your 5 digit PIN using finger gestures")
                self.start_camera()
    
    def start_setpin(self):
        name = simpledialog.askstring("Set PIN", "Enter your name:", parent=self.root)
        if name:
            if not self.db.user_exists(name):
                messagebox.showerror("Error", "User not found! Please register first.", parent=self.root)
                self.speak("User not found. Please register first")
            else:
                messagebox.showinfo("Info", "This user already has a PIN set!", parent=self.root)
                self.speak("This user already has a PIN set")
    
    def start_login(self):
        name = simpledialog.askstring("Login", "Enter your name:", parent=self.root)
        if name:
            if not self.db.user_exists(name):
                messagebox.showerror("Error", "User not found!", parent=self.root)
                self.speak("User not found")
            else:
                self.current_user = name
                self.current_mode = 'login'
                self.pin_buffer = []
                self.update_status(f"Login as {name}\nShow RIGHT HAND and enter your 5-digit PIN")
                self.update_pin_display()
                self.speak(f"Please enter your PIN using finger gestures")
                self.start_camera()
    
    def start_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_running = True
            self.update_frame()
    
    def stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def update_frame(self):
        if not self.camera_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            finger_count, hand_detected, fingers_status = self.recognizer.detect_gesture(frame)
            
            # Cooldown management (1 second = ~30 frames at 30fps)
            if self.in_cooldown:
                self.cooldown_frames += 1
                if self.cooldown_frames >= 30:  # 1 second cooldown
                    self.in_cooldown = False
                    self.cooldown_frames = 0
                    self.stable_count = -1
                    self.stable_frames = 0
            
            if hand_detected and finger_count >= 0 and not self.in_cooldown:
                # Stability check
                if finger_count == self.stable_count:
                    self.stable_frames += 1
                else:
                    self.stable_count = finger_count
                    self.stable_frames = 1
                
                # If stable for 30 frames (~1 second) and we need more digits
                if self.stable_frames == 30 and len(self.pin_buffer) < 5:
                    self.pin_buffer.append(finger_count)
                    self.update_pin_display()
                    self.speak(str(finger_count))
                    
                    # Start cooldown period
                    self.in_cooldown = True
                    self.cooldown_frames = 0
                    
                    # Check if PIN is complete
                    if len(self.pin_buffer) == 5:
                        self.process_complete_pin()
                
                # Draw finger status boxes on frame
                self.draw_finger_status(frame, finger_count, fingers_status)
                
            else:
                if self.in_cooldown:
                    remaining = 1 - (self.cooldown_frames / 30)
                    # Draw cooldown overlay
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (0, 100, 255), -1)
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    cv2.putText(frame, f"COOLDOWN: {remaining:.1f}s", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.putText(frame, ">>> CHANGE FINGER GESTURE <<<", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Show instruction when no hand detected
                    cv2.putText(frame, "SHOW RIGHT HAND", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Draw top info bar
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
            cv2.putText(frame, f"MODE: {self.current_mode.upper()}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"PIN CAPTURED: {len(self.pin_buffer)}/5", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Show current PIN
            pin_display = " ".join([str(d) for d in self.pin_buffer]) + " _" * (5 - len(self.pin_buffer))
            cv2.putText(frame, f"PIN: {pin_display}", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Progress bar
            progress = len(self.pin_buffer) / 5
            bar_width = int(progress * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, 120), (frame.shape[1] - 20, 140), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 120), (20 + bar_width, 140), (0, 255, 0), -1)
            
            # Convert to PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)
    
    def draw_finger_status(self, frame, finger_count, fingers_status):
        """Draw clear visual feedback for finger detection"""
        h, w = frame.shape[:2]
        
        # Draw large finger count in center
        cv2.putText(frame, str(finger_count), (w//2 - 50, h//2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
        
        # Draw individual finger status at bottom
        finger_names = ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']
        box_width = w // 5
        y_pos = h - 80
        
        for i, (name, status) in enumerate(zip(finger_names, fingers_status)):
            x_start = i * box_width
            color = (0, 255, 0) if status else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(frame, (x_start + 5, y_pos), (x_start + box_width - 5, y_pos + 70), color, -1)
            
            # Draw border
            cv2.rectangle(frame, (x_start + 5, y_pos), (x_start + box_width - 5, y_pos + 70), (255, 255, 255), 2)
            
            # Draw finger name
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0]
            text_x = x_start + (box_width - text_size[0]) // 2
            cv2.putText(frame, name, (text_x, y_pos + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            
            # Draw status
            status_text = "UP" if status else "DOWN"
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x_start + (box_width - text_size[0]) // 2
            cv2.putText(frame, status_text, (text_x, y_pos + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def process_complete_pin(self):
        pin_str = ''.join(map(str, self.pin_buffer))
        
        if self.current_mode == 'setpin':
            # Register new user with PIN
            if self.db.register_user(self.current_user, pin_str):
                self.speak("PIN set successfully. Registration complete")
                messagebox.showinfo("Success", f"User {self.current_user} registered with PIN!", parent=self.root)
                self.update_status("Registration successful!\nYou can now login.")
            else:
                self.speak("Registration failed")
                messagebox.showerror("Error", "Registration failed!", parent=self.root)
            
            self.stop_camera()
            self.current_mode = None
            self.current_user = None
            self.pin_buffer = []
            
        elif self.current_mode == 'login':
            # Verify PIN
            if self.db.verify_user(self.current_user, pin_str):
                self.speak("Device unlocked. Welcome")
                messagebox.showinfo("Success", f"✅ DEVICE UNLOCKED!\n\nWelcome, {self.current_user}!", parent=self.root)
                self.update_status(f"✅ UNLOCKED\nWelcome {self.current_user}!")
            else:
                self.speak("Incorrect PIN. Access denied")
                messagebox.showerror("Error", "❌ Incorrect PIN!\nAccess Denied", parent=self.root)
                self.update_status("❌ ACCESS DENIED\nIncorrect PIN!")
            
            self.stop_camera()
            self.current_mode = None
            self.current_user = None
            self.pin_buffer = []
    
    def run(self):
        self.speak("Gesture lock system ready")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    app = GestureLockApp()
    app.run()
