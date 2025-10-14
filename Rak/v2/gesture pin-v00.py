import cv2
import mediapipe as mp
import pyttsx3
import time
import math
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import sqlite3

class FingerCounter:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        # Landmark indices
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_pips = [2, 6, 10, 14, 18]  # Lower joints
        self.finger_mcp = [1, 5, 9, 13, 17]    # Knuckles
        
        # For preventing repeated announcements
        self.last_count = -1
        self.last_announcement_time = 0
        self.announcement_delay = 0.8
        self.stable_count = -1
        self.stable_frames = 0
        self.stability_threshold = 5  # Need 5 consistent frames
        
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def is_thumb_open(self, landmarks):
        """Check if thumb is extended with high accuracy for folding detection"""
        thumb_tip = landmarks[4]      # Thumb tip
        thumb_ip = landmarks[3]       # Thumb IP joint
        thumb_mcp = landmarks[2]      # Thumb MCP joint
        thumb_cmc = landmarks[1]      # Thumb CMC joint (base)
        index_mcp = landmarks[5]      # Index finger base
        wrist = landmarks[0]          # Wrist
        
        # Method 1: Check if thumb tip is far from palm center
        palm_center_x = (wrist.x + index_mcp.x) / 2
        palm_center_y = (wrist.y + index_mcp.y) / 2
        
        tip_to_palm = math.sqrt((thumb_tip.x - palm_center_x)**2 + 
                                (thumb_tip.y - palm_center_y)**2)
        
        cmc_to_palm = math.sqrt((thumb_cmc.x - palm_center_x)**2 + 
                                (thumb_cmc.y - palm_center_y)**2)
        
        distance_check = tip_to_palm > cmc_to_palm * 1.8
        
        # Method 2: Check thumb extension using joint distances
        tip_to_mcp = self.calculate_distance(thumb_tip, thumb_mcp)
        ip_to_mcp = self.calculate_distance(thumb_ip, thumb_mcp)
        
        extension_check = tip_to_mcp > ip_to_mcp * 1.3
        
        # Method 3: Check horizontal distance (for right hand)
        horizontal_check = thumb_tip.x < thumb_mcp.x - 0.05
        
        # Method 4: Check if thumb tip is away from index finger
        tip_to_index = self.calculate_distance(thumb_tip, index_mcp)
        ip_to_index = self.calculate_distance(thumb_ip, index_mcp)
        
        separation_check = tip_to_index > ip_to_index * 0.9
        
        checks_passed = sum([distance_check, extension_check, horizontal_check, separation_check])
        
        return checks_passed >= 3
    
    def is_finger_open(self, landmarks, finger_idx):
        """Check if a finger is extended using multiple checks"""
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
        """Count extended fingers with improved accuracy"""
        fingers_up = []
        
        fingers_up.append(1 if self.is_thumb_open(landmarks) else 0)
        
        for i in range(1, 5):
            fingers_up.append(1 if self.is_finger_open(landmarks, i) else 0)
        
        return sum(fingers_up), fingers_up
    
    def announce(self, count):
        """Announce finger count via text-to-speech"""
        current_time = time.time()
        
        if count != self.last_count and (current_time - self.last_announcement_time) > self.announcement_delay:
            if count == 0:
                text = "Zero fingers"
            elif count == 1:
                text = "One finger"
            elif count == 2:
                text = "Two fingers"
            elif count == 3:
                text = "Three fingers"
            elif count == 4:
                text = "Four fingers"
            elif count == 5:
                text = "Five fingers"
            else:
                return
            
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_count = count
            self.last_announcement_time = current_time
            print(f"Announced: {text}")
    
    def run(self):
        """Main loop for finger counting (original mode)"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("=" * 50)
        print("Hand Finger Recognition Started")
        print("Please show your RIGHT hand to the camera")
        print("Keep your hand steady for accurate detection")
        print("Press 'q' to quit")
        print("=" * 50)
        
        self.engine.say("Please show your right hand")
        self.engine.runAndWait()
        
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(img_rgb)
            
            finger_count = 0
            fingers_status = []
            detected_hand = False
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    if hand_label == "Right":
                        detected_hand = True
                        
                        self.mp_draw.draw_landmarks(
                            img, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )
                        
                        finger_count, fingers_status = self.count_fingers(hand_landmarks.landmark)
                        
                        if finger_count == self.stable_count:
                            self.stable_frames += 1
                        else:
                            self.stable_count = finger_count
                            self.stable_frames = 1
                        
                        if self.stable_frames >= self.stability_threshold:
                            self.announce(finger_count)
            
            if detected_hand:
                cv2.rectangle(img, (5, 5), (400, 150), (0, 0, 0), -1)
                cv2.rectangle(img, (5, 5), (400, 150), (0, 255, 0), 3)
                
                cv2.putText(img, 'RIGHT HAND', (15, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, f'Fingers: {finger_count}', (15, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                y_pos = 130
                for i, (name, status) in enumerate(zip(finger_names, fingers_status)):
                    color = (0, 255, 0) if status else (0, 0, 255)
                    cv2.putText(img, f'{name}: {"UP" if status else "DOWN"}', 
                               (15, y_pos + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.rectangle(img, (5, 5), (450, 80), (0, 0, 255), 3)
                cv2.putText(img, 'SHOW RIGHT HAND', (15, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.putText(img, 'Press Q to quit', (img.shape[1] - 250, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Accurate Hand Finger Counter', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nProgram ended. Goodbye!")

    def capture_pin(self):
        """Capture a 5-digit PIN using finger gestures"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.engine.say("Start entering your PIN. Show right hand for each digit.")
        self.engine.runAndWait()
        
        pin = []
        current_digit = 0
        last_confirm_time = 0
        confirm_delay = 1.0  # Delay after confirm to reset
        
        print("=" * 50)
        print("PIN Input Mode Started")
        print("Show 0-5 fingers for each digit")
        print("Press SPACE to confirm stable count")
        print("Press 'q' to cancel")
        print("=" * 50)
        
        while current_digit < 5:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(img_rgb)
            
            finger_count = 0
            fingers_status = []
            detected_hand = False
            
            current_time = time.time()
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    
                    if hand_label == "Right":
                        detected_hand = True
                        
                        self.mp_draw.draw_landmarks(
                            img, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )
                        
                        finger_count, fingers_status = self.count_fingers(hand_landmarks.landmark)
                        
                        if finger_count == self.stable_count:
                            self.stable_frames += 1
                        else:
                            self.stable_count = finger_count
                            self.stable_frames = 1
            
            # Display
            cv2.rectangle(img, (5, 5), (500, 200), (0, 0, 0), -1)
            cv2.rectangle(img, (5, 5), (500, 200), (0, 255, 0), 3)
            
            cv2.putText(img, f'Entering Digit {current_digit + 1}/5', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f'PIN so far: {"*" * len(pin)}', (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if detected_hand:
                cv2.putText(img, f'Fingers: {finger_count}', (15, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                if self.stable_frames >= self.stability_threshold and (current_time - last_confirm_time) > confirm_delay:
                    cv2.putText(img, f'Press SPACE to confirm {finger_count}', (15, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(img, 'SHOW RIGHT HAND', (15, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.putText(img, 'Press Q to cancel', (img.shape[1] - 250, img.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Gesture PIN Input', img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and detected_hand and self.stable_frames >= self.stability_threshold and (current_time - last_confirm_time) > confirm_delay:
                pin.append(str(finger_count))
                self.engine.say(f"Digit {current_digit + 1} set to {finger_count}")
                self.engine.runAndWait()
                current_digit += 1
                last_confirm_time = current_time
                self.stable_count = -1
                self.stable_frames = 0  # Reset stability
                print(f"Digit {current_digit} confirmed: {finger_count}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(pin) == 5:
            print("PIN captured successfully")
            return ''.join(pin)
        else:
            print("PIN input cancelled")
            return None

# Tkinter GUI functions
def register(counter, conn, c):
    name = simpledialog.askstring("Register", "Enter your name:")
    if name:
        c.execute("SELECT * FROM users WHERE name=?", (name,))
        if c.fetchone():
            messagebox.showerror("Error", "User already exists")
            return
        messagebox.showinfo("Set PIN", "Now set your 5-digit PIN (0-5 per digit) using finger gestures in the camera window.")
        pin = counter.capture_pin()
        if pin:
            c.execute("INSERT INTO users VALUES (?, ?)", (name, pin))
            conn.commit()
            messagebox.showinfo("Success", "User registered successfully!")

def login(counter, conn, c):
    name = simpledialog.askstring("Login", "Enter your name:")
    if name:
        c.execute("SELECT pin FROM users WHERE name=?", (name,))
        row = c.fetchone()
        if not row:
            messagebox.showerror("Error", "User not found")
            return
        stored_pin = row[0]
        messagebox.showinfo("Enter PIN", "Now enter your 5-digit PIN using finger gestures in the camera window.")
        entered_pin = counter.capture_pin()
        if entered_pin:
            if entered_pin == stored_pin:
                counter.engine.say("Device unlocked")
                counter.engine.runAndWait()
                messagebox.showinfo("Success", "Device unlocked!")
            else:
                messagebox.showerror("Error", "Incorrect PIN")

if __name__ == "__main__":
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (name TEXT PRIMARY KEY, pin TEXT)''')
    
    counter = FingerCounter()
    
    root = tk.Tk()
    root.title("Gesture PIN Lock System")
    root.geometry("300x200")
    
    tk.Button(root, text="Register", command=lambda: register(counter, conn, c), width=20).pack(pady=10)
    tk.Button(root, text="Login", command=lambda: login(counter, conn, c), width=20).pack(pady=10)
    tk.Button(root, text="Run Finger Counter (Demo)", command=counter.run, width=20).pack(pady=10)
    tk.Button(root, text="Quit", command=root.quit, width=20).pack(pady=10)
    
    root.mainloop()
    
    conn.close()
