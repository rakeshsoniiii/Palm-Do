import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
import pickle
import os
from PIL import Image, ImageTk
import time
import mediapipe as mp

class FingerAuthSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Finger Authentication System")
        self.root.geometry("600x500")
        self.root.configure(bg='#2c3e50')
        
        # Data storage
        self.users_file = "users.pkl"
        self.users = self.load_users()
        
        # Current user tracking
        self.current_user = None
        self.registration_name = ""
        self.registration_pin = ""
        self.login_attempt_pin = ""
        
        # Camera variables
        self.cap = None
        self.camera_window = None
        self.camera_label = None
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Finger counting optimization
        self.last_finger_count = 0
        self.stable_count = 0
        self.stable_threshold = 3
        
        self.setup_main_menu()
    
    def load_users(self):
        """Load user data from file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_users(self):
        """Save user data to file"""
        with open(self.users_file, 'wb') as f:
            pickle.dump(self.users, f)
    
    def setup_main_menu(self):
        """Setup the main menu interface"""
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Title
        title_label = tk.Label(self.root, text="Finger Authentication System", 
                              font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=30)
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=50)
        
        # Register button
        register_btn = tk.Button(button_frame, text="Register", font=('Arial', 14),
                                bg='#3498db', fg='white', width=15, height=2,
                                command=self.start_registration)
        register_btn.grid(row=0, column=0, padx=20, pady=10)
        
        # Login button
        login_btn = tk.Button(button_frame, text="Login", font=('Arial', 14),
                             bg='#2ecc71', fg='white', width=15, height=2,
                             command=self.start_login)
        login_btn.grid(row=0, column=1, padx=20, pady=10)
        
        # Logout button
        logout_btn = tk.Button(button_frame, text="Log Out", font=('Arial', 14),
                              bg='#e74c3c', fg='white', width=15, height=2,
                              command=self.logout)
        logout_btn.grid(row=1, column=0, padx=20, pady=10)
        
        # Exit button
        exit_btn = tk.Button(button_frame, text="Exit", font=('Arial', 14),
                            bg='#95a5a6', fg='white', width=15, height=2,
                            command=self.root.quit)
        exit_btn.grid(row=1, column=1, padx=20, pady=10)
        
        # Status label
        self.status_label = tk.Label(self.root, text="", font=('Arial', 12), 
                                    bg='#2c3e50', fg='white')
        self.status_label.pack(pady=20)
        
        # Update status
        if self.current_user:
            self.status_label.config(text=f"Logged in as: {self.current_user}")
        else:
            self.status_label.config(text="Not logged in")
    
    def start_registration(self):
        """Start the registration process"""
        self.registration_name = ""
        self.registration_pin = ""
        self.get_name_for_registration()
    
    def get_name_for_registration(self):
        """Get name for registration"""
        name_window = tk.Toplevel(self.root)
        name_window.title("Registration")
        name_window.geometry("400x200")
        name_window.configure(bg='#34495e')
        name_window.transient(self.root)
        name_window.grab_set()
        
        tk.Label(name_window, text="Enter Your Name:", font=('Arial', 14), 
                bg='#34495e', fg='white').pack(pady=20)
        
        name_entry = tk.Entry(name_window, font=('Arial', 14), width=20)
        name_entry.pack(pady=10)
        name_entry.focus()
        
        def submit_name():
            name = name_entry.get().strip()
            if name:
                if name in self.users:
                    messagebox.showerror("Error", "User already exists!")
                else:
                    self.registration_name = name
                    name_window.destroy()
                    self.setup_pin_entry("registration")
            else:
                messagebox.showerror("Error", "Please enter a valid name!")
        
        submit_btn = tk.Button(name_window, text="Submit", font=('Arial', 12),
                              bg='#3498db', fg='white', command=submit_name)
        submit_btn.pack(pady=10)
        
        name_entry.bind('<Return>', lambda e: submit_name())
    
    def start_login(self):
        """Start the login process"""
        if not self.users:
            messagebox.showerror("Error", "No users registered yet!")
            return
        
        self.login_attempt_pin = ""
        self.setup_pin_entry("login")
    
    def setup_pin_entry(self, mode):
        """Setup PIN entry interface (camera + manual option)"""
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title(f"{mode.capitalize()} - Enter PIN")
        self.camera_window.geometry("900x700")
        self.camera_window.configure(bg='#34495e')
        
        # Title
        title_text = "Set Your 4-Digit PIN" if mode == "registration" else "Enter Your 4-Digit PIN"
        tk.Label(self.camera_window, text=title_text, font=('Arial', 16, 'bold'),
                bg='#34495e', fg='white').pack(pady=10)
        
        # Camera display
        self.camera_label = tk.Label(self.camera_window, bg='black')
        self.camera_label.pack(pady=10)
        
        # PIN display
        pin_frame = tk.Frame(self.camera_window, bg='#34495e')
        pin_frame.pack(pady=10)
        
        if mode == "registration":
            self.pin_display = tk.Label(pin_frame, text="PIN: ____", font=('Arial', 18, 'bold'),
                                       bg='#34495e', fg='white')
        else:
            self.pin_display = tk.Label(pin_frame, text="Enter PIN: ____", font=('Arial', 18, 'bold'),
                                       bg='#34495e', fg='white')
        self.pin_display.pack()
        
        # Detection status
        self.detection_status = tk.Label(pin_frame, text="Show your hand(s) to camera", 
                                        font=('Arial', 12), bg='#34495e', fg='#f39c12')
        self.detection_status.pack()
        
        # Current finger count display
        self.finger_count_label = tk.Label(pin_frame, text="Fingers: 0", font=('Arial', 12),
                                          bg='#34495e', fg='#3498db')
        self.finger_count_label.pack()
        
        # Instructions
        instructions = """
INSTRUCTIONS:
• Show your hand(s) to the camera with fingers extended
• Single hand: 1-5 fingers = digits 1-5
• Both hands: Left hand (1-5) + Right hand (1-5) = digits 6-10
• Make sure your hands are clearly visible
• Press 'Space' to capture the current finger count
• Press 'Enter' to submit PIN
• Use 'Manual PIN Entry' if finger detection doesn't work
        """
        tk.Label(self.camera_window, text=instructions, font=('Arial', 11),
                bg='#34495e', fg='#bdc3c7', justify=tk.LEFT).pack(pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(self.camera_window, bg='#34495e')
        button_frame.pack(pady=10)
        
        # Manual PIN entry button
        manual_btn = tk.Button(button_frame, text="Enter PIN Manually", font=('Arial', 12),
                              bg='#e67e22', fg='white', command=lambda: self.manual_pin_entry(mode))
        manual_btn.grid(row=0, column=0, padx=10)
        
        # Clear button
        clear_btn = tk.Button(button_frame, text="Clear PIN", font=('Arial', 12),
                             bg='#e74c3c', fg='white', command=self.clear_pin)
        clear_btn.grid(row=0, column=1, padx=10)
        
        # Submit button
        submit_btn = tk.Button(button_frame, text="Submit", font=('Arial', 12),
                              bg='#2ecc71', fg='white', command=lambda: self.submit_pin(mode))
        submit_btn.grid(row=0, column=2, padx=10)
        
        # Start camera
        self.start_camera()
        
        # Bind keys
        self.camera_window.bind('<space>', lambda e: self.capture_finger_count(mode))
        self.camera_window.bind('<Return>', lambda e: self.submit_pin(mode))
        self.camera_window.bind('<m>', lambda e: self.manual_pin_entry(mode))
        self.camera_window.bind('<c>', lambda e: self.clear_pin())
        
        self.camera_window.protocol("WM_DELETE_WINDOW", self.stop_camera)
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera!")
            return
        
        # Set camera resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.update_camera()
    
    def count_fingers_single_hand(self, landmarks):
        """Count extended fingers for a single hand with improved accuracy"""
        try:
            # Finger tip landmarks
            tip_ids = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
            pip_ids = [3, 6, 10, 14, 18]  # PIP joints
            
            fingers = []
            
            # Thumb - improved detection
            thumb_tip = landmarks[tip_ids[0]]
            thumb_ip = landmarks[tip_ids[0] - 1]
            thumb_mcp = landmarks[2]
            
            # For thumb, use both x and y coordinates for better accuracy
            if thumb_tip.x < thumb_mcp.x:  # Thumb is extended to the left
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other four fingers - improved detection using multiple joints
            for i in range(1, 5):
                tip = landmarks[tip_ids[i]]
                pip = landmarks[pip_ids[i]]
                mcp = landmarks[pip_ids[i] - 1]
                
                # Check if finger is extended using multiple criteria
                is_extended = (
                    tip.y < pip.y and  # Tip above PIP
                    abs(tip.x - pip.x) < 0.1 and  # Not too far sideways
                    tip.y < mcp.y  # Tip above MCP
                )
                
                fingers.append(1 if is_extended else 0)
            
            return sum(fingers)
        except Exception as e:
            print(f"Error counting fingers for single hand: {e}")
            return 0
    
    def count_fingers_both_hands(self, hand_landmarks_list):
        """Count total fingers from both hands with stability check"""
        total_fingers = 0
        
        for hand_landmarks in hand_landmarks_list:
            total_fingers += self.count_fingers_single_hand(hand_landmarks.landmark)
        
        # Apply stability check
        if total_fingers == self.last_finger_count:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.last_finger_count = total_fingers
        
        # Only return count if stable for threshold frames
        if self.stable_count >= self.stable_threshold:
            return min(total_fingers, 9)
        else:
            return self.last_finger_count
    
    def update_camera(self):
        """Update camera feed with optimized hand detection"""
        if self.cap and self.camera_window and self.camera_label:
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror view
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.hands.process(rgb_frame)
                
                current_finger_count = 0
                detection_text = "Show your hand(s) to camera"
                text_color = '#f39c12'  # Orange
                hand_count = 0
                
                if results.multi_hand_landmarks:
                    hand_count = len(results.multi_hand_landmarks)
                    current_finger_count = self.count_fingers_both_hands(results.multi_hand_landmarks)
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks with different colors
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))
                    
                    # Update detection status
                    stability_indicator = "✓" if self.stable_count >= self.stable_threshold else "…"
                    detection_text = f"Hands: {hand_count}, Fingers: {current_finger_count} {stability_indicator} - Press SPACE to capture"
                    text_color = '#2ecc71'  # Green
                
                # Update detection status label
                if hasattr(self, 'detection_status') and self.detection_status.winfo_exists():
                    self.detection_status.config(text=detection_text, fg=text_color)
                
                # Update finger count label
                if hasattr(self, 'finger_count_label') and self.finger_count_label.winfo_exists():
                    self.finger_count_label.config(text=f"Fingers: {current_finger_count}")
                
                # Display information on frame
                cv2.putText(frame, f"Hands: {hand_count}, Fingers: {current_finger_count}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Get current PIN for display
                if hasattr(self, 'registration_pin') and hasattr(self, 'login_attempt_pin'):
                    current_pin = self.registration_pin if hasattr(self, 'registration_mode') else self.login_attempt_pin
                    cv2.putText(frame, f"Current PIN: {current_pin}", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Instructions on frame
                cv2.putText(frame, "SPACE: Capture | ENTER: Submit | M: Manual | C: Clear", (20, 450),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Stability: {self.stable_count}/{self.stable_threshold}", (20, 480),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Convert to RGB for display
                rgb_frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame_display)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            
            if self.camera_window.winfo_exists():
                self.camera_window.after(5, self.update_camera)  # Faster update for smoother experience
    
    def capture_finger_count(self, mode):
        """Capture the current finger count and add to PIN with improved accuracy"""
        if not self.cap:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        try:
            # Flip frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks and self.stable_count >= self.stable_threshold:
                finger_count = self.count_fingers_both_hands(results.multi_hand_landmarks)
                
                if 1 <= finger_count <= 9:
                    if mode == "registration":
                        if len(self.registration_pin) < 4:
                            self.registration_pin += str(finger_count)
                            self.update_pin_display(mode)
                            self.detection_status.config(text=f"Added {finger_count} to PIN!", fg="#2ecc71")
                            
                            # Auto-submit if PIN is complete
                            if len(self.registration_pin) == 4:
                                self.camera_window.after(100, lambda: self.submit_pin(mode))
                    else:  # login
                        if len(self.login_attempt_pin) < 4:
                            self.login_attempt_pin += str(finger_count)
                            self.update_pin_display(mode)
                            self.detection_status.config(text=f"Added {finger_count} to PIN!", fg="#2ecc71")
                            
                            # Auto-submit if PIN is complete
                            if len(self.login_attempt_pin) == 4:
                                self.camera_window.after(100, lambda: self.submit_pin(mode))
                else:
                    self.detection_status.config(text="Please show 1-9 fingers", fg="#e74c3c")
            else:
                if not results.multi_hand_landmarks:
                    self.detection_status.config(text="No hand detected!", fg="#e74c3c")
                else:
                    self.detection_status.config(text="Keep hand steady...", fg="#f39c12")
                    
        except Exception as e:
            print(f"Error in capture_finger_count: {e}")
            self.detection_status.config(text="Detection failed. Use manual entry.", fg="#e74c3c")
    
    def update_pin_display(self, mode):
        """Update the PIN display label"""
        if mode == "registration":
            display_text = "PIN: " + self.registration_pin + "_" * (4 - len(self.registration_pin))
        else:
            display_text = "Enter PIN: " + self.login_attempt_pin + "_" * (4 - len(self.login_attempt_pin))
        
        self.pin_display.config(text=display_text)
    
    def clear_pin(self):
        """Clear the current PIN entry"""
        if hasattr(self, 'registration_pin'):
            self.registration_pin = ""
        if hasattr(self, 'login_attempt_pin'):
            self.login_attempt_pin = ""
        
        # Reset display
        if hasattr(self, 'pin_display'):
            self.pin_display.config(text="PIN: ____")
        if hasattr(self, 'detection_status'):
            self.detection_status.config(text="Show your hand(s) to camera", fg="#f39c12")
        if hasattr(self, 'finger_count_label'):
            self.finger_count_label.config(text="Fingers: 0")
        
        # Reset stability counter
        self.stable_count = 0
        self.last_finger_count = 0
    
    def manual_pin_entry(self, mode):
        """Manual PIN entry window with Enter key support"""
        manual_window = tk.Toplevel(self.camera_window)
        manual_window.title("Manual PIN Entry")
        manual_window.geometry("350x300")
        manual_window.configure(bg='#34495e')
        manual_window.transient(self.camera_window)
        manual_window.grab_set()
        
        tk.Label(manual_window, text="Enter 4-Digit PIN:", font=('Arial', 16, 'bold'),
                bg='#34495e', fg='white').pack(pady=20)
        
        # PIN entry with validation
        pin_var = tk.StringVar()
        
        def validate_pin(char):
            return char.isdigit() and len(pin_var.get()) < 4
        
        def on_pin_change(*args):
            current_pin = pin_var.get()
            if len(current_pin) == 4:
                submit_manual_pin()
        
        pin_var.trace('w', on_pin_change)
        
        pin_entry = tk.Entry(manual_window, font=('Arial', 20, 'bold'), width=10, 
                            textvariable=pin_var, show='•', justify='center')
        pin_entry.pack(pady=20)
        pin_entry.focus()
        
        # Register validation
        vcmd = (manual_window.register(validate_pin), '%S')
        pin_entry.config(validate="key", validatecommand=vcmd)
        
        def submit_manual_pin():
            pin = pin_var.get().strip()
            if len(pin) == 4 and pin.isdigit():
                if mode == "registration":
                    self.registration_pin = pin
                else:
                    self.login_attempt_pin = pin
                self.update_pin_display(mode)
                manual_window.destroy()
                self.detection_status.config(text="Manual PIN entered successfully!", fg="#2ecc71")
                
                # Auto-submit after manual entry
                self.camera_window.after(500, lambda: self.submit_pin(mode))
            else:
                messagebox.showerror("Error", "Please enter a valid 4-digit PIN!")
        
        # Button frame
        button_frame = tk.Frame(manual_window, bg='#34495e')
        button_frame.pack(pady=10)
        
        submit_btn = tk.Button(button_frame, text="Submit PIN", font=('Arial', 12),
                              bg='#3498db', fg='white', command=submit_manual_pin)
        submit_btn.grid(row=0, column=0, padx=10)
        
        clear_btn = tk.Button(button_frame, text="Clear", font=('Arial', 12),
                             bg='#e74c3c', fg='white', command=lambda: pin_var.set(""))
        clear_btn.grid(row=0, column=1, padx=10)
        
        # Bind Enter key
        manual_window.bind('<Return>', lambda e: submit_manual_pin())
        pin_entry.bind('<Return>', lambda e: submit_manual_pin())
        
        # Instructions
        tk.Label(manual_window, text="Press Enter or type 4 digits to auto-submit", 
                font=('Arial', 10), bg='#34495e', fg='#bdc3c7').pack(pady=5)
    
    def save_user_data(self):
        """Save user data after PIN is set"""
        if self.registration_name and len(self.registration_pin) == 4:
            self.users[self.registration_name] = self.registration_pin
            self.save_users()
            return True
        return False
    
    def submit_pin(self, mode):
        """Submit the entered PIN"""
        if mode == "registration":
            pin = self.registration_pin
            if len(pin) != 4:
                messagebox.showerror("Error", "Please enter a 4-digit PIN!")
                return
            
            # Save user registration
            if self.save_user_data():
                self.stop_camera()
                if self.camera_window:
                    self.camera_window.destroy()
                
                messagebox.showinfo("Success", f"Registration completed for {self.registration_name}!\nPIN: {pin}")
                self.setup_main_menu()
            else:
                messagebox.showerror("Error", "Failed to save user data!")
            
        else:  # login
            pin = self.login_attempt_pin
            if len(pin) != 4:
                messagebox.showerror("Error", "Please enter a 4-digit PIN!")
                return
            
            # Check credentials
            authenticated = False
            user_name = ""
            
            for name, stored_pin in self.users.items():
                if stored_pin == pin:
                    authenticated = True
                    user_name = name
                    break
            
            if authenticated:
                self.current_user = user_name
                self.stop_camera()
                if self.camera_window:
                    self.camera_window.destroy()
                
                self.show_welcome_screen()
            else:
                messagebox.showerror("Error", "Invalid PIN!")
                self.clear_pin()
    
    def show_welcome_screen(self):
        """Show welcome screen after successful login"""
        for widget in self.root.winfo_children():
            widget.destroy()
        
        welcome_label = tk.Label(self.root, text=f"Welcome {self.current_user}", 
                                font=('Arial', 24, 'bold'), bg='#2c3e50', fg='white')
        welcome_label.pack(expand=True, pady=50)
        
        user_info = tk.Label(self.root, text=f"You are successfully logged in!\nYour PIN: {self.users[self.current_user]}",
                           font=('Arial', 14), bg='#2c3e50', fg='#bdc3c7')
        user_info.pack(pady=20)
        
        back_btn = tk.Button(self.root, text="Back to Main Menu", font=('Arial', 14),
                            bg='#3498db', fg='white', command=self.setup_main_menu)
        back_btn.pack(pady=20)
    
    def logout(self):
        """Logout current user"""
        if self.current_user:
            previous_user = self.current_user
            self.current_user = None
            messagebox.showinfo("Success", f"Logged out successfully from {previous_user}!")
        else:
            messagebox.showinfo("Info", "No user is currently logged in.")
        
        self.setup_main_menu()
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        
        # Reset stability counters
        self.stable_count = 0
        self.last_finger_count = 0
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            self.stop_camera()

if __name__ == "__main__":
    app = FingerAuthSystem()
    app.run()
