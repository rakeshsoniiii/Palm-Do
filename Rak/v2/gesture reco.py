import pygame
import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import sys
import threading

# Initialize PyGame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Recognition - Both Hands")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)

# Font
font = pygame.font.SysFont('Arial', 36)
small_font = pygame.font.SysFont('Arial', 20)

# Global camera variable
cap = None

def initialize_camera():
    """Initialize camera with error handling"""
    global cap
    try:
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

# State variables
left_count = 0
right_count = 0
stable_left_count = 0
stable_right_count = 0
stable_frames_left = 0
stable_frames_right = 0
STABLE_THRESHOLD = 15

# Threading lock for thread-safe operations
tts_lock = threading.Lock()
is_speaking = False

def count_fingers(hand_landmarks, handedness):
    """Count the number of extended fingers in a hand"""
    try:
        finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp_hands.HandLandmark.RING_FINGER_TIP,
                      mp_hands.HandLandmark.PINKY_TIP]
        
        finger_pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                      mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                      mp_hands.HandLandmark.RING_FINGER_PIP,
                      mp_hands.HandLandmark.PINKY_PIP]
        
        thumb_tip = mp_hands.HandLandmark.THUMB_TIP
        thumb_mcp = mp_hands.HandLandmark.THUMB_MCP
        
        # Count fingers (excluding thumb)
        count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                count += 1
        
        # Count thumb
        if handedness == "Right":
            # For right hand, thumb is extended if thumb tip is to the left of MCP
            if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_mcp].x - 0.05:
                count += 1
        else:  # Left hand
            # For left hand, thumb is extended if thumb tip is to the right of MCP
            if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_mcp].x + 0.05:
                count += 1
        
        return min(count, 5)  # Ensure count doesn't exceed 5
    except Exception as e:
        print(f"Error counting fingers: {e}")
        return 0

def get_hand_label(handedness):
    """Extract hand label from MediaPipe handedness data"""
    try:
        if handedness and handedness.classification:
            return handedness.classification[0].label
        return "Unknown"
    except Exception as e:
        print(f"Error getting hand label: {e}")
        return "Unknown"

def speak_count_async(left_count, right_count):
    """Speak the finger counts in a separate thread to avoid blocking"""
    def speak_thread():
        global is_speaking
        try:
            with tts_lock:
                is_speaking = True
            
            if left_count > 0 and right_count > 0:
                tts_engine.say(f"Left hand: {left_count}, Right hand: {right_count}")
            elif left_count > 0:
                tts_engine.say(f"Left hand: {left_count}")
            elif right_count > 0:
                tts_engine.say(f"Right hand: {right_count}")
            
            tts_engine.runAndWait()
            
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            with tts_lock:
                is_speaking = False
    
    # Only start new thread if not already speaking
    with tts_lock:
        if not is_speaking and (left_count > 0 or right_count > 0):
            threading.Thread(target=speak_thread, daemon=True).start()

def draw_landmarks_on_frame(frame, hand_landmarks, handedness):
    """Draw hand landmarks on the frame with color coding"""
    try:
        h, w, _ = frame.shape
        color = (0, 255, 0) if handedness == "Left" else (255, 0, 0)
        
        # Draw landmarks
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw connections
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        )
    except Exception as e:
        print(f"Error drawing landmarks: {e}")

def process_frame(frame):
    """Process frame with MediaPipe and return results"""
    try:
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        return frame, results
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, None

def main():
    global left_count, right_count, stable_left_count, stable_right_count
    global stable_frames_left, stable_frames_right, cap
    
    # Initialize camera
    if not initialize_camera():
        print("Failed to initialize camera. Exiting...")
        pygame.quit()
        sys.exit(1)
    
    clock = pygame.time.Clock()
    running = True
    
    # For tracking previous counts
    last_left_count = 0
    last_right_count = 0
    
    print("Starting hand gesture recognition...")
    print("Press ESC to exit")
    
    while running:
        try:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Reinitializing camera...")
                if not initialize_camera():
                    print("Could not reinitialize camera. Exiting...")
                    running = False
                pygame.time.delay(1000)
                continue
            
            # Process frame
            processed_frame, results = process_frame(frame)
            
            # Reset counts
            current_left_count = 0
            current_right_count = 0
            
            # Process hand results
            if results and results.multi_hand_landmarks and results.multi_handedness:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if i >= 2:  # Only process up to 2 hands
                        break
                        
                    handedness = get_hand_label(results.multi_handedness[i])
                    count = count_fingers(hand_landmarks, handedness)
                    
                    if handedness == "Left":
                        current_left_count = count
                    elif handedness == "Right":
                        current_right_count = count
                    
                    # Draw landmarks
                    draw_landmarks_on_frame(processed_frame, hand_landmarks, handedness)
                    
                    # Add hand label to frame
                    h, w, _ = processed_frame.shape
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    x, y = int(wrist.x * w), int(wrist.y * h)
                    color = (0, 255, 0) if handedness == "Left" else (255, 0, 0)
                    cv2.putText(processed_frame, f"{handedness}: {count}", 
                               (x - 60, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update counts
            left_count = current_left_count
            right_count = current_right_count
            
            # Check stability for left hand
            if left_count == last_left_count and left_count > 0:
                stable_frames_left += 1
            else:
                stable_frames_left = 0
            
            # Check stability for right hand
            if right_count == last_right_count and right_count > 0:
                stable_frames_right += 1
            else:
                stable_frames_right = 0
            
            # Update last counts
            last_left_count = left_count
            last_right_count = right_count
            
            # Trigger speech if counts are stable
            left_stable = stable_frames_left >= STABLE_THRESHOLD and left_count != stable_left_count
            right_stable = stable_frames_right >= STABLE_THRESHOLD and right_count != stable_right_count
            
            if left_stable or right_stable:
                stable_left_count = left_count
                stable_right_count = right_count
                speak_count_async(left_count, right_count)
            
            # Convert frame to PyGame surface
            try:
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            except Exception as e:
                print(f"Error converting frame: {e}")
                continue
            
            # Clear screen
            screen.fill(BLACK)
            
            # Draw camera feed
            screen.blit(frame_surface, (WIDTH//2 - frame_surface.get_width()//2, 20))
            
            # Draw UI elements
            pygame.draw.rect(screen, (40, 40, 40), (0, 0, WIDTH, HEIGHT), 8)
            
            # Draw title
            title = font.render("Hand Gesture Recognition - Both Hands", True, WHITE)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, 5))
            
            # Draw finger counts
            left_text = font.render(f"Left: {left_count}", True, GREEN)
            right_text = font.render(f"Right: {right_count}", True, BLUE)
            
            screen.blit(left_text, (WIDTH//4 - left_text.get_width()//2, HEIGHT - 160))
            screen.blit(right_text, (3*WIDTH//4 - right_text.get_width()//2, HEIGHT - 160))
            
            # Draw stability indicators
            stability_left = min(stable_frames_left / STABLE_THRESHOLD, 1.0)
            stability_right = min(stable_frames_right / STABLE_THRESHOLD, 1.0)
            
            # Left stability
            pygame.draw.rect(screen, (60, 60, 60), (WIDTH//4 - 80, HEIGHT - 120, 160, 15))
            pygame.draw.rect(screen, GREEN, (WIDTH//4 - 80, HEIGHT - 120, 160 * stability_left, 15))
            
            # Right stability
            pygame.draw.rect(screen, (60, 60, 60), (3*WIDTH//4 - 80, HEIGHT - 120, 160, 15))
            pygame.draw.rect(screen, BLUE, (3*WIDTH//4 - 80, HEIGHT - 120, 160 * stability_right, 15))
            
            # Instructions
            instructions = [
                "Show one or both hands to the camera",
                "Hold gesture steady for voice feedback",
                "Press ESC to exit"
            ]
            
            for i, instruction in enumerate(instructions):
                text = small_font.render(instruction, True, WHITE)
                screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 80 + i*20))
            
            # Legend
            legend_left = small_font.render("Green: Left Hand", True, GREEN)
            legend_right = small_font.render("Blue: Right Hand", True, BLUE)
            
            screen.blit(legend_left, (WIDTH//4 - legend_left.get_width()//2, HEIGHT - 40))
            screen.blit(legend_right, (3*WIDTH//4 - legend_right.get_width()//2, HEIGHT - 40))
            
            pygame.display.flip()
            clock.tick(30)  # Cap at 30 FPS
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            pygame.time.delay(100)
    
    # Cleanup
    print("Shutting down...")
    if cap is not None:
        cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
