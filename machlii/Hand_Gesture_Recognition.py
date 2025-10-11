import pygame
import cv2
import mediapipe as mp
import pyttsx3
import numpy as np
import sys

# Initialize PyGame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Recognition")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 120, 255)

# Font
font = pygame.font.SysFont('Arial', 36)
small_font = pygame.font.SysFont('Arial', 24)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# State variables
last_count = 0
count = 0
stable_count = 0
stable_frames = 0
STABLE_THRESHOLD = 10  # Number of consecutive frames with same count to trigger speech

def count_fingers(hand_landmarks):
    """Count the number of extended fingers in a hand"""
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    
    finger_pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, 
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                   mp_hands.HandLandmark.RING_FINGER_PIP,
                   mp_hands.HandLandmark.PINKY_PIP]
    
    thumb_tip = mp_hands.HandLandmark.THUMB_TIP
    thumb_ip = mp_hands.HandLandmark.THUMB_IP
    
    # Count fingers (excluding thumb)
    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1
    
    # Count thumb - different approach due to thumb's range of motion
    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    thumb_tip_x = hand_landmarks.landmark[thumb_tip].x
    
    # For right hand, thumb is extended if thumb tip is to the right of wrist
    # For left hand, thumb is extended if thumb tip is to the left of wrist
    # We'll use a more robust approach
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    
    # Calculate distance from thumb tip to wrist
    thumb_tip_to_wrist = abs(hand_landmarks.landmark[thumb_tip].x - wrist_x)
    
    # If thumb is extended away from hand
    if thumb_tip_to_wrist > 0.1:  # Adjust this threshold as needed
        count += 1
    
    return count

def speak_count(count):
    """Speak the finger count using text-to-speech"""
    try:
        tts_engine.say(f"{count}")
        tts_engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

def draw_landmarks_on_frame(frame, hand_landmarks):
    """Draw hand landmarks on the frame"""
    h, w, _ = frame.shape
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def main():
    global last_count, count, stable_count, stable_frames
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        
        # Reset count for this frame
        count = 0
        
        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Count fingers
                count = count_fingers(hand_landmarks)
                
                # Draw landmarks on frame
                draw_landmarks_on_frame(frame, hand_landmarks)
                
                # Draw hand connections
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Check if count is stable
        if count == last_count:
            stable_frames += 1
        else:
            stable_frames = 0
            last_count = count
        
        # If count is stable for enough frames and different from last spoken count
        if stable_frames >= STABLE_THRESHOLD and count != stable_count:
            stable_count = count
            speak_count(count)
        
        # Convert frame to PyGame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)
        
        # Clear screen
        screen.fill(BLACK)
        
        # Draw camera feed
        screen.blit(frame, (WIDTH//2 - frame.get_width()//2, 20))
        
        # Draw UI elements
        pygame.draw.rect(screen, (40, 40, 40), (0, 0, WIDTH, HEIGHT), 10)
        
        # Draw title
        title = font.render("Hand Gesture Recognition", True, WHITE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 10))
        
        # Draw finger count
        count_text = font.render(f"Fingers: {count}", True, GREEN if count > 0 else RED)
        screen.blit(count_text, (WIDTH//2 - count_text.get_width()//2, HEIGHT - 150))
        
        # Draw stability indicator
        stability = min(stable_frames / STABLE_THRESHOLD, 1.0)
        pygame.draw.rect(screen, (60, 60, 60), (WIDTH//2 - 100, HEIGHT - 100, 200, 20))
        pygame.draw.rect(screen, BLUE, (WIDTH//2 - 100, HEIGHT - 100, 200 * stability, 20))
        
        stability_text = small_font.render("Stability", True, WHITE)
        screen.blit(stability_text, (WIDTH//2 - stability_text.get_width()//2, HEIGHT - 120))
        
        # Draw instructions
        instructions = [
            "Show your hand to the camera",
            "Hold your gesture steady for detection",
            "Press ESC to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = small_font.render(instruction, True, WHITE)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT - 60 + i*25))
        
        pygame.display.flip()
        clock.tick(30)
    
    # Clean up
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
