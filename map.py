



import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Constants
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
MIN_CONFIDENCE = 0.7
CLICK_DISTANCE = 50
APP_HOTKEY = ('win', 'space')  # Hotkey to open the application (Windows key + Spacebar)

# Initialize Mediapipe hand tracking
hand_tracker = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=MIN_CONFIDENCE)
drawing_utils = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def process_hand_tracking(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_tracker.process(rgb_frame)
    return results

def move_cursor(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    cursor_x = int(index_tip.x * SCREEN_WIDTH)
    cursor_y = int(index_tip.y * SCREEN_HEIGHT)
    pyautogui.moveTo(cursor_x, cursor_y)

def check_click(hand_landmarks, is_moving):
    if not is_moving:
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        index_tip_array = np.array([index_tip.x, index_tip.y, index_tip.z])
        middle_tip_array = np.array([middle_tip.x, middle_tip.y, middle_tip.z])
        distance = np.linalg.norm(index_tip_array - middle_tip_array)
        if distance < CLICK_DISTANCE:
            pyautogui.click()

def check_open_app(hand_landmarks, is_moving):
    if not is_moving:
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        if index_tip.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y and \
           middle_tip.y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y:
            pyautogui.hotkey(*APP_HOTKEY)

last_hand_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = process_hand_tracking(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            move_cursor(hand_landmarks)

            is_moving = False
            if last_hand_landmarks:
                index_tip_last = last_hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_tip_current = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                distance = np.linalg.norm(np.array([index_tip_last.x, index_tip_last.y, index_tip_last.z]) - 
                                          np.array([index_tip_current.x, index_tip_current.y, index_tip_current.z]))
                if distance > 0.01:  # adjust this threshold as needed
                    is_moving = True

            check_click(hand_landmarks, is_moving)
            check_open_app(hand_landmarks, is_moving)

            last_hand_landmarks = hand_landmarks

    cv2.imshow('Virtual Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
