import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
import os
import sys
from collections import deque

# Src import settings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks


# Load model and label encoder
MODEL_PATH = "data/models/gesture_rf.pkl"
ENCODER_PATH = "data/models/label_encoder.pkl"

clf = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# === Command logic ===

# Find last N predictions for smoothing
PRED_HISTORY_SIZE = 3  # decreased from 5 to 3 to make it more responsive
pred_history = deque(maxlen=PRED_HISTORY_SIZE)

# Cooldown between commands (in seconds)
COMMAND_COOLDOWN = 0.6
last_command_time = 0

# Wrist y position for scroll gesture
prev_wrist_y = None
SCROLL_THRESHOLD = 30   # Scroll trigger threshold in pixels
SCROLL_AMOUNT = 5       # Scroll amount per command

# 'Same gesture twice in a row’ filter for INDEX_* commands
last_gesture_for_command = None


def majority_vote(history):
    if not history:
        return None
    values, counts = np.unique(history, return_counts=True)
    return values[np.argmax(counts)]


def is_open_palm(pixel_coords):
    """
    pixel_coords: list of (x, y) pixel coordinates, length = 21
    Açık el (en az 3 parmak belirgin uzamış) durumunu tespit eder.
    Y ekseninde: yukarı = küçük y, aşağı = büyük y
    """
    coords = np.array(pixel_coords, dtype=np.float32)

    # Fingertips (TIP) and PIP joints
    tip_ids = [8, 12, 16, 20]   # index, middle, ring, little tips
    pip_ids = [6, 10, 14, 18]   # their corresponding PIP joints

    extended = 0
    for tip, pip in zip(tip_ids, pip_ids):
        tip_y = coords[tip, 1]
        pip_y = coords[pip, 1]
        # # If the finger is extended: the tip is higher than the joint (smaller y)
        if tip_y < pip_y:
            extended += 1

    # If at least 3 fingers are extended, consider it an open palm
    return extended >= 3


def send_command(gesture, wrist_y):
    global last_command_time, prev_wrist_y, last_gesture_for_command

    now = time.time()
    if now - last_command_time < COMMAND_COOLDOWN:
        return

    # === FIST: scroll logic ===
    if gesture == "FIST":
        if prev_wrist_y is not None and wrist_y is not None:
            dy = wrist_y - prev_wrist_y
            if dy > SCROLL_THRESHOLD:
                print("Scroll down")
                pyautogui.scroll(-SCROLL_AMOUNT)
                last_command_time = now
            elif dy < -SCROLL_THRESHOLD:
                print("Scroll up")
                pyautogui.scroll(SCROLL_AMOUNT)
                last_command_time = now
        prev_wrist_y = wrist_y
        # Return here because FIST is used only for scrolling
        return

    # If not FIST, set prev_wrist_y to zero
    prev_wrist_y = None

    # === For INDEX_* commands: require the same gesture twice in a row ===
    if gesture in ["INDEX_RIGHT", "INDEX_LEFT", "INDEX_UP", "INDEX_DOWN"]:
        if gesture != last_gesture_for_command:
            # # First time seeing this gesture: store it as a candidate only, do not send a command
            last_gesture_for_command = gesture
            return
        # Same gesture detected twice in a row → now we can send the command
    else:
        # If a different gesture is detected (non-FIST), reset the candidate
        last_gesture_for_command = None

    # === Statik gesture → arrow commands ===
    if gesture == "INDEX_RIGHT":
        print("Command: RIGHT_ARROW")
        pyautogui.press("right")
        last_command_time = now
    elif gesture == "INDEX_LEFT":
        print("Command: LEFT_ARROW")
        pyautogui.press("left")
        last_command_time = now
    elif gesture == "INDEX_UP":
        print("Command: UP_ARROW")
        pyautogui.press("up")
        last_command_time = now
    elif gesture == "INDEX_DOWN":
        print("Command: DOWN_ARROW")
        pyautogui.press("down")
        last_command_time = now
    elif gesture == "FIST":
        # We handled FIST above for scrolling
        pass
    else:
        # Unknown gesture -> no command
        pass


def run_realtime_control():
    global prev_wrist_y, last_gesture_for_command

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("!!! Camera could not have opened!")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print("▶ Realtime gesture control started.")
        print("   Focus the window you want to control (Explorer, browser, etc.).")
        print("   Press 'q' or ESC in the video window to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("!! Frame could not have read, exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, c = frame.shape
            wrist_y = None
            predicted_label = None
            predicted_proba = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Extract landmark coordinates (pixels)
                coords = []
                for lm in hand_landmarks.landmark:
                    x_pix = int(lm.x * w)
                    y_pix = int(lm.y * h)
                    coords.append((x_pix, y_pix))

                # Wrist y (for scroll gesture)
                wrist_y = coords[0][1]

                # Extract features (for model prediction)
                norm_coords_flat = normalize_landmarks(coords)
                feat = norm_coords_flat.reshape(1, -1)

                # Prediction
                probs = clf.predict_proba(feat)[0]
                class_idx = np.argmax(probs)
                predicted_label = label_encoder.inverse_transform([class_idx])[0]
                predicted_proba = probs[class_idx]

                # Insert prediction into history for smoothing
                pred_history.append(predicted_label)
                smooth_label = majority_vote(pred_history)

                # If it's an open hand, don't generate a command (IDLE) + reset state
                if is_open_palm(coords):
                    smooth_label = "IDLE"
                    prev_wrist_y = None
                    last_gesture_for_command = None

                # Display prediction
                text = f"{smooth_label} ({predicted_proba*100:.1f}%)"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Send command (if not IDLE)
                if smooth_label != "IDLE":
                    send_command(smooth_label, wrist_y)

            else:
                # Reset states if no hand is detected
                pred_history.clear()
                prev_wrist_y = None
                last_gesture_for_command = None

            cv2.imshow("Realtime Gesture Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("⏹ Exiting realtime control.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_control()
