import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Path settings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks

mp_hands = mp.solutions.hands


def visualize_normalized(output_path="assets/normalized_landmarks.png"):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("!!! Camera could not be opened.")
        return

    print("Camera opened.")
    print("➡ Put your hand in front of the camera.")
    print("➡ Press SPACE to capture a frame.")
    print("➡ Press Q or ESC to quit without saving.")

    captured_frame = None

    # Preview cycle
    while True:
        ret, frame = cap.read()
        if not ret:
            print("!! Could not read frame.")
            break

        cv2.imshow("Preview - Press SPACE to Capture (Normalized)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_frame = frame.copy()
            print("📸 Frame captured for normalized landmarks!")
            break
        elif key in [27, ord('q')]:  # ESC or Q
            print("Exit without capturing.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        print("!! No frame captured.")
        return

    # BGR -> RGB
    frame = captured_frame
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Landmark detection with MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("!! No hand detected in captured frame.")
        return

    hand_landmarks = results.multi_hand_landmarks[0]

    # Pixel coordinates
    coords = []
    for lm in hand_landmarks.landmark:
        x_pix = int(lm.x * w)
        y_pix = int(lm.y * h)
        coords.append((x_pix, y_pix))

    # Normalization
    norm = normalize_landmarks(coords)  # shape (42,)
    norm = norm.reshape(-1, 2)          # (21, 2)

    x = norm[:, 0]
    y = norm[:, 1]

    # Scatter plot (fixed layout)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, -y, s=30)

    # Landmark index labels
    for i, (xi, yi) in enumerate(zip(x, -y)):
        ax.text(xi, yi, str(i), fontsize=8)

    ax.set_title("Normalized Hand Landmarks (Feature Space)", fontsize=12)
    ax.set_xlabel("x (normalized)", fontsize=10)
    ax.set_ylabel("y (normalized)", fontsize=10)

    # Center grid + reference lines
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Adjust limits for more padding (no cropping)
    ax.set_xlim(min(x) - 0.1, max(x) + 0.1)
    ax.set_ylim(min(-y) - 0.1, max(-y) + 0.1)

    # Save without cropping
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved normalized landmarks plot to {output_path}")

if __name__ == "__main__":
    visualize_normalized()
