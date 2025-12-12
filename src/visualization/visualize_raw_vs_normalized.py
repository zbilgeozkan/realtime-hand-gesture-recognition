import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks

mp_hands = mp.solutions.hands

def visualize_raw_vs_normalized(output_path="assets/raw_vs_normalized_landmarks.png"):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("!!! Camera could not have opened.")
        return

    print("Camera opened.")
    print("➡ Put your hand in front of the camera.")
    print("➡ Press SPACE to capture a frame.")
    print("➡ Press Q or ESC to quit without saving.")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("!! Could not read frame.")
            break

        cv2.imshow("Preview - Raw vs Normalized", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_frame = frame.copy()
            print("Frame captured for raw vs normalized.")
            break
        elif key in [27, ord('q')]:
            print("Exit without capturing.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        print("!! No frame captured.")
        return

    h, w, c = captured_frame.shape
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("!! No hand detected.")
        return

    hand_landmarks = results.multi_hand_landmarks[0]

    # Raw pixel coords
    coords = []
    for lm in hand_landmarks.landmark:
        x_pix = int(lm.x * w)
        y_pix = int(lm.y * h)
        coords.append((x_pix, y_pix))

    coords = np.array(coords, dtype=np.float32)
    norm = normalize_landmarks(coords).reshape(-1, 2)  # (21, 2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Raw
    axes[0].scatter(coords[:, 0], coords[:, 1])
    for i, (x, y) in enumerate(coords):
        axes[0].text(x, y, str(i), fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_title("Raw Hand Landmarks (Pixel Space)")
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")
    axes[0].grid(True, alpha=0.4)

    # Normalized
    x_n = norm[:, 0]
    y_n = -norm[:, 1]
    axes[1].scatter(x_n, y_n)
    for i, (x, y) in enumerate(zip(x_n, y_n)):
        axes[1].text(x, y, str(i), fontsize=7)
    axes[1].set_title("Normalized Hand Landmarks (Feature Space)")
    axes[1].set_xlabel("x (normalized)")
    axes[1].set_ylabel("y (normalized)")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(0, color="gray", linewidth=0.5)
    axes[1].grid(True, alpha=0.4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved raw vs normalized visualization to {output_path}")


if __name__ == "__main__":
    visualize_raw_vs_normalized()
