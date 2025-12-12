import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import deque
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks

MODEL_PATH = "data/models/gesture_rf.pkl"
ENCODER_PATH = "data/models/label_encoder.pkl"

mp_hands = mp.solutions.hands

def majority_vote_window(history):
    if not history:
        return None
    values, counts = np.unique(history, return_counts=True)
    return values[np.argmax(counts)]

def visualize_prediction_smoothing(
    num_frames=150,
    window_size=5,
    output_path="assets/prediction_smoothing.png"
):
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("!!! Camera could not be opened.")
        return

    print("Camera opened.")
    print("➡ Perform a gesture (e.g. INDEX_RIGHT) for a few seconds.")
    print("➡ Recording", num_frames, "frames...")

    raw_labels = []
    smooth_labels = []
    history = deque(maxlen=window_size)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        frame_idx = 0
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                print("!! Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = []
                for lm in hand_landmarks.landmark:
                    x_pix = int(lm.x * w)
                    y_pix = int(lm.y * h)
                    coords.append((x_pix, y_pix))
                feat = normalize_landmarks(coords).reshape(1, -1)

                probs = clf.predict_proba(feat)[0]
                class_idx = np.argmax(probs)
                raw_label = label_encoder.inverse_transform([class_idx])[0]
            else:
                raw_label = "NO_HAND"

            history.append(raw_label)
            smooth_label = majority_vote_window(history)

            raw_labels.append(raw_label)
            smooth_labels.append(smooth_label if smooth_label else "NO_HAND")

            cv2.putText(frame, f"Raw: {raw_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Smoothed: {smooth_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cv2.imshow("Prediction Smoothing Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                print("⏹ Stopped by user.")
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(raw_labels) == 0:
        print("!! No predictions recorded.")
        return

    # Map labels to integers for plotting
    all_labels = sorted(list(set(raw_labels + smooth_labels)))
    label_to_int = {lab: i for i, lab in enumerate(all_labels)}

    x = np.arange(len(raw_labels))
    y_raw = [label_to_int[l] for l in raw_labels]
    y_smooth = [label_to_int[l] for l in smooth_labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(x, y_raw, where="post", label="Raw Predictions", alpha=0.6)
    ax.step(x, y_smooth, where="post", label="Smoothed Predictions", linewidth=2)

    ax.set_yticks(list(label_to_int.values()))
    ax.set_yticklabels(list(label_to_int.keys()))
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Predicted Class")
    ax.set_title("Effect of Temporal Smoothing (Majority Vote)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved prediction smoothing plot to {output_path}")


if __name__ == "__main__":
    visualize_prediction_smoothing()
