import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

mp_hands = mp.solutions.hands

def record_wrist_motion(num_frames=120, output_path="assets/wrist_motion.png"):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("!!! Camera could not have opened.")
        return

    print("Camera opened.")
    print("➡ Show your fist and move it up/down slowly.")
    print("➡ Recording", num_frames, "frames...")
    print("➡ Press Q or ESC to abort early.")

    wrist_ys = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        frame_count = 0
        while frame_count < num_frames:
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
                wrist = hand_landmarks.landmark[0]
                wrist_y_pix = int(wrist.y * h)
                wrist_ys.append(wrist_y_pix)
            else:
                wrist_ys.append(np.nan)

            cv2.imshow("Wrist Motion Recording", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                print("⏹ Stopped by user.")
                break

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(wrist_ys) == 0:
        print("!! No data recorded.")
        return

    # Plot trajectory
    frames = np.arange(len(wrist_ys))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frames, wrist_ys, marker="o", linestyle="-")
    ax.set_title("Wrist Vertical Position Over Time")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Wrist y (pixels)")
    ax.invert_yaxis()  # Matching image coordinates
    ax.grid(True, alpha=0.4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved wrist motion plot to {output_path}")


if __name__ == "__main__":
    record_wrist_motion()
