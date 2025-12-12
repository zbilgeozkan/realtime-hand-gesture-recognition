import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def capture_and_visualize(output_path="assets/mediapipe_landmarks.png"):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("!!! Camera could not have opened.")
        return

    print("Camera opened.")
    print("➡ Press SPACE to capture a frame.")
    print("➡ Press Q or ESC to quit without saving.")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("!! Could not read frame.")
            break

        cv2.imshow("Preview - Press SPACE to Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE key
            captured_frame = frame.copy()
            print("Frame captured!")
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

    # Convert to RGB
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(image_rgb)
        image_with_landmarks = image_rgb.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_with_landmarks,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        else:
            print("!! No hand detected in captured frame.")

    # Plotting side by side
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Frame")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_with_landmarks)
    plt.title("MediaPipe Hand Landmarks")
    plt.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    capture_and_visualize()