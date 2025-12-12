import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def visualize_bounding_box_pipeline(output_path="assets/hand_bounding_box_pipeline.png"):
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

        cv2.imshow("Preview - Press SPACE to Capture (Bounding Box)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_frame = frame.copy()
            print("Success! Frame captured for bounding box pipeline!")
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

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    h, w, c = image_rgb.shape

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

    # Original frame (RGB for matplotlib)
    orig_vis = image_rgb.copy()

    # Copy for bounding box
    bbox_vis = image_rgb.copy()

    # Copy for landmarks
    landmarks_vis = image_rgb.copy()

    # Landmark pixel coordinates
    xs = []
    ys = []
    for lm in hand_landmarks.landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Draw bounding box
    cv2.rectangle(
        bbox_vis,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2
    )

    # Draw landmarks + connections on landmarks_vis
    mp_drawing.draw_landmarks(
        landmarks_vis,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )

    # Compose 3-step figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_vis)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(bbox_vis)
    axes[1].set_title("Hand Region (Bounding Box)")
    axes[1].axis("off")

    axes[2].imshow(landmarks_vis)
    axes[2].set_title("Hand Landmarks & Skeleton")
    axes[2].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved bounding box pipeline visualization to {output_path}")


if __name__ == "__main__":
    visualize_bounding_box_pipeline()