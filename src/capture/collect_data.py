import cv2
import mediapipe as mp
import csv
import os
import sys

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def collect_data(label):
    # Output CSV file
    os.makedirs("data/raw", exist_ok=True)
    csv_path = "data/raw/gestures.csv"

    # Write header if file doesn't exist
    file_exists = os.path.isfile(csv_path)
    if not file_exists:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"]
            for i in range(21):
                header.append(f"x_{i}")
                header.append(f"y_{i}")
            writer.writerow(header)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("!!! Camera could not be opened!")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print(f"▶ Data collection started. Label: '{label}'")
        print("   Press 'c' to capture the current hand pose")
        print("   Press 'q' or ESC to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("!! Frame could not be read. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, c = frame.shape
            landmarks = None

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []

                # Save x, y coordinates
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))

                # Draw landmarks + hand connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # UI text
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Collect Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if landmarks is not None:
                    row = [label]
                    for (x, y) in landmarks:
                        row.extend([x, y])
                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    print(f"Sample saved. Label = {label}, landmark count = {len(landmarks)}")
                else:
                    print("!! No hand detected. Capture skipped.")
            elif key == ord('q') or key == 27:
                print("⏹ Exiting data collection.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/capture/collect_data.py <LABEL>")
        print("Example: python src/capture/collect_data.py FIST")
        sys.exit(1)

    label = sys.argv[1]
    collect_data(label)
