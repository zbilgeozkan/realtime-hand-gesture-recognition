import cv2

def run_camera_test():
    # Try default camera (index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Camera could not be opened.")
        return

    # Try to read one frame to get resolution
    ret, frame = cap.read()
    if not ret:
        print("Could not read from camera.")
        cap.release()
        return

    h, w, c = frame.shape
    print(f"Camera test started. Resolution: {w}x{h}")
    print("   Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame could not have read. Exiting.")
            break

        # Flip for selfie-view
        frame = cv2.flip(frame, 1)

        cv2.imshow("Camera Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("⏹ Exiting camera test.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_test()
