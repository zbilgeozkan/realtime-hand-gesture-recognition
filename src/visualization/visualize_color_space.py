import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def capture_and_visualize_color_space(output_path="assets/color_space_bgr_rgb.png"):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("!!! Camera could not have opened.")
        return

    print("Camera opened.")
    print("➡ Put your hand / scene in front of the camera.")
    print("➡ Press SPACE to capture a frame.")
    print("➡ Press Q or ESC to quit without saving.")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("!! Could not read frame.")
            break

        cv2.imshow("Preview - Press SPACE to Capture (Color Space)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            captured_frame = frame.copy()
            print("Frame captured for color space visualization!")
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

    # OpenCV -> BGR
    bgr = captured_frame

    # "Wrong" way: passing BGR directly to matplotlib (interpreted as RGB)
    img_wrong = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[:, :, ::-1]  # deliberately shuffle back

    # Correct way: BGR -> RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Red channel from RGB
    red_channel = rgb[:, :, 0]

    # Plot 3 images side by side
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_wrong)
    axes[0].set_title("Wrong Interpretation (BGR as RGB)")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].set_title("Correct RGB (after BGR→RGB)")
    axes[1].axis("off")

    axes[2].imshow(red_channel, cmap="gray")
    axes[2].set_title("Red Channel (Grayscale)")
    axes[2].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved color space visualization to {output_path}")


if __name__ == "__main__":
    capture_and_visualize_color_space()
