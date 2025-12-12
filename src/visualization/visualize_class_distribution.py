import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "data/raw/gestures.csv"

def visualize_class_distribution(output_path="assets/class_distribution.png"):
    df = pd.read_csv(CSV_PATH)
    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Gesture Class Distribution")
    plt.xlabel("Gesture Label")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved class distribution plot to {output_path}")


if __name__ == "__main__":
    visualize_class_distribution()
