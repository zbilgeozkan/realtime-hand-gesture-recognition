import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks

CSV_PATH = "data/raw/gestures.csv"

def visualize_pca(output_path="assets/pca_gestures.png"):
    if not os.path.isfile(CSV_PATH):
        print(f"!!! {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH).dropna()
    labels = df["label"].values

    X_features = []
    for _, row in df.iterrows():
        coords = []
        for i in range(21):
            x = row[f"x_{i}"]
            y = row[f"y_{i}"]
            coords.append((x, y))
        feat = normalize_landmarks(coords)
        X_features.append(feat)

    X = np.vstack(X_features)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    unique_labels = sorted(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=lab,
            alpha=0.7,
            s=30,
            color=colors(idx)
        )

    ax.set_title("PCA Projection of Gesture Features (2D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved PCA gesture visualization to {output_path}")


if __name__ == "__main__":
    visualize_pca()
