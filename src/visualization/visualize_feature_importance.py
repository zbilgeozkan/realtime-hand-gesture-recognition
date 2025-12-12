import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

MODEL_PATH = "data/models/gesture_rf.pkl"

def visualize_feature_importance(output_path="assets/feature_importance.png"):
    clf = joblib.load(MODEL_PATH)

    importances = clf.feature_importances_  # length 42 (21 x + 21 y)
    indices = np.arange(len(importances))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(indices, importances)
    ax.set_title("RandomForest Feature Importances")
    ax.set_xlabel("Feature Index (0–41)")
    ax.set_ylabel("Importance")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    print(f"Saved feature importance plot to {output_path}")


if __name__ == "__main__":
    visualize_feature_importance()
