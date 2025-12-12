import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Path settings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks

MODEL_PATH = "data/models/gesture_rf.pkl"
ENCODER_PATH = "data/models/label_encoder.pkl"
CSV_PATH = "data/raw/gestures.csv"


def load_dataset(csv_path=CSV_PATH):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    df = df.dropna()

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
    y = labels

    return X, y


def plot_and_save_confusion_matrix(cm, class_names, save_path="assets/confusion_matrix.png"):
    """Plot and save confusion matrix as PNG."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Labeling each cell
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Ensure assets folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def evaluate_model():
    print("- Loading dataset...")
    X, y = load_dataset()
    print(f"Dataset shape: X = {X.shape}, y = {y.shape}")

    print("- Loading trained model...")
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("- Evaluating model on test data...")
    y_pred = clf.predict(X_test)

    class_names = label_encoder.classes_

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm)

    # Save PNG
    plot_and_save_confusion_matrix(cm, class_names)

if __name__ == "__main__":
    evaluate_model()