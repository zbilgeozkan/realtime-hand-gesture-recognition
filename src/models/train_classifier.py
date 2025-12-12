import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Path settings
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..", "..")
sys.path.append(PROJECT_ROOT)

from src.preprocessing.feature_extraction import normalize_landmarks


def load_dataset(csv_path="data/raw/gestures.csv"):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Collect some data first.")

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


def train_model():
    X, y = load_dataset()

    print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
    print("Classes:", np.unique(y))

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model + label encoder
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(clf, "data/models/gesture_rf.pkl")
    joblib.dump(le, "data/models/label_encoder.pkl")

    print("\n Model saved to data/models/gesture_rf.pkl")
    print(" Label encoder saved to data/models/label_encoder.pkl")


if __name__ == "__main__":
    train_model()
