import numpy as np


def compute_euclidean_distance(p1, p2):
    """
    Compute Euclidean distance between two 2D points.
    p1, p2: (x, y)
    """
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))


def compute_angle(a, b, c):
    """
    Compute angle (in degrees) at point b given three 2D points a, b, c.
    a, b, c: (x, y)
    Returns angle ABC in degrees.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    # Normalization
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0

    ba_norm = ba / np.linalg.norm(ba)
    bc_norm = bc / np.linalg.norm(bc)

    cos_angle = np.clip(np.dot(ba_norm, bc_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return float(angle_deg)


def hand_center(landmarks):
    """
    Compute approximate center of the hand given a list of (x, y) landmarks.
    landmarks: list of (x, y), length usually 21.
    Returns (cx, cy).
    """
    if len(landmarks) == 0:
        return (0.0, 0.0)

    coords = np.array(landmarks, dtype=np.float32)
    cx = float(np.mean(coords[:, 0]))
    cy = float(np.mean(coords[:, 1]))
    return (cx, cy)


if __name__ == "__main__":
    # Simple quick test
    p1 = (0, 0)
    p2 = (3, 4)
    print("Distance example:", compute_euclidean_distance(p1, p2))  # ~5.0

    a = (0, 0)
    b = (1, 0)
    c = (1, 1)
    print("Angle example (should be ~90):", compute_angle(a, b, c))