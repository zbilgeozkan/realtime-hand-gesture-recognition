import numpy as np

def normalize_landmarks(coords):
    """
    coords: list of (x, y) pixel coordinates, length = 21
    Returns: flattened normalized numpy array [x0, y0, x1, y1, ...].

    - Uses wrist (landmark 0) as origin.
    - Scales so the hand roughly fits into [-1, 1].
    """
    if len(coords) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(coords)}")

    # Use wrist as origin
    wrist_x, wrist_y = coords[0]
    rel = [(x - wrist_x, y - wrist_y) for (x, y) in coords]

    xs = [x for x, _ in rel]
    ys = [y for _, y in rel]

    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    scale = max(range_x, range_y, 1e-6)  # avoid division by zero

    norm = [(x / scale, y / scale) for (x, y) in rel]

    flat = []
    for x, y in norm:
        flat.append(x)
        flat.append(y)

    return np.array(flat, dtype=np.float32)
