import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis

# =============================================================================
# FEATURE EXTRACTION FUNCTION
# Must be identical to what you used during training in Step 5!
# =============================================================================

def extract_features(window):
    """
    Takes a window of shape (n_samples, 3) — columns are x, y, z.
    Returns a 1D feature vector.
    """
    features = []
    for axis in range(3):  # x=0, y=1, z=2
        signal = window[:, axis]
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.var(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.max(signal) - np.min(signal)) # range
        features.append(np.median(signal))
        
        """features.append(kurtosis(signal))
        features.append(np.sqrt(np.mean(signal**2))) # RMS
        features.append(np.mean(np.abs(signal - np.mean(signal))))  # MAD
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features.append(zero_crossings / len(signal)) # ZCR"""
    return np.array(features)


def segment_and_predict(csv_path, model_path, sampling_rate=50, window_seconds=5):
    clf = joblib.load(model_path)

    df = pd.read_csv(csv_path)

    # Print columns so you can verify during testing
    print(f"CSV columns found: {df.columns.tolist()}")
    print(f"CSV shape: {df.shape}")

    # Select x, y, z by position (column 1, 2, 3) regardless of header names
    # This assumes: col 0 = time, col 1 = x, col 2 = y, col 3 = z, col 4 = magnitude (optional)
    data = df.iloc[:, 1:4].values  # shape: (n_samples, 3)

    print(f"Using columns: {df.columns[1]}, {df.columns[2]}, {df.columns[3]}")

    # Segment into windows
    window_size = sampling_rate * window_seconds
    n_windows = len(data) // window_size

    if n_windows == 0:
        print(f"ERROR: Not enough data. Got {len(data)} samples, need at least {window_size}.")
        print(f"Check your sampling_rate ({sampling_rate} Hz) is correct.")
        return []

    predictions = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = data[start:end]
        features = extract_features(window).reshape(1, -1)
        pred = clf.predict(features)[0]
        label = "jumping" if pred == 1 else "walking"
        predictions.append(label)

    print(f"Processed {n_windows} windows from {csv_path}")
    return predictions


def save_predictions(predictions, output_path):
    """Saves the list of predictions to a CSV file."""
    df = pd.DataFrame({
        "window": range(1, len(predictions) + 1),
        "label": predictions
    })
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    preds = segment_and_predict(
        csv_path="Data\\CSV\\Data_Jumping_Ben.csv",
        model_path="final_model.joblib",
        sampling_rate=50,
        window_seconds=5
    )
    save_predictions(preds, "output.csv")