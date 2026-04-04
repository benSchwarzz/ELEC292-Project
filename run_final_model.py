import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from preprocess import load_csv, remove_duplicates, fill_time_gaps, moving_average

# =============================================================================
# FEATURE EXTRACTION — must be identical to Feature_Extraction.py
# =============================================================================

def extract_features(window):
    """
    Extract features from one window.

    Parameters
    ----------
    window : np.ndarray, shape (n_samples, 3)
        Columns are acc_x, acc_y, acc_z  (time column already dropped).

    Returns
    -------
    np.ndarray, shape (30,)  — 10 features × 3 axes
    """
    features = []
    for axis in range(3):           # x=0, y=1, z=2
        signal = window[:, axis].astype(float)
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.var(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.max(signal) - np.min(signal))
        features.append(np.median(signal))
        features.append(skew(signal) if not np.allclose(signal, signal[0]) else 0.0)
        features.append(kurtosis(signal) if not np.allclose(signal, signal[0]) else 0.0)
        features.append(np.sqrt(np.mean(signal ** 2)))
    return np.array(features)


def get_sample_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate from the time column of a loaded DataFrame."""
    diffs = np.diff(df.iloc[:, 0].values)
    diffs = diffs[diffs > 0]
    return float(1.0 / np.median(diffs))


def segment_and_predict(csv_path, model_path, window_seconds=5):
    """
    Load a CSV file, preprocess it, segment into windows, and predict each window.

    Parameters
    ----------
    csv_path : str
        Path to the input accelerometer CSV (time, x, y, z[, abs]).
    model_path : str
        Path to the saved joblib model file.
    window_seconds : int
        Window length in seconds (default 5, matching training).

    Returns
    -------
    list of str
        One label ('walking' or 'jumping') per window.
    """
    clf = joblib.load(model_path)

    # --- Load and preprocess ------------------------------------------------
    df = load_csv(csv_path)
    df = remove_duplicates(df)
    df = fill_time_gaps(df)
    df = moving_average(df, window=15)

    print(f"CSV columns : {df.columns.tolist()}")
    print(f"CSV shape   : {df.shape}")

    # Detect sampling rate from data rather than using a hard-coded value
    # FIX: was hard-coded to 50 Hz, which gave wrong window_size for 100 Hz data
    sampling_rate = get_sample_rate(df)
    print(f"Detected sampling rate: {sampling_rate:.1f} Hz")

    # Select acc_x, acc_y, acc_z (columns 1-3); drop time and optional magnitude
    data = df.iloc[:, 1:4].values   # shape: (n_samples, 3)

    # Segment into windows
    window_size = int(round(sampling_rate * window_seconds))
    n_windows   = len(data) // window_size

    if n_windows == 0:
        print(f"ERROR: Not enough data. Got {len(data)} samples, "
              f"need at least {window_size} for one {window_seconds}-s window "
              f"at {sampling_rate:.0f} Hz.")
        return []

    predictions = []
    for i in range(n_windows):
        start    = i * window_size
        end      = start + window_size
        window   = data[start:end]
        features = extract_features(window).reshape(1, -1)
        pred     = clf.predict(features)[0]
        label    = "jumping" if pred == 1 else "walking"
        predictions.append(label)

    print(f"Processed {n_windows} windows from {csv_path}")
    return predictions


def save_predictions(predictions, output_path):
    """Save the list of predictions to a CSV file."""
    df = pd.DataFrame({
        "window": range(1, len(predictions) + 1),
        "label":  predictions,
    })
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    preds = segment_and_predict(
        csv_path="Data/CSV/ELEC292_Jumping_RightHand_Data_Christian.csv",
        model_path="final_model.joblib",
        sampling_rate=50,
        window_seconds=5
    )
    save_predictions(preds, "ELEC292-Project/output.csv")

