import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from preprocess import load_csv, remove_duplicates, fill_time_gaps, moving_average
from Feature_Extraction import safe_kurtosis, safe_skew

def extract_features(window):
    features = []
    for axis in range(3):
        signal = window[:, axis].astype(float)
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.var(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.max(signal) - np.min(signal))
        features.append(np.median(signal))
        features.append(safe_skew(signal))
        features.append(safe_kurtosis(signal))
        features.append(np.sqrt(np.mean(signal ** 2)))
    return np.array(features)

def get_sample_rate(df: pd.DataFrame) -> float:
    """Estimate sampling rate from the time column of a loaded DataFrame."""
    diffs = np.diff(df.iloc[:, 0].values)
    diffs = diffs[diffs > 0]
    return float(1.0 / np.median(diffs))

def segment_and_predict(csv_path, model_path, window_seconds=5):
    clf = joblib.load(model_path)

    # load and preprocess
    df = load_csv(csv_path)
    df = remove_duplicates(df)
    df = fill_time_gaps(df)
    df = moving_average(df, window=15)

    print(f"CSV columns : {df.columns.tolist()}")
    print(f"CSV shape   : {df.shape}")

    # sampling rate
    sampling_rate = get_sample_rate(df)
    print(f"Detected sampling rate: {sampling_rate:.1f} Hz")

    data = df.iloc[:, 1:4].values

    # segment into windows
    window_size = int(round(sampling_rate * window_seconds))
    n_windows   = len(data) // window_size

    if n_windows == 0:
        print(f"not enough data")
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

    print(f"Processed {n_windows} windows")
    return predictions

def save_predictions(predictions, output_path):
    """Save the list of predictions to a CSV file."""
    df = pd.DataFrame({
        "window": range(1, len(predictions) + 1),
        "label":  predictions,
    })
    df.to_csv(output_path, index=False)
    print(f"Predictions saved")



#############################################################################3

if __name__ == "__main__":
    preds = segment_and_predict(
        csv_path="Data/CSV/ELEC292_Jumping_RightHand_Data_Christian.csv",
        model_path="final_model.joblib",
        sampling_rate=50,
        window_seconds=5
    )
    save_predictions(preds, "ELEC292-Project/output.csv")

