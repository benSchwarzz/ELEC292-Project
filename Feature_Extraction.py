import numpy as np
import pandas as pd
import h5py
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

print("SCRIPT IS RUNNING - HDF5 VERSION")

H5_FILE = "data_storage.h5"



# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def decode_labels(labels):
    decoded = []
    for label in labels:
        if isinstance(label, (bytes, np.bytes_)):
            decoded.append(label.decode("utf-8"))
        else:
            decoded.append(str(label))
    return decoded


def safe_skew(signal):
    if np.allclose(signal, signal[0], atol=1e-8):
        return 0.0
    value = skew(signal)
    return 0.0 if np.isnan(value) else float(value)


def safe_kurtosis(signal):
    if np.allclose(signal, signal[0], atol=1e-8):
        return 0.0
    value = kurtosis(signal)
    return 0.0 if np.isnan(value) else float(value)


def calculate_stats(signal, prefix):
    signal = np.array(signal, dtype=float)
    return {
        f"mean_{prefix}":     np.mean(signal),
        f"std_{prefix}":      np.std(signal),
        f"max_{prefix}":      np.max(signal),
        f"min_{prefix}":      np.min(signal),
        f"range_{prefix}":    np.max(signal) - np.min(signal),
        f"median_{prefix}":   np.median(signal),
        f"var_{prefix}":      np.var(signal),
        f"skew_{prefix}":     safe_skew(signal),
        f"kurtosis_{prefix}": safe_kurtosis(signal),
        f"rms_{prefix}":      np.sqrt(np.mean(signal ** 2)),
    }


def extract_features(segment):
    """
    Extract 30 statistical features (10 per axis) from a single window.

    Parameters
    ----------
    segment : np.ndarray, shape (window_len, 4)
        Columns are [time, acc_x, acc_y, acc_z].

    Returns
    -------
    dict of feature_name -> float
    """
    if segment.ndim != 2:
        raise ValueError(f"Expected 2D segment, got shape {segment.shape}")

    # Transpose if segment was stored as (4, N) instead of (N, 4)
    if segment.shape[0] <= 5 and segment.shape[1] > 5:
        segment = segment.T

    if segment.shape[1] < 4:
        raise ValueError(
            f"Segment must have at least 4 columns [time, x, y, z]. "
            f"Got shape {segment.shape}"
        )

    # BUG FIX: columns are [time=0, x=1, y=2, z=3].
    # Previously this used indices 0,1,2 which included the time column as a
    # feature and excluded z entirely. Corrected to indices 1, 2, 3.
    x = segment[:, 1]   # acc_x
    y = segment[:, 2]   # acc_y
    z = segment[:, 3]   # acc_z

    features = {}
    features.update(calculate_stats(x, "x"))
    features.update(calculate_stats(y, "y"))
    features.update(calculate_stats(z, "z"))
    return features


# ---------------------------------------------------------------------------
# Load segmented data from HDF5
# ---------------------------------------------------------------------------

with h5py.File(H5_FILE, "r") as f:
    train_segments = f["segmented/train"][:]
    test_segments  = f["segmented/test"][:]
    train_labels   = f["segmented/train_labels"][:]
    test_labels    = f["segmented/test_labels"][:]

train_labels = decode_labels(train_labels)
test_labels  = decode_labels(test_labels)

print("Train segments :", train_segments.shape)
print("Test  segments :", test_segments.shape)

# ---------------------------------------------------------------------------
# Extract features
# ---------------------------------------------------------------------------

train_rows = []
for i, segment in enumerate(train_segments):
    row = extract_features(segment)
    row["label"] = train_labels[i]
    train_rows.append(row)

train_features = pd.DataFrame(train_rows)
print("\nTrain features created")
print(train_features.head())
print(train_features.shape)

test_rows = []
for i, segment in enumerate(test_segments):
    row = extract_features(segment)
    row["label"] = test_labels[i]
    test_rows.append(row)

test_features = pd.DataFrame(test_rows)
print("\nTest features created")
print(test_features.head())
print(test_features.shape)

# ---------------------------------------------------------------------------
# Normalize features (z-score standardisation)
# Scaler is fit on training data only to prevent data leakage.
# ---------------------------------------------------------------------------

X_train = train_features.drop(columns=["label"])
X_test  = test_features.drop(columns=["label"])
y_train = train_features["label"]
y_test  = test_features["label"]

feature_names = X_train.columns.tolist()

scaler        = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

train_features = pd.DataFrame(X_train_scaled, columns=feature_names)
train_features["label"] = y_train.values

test_features = pd.DataFrame(X_test_scaled, columns=feature_names)
test_features["label"] = y_test.values

print("\nNormalized train features")
print(train_features.head())
print(train_features.shape)

print("\nNormalized test features")
print(test_features.head())
print(test_features.shape)

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------

train_features.to_csv("train_features.csv", index=False)
test_features.to_csv("test_features.csv",   index=False)
print("\nCSV files saved: train_features.csv, test_features.csv")

# ---------------------------------------------------------------------------
# Save features back into HDF5
# ---------------------------------------------------------------------------

with h5py.File(H5_FILE, "a") as f:
    features_grp = f.require_group("features")

    for name in ["train", "test", "train_labels", "test_labels"]:
        if name in features_grp:
            del features_grp[name]

    features_grp.create_dataset(
        "train", data=train_features.drop(columns=["label"]).values
    )
    features_grp.create_dataset(
        "test",  data=test_features.drop(columns=["label"]).values
    )

    # Store labels as integers (0=walking, 1=jumping)
    label_map = {"0": 0, "1": 1, "walking": 0, "jumping": 1}
    train_label_arr = np.array(
        [label_map.get(str(l), int(float(l))) for l in train_features["label"]]
    )
    test_label_arr  = np.array(
        [label_map.get(str(l), int(float(l))) for l in test_features["label"]]
    )
    features_grp.create_dataset("train_labels", data=train_label_arr)
    features_grp.create_dataset("test_labels",  data=test_label_arr)
    features_grp.attrs["feature_names"]   = feature_names
    features_grp.attrs["label_encoding"]  = "0 = walking, 1 = jumping"
    features_grp.attrs["normalisation"]   = "z-score (StandardScaler fit on train only)"

print("Features saved to HDF5.")