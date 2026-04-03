import numpy as np
import pandas as pd
import h5py
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

print("SCRIPT IS RUNNING - HDF5 VERSION")

h5_file = "data_storage.h5"

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
    return 0.0 if np.isnan(value) else value

def safe_kurtosis(signal):
    if np.allclose(signal, signal[0], atol=1e-8):
        return 0.0
    value = kurtosis(signal)
    return 0.0 if np.isnan(value) else value

def calculate_stats(signal, prefix):
    signal = np.array(signal, dtype=float)

    return {
        f"mean_{prefix}": np.mean(signal),
        f"std_{prefix}": np.std(signal),
        f"max_{prefix}": np.max(signal),
        f"min_{prefix}": np.min(signal),
        f"range_{prefix}": np.max(signal) - np.min(signal),
        f"median_{prefix}": np.median(signal),
        f"var_{prefix}": np.var(signal),
        f"skew_{prefix}": safe_skew(signal),
        f"kurtosis_{prefix}": safe_kurtosis(signal),
        f"rms_{prefix}": np.sqrt(np.mean(signal**2))
    }

def extract_features(segment):
    if segment.ndim != 2:
        raise ValueError(f"Expected 2D segment, got shape {segment.shape}")

    # Fix shape if segment is flipped like (4, N)
    if segment.shape[0] <= 5 and segment.shape[1] > 5:
        segment = segment.T

    if segment.shape[1] < 3:
        raise ValueError(f"Segment must have at least 3 columns. Got shape {segment.shape}")

    x = segment[:, 0]
    y = segment[:, 1]
    z = segment[:, 2]

    if segment.shape[1] >= 4:
        abs_acc = segment[:, 3]
    else:
        abs_acc = np.sqrt(x**2 + y**2 + z**2)

    features = {}
    features.update(calculate_stats(x, "x"))
    features.update(calculate_stats(y, "y"))
    features.update(calculate_stats(z, "z"))
    features.update(calculate_stats(abs_acc, "abs"))

    return features

# LOAD SEGMENTED DATA FROM HDF5
with h5py.File(h5_file, "r") as f:
    train_segments = f["segmented/train"][:]
    test_segments = f["segmented/test"][:]
    train_labels = f["segmented/train_labels"][:]
    test_labels = f["segmented/test_labels"][:]

train_labels = decode_labels(train_labels)
test_labels = decode_labels(test_labels)

print("Train segments:", train_segments.shape)
print("Test segments:", test_segments.shape)

# EXTRACT TRAIN FEATURES
train_rows = []
for i, segment in enumerate(train_segments):
    row = extract_features(segment)
    row["label"] = train_labels[i]
    train_rows.append(row)

train_features = pd.DataFrame(train_rows)

print("\nTrain features created")
print(train_features.head())
print(train_features.shape)

# EXTRACT TEST FEATURES
test_rows = []
for i, segment in enumerate(test_segments):
    row = extract_features(segment)
    row["label"] = test_labels[i]
    test_rows.append(row)

test_features = pd.DataFrame(test_rows)

print("\nTest features created")
print(test_features.head())
print(test_features.shape)

# NORMALIZE FEATURES
X_train = train_features.drop(columns=["label"])
X_test = test_features.drop(columns=["label"])

y_train = train_features["label"]
y_test = test_features["label"]

feature_names = X_train.columns

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_features = pd.DataFrame(X_train_scaled, columns=feature_names)
train_features["label"] = y_train.values

test_features = pd.DataFrame(X_test_scaled, columns=feature_names)
test_features["label"] = y_test.values

print("\nNormalized train features created")
print(train_features.head())
print(train_features.shape)

print("\nNormalized test features created")
print(test_features.head())
print(test_features.shape)

# SAVE CSV FILES
train_features.to_csv("train_features.csv", index=False)
test_features.to_csv("test_features.csv", index=False)

print("\nCSV files saved:")
print("train_features.csv")
print("test_features.csv")

# SAVE FEATURES INTO HDF5
with h5py.File(h5_file, "a") as f:
    if "features" not in f:
        features_group = f.create_group("features")
    else:
        features_group = f["features"]

    # Delete old datasets if they already exist
    for name in ["train", "test", "train_labels", "test_labels"]:
        if name in features_group:
            del features_group[name]

    features_group.create_dataset(
        "train",
        data=train_features.drop(columns=["label"]).values
    )
    features_group.create_dataset(
        "test",
        data=test_features.drop(columns=["label"]).values
    )

    # convert labels to numbers if needed
    try:
        train_label_data = train_features["label"].astype(float).values
        test_label_data = test_features["label"].astype(float).values
    except:
        label_map = {"walking": 0, "jumping": 1}
        train_label_data = train_features["label"].map(label_map).values
        test_label_data = test_features["label"].map(label_map).values

    features_group.create_dataset("train_labels", data=train_label_data)
    features_group.create_dataset("test_labels", data=test_label_data)
