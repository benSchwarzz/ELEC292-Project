import numpy as np
import pandas as pd
import h5py
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

h5_file = "data_storage.h5"


def calculate_stats(signal, prefix):
    signal = np.array(signal, dtype=float)

    mean_value = np.mean(signal)
    std_value = np.std(signal)
    max_value = np.max(signal)
    min_value = np.min(signal)
    range_value = max_value - min_value
    median_value = np.median(signal)
    var_value = np.var(signal)

    # Safe skew for constant or nearly constant signals
    if np.allclose(signal, signal[0], atol=1e-8):
        skew_value = 0.0
    else:
        skew_value = skew(signal)
        if np.isnan(skew_value):
            skew_value = 0.0

    return {
        f"mean_{prefix}": mean_value,
        f"std_{prefix}": std_value,
        f"max_{prefix}": max_value,
        f"min_{prefix}": min_value,
        f"range_{prefix}": range_value,
        f"median_{prefix}": median_value,
        f"var_{prefix}": var_value,
        f"skew_{prefix}": skew_value
    }


def decode_labels(labels):
    decoded = []
    for label in labels:
        if isinstance(label, (bytes, np.bytes_)):
            decoded.append(label.decode("utf-8"))
        else:
            decoded.append(str(label))
    return decoded


def extract_features(segment):
    if segment.ndim != 2:
        raise ValueError(f"Expected 2D segment, got shape {segment.shape}")

    # If data is flipped like (4, 50), transpose to (50, 4)
    if segment.shape[0] <= 5 and segment.shape[1] > 5:
        segment = segment.T

    if segment.shape[1] < 3:
        raise ValueError(f"Segment must have at least 3 columns. Got shape {segment.shape}")

    x = segment[:, 0]
    y = segment[:, 1]
    z = segment[:, 2]

    # Use stored absolute acceleration if present
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


# ------------------------
# LOAD DATA FROM HDF5
# ------------------------

with h5py.File(h5_file, "r") as f:
    train_segments = f["segmented/train"][:]
    test_segments = f["segmented/test"][:]
    train_labels = f["segmented/train_labels"][:]
    test_labels = f["segmented/test_labels"][:]

train_labels = decode_labels(train_labels)
test_labels = decode_labels(test_labels)

print("Train segments:", train_segments.shape)
print("Test segments:", test_segments.shape)


# ------------------------
# EXTRACT TRAIN FEATURES
# ------------------------

train_rows = []

for i, segment in enumerate(train_segments):
    row = extract_features(segment)
    row["label"] = train_labels[i]
    train_rows.append(row)

train_features = pd.DataFrame(train_rows)

print("\nTrain features created")
print(train_features.head())
print(train_features.shape)


# ------------------------
# EXTRACT TEST FEATURES
# ------------------------

test_rows = []

for i, segment in enumerate(test_segments):
    row = extract_features(segment)
    row["label"] = test_labels[i]
    test_rows.append(row)

test_features = pd.DataFrame(test_rows)

print("\nTest features created")
print(test_features.head())
print(test_features.shape)


# ------------------------
# NORMALIZE FEATURES
# ------------------------

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


# ------------------------
# SAVE CSV FILES
# ------------------------

train_features.to_csv("train_features.csv", index=False)
test_features.to_csv("test_features.csv", index=False)

print("\nCSV files saved:")
print("train_features.csv")
print("test_features.csv")


# ------------------------
# SAVE FEATURES INTO HDF5
# ------------------------

with h5py.File(h5_file, "a") as f:
    if "features" not in f:
        features_group = f.create_group("features")
    else:
        features_group = f["features"]

    # Delete old feature datasets if they already exist
    for name in ["train", "test", "train_labels", "test_labels"]:
        if name in features_group:
            del features_group[name]

    # Save normalized feature arrays
    features_group.create_dataset(
        "train",
        data=train_features.drop(columns=["label"]).values
    )
    features_group.create_dataset(
        "test",
        data=test_features.drop(columns=["label"]).values
    )

    # Save labels separately
    features_group.create_dataset(
        "train_labels",
        data=train_features["label"].astype(float).values
    )
    features_group.create_dataset(
        "test_labels",
        data=test_features["label"].astype(float).values
    )

print("\nNormalized features added to data_storage.h5")
print("Saved in:")
print("features/train")
print("features/test")
print("features/train_labels")
print("features/test_labels")
