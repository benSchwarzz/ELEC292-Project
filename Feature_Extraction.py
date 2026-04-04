import numpy as np
import pandas as pd
import h5py
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

h5_file = "data_storage.h5"

def decode_labels(labels):
    decoded = []
    for label in labels:
        if isinstance(label, (bytes, np.bytes_)):
            decoded.append(label.decode("utf-8"))
        else:
            decoded.append(str(label))
    return decoded

#Skew feature
def skew_feature(signal):
    if np.allclose(signal, signal[0], atol=1e-8):
        return 0.0
    value = skew(signal)
    return 0.0 if np.isnan(value) else value

#Kurtosis feature
def kurtosis_feature(signal):
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
        f"skew_{prefix}": skew_feature(signal),
        f"kurtosis_{prefix}": kurtosis_feature(signal),
        f"rms_{prefix}": np.sqrt(np.mean(signal**2))
    }

def extract_features(segment):
    if segment.ndim != 2:
        raise ValueError(f"Expected 2D segment, got shape {segment.shape}")

    if segment.shape[1] < 3:
        raise ValueError(f"Segment must have at least 3 columns. Got shape {segment.shape}")

    x = segment[:, 1]
    y = segment[:, 2]
    z = segment[:, 3]

    features = {}
    features.update(calculate_stats(x, "x"))
    features.update(calculate_stats(y, "y"))
    features.update(calculate_stats(z, "z"))
    
    return features

#Load Segmented Data From HDF5
with h5py.File(h5_file, "r") as f:
    train_segments = f["segmented/train"][:]
    test_segments = f["segmented/test"][:]
    train_labels = f["segmented/train_labels"][:]
    test_labels = f["segmented/test_labels"][:]

train_labels = decode_labels(train_labels)
test_labels = decode_labels(test_labels)


#Extract Train Features
train_rows = []
for i, segment in enumerate(train_segments):
    row = extract_features(segment)
    row["label"] = train_labels[i]
    train_rows.append(row)

train_features = pd.DataFrame(train_rows)


#Extract Test Features
test_rows = []
for i, segment in enumerate(test_segments):
    row = extract_features(segment)
    row["label"] = test_labels[i]
    test_rows.append(row)

test_features = pd.DataFrame(test_rows)


# Normalize Features
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


#Save CSV Files
train_features.to_csv("train_features.csv", index=False)
test_features.to_csv("test_features.csv", index=False)

print("\nCSV files saved:")
print("train_features.csv")
print("test_features.csv")

#Save Features into HDF5
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
print("\nHDF5 files saved:")
print("features")