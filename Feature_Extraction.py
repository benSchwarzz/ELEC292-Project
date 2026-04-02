import os
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

print("SCRIPT IS RUNNING - NEW VERSION")

csv_folder = "Data\\CSV"
window_size = 500

files_and_labels = [
    ("Data_Jumping_Ben.csv", "jumping"),
    ("Data_Jumping_SweaterPocket_Sachin.csv", "jumping"),
    ("ELEC292_Jumping_RightHand_Data_Christian.csv", "jumping"),
    ("Data_WalkingOutside_Ben.csv", "walking"),
    ("Data_Walking_SweaterPockzet_Sachin.csv", "walking"),
    ("ELEC292_Walking_LeftPocket_Data_Christian.csv", "walking")
]

def get_axis_columns(df):
    columns = df.columns.tolist()

    if "Acceleration x (m/s^2)" in columns:
        x_col = "Acceleration x (m/s^2)"
        y_col = "Acceleration y (m/s^2)"
        z_col = "Acceleration z (m/s^2)"
    elif "Linear Acceleration x (m/s^2)" in columns:
        x_col = "Linear Acceleration x (m/s^2)"
        y_col = "Linear Acceleration y (m/s^2)"
        z_col = "Linear Acceleration z (m/s^2)"
    else:
        raise ValueError(f"Could not find acceleration columns. Columns found: {columns}")

    return x_col, y_col, z_col

def calculate_stats(signal, prefix):
    return {
        f"mean_{prefix}": np.mean(signal),
        f"std_{prefix}": np.std(signal),
        f"max_{prefix}": np.max(signal),
        f"min_{prefix}": np.min(signal),
        f"range_{prefix}": np.max(signal) - np.min(signal),
        f"median_{prefix}": np.median(signal),
        f"var_{prefix}": np.var(signal),
        f"skew_{prefix}": skew(signal),
        f"kurtosis_{prefix}": pd.Series(signal).kurtosis(),
        f"rms_{prefix}": np.sqrt(np.mean(signal**2))
    }

def extract_features(window_df, label, source_file, x_col, y_col, z_col):
    x = window_df[x_col].values
    y = window_df[y_col].values
    z = window_df[z_col].values
    abs_acc = window_df["Absolute acceleration (m/s^2)"].values

    features = {}
    features.update(calculate_stats(x, "x"))
    features.update(calculate_stats(y, "y"))
    features.update(calculate_stats(z, "z"))
    features.update(calculate_stats(abs_acc, "abs"))

    features["label"] = label
    features["source_file"] = source_file

    return features

feature_rows = []

for filename, label in files_and_labels:
    file_path = os.path.join(csv_folder, filename)
    df = pd.read_csv("Data\\CSV" + filename)
    df.columns = df.columns.str.strip()

    print(f"Processing {filename}...")

    x_col, y_col, z_col = get_axis_columns(df)

    for start in range(0, len(df) - window_size + 1, window_size):
        window_df = df.iloc[start:start + window_size]

        if len(window_df) == window_size:
            feature_row = extract_features(window_df, label, filename, x_col, y_col, z_col)
            feature_rows.append(feature_row)

feature_df = pd.DataFrame(feature_rows)

print("\nALL FEATURE COLUMNS:")
print(feature_df.columns.tolist())

print("\nFeature table created:")
print(feature_df.head())
print("\nShape:", feature_df.shape)

X = feature_df.drop(columns=["label", "source_file"])
y = feature_df["label"]
source = feature_df["source_file"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_feature_df = pd.DataFrame(X_scaled, columns=X.columns)
scaled_feature_df["label"] = y.values
scaled_feature_df["source_file"] = source.values

print("\nALL NORMALIZED FEATURE COLUMNS:")
print(scaled_feature_df.columns.tolist())

print("\nNormalized feature table:")
print(scaled_feature_df.head())

feature_df.to_csv("features_raw.csv", index=False)
scaled_feature_df.to_csv("features_scaled.csv", index=False)

print("\nSaved files:")
print("features_raw.csv")
print("features_scaled.csv")