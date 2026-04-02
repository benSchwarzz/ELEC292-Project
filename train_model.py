import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import learning_curve
import joblib

# =============================================================================
# 1. LOAD FEATURES AND LABELS FROM HDF5
# =============================================================================
# This assumes you saved your extracted features and labels into the HDF5 file
# in Step 5. Adjust the dataset paths to match your actual HDF5 structure.

HDF5_PATH = "data_storage.h5"

with h5py.File(HDF5_PATH, "r") as f:
    X_train = f["features/train"][:]   # shape: (n_train_windows, n_features)
    y_train = f["features/train_labels"][:]     # shape: (n_train_windows,)
    X_test  = f["features/test"][:]    # shape: (n_test_windows, n_features)
    y_test  = f["features/test_labels"][:]      # shape: (n_test_windows,)

# Labels should be 0 = walking, 1 = jumping (or whatever encoding you used)
print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Features per window: {X_train.shape[1]}")

# =============================================================================
# 2. DEFINE THE MODEL (StandardScaler + LogisticRegression pipeline)
# =============================================================================
# Using a pipeline ensures the scaler is fit ONLY on training data and then
# applied to the test set — preventing data leakage (as covered in the slides).

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000, random_state=0)
)

# =============================================================================
# 3. LEARNING CURVES
# =============================================================================
# Learning curves show how training and validation accuracy evolve as more
# training data is used. This helps detect overfitting or underfitting.

print("\nComputing learning curves...")

train_sizes, train_scores, val_scores = learning_curve(
    clf,
    X_train, y_train,
    cv=5,                          # 5-fold cross-validation on the training set
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True,
    random_state=0
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", color="royalblue", label="Training accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.15, color="royalblue")
plt.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation accuracy (CV)")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.15, color="darkorange")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.title("Learning Curves — Logistic Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()
print("Learning curves saved to learning_curves.png")

# =============================================================================
# 4. TRAIN THE FINAL MODEL ON ALL TRAINING DATA
# =============================================================================

print("\nTraining final model on full training set...")
clf.fit(X_train, y_train)

# =============================================================================
# 5. EVALUATE ON THE TEST SET
# =============================================================================

y_pred      = clf.predict(X_test)
y_prob      = clf.predict_proba(X_test)[:, 1]   # probability of positive class (jumping)

accuracy    = accuracy_score(y_test, y_pred)
recall      = recall_score(y_test, y_pred)
f1          = f1_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_prob)

train_acc   = accuracy_score(y_train, clf.predict(X_train))

print(f"\n{'='*40}")
print(f"  Training accuracy : {train_acc:.4f}")
print(f"  Test accuracy     : {accuracy:.4f}")
print(f"  Recall (sensitivity): {recall:.4f}")
print(f"  F1 score          : {f1:.4f}")
print(f"  AUC               : {auc:.4f}")
print(f"{'='*40}\n")

# =============================================================================
# 6. CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Walking", "Jumping"])

plt.figure(figsize=(6, 5))
disp.plot(colorbar=True, cmap="Blues")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png")

# Print the raw TP / TN / FP / FN values for the report
tn, fp, fn, tp = cm.ravel()
print(f"TP = {tp}  |  TN = {tn}  |  FP = {fp}  |  FN = {fn}")

# Derived metrics (matching slide formulas exactly)
specificity = tn / (tn + fp)
fpr_manual  = fp / (fp + tn)
precision   = tp / (tp + fp)
print(f"Specificity : {specificity:.4f}")
print(f"FPR         : {fpr_manual:.4f}")
print(f"Precision   : {precision:.4f}")

# =============================================================================
# 7. ROC CURVE AND AUC
# =============================================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="red", lw=1.5, linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve — Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("ROC curve saved to roc_curve.png")

# =============================================================================
# 8. SUMMARY TABLE (for quick reference in report)
# =============================================================================

print("\n--- Summary of Evaluation Metrics ---")
print(f"{'Metric':<25} {'Value':>10}")
print("-" * 37)
print(f"{'Training Accuracy':<25} {train_acc:>10.4f}")
print(f"{'Test Accuracy':<25} {accuracy:>10.4f}")
print(f"{'Sensitivity / Recall':<25} {recall:>10.4f}")
print(f"{'Specificity':<25} {specificity:>10.4f}")
print(f"{'Precision':<25} {precision:>10.4f}")
print(f"{'F1 Score':<25} {f1:>10.4f}")
print(f"{'False Positive Rate':<25} {fpr_manual:>10.4f}")
print(f"{'AUC':<25} {auc:>10.4f}")
print(f"{'TP':<25} {tp:>10}")
print(f"{'TN':<25} {tn:>10}")
print(f"{'FP':<25} {fp:>10}")
print(f"{'FN':<25} {fn:>10}")

# =============================================================================
# 9. Save the model
# =============================================================================

joblib.dump(clf, "final_model.joblib")
print("\nFinal model saved to final_model.joblib")