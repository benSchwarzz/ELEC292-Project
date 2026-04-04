import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, f1_score

from sklearn.model_selection import learning_curve
import joblib

# loading data

HDF5_PATH = "data_storage.h5"

with h5py.File(HDF5_PATH, "r") as f:
    X_train = f["features/train"][:]
    y_train = f["features/train_labels"][:].astype(int)
    X_test = f["features/test"][:]
    y_test = f["features/test_labels"][:].astype(int)

# create pipeline
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000, random_state=0)
)

# learning curves

print("\nComputing learning curves...")

train_sizes, train_scores, val_scores = learning_curve(
    clf,
    X_train, 
    y_train,
    cv = 5,
    scoring = "accuracy",
    train_sizes = np.linspace(0.1, 1.0, 10),
    shuffle = True,
    random_state = 0,
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", color="royalblue", label="Training accuracy")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="royalblue")
plt.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation accuracy (CV)")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="darkorange")
plt.xlabel("Number of training samples")
plt.ylabel("Accuracy")
plt.title("Learning Curves — Logistic Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.close()

# training final model
print("training")
clf.fit(X_train, y_train)

# evaluating on test set
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
train_acc = accuracy_score(y_train, clf.predict(X_train))

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Recall (sensitivity): {recall:.4f}")
print(f"F1 score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Walking", "Jumping"])

plt.figure(figsize=(6, 5))
disp.plot(colorbar=True, cmap="Blues")
plt.title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

tn, fp, fn, tp = cm.ravel()
print(f"TP = {tp}  |  TN = {tn}  |  FP = {fp}  |  FN = {fn}")

specificity = tn / (tn + fp)
fpr_manual  = fp / (fp + tn)
precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
print(f"Specificity: {specificity:.4f}")
print(f"FPR: {fpr_manual:.4f}")
print(f"Precision: {precision:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="royalblue", lw=2,
         label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="red", lw=1.5, linestyle="--",
         label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve — Logistic Regression")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# results
print("\nSummary of Evaluation Metrics")

for name, val in [
    ("Training Accuracy", train_acc),
    ("Test Accuracy", accuracy),
    ("Sensitivity / Recall", recall),
    ("Specificity", specificity),
    ("Precision", precision),
    ("F1 Score", f1),
    ("False Positive Rate", fpr_manual),
    ("AUC", auc),
]:
    print(f"{name}:   {val:0.4f}")

for name, val in [("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn)]:
    print(f"{name}:   {val}")

joblib.dump(clf, "final_model.joblib")
print("\nFinal model saved to final_model.joblib")