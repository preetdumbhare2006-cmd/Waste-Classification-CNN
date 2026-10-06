from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data_split"
MODEL_PATH = ROOT / "models" / "mobilenetv2_waste.keras"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "test",
    image_size=(224, 224),
    batch_size=32,
    label_mode="binary",
    shuffle=False
)

class_names = test_ds.class_names

print("\nClass names:")
print(class_names)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_true = []
y_prob = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    ).ravel()

    y_prob.extend(predictions)

    y_true.extend(
        labels.numpy()
        .astype(int)
        .ravel()
    )


y_true = np.array(y_true)

y_prob = np.array(y_prob)

y_pred = (
    y_prob >= 0.5
).astype(int)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n========================================")
print("MOBILENETV2 EVALUATION RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "MobileNetV2 Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    RESULTS / "mobilenet_confusion_matrix.png",
    dpi=200
)

plt.close()


print(
    "\nConfusion matrix saved at:"
)

print(
    RESULTS / "mobilenet_confusion_matrix.png"
)