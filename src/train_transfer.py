from pathlib import Path
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data_split"
MODEL_DIR = ROOT / "models"
RESULTS = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
SEED = 42

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names

print("\nClass names:")
print(class_names)

# --------------------------------------------------
# PERFORMANCE OPTIMIZATION
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
])

# --------------------------------------------------
# MOBILE NET V2 BASE MODEL
# --------------------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# --------------------------------------------------
# BUILD MODEL
# --------------------------------------------------

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

# --------------------------------------------------
# COMPILE
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n========== MOBILE NET V2 MODEL ==========\n")

model.summary()

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

checkpoint_path = MODEL_DIR / "mobilenetv2_waste.keras"

cb = [

    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7
    ),

    callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True
    )
]

# --------------------------------------------------
# PHASE 1
# TRAIN CLASSIFICATION HEAD
# --------------------------------------------------

print("\n========================================")
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("========================================\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=cb
)

# --------------------------------------------------
# PHASE 2
# FINE TUNING
# --------------------------------------------------

print("\n========================================")
print("PHASE 2: FINE TUNING MOBILE NET V2")
print("========================================\n")

base_model.trainable = True

# Freeze early layers
fine_tune_from = 100

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

# Recompile with small learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=cb
)

# --------------------------------------------------
# SAVE CLASS NAMES
# --------------------------------------------------

with open(
    MODEL_DIR / "mobilenet_class_names.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=2
    )

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================\n")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(
    f"\nMobileNetV2 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"MobileNetV2 Test Loss: "
    f"{test_loss:.4f}"
)

# --------------------------------------------------
# COMBINE TRAINING HISTORY
# --------------------------------------------------

acc = history1.history["accuracy"] + history2.history["accuracy"]

val_acc = (
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

loss = history1.history["loss"] + history2.history["loss"]

val_loss = (
    history1.history["val_loss"]
    + history2.history["val_loss"]
)

epochs_range = range(1, len(acc) + 1)

# --------------------------------------------------
# ACCURACY GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    acc,
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_acc,
    label="Validation Accuracy"
)

plt.axvline(
    INITIAL_EPOCHS,
    linestyle="--",
    label="Fine-Tuning Starts"
)

plt.title(
    "MobileNetV2 Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "mobilenet_accuracy_curve.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# LOSS GRAPH
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    loss,
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_loss,
    label="Validation Loss"
)

plt.axvline(
    INITIAL_EPOCHS,
    linestyle="--",
    label="Fine-Tuning Starts"
)

plt.title(
    "MobileNetV2 Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS / "mobilenet_loss_curve.png",
    dpi=200
)

plt.close()

print("\nTraining completed successfully!")

print(
    f"Model saved at:\n"
    f"{checkpoint_path}"
)