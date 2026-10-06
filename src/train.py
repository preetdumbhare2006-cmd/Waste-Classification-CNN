from pathlib import Path
import json, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras import layers,models,callbacks
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data_split"; MODELS=ROOT/"models"; RES=ROOT/"results"
MODELS.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
SIZE=(224,224); BATCH=32
train=tf.keras.utils.image_dataset_from_directory(DATA/"train",image_size=SIZE,batch_size=BATCH,label_mode="binary",shuffle=True,seed=42)
val=tf.keras.utils.image_dataset_from_directory(DATA/"val",image_size=SIZE,batch_size=BATCH,label_mode="binary",shuffle=False)
test=tf.keras.utils.image_dataset_from_directory(DATA/"test",image_size=SIZE,batch_size=BATCH,label_mode="binary",shuffle=False)
print("Classes:",train.class_names)
aug=tf.keras.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(.10),layers.RandomZoom(.10),layers.RandomContrast(.10)])
model=models.Sequential([layers.Input((224,224,3)),aug,layers.Rescaling(1/255),
layers.Conv2D(32,3,activation="relu",padding="same"),layers.MaxPooling2D(),
layers.Conv2D(64,3,activation="relu",padding="same"),layers.MaxPooling2D(),
layers.Conv2D(128,3,activation="relu",padding="same"),layers.MaxPooling2D(),
layers.Conv2D(256,3,activation="relu",padding="same"),layers.MaxPooling2D(),
layers.Flatten(),layers.Dense(256,activation="relu"),layers.Dropout(.5),layers.Dense(1,activation="sigmoid")])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"])
model.summary()
h=model.fit(train,validation_data=val,epochs=25,callbacks=[
callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.3,patience=2,min_lr=1e-6),
callbacks.ModelCheckpoint(MODELS/"waste_cnn.keras",monitor="val_accuracy",save_best_only=True)])
json.dump(train.class_names,open(MODELS/"class_names.json","w"))
for metric,title,name in [("accuracy","CNN Accuracy","accuracy_curve.png"),("loss","CNN Loss","loss_curve.png")]:
    plt.figure(figsize=(8,5)); plt.plot(h.history[metric],label="Training"); plt.plot(h.history["val_"+metric],label="Validation")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(metric.title()); plt.legend(); plt.tight_layout(); plt.savefig(RES/name,dpi=200); plt.close()
loss,acc=model.evaluate(test); print(f"Test accuracy: {acc:.4f}")
