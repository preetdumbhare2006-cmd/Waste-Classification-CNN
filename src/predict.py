from pathlib import Path
import sys, numpy as np, tensorflow as tf
ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2: raise SystemExit("Usage: python src/predict.py image.jpg")
p=Path(sys.argv[1])
model=tf.keras.models.load_model(ROOT/"models/waste_cnn.keras")
img=tf.keras.utils.load_img(p,target_size=(224,224)); a=np.expand_dims(tf.keras.utils.img_to_array(img),0)
prob=float(model.predict(a,verbose=0)[0][0]); label="Non-Degradable" if prob>=.5 else "Degradable"; conf=prob if prob>=.5 else 1-prob
print(f"Prediction: {label}"); print(f"Confidence: {conf*100:.2f}%")
