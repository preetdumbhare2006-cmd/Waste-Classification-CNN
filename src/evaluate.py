from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, seaborn as sns, tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data_split"; RES=ROOT/"results"
test=tf.keras.utils.image_dataset_from_directory(DATA/"test",image_size=(224,224),batch_size=32,label_mode="binary",shuffle=False)
model=tf.keras.models.load_model(ROOT/"models/waste_cnn.keras"); names=test.class_names
yt=[]; yp=[]
for x,y in test: yt.extend(y.numpy().ravel()); yp.extend(model.predict(x,verbose=0).ravel())
yt=np.array(yt).astype(int); yp=(np.array(yp)>=.5).astype(int)
print("Accuracy :",accuracy_score(yt,yp)); print("Precision:",precision_score(yt,yp,zero_division=0))
print("Recall   :",recall_score(yt,yp,zero_division=0)); print("F1-score :",f1_score(yt,yp,zero_division=0))
print(classification_report(yt,yp,target_names=names,zero_division=0))
cm=confusion_matrix(yt,yp); plt.figure(figsize=(7,5)); sns.heatmap(cm,annot=True,fmt="d",xticklabels=names,yticklabels=names)
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(RES/"confusion_matrix.png",dpi=200); plt.close()
