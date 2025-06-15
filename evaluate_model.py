import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model dan history
model = tf.keras.models.load_model("model/rps_model.h5")
with open("model/history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot accuracy & loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("model/training_metrics.png")
plt.close()

# Ganti ukuran gambar sesuai model
img_size = (224, 224)
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "organized_dataset",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Prediksi & evaluasi
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes
labels = list(test_generator.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()

# Classification report
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
report_summary = "\n".join(
    f"{label}: precision={value['precision']:.2f}, recall={value['recall']:.2f}, f1-score={value['f1-score']:.2f}"
    for label, value in report.items() if label in labels
)
report_summary += f"\n\nAccuracy: {report['accuracy']:.2f}"

with open("model/classification_report.txt", "w") as f:
    f.write(report_summary)
