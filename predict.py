import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("model/rps_model.h5")
class_names = ['paper', 'rock', 'scissors']

# Dataset path
dataset_path = 'organized_dataset'

# Folder untuk simpan hasil grafik
output_dir = 'prediction_graphs'
os.makedirs(output_dir, exist_ok=True)

# Ukuran input sesuai model
target_size = (224, 224)

# Loop semua folder kelas
for label in class_names:
    folder_path = os.path.join(dataset_path, label)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = prediction[predicted_index] * 100

        # Print info
        print(f"Gambar     : {img_name}")
        print(f"Label Asli : {label}")
        print(f"Prediksi   : {predicted_class} ({confidence:.2f}%)")
        print("-" * 40)

        # Plot dan simpan grafik
        plt.figure(figsize=(6, 4))
        plt.bar(class_names, prediction, color=['skyblue', 'orange', 'lightgreen'])
        plt.ylim([0, 1])
        plt.title(f"{img_name}\nAsli: {label} | Prediksi: {predicted_class} ({confidence:.2f}%)")
        plt.xlabel("Kelas")
        plt.ylabel("Probabilitas")
        plt.tight_layout()

        # Simpan file grafik
        safe_filename = f"{label}_{os.path.splitext(img_name)[0]}.png"
        plt.savefig(os.path.join(output_dir, safe_filename))
        plt.close()
