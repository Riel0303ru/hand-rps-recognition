import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tqdm import tqdm

# Load model
model = tf.keras.models.load_model("model/rps_model.h5")

# Daftar kelas & path dataset
class_names = ['paper', 'rock', 'scissors']
base_dir = "organized_dataset"

# Variabel evaluasi
correct = 0
total = 0
per_class_result = {k: {"correct": 0, "total": 0} for k in class_names}

print("\n🚀 Memulai evaluasi model...\n")

# Loop gambar untuk setiap kelas
for label in class_names:
    path = os.path.join(base_dir, label)
    file_list = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for fname in tqdm(file_list, desc=f"🔍 Mengecek '{label}'", unit="gambar"):
        img_path = os.path.join(path, fname)
        try:
            # ✅ Ukuran gambar sesuai input model
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi
            prediction = model.predict(img_array, verbose=0)[0]
            predicted_class = class_names[np.argmax(prediction)]

            # Evaluasi akurasi
            if predicted_class == label:
                correct += 1
                per_class_result[label]["correct"] += 1
            per_class_result[label]["total"] += 1
            total += 1

        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

# Output hasil
print(f"\n📊 Evaluasi Selesai!")
print(f"🖼️  Total Gambar   : {total}")
print(f"✅ Prediksi Benar : {correct}")
print(f"🎯 Akurasi Total  : {correct / total * 100:.2f}%\n")

print("📋 Akurasi per Kelas:")
for label in class_names:
    total_kelas = per_class_result[label]["total"]
    benar_kelas = per_class_result[label]["correct"]
    akurasi = (benar_kelas / total_kelas) * 100 if total_kelas > 0 else 0
    print(f"  - {label.capitalize():<10}: {akurasi:.2f}% ({benar_kelas}/{total_kelas})")
