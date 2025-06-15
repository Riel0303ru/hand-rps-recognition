import os
import shutil
import pandas as pd

csv_path = "rps_dataset/train/train/_annotations.csv"
img_dir = "rps_dataset/train/train"
output_dir = "organized_dataset"

# Baca data CSV
df = pd.read_csv(csv_path)

# Buat folder output
for label in df['class'].unique():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Salin gambar ke folder berdasarkan label
for _, row in df.iterrows():
    src = os.path.join(img_dir, row['filename'])
    dst = os.path.join(output_dir, row['class'], row['filename'])
    if os.path.exists(src):
        shutil.copy(src, dst)

print("Dataset berhasil diorganisasi ulang ke folder 'organized_dataset'")
