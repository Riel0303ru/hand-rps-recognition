import os

base_dir = 'organized_dataset'
categories = ['paper', 'rock', 'scissors']

for category in categories:
    folder_path = os.path.join(base_dir, category)
    for i, filename in enumerate(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[-1]
        new_name = f"{category}_{i+1:04d}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)

print("✅ Semua gambar berhasil di-rename!")
