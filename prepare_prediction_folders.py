import os
import shutil

def prepare_folders(base_dir='prediction_graphs'):
    labels = ['paper', 'rock', 'scissors']

    # Bikin folder utama dan subfolder
    os.makedirs(base_dir, exist_ok=True)

    for label in labels:
        folder_path = os.path.join(base_dir, label)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✅ Folder disiapkan: {folder_path}")

    return labels

def sort_graphs(base_dir='prediction_graphs', labels=None):
    if labels is None:
        labels = ['paper', 'rock', 'scissors']

    # Loop semua file di folder utama
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)

        # Hanya proses file gambar PNG
        if os.path.isfile(file_path) and filename.lower().endswith('.png'):
            filename_lower = filename.lower()
            for label in labels:
                if filename_lower.startswith(label.lower()):
                    target_path = os.path.join(base_dir, label, filename)
                    shutil.move(file_path, target_path)
                    print(f"📦 {filename} → {label}/")
                    break

if __name__ == "__main__":
    labels = prepare_folders()
    sort_graphs(labels=labels)
