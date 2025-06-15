import pickle
import matplotlib.pyplot as plt
import os

# Path file history
history_path = "model/history.pkl"

# Cek file exist dulu
if not os.path.exists(history_path):
    raise FileNotFoundError("File history.pkl tidak ditemukan di folder 'model'.")

# Load history
with open(history_path, "rb") as f:
    history = pickle.load(f)

# Fungsi bantu untuk plotting
def plot_metric(metric_name, title, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(history.get(metric_name, []), label=f'Training {ylabel}')
    plt.plot(history.get(f'val_{metric_name}', []), label=f'Validation {ylabel}')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot akurasi
plot_metric("accuracy", "Training & Validation Accuracy", "Accuracy")

# Plot loss
plot_metric("loss", "Training & Validation Loss", "Loss")
