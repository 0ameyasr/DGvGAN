import pandas as pd
import matplotlib.pyplot as plt
import glob

# 1. Load and aggregate data
file_pattern = 'models/saves/hybrid/history/*.csv'
files = glob.glob(file_pattern)
print(files)

dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs)
avg_history = combined.groupby('epoch').mean().reset_index()

# 2. Plotting (VERTICAL + LARGER)
fig, axes = plt.subplots(2, 1, figsize=(6,6))  # taller figure

# Generator Loss Plot
axes[0].plot(avg_history['epoch'], avg_history['g_loss'], linewidth=2, color='red')
axes[0].set_title('Generator Loss', fontsize=16)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle='--', alpha=0.7)

# Discriminator Loss Plot
axes[1].plot(avg_history['epoch'], avg_history['d_loss'], linewidth=2)
axes[1].set_title('Discriminator Loss', fontsize=16)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()