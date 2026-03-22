

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # up to src/
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cpe487587hw import deepl


DATA_DIR    = '/data/CPE_487-587/ACCDataset'
EPOCHS      = 100
TRAIN_RATIO = 0.8
BATCH_SIZE  = 512
K           = 10              # history length (spec: k=10)
WINDOW_SIZE = (K + 1) * 4    # 44 = 4 feature groups x 11 timesteps
LR          = 3e-4



def get_best_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    utilizations = [int(x.strip()) for x in result.stdout.strip().split("\n")]
    return utilizations.index(min(utilizations))



if torch.cuda.is_available():
    device_id = get_best_gpu()
    device    = torch.device(f"cuda:{device_id}")
    print(f"Selected GPU: {device_id}")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU.")

# 1. Build CSV pairs
csv_pairs = deepl.build_csv_pairs(DATA_DIR)

# 2. Compute normalisation coefficients over all speed data
all_speeds = []
for sf, _ in csv_pairs:
    df = pd.read_csv(sf, usecols=['Message'])
    all_speeds.append(df['Message'].values / 3.6)   # km/h -> m/s

flat      = np.concatenate(all_speeds)
norm_mean = float(np.mean(flat))
norm_std  = float(np.std(flat))
print(f"Speed normalisation - mean: {norm_mean:.4f} m/s, std: {norm_std:.4f} m/s")

# Save normalisation coefficients alongside the ONNX model
np.savez(os.path.join(SCRIPT_DIR, 'acc_norm_coeffs.npz'),
         mean=norm_mean, std=norm_std)
print("Normalisation coefficients saved.")

# 3. Build dataset
full_dataset = deepl.ACCDataset(
    csv_pairs,
    k=K,
    norm_mean=norm_mean,
    norm_std=norm_std,
)
print(f"Total samples: {len(full_dataset)}")

# Print class balance
total = len(full_dataset)
pos   = int(full_dataset.y.sum())
neg   = total - pos
print(f"Class balance - ACC=1: {pos/total*100:.1f}%  ACC=0: {neg/total*100:.1f}%")

# 4. Train / validation split
n_train = int(len(full_dataset) * TRAIN_RATIO)
n_val   = len(full_dataset) - n_train

train_dataset, val_dataset = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
print(f"Train samples: {n_train}, Val samples: {n_val}")

# 5. DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=(device.type == 'cuda'),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=(device.type == 'cuda'),
)

# 6. Instantiate model
model = deepl.ACCNet(window_size=WINDOW_SIZE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ACCNet trainable parameters: {n_params:,}")

# 7. Instantiate trainer and train
trainer = deepl.ACCTrainer(
    model        = model,
    train_loader = train_loader,
    val_loader   = val_loader,
    device       = device,
    lr           = LR,
)

print(f"\nStarting training for {EPOCHS} epochs...\n")
trainer.fit(num_epochs=EPOCHS)

# 8. Save ONNX model
trainer.save_onnx(
    os.path.join(SCRIPT_DIR, 'acc_model.onnx'),
    window_size=WINDOW_SIZE
)

# 9. Plot loss and accuracy vs epochs
epochs_axis = list(range(1, EPOCHS + 1))
fig, axes   = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(epochs_axis, trainer.history['train_loss'],
             label='Train Loss', color='tab:red')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Focal Tversky Loss')
axes[0].set_title('Training Loss vs Epochs')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_axis, [a * 100 for a in trainer.history['train_acc']],
             label='Train Accuracy', color='tab:blue')
axes[1].plot(epochs_axis, [a * 100 for a in trainer.history['val_acc']],
             label='Val Accuracy', color='tab:orange', linestyle='--')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Balanced Accuracy (%)')
axes[1].set_title('Accuracy vs Epochs')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'acc_training_plot.png'), dpi=150)
plt.close()
print("Training plot saved.")