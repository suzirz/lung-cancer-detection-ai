"""Generate training curve visualization from training log data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Training data from Colab output (group-aware splits, 15 epochs)
epochs = list(range(1, 16))

train_loss = [0.2393, 0.0651, 0.0351, 0.0240, 0.0258, 0.0259, 0.0134, 0.0229, 0.0218, 0.0088, 0.0181, 0.0225, 0.0124, 0.0083, 0.0057]
train_acc  = [90.07, 97.62, 98.80, 99.28, 99.15, 99.08, 99.53, 99.33, 99.20, 99.67, 99.46, 99.39, 99.63, 99.76, 99.80]
val_loss   = [0.0410, 0.0618, 0.0057, 0.0030, 0.0590, 0.0062, 0.0012, 0.0025, 0.0028, 0.0104, 0.0032, 0.0027, 0.0083, 0.0010, 0.0004]
val_acc    = [98.61, 96.66, 99.89, 99.94, 98.22, 99.78, 99.94, 100.00, 100.00, 99.56, 99.89, 99.94, 99.56, 100.00, 100.00]

# Style
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

TRAIN_COLOR = '#2196F3'
VAL_COLOR = '#FF5722'

# --- Loss Curve ---
ax1.plot(epochs, train_loss, 'o-', color=TRAIN_COLOR, linewidth=2, markersize=5, label='Training Loss', alpha=0.9)
ax1.plot(epochs, val_loss, 's-', color=VAL_COLOR, linewidth=2, markersize=5, label='Validation Loss', alpha=0.9)
ax1.fill_between(epochs, train_loss, alpha=0.1, color=TRAIN_COLOR)
ax1.fill_between(epochs, val_loss, alpha=0.1, color=VAL_COLOR)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12, fontweight='bold')
ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.set_xticks(epochs)
ax1.set_ylim(bottom=0)

ax1.annotate(f'{train_loss[-1]:.4f}', xy=(15, train_loss[-1]),
             xytext=(13, train_loss[-1] + 0.015), fontsize=9, color=TRAIN_COLOR, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=TRAIN_COLOR, lw=1.2))
ax1.annotate(f'{val_loss[-1]:.4f}', xy=(15, val_loss[-1]),
             xytext=(13, val_loss[-1] + 0.008), fontsize=9, color=VAL_COLOR, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=VAL_COLOR, lw=1.2))

# --- Accuracy Curve ---
ax2.plot(epochs, train_acc, 'o-', color=TRAIN_COLOR, linewidth=2, markersize=5, label='Training Accuracy', alpha=0.9)
ax2.plot(epochs, val_acc, 's-', color=VAL_COLOR, linewidth=2, markersize=5, label='Validation Accuracy', alpha=0.9)
ax2.fill_between(epochs, train_acc, alpha=0.1, color=TRAIN_COLOR)
ax2.fill_between(epochs, val_acc, alpha=0.1, color=VAL_COLOR)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='lower right')
ax2.set_xticks(epochs)
ax2.set_ylim(88, 101)
ax2.axhline(y=98.48, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(1.5, 98.0, 'Test Acc: 98.48%', fontsize=9, color='gray', fontstyle='italic')

ax2.annotate(f'{train_acc[-1]:.1f}%', xy=(15, train_acc[-1]),
             xytext=(13, train_acc[-1] - 1.5), fontsize=9, color=TRAIN_COLOR, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=TRAIN_COLOR, lw=1.2))
ax2.annotate(f'{val_acc[-1]:.1f}%', xy=(15, val_acc[-1]),
             xytext=(13, val_acc[-1] + 0.5), fontsize=9, color=VAL_COLOR, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=VAL_COLOR, lw=1.2))

fig.suptitle('EfficientNet-B0 Training Curves (Group-Aware Splits, Tesla T4 FP16)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

os.makedirs('reports', exist_ok=True)
fig.savefig('reports/training_curves.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('[OK] Training curves saved to reports/training_curves.png')
plt.close()
