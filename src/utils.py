import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
from sklearn.metrics import classification_report

def save_hyperparameters(args, save_path):
    """
    Saves the training configuration (Hyperparameters) to a JSON file.
    Useful for reproducibility.
    """
    with open(save_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f" [Log] Hyperparameters saved to {save_path}")

def save_classification_report(y_true, y_pred, class_names, save_path):
    """
    Generates and saves a detailed text report of Precision, Recall, F1 for each class.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f" [Log] Classification Report saved to {save_path}")

def plot_comprehensive_metrics(history, cm, class_names, filename):
    """
    Plots:
    1. Train vs Val Loss
    2. Validation Accuracy & F1-Score over epochs
    3. Confusion Matrix
    """
    fig = plt.figure(figsize=(18, 5))
    
    # 1. Loss Curve
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['val_loss'], label='Val Loss', marker='o', markersize=3)
    ax1.set_title('Learning Curve (Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Metrics Curve (Accuracy & F1)
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(history['val_acc'], label='Val Accuracy', color='green', marker='s', markersize=3)
    ax2.plot(history['val_f1'], label='Val F1-Score', color='purple', linestyle='--', marker='s', markersize=3)
    ax2.set_title('Validation Performance')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Confusion Matrix
    ax3 = plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Final Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=300) # High DPI for papers/reports
    plt.close()
    print(f" [Log] Graphs saved to {filename}")