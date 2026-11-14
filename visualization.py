import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from config import PLOTS_DIR

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

def plot_dataset_analysis(X_raw, Y_raw, output_dir=PLOTS_DIR):
    normal_idx = np.where(Y_raw == 0)[0][0]
    arc_idx = np.where(Y_raw == 1)[0][0]

    normal_signal = X_raw[normal_idx]
    arc_signal = X_raw[arc_idx]

    time_ms = np.linspace(0, 20, len(normal_signal))
    freqs = np.fft.rfftfreq(len(normal_signal), d=20e-3 / 160)
    normal_fft = np.abs(np.fft.rfft(normal_signal))
    arc_fft = np.abs(np.fft.rfft(arc_signal))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(time_ms, normal_signal, label='Normal (50Hz Fundamental)', color='#1f77b4', linewidth=2)
    axes[0].plot(time_ms, arc_signal, label='Arc Fault (Shoulders + Noise)', color='#d62728', alpha=0.85, linewidth=1.5)
    axes[0].set_title('AC Current Waveform: Normal vs. Arc Fault', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Time (ms)', fontsize=11)
    axes[0].set_ylabel('Current Amplitude (A)', fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)

    axes[1].plot(freqs, normal_fft, label='Normal Spectrum', color='#1f77b4', linewidth=2)
    axes[1].plot(freqs, arc_fft, label='Arc Fault Spectrum', color='#d62728', alpha=0.85, linewidth=1.5)
    axes[1].set_title('Frequency Spectrum (FFT Analysis)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1].set_ylabel('Magnitude', fontsize=11)
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_dataset_signal_analysis.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_training_curves(history, output_dir=PLOTS_DIR):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Loss Convergence Curve', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=11)
    axes[0].legend(frameon=True)

    axes[1].plot(epochs, [acc * 100 for acc in history['train_acc']], 'o-', label='Train Acc', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, [acc * 100 for acc in history['val_acc']], 's--', label='Val Acc', color='#9467bd', linewidth=2)
    axes[1].set_title('Accuracy Convergence Curve', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, "02_training_validation_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_confusion_matrices(metrics_dict, output_dir=PLOTS_DIR):
    models = ['FP32 Baseline', 'PTQ (INT8)', 'QAT (INT8)']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, model_name in enumerate(models):
        if model_name not in metrics_dict:
            continue
        cm = np.array(metrics_dict[model_name]['confusion_matrix'])
        acc = metrics_dict[model_name]['accuracy'] * 100
        f1 = metrics_dict[model_name]['f1']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                    annot_kws={'size': 14, 'weight': 'bold'},
                    xticklabels=['Normal (0)', 'Arc (1)'],
                    yticklabels=['Normal (0)', 'Arc (1)'])
        axes[idx].set_title(f"{model_name}\nAcc: {acc:.2f}% | F1: {f1:.4f}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=11)
        axes[idx].set_ylabel('True Label' if idx == 0 else '', fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "03_confusion_matrices.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_roc_pr_curves(eval_data_dict, output_dir=PLOTS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'FP32 Baseline': '#1f77b4', 'PTQ (INT8)': '#d62728', 'QAT (INT8)': '#2ca02c'}

    for name, data in eval_data_dict.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        auc = data['metrics']['roc_auc']
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=colors.get(name, '#333333'), linewidth=2)

    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate', fontsize=11)
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].legend(loc='lower right', frameon=True)

    for name, data in eval_data_dict.items():
        prec, rec, _ = precision_recall_curve(data['y_true'], data['y_prob'])
        pr_auc = data['metrics']['pr_auc']
        axes[1].plot(rec, prec, label=f"{name} (AP={pr_auc:.4f})", color=colors.get(name, '#333333'), linewidth=2)

    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].legend(loc='lower left', frameon=True)

    plt.tight_layout()
    path = os.path.join(output_dir, "04_roc_pr_curves.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_model_comparison_bar(summary_df, output_dir=PLOTS_DIR):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(summary_df))
    width = 0.35
    axes[0].bar(x - width/2, summary_df['Accuracy (%)'], width, label='Accuracy (%)', color='#2b5c8f')
    axes[0].bar(x + width/2, summary_df['F1-Score'] * 100, width, label='F1 (x100)', color='#388e3c')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df['Model'])
    axes[0].set_title('Accuracy and F1-Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=11)
    axes[0].set_ylim([min(summary_df['Accuracy (%)']) - 5, 102])
    axes[0].legend(frameon=True)

    bars2 = axes[1].bar(summary_df['Model'], summary_df['Model Size (KB)'], color=['#1976d2', '#e64a19', '#388e3c'])
    axes[1].set_title('Memory Footprint (KB in BRAM)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Kilobytes (KB)', fontsize=11)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f} KB", ha='center', va='bottom')

    bars3 = axes[2].bar(summary_df['Model'], summary_df['False Alarm Rate (%)'], color='#d32f2f')
    axes[2].set_title('False Alarm Rate (Lower is Better)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('FAR (%)', fontsize=11)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}%", ha='center', va='bottom')

    plt.tight_layout()
    path = os.path.join(output_dir, "05_model_comparison_bar.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def plot_weight_distributions(fp32_model, qat_model, output_dir=PLOTS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fp32_weights = []
    for p in fp32_model.parameters():
        fp32_weights.extend(p.detach().cpu().numpy().flatten())
    fp32_weights = np.array(fp32_weights)

    int8_weights = []
    for name, p in qat_model.named_parameters():
        if "weight" in name and "bn" not in name:
            data = p.detach().cpu().numpy().flatten()
            scale = np.max(np.abs(data)) / 127.0 if np.max(np.abs(data)) > 0 else 1.0
            q_data = np.clip(np.round(data / scale), -128, 127)
            int8_weights.extend(q_data.tolist())
    int8_weights = np.array(int8_weights)

    axes[0].hist(fp32_weights, bins=80, color='#1976d2', edgecolor='black', alpha=0.7)
    axes[0].set_title('FP32 Continuous Weight Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Weight Value', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)

    axes[1].hist(int8_weights, bins=64, color='#388e3c', edgecolor='black', alpha=0.7)
    axes[1].set_title('QAT INT8 Quantized Weight Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Quantized Level (-128 to +127)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, "06_weight_distribution_quantization.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path
