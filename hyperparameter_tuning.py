import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, PLOTS_DIR, get_device
from models import ArcFaultCNN_FPGA
from quantization_engine import evaluate_model_comprehensive

def run_hyperparameter_tuning(train_loader, val_loader, device=None, max_trials=6):
    if device is None:
        device = get_device()

    print("\n[TUNING] Starting Hyperparameter Search...")

    trial_configs = [
        {"id": "Trial_1", "lr": 0.001, "weight_decay": 1e-4, "dropout": 0.1, "opt": "Adam", "epochs": 4},
        {"id": "Trial_2", "lr": 0.003, "weight_decay": 1e-4, "dropout": 0.2, "opt": "Adam", "epochs": 4},
        {"id": "Trial_3", "lr": 0.005, "weight_decay": 1e-5, "dropout": 0.2, "opt": "AdamW", "epochs": 4},
        {"id": "Trial_4", "lr": 0.008, "weight_decay": 1e-4, "dropout": 0.3, "opt": "Adam", "epochs": 4},
        {"id": "Trial_5", "lr": 0.003, "weight_decay": 1e-5, "dropout": 0.3, "opt": "AdamW", "epochs": 4},
        {"id": "Trial_6", "lr": 0.002, "weight_decay": 5e-5, "dropout": 0.15, "opt": "Adam", "epochs": 4},
    ][:max_trials]

    results = []

    for cfg in trial_configs:
        print(f"Running {cfg['id']}: LR={cfg['lr']}, WD={cfg['weight_decay']}, Dropout={cfg['dropout']}, Opt={cfg['opt']}")
        model = ArcFaultCNN_FPGA(dropout_rate=cfg['dropout']).to(device)
        criterion = nn.BCELoss()

        if cfg['opt'] == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

        start_time = time.time()
        for epoch in range(cfg['epochs']):
            model.train()
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

        elapsed = time.time() - start_time
        val_metrics, _, _, _ = evaluate_model_comprehensive(model, val_loader, device, criterion)

        trial_record = {
            "Trial": cfg['id'],
            "Learning_Rate": cfg['lr'],
            "Weight_Decay": cfg['weight_decay'],
            "Dropout": cfg['dropout'],
            "Optimizer": cfg['opt'],
            "Val_Accuracy": round(val_metrics['accuracy'] * 100, 2),
            "Val_F1_Score": round(val_metrics['f1'], 4),
            "Val_Loss": round(val_metrics['loss'], 4),
            "False_Alarm_Rate": round(val_metrics['false_alarm_rate'] * 100, 2),
            "Train_Time_Sec": round(elapsed, 2)
        }
        results.append(trial_record)

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "hyperparameter_tuning_results.csv")
    df_results.to_csv(csv_path, index=False)

    plot_tuning_results(df_results)

    best_trial = df_results.sort_values(by="Val_F1_Score", ascending=False).iloc[0]
    print(f"[TUNING] Selected Optimal: {best_trial['Trial']} (Val F1: {best_trial['Val_F1_Score']})\n")

    best_config = {
        "lr": float(best_trial['Learning_Rate']),
        "weight_decay": float(best_trial['Weight_Decay']),
        "dropout": float(best_trial['Dropout']),
        "optimizer": str(best_trial['Optimizer'])
    }
    return best_config, df_results

def plot_tuning_results(df_results):
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    plt.subplot(1, 2, 1)
    bars1 = plt.bar(df_results['Trial'], df_results['Val_Accuracy'], color='#2b5c8f')
    plt.title('Validation Accuracy across Trials (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim([min(df_results['Val_Accuracy']) - 5, 100])
    plt.xticks(rotation=30)
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.subplot(1, 2, 2)
    bars2 = plt.bar(df_results['Trial'], df_results['Val_F1_Score'], color='#2e8b57')
    plt.title('Validation F1-Score across Trials', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score')
    plt.ylim([min(df_results['Val_F1_Score']) - 0.05, 1.0])
    plt.xticks(rotation=30)
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "07_hyperparameter_tuning.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
