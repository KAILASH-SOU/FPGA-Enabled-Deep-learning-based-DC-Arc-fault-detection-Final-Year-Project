import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
)
from models import ArcFaultCNN_QAT

def evaluate_model_comprehensive(model, dataloader, device, criterion=None):
    if criterion is None:
        criterion = nn.BCELoss()

    model.eval()
    all_preds_prob = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            total_loss += loss.item()
            num_batches += 1

            all_preds_prob.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    total_eval_time = time.time() - start_time
    total_samples = len(all_labels)
    latency_ms_per_sample = (total_eval_time / total_samples) * 1000.0

    all_preds_prob = np.array(all_preds_prob)
    all_labels = np.array(all_labels)
    all_preds_binary = (all_preds_prob >= 0.5).astype(int)

    cm = confusion_matrix(all_labels, all_preds_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        'loss': float(total_loss / max(num_batches, 1)),
        'accuracy': float(accuracy_score(all_labels, all_preds_binary)),
        'precision': float(precision_score(all_labels, all_preds_binary, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds_binary, zero_division=0)),
        'specificity': float(specificity),
        'f1': float(f1_score(all_labels, all_preds_binary, zero_division=0)),
        'false_alarm_rate': float(false_alarm_rate),
        'roc_auc': float(roc_auc_score(all_labels, all_preds_prob)) if len(np.unique(all_labels)) > 1 else 0.0,
        'pr_auc': float(average_precision_score(all_labels, all_preds_prob)) if len(np.unique(all_labels)) > 1 else 0.0,
        'latency_ms': float(latency_ms_per_sample),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

    return metrics, all_preds_prob, all_preds_binary, all_labels

def apply_post_training_quantization(fp32_model, calibration_loader, device):
    print("\n[PTQ] Applying Post-Training Quantization (INT8)...")
    ptq_model = ArcFaultCNN_QAT(fp32_model=fp32_model).to(device)

    ptq_model.train()
    with torch.no_grad():
        for idx, (inputs, _) in enumerate(calibration_loader):
            inputs = inputs.to(device)
            _ = ptq_model(inputs)
            if idx >= 20:
                break

    ptq_model.eval()
    print("[PTQ] Calibration complete.")
    return ptq_model

def train_qat_model(fp32_model, train_loader, val_loader, device, epochs=5, lr=0.0008):
    print(f"\n[QAT] Training Quantization-Aware Model (INT8, {epochs} epochs)...")
    qat_model = ArcFaultCNN_QAT(fp32_model=fp32_model).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_qat_weights = None

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(epochs):
        qat_model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = qat_model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        scheduler.step()

        val_metrics, _, _, _ = evaluate_model_comprehensive(qat_model, val_loader, device, criterion)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        print(f"QAT Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_qat_weights = copy.deepcopy(qat_model.state_dict())

    if best_qat_weights is not None:
        qat_model.load_state_dict(best_qat_weights)

    return qat_model, history
