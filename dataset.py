import os
import scipy.io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from config import (
    DATASET_PATH, SAMPLE_RATE, GRID_FREQ, WINDOW_LEN,
    TOTAL_FEATURES, RANDOM_SEED, BATCH_SIZE, DEFAULT_NUM_SAMPLES
)

def generate_synthetic_arc_dataset(filepath=DATASET_PATH, num_samples=DEFAULT_NUM_SAMPLES):
    print(f"Generating dataset ({num_samples} samples)...")
    np.random.seed(RANDOM_SEED)

    num_features = WINDOW_LEN
    t = np.linspace(0, (WINDOW_LEN - 1) / SAMPLE_RATE, num_features)

    Y_data = np.random.randint(0, 2, size=num_samples).astype(np.int64)

    fundamental = 100.0 * np.sin(2 * np.pi * GRID_FREQ * t)
    third_harmonic = 5.0 * np.sin(2 * np.pi * 3 * GRID_FREQ * t)
    base_current = fundamental + third_harmonic

    X_data = np.tile(base_current, (num_samples, 1))
    X_data += np.random.normal(0, 3.0, (num_samples, num_features))

    arc_indices = np.where(Y_data == 1)[0]
    num_arcs = len(arc_indices)

    if num_arcs > 0:
        shoulder_threshold = 20.0
        arc_signals = X_data[arc_indices]
        shoulder_mask = np.abs(arc_signals) < shoulder_threshold
        arc_signals[shoulder_mask] = arc_signals[shoulder_mask] * 0.15

        hf_bursts = np.random.normal(0, 25.0, (num_arcs, num_features))
        dc_offsets = np.random.uniform(-15.0, 15.0, (num_arcs, 1))
        arc_signals += hf_bursts + dc_offsets
        X_data[arc_indices] = arc_signals

    scipy.io.savemat(filepath, {'X_data': X_data, 'Y_data': Y_data})
    print(f"Dataset saved to: {filepath}\n")
    return X_data, Y_data

def load_or_create_dataset(filepath=DATASET_PATH, num_samples=DEFAULT_NUM_SAMPLES):
    if os.path.exists(filepath):
        print(f"Loading dataset from: {filepath}")
        mat_data = scipy.io.loadmat(filepath)
        X_data = mat_data['X_data']
        Y_data = mat_data['Y_data'].ravel().astype(np.int64)
        return X_data, Y_data
    return generate_synthetic_arc_dataset(filepath=filepath, num_samples=num_samples)

def preprocess_features(X_raw):
    max_current = np.percentile(np.abs(X_raw), 99.9)
    if max_current < 1e-6:
        max_current = 1.0
    X_time = np.clip(X_raw / max_current, -1.0, 1.0)

    X_fft_raw = np.abs(np.fft.fft(X_raw, axis=1))
    X_fft_raw[:, 0] = X_fft_raw[:, 0] * 0.1
    max_fft = np.percentile(X_fft_raw, 99.9)
    if max_fft < 1e-6:
        max_fft = 1.0
    X_fft = np.clip(X_fft_raw / max_fft, 0.0, 1.0)

    X_processed = np.concatenate((X_fft, X_time), axis=1).astype(np.float32)
    return X_processed, float(max_current), float(max_fft)

def prepare_dataloaders(filepath=DATASET_PATH, batch_size=BATCH_SIZE, num_samples=DEFAULT_NUM_SAMPLES):
    X_raw, Y_raw = load_or_create_dataset(filepath, num_samples=num_samples)
    X_processed, max_current, max_fft = preprocess_features(X_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, Y_raw, test_size=0.30, random_state=RANDOM_SEED, stratify=Y_raw
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        'max_current': max_current,
        'max_fft': max_fft,
        'X_raw': X_raw,
        'Y_raw': Y_raw,
        'X_test': X_test,
        'y_test': y_test
    }
    return train_loader, val_loader, test_loader, metadata
