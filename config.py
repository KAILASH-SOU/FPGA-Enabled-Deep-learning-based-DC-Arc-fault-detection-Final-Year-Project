import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EXPORT_DIR = os.path.join(RESULTS_DIR, "hardware_export")

for d in [RESULTS_DIR, PLOTS_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_PATH = os.path.join(BASE_DIR, "arc_dataset_synthetic.mat")
SAMPLE_RATE = 8000
GRID_FREQ = 50
WINDOW_LEN = 160
NUM_FFT_FEATURES = 160
NUM_TIME_FEATURES = 160
TOTAL_FEATURES = 320
DEFAULT_NUM_SAMPLES = 100000

RANDOM_SEED = 42
BATCH_SIZE = 64
DEFAULT_EPOCHS = 12
QAT_EPOCHS = 5
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

FPGA_CONFIG = {
    "target_chip": "Xilinx Artix-7 (XC7A100T-1CSG324C)",
    "clock_freq_mhz": 100.0,
    "bit_width": 8,
    "weight_precision": "INT8 (-128 to 127)",
    "activation_precision": "INT8 (-128 to 127) / UINT8 (0 to 255)",
    "dsp_slice_type": "DSP48E1 (25x18 multiplier)",
    "bram_size_kb": 36,
    "available_dsps": 240,
    "available_brams": 135
}
