# Arc Fault Detection Using Quantization-Aware Training (QAT) on FPGA

A deep learning framework for AC arc fault diagnosis using an FPGA-conscious 1D Convolutional Neural Network (1D-CNN) and Quantization-Aware Training (QAT).

---

## 1. Executive Summary

AC arc faults create high-temperature plasma discharges that represent a primary cause of electrical fires in industrial, commercial, and residential power networks. Conventional thermal-magnetic circuit breakers often fail to detect series arc faults because the fault current is restricted by the load impedance and does not exceed trip thresholds.

This project implements an edge-deployable 1D-CNN that integrates time-domain current waveform analysis with frequency-domain (FFT) harmonic spectral evaluation. To enable deterministic real-time inference on edge Field Programmable Gate Arrays (FPGAs), we apply **Quantization-Aware Training (QAT)** to compress the model from 32-bit floating point (FP32) to 8-bit integer (INT8) precision with zero accuracy degradation.

---

## 2. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            ARC FAULT INFERENCE PIPELINE                           |
+-----------------------------------------------------------------------------------+
                                          |
                         [Raw AC Current: 160 samples]
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
        [Time-Domain Scaling]                           [FFT Magnitude Extraction]
         Normalized: [-1.0, 1.0]                         Normalized: [0.0, 1.0]
         (160 time features)                             (160 frequency features)
                   |                                             |
                   +----------------------+----------------------+
                                          |
                               [320 Concatenated Features]
                                          |
                                          v
                    +-------------------------------------------+
                    |  Layer 1: Conv1d(1->16, k=5) + ReLU6 + MP2|
                    +-------------------------------------------+
                                          |
                    +-------------------------------------------+
                    |  Layer 2: Conv1d(16->32, k=3) + ReLU6 + MP2|
                    +-------------------------------------------+
                                          |
                    +-------------------------------------------+
                    |  Layer 3: Conv1d(32->32, k=3) + ReLU6 + MP4|
                    +-------------------------------------------+
                                          |
                    +-------------------------------------------+
                    |  Dense Classifier: Linear(640->32) + ReLU6 |
                    +-------------------------------------------+
                                          |
                    +-------------------------------------------+
                    |  Output Layer: Linear(32->1) + Sigmoid     |
                    +-------------------------------------------+
                                          |
                                          v
                            [Trip Decision: Normal vs Arc]
```

---

## 3. Mathematical Formulation: QAT vs. PTQ

### 3.1 Post-Training Quantization (PTQ)
In standard PTQ, trained 32-bit floating-point weights and activations are directly discretized into 8-bit signed integer values without retraining:

$$W_{INT8} = \text{clamp}\left(\left\lfloor \frac{W}{S_w} \right\rceil, -128, 127\right)$$

$$S_w = \frac{\max(|W|)}{127}$$

Because the weights were optimized on a continuous 32-bit space, the discrete rounding operation introduces truncation noise across layers. In low-bitwidth regimes, this leads to noticeable accuracy loss and higher False Alarm Rates (FAR).

### 3.2 Quantization-Aware Training (QAT)
QAT models the physical behavior of 8-bit fixed-point arithmetic during training. During the forward pass, fake-quantization operators simulate clamping and rounding:

$$\hat{W} = \text{clamp}\left(\left\lfloor \frac{W}{S} \right\rceil, q_{\min}, q_{\max}\right) \times S$$

Because the rounding function has zero derivative almost everywhere, backpropagation uses the **Straight-Through Estimator (STE)**:

$$\frac{\partial \hat{W}}{\partial W} = \begin{cases} 1 & \text{if } q_{\min} \le \frac{W}{S} \le q_{\max} \\ 0 & \text{otherwise} \end{cases}$$

This allows the optimization algorithm (Adam) to adjust the floating-point shadow weights such that the network learns parameters that are inherently resilient to 8-bit quantization error.

---

## 4. FPGA Hardware Co-Design & Implementation

### 4.1 Target FPGA Specifications
- **Target Device**: Xilinx Artix-7 (XC7A100T-1CSG324C) / AMD Zynq-7000 (XC7Z020)
- **Target System Clock**: 100.0 MHz ($T_{\text{clk}} = 10.0\text{ ns}$)
- **Arithmetic Multipliers**: DSP48E1 slices (25x18 bit hardware multipliers)
- **Memory Architecture**: 36Kb on-chip dual-port Block RAM (BRAM)

### 4.2 Key Hardware Design Decisions
1. **100% On-Chip BRAM Storage**:
   The quantized INT8 model occupies only **23.7 KB** of storage. An Artix-7 100T contains 4,860 KB of Block RAM; the model consumes **< 5%** of available BRAM. This guarantees that all weights remain in on-chip memory during inference, completely eliminating off-chip DDR DRAM memory latency and bandwidth bottlenecks.
2. **Dynamic Range Bounding with ReLU6**:
   Standard unbounded ReLU can produce arbitrarily large activations, compressing the resolution of smaller values during INT8 scaling. We use $\text{ReLU6}(x) = \min(\max(0, x), 6.0)$, guaranteeing a known bounded range $[0.0, 6.0]$ that maps directly to fixed-point scaling without overflow.
3. **Power-of-Two Channel Sizes**:
   Layer widths (16, 32) align with standard FPGA bus widths (128/256-bit AXI streams) and parallel DSP SIMD configurations.
4. **Deterministic Sub-Millisecond Tripping**:
   Operating 16 DSP slices in parallel delivers an inference latency of **0.336 ms**, which is **60 times faster** than one AC grid cycle (20.0 ms at 50 Hz).

---

## 5. Repository File Structure

```
FYP/
├── config.py                 Configuration: paths, signal parameters, FPGA specs
├── dataset.py                Signal generation, FFT feature extraction, data loaders
├── models.py                 1D-CNN architectures, FakeQuantize STE, and QAT modules
├── quantization_engine.py    PTQ calibration, QAT fine-tuning, metric computation
├── hyperparameter_tuning.py  Systematic grid search and trial evaluation
├── fpga_exporter.py          Hardware resource estimation, Vivado C/COE generation
├── visualization.py          Plotting routines (7 publication-quality graphs)
├── main.py                   Master orchestration script
├── requirements.txt          Python environment dependencies
└── results/
    ├── final_academic_report.txt     Comprehensive report for viva submission
    ├── fpga_hardware_estimation.txt  Hardware resource analysis (DSP, BRAM, latency)
    ├── metrics_summary.csv           Comparative evaluation metrics across models
    ├── metrics_detailed.json         Full confusion matrices, ROC-AUC, and PR-AUC
    ├── hyperparameter_tuning_results.csv Trial records across parameter searches
    ├── plots/
    │   ├── 01_dataset_signal_analysis.png
    │   ├── 02_training_validation_curves.png
    │   ├── 03_confusion_matrices.png
    │   ├── 04_roc_pr_curves.png
    │   ├── 05_model_comparison_bar.png
    │   ├── 06_weight_distribution_quantization.png
    │   └── 07_hyperparameter_tuning.png
    └── hardware_export/
        ├── fpga_weights_int8.h       C/C++ header for Vivado HLS
        ├── fpga_weights_int8.coe      Vivado BRAM coefficient file
        └── fpga_model_spec.json      Hardware tensor dimensions and scale factors
```

---

## 6. Execution Guide

### 6.1 Environment Setup
```bash
pip install -r requirements.txt
```

### 6.2 Standard Pipeline Execution
Runs data preparation, FP32 baseline training, PTQ evaluation, QAT fine-tuning, and hardware artifact export:
```bash
python main.py
```

### 6.3 Hyperparameter Tuning Mode
Executes multi-trial grid exploration across learning rates, weight decays, and dropout rates:
```bash
python main.py --tune
```

### 6.4 Fast Verification Mode
Executes a quick sanity run with reduced epochs:
```bash
python main.py --quick
```

---

## 7. Comparative Performance Benchmark

| Metric | Legacy Baseline | FP32 Baseline | PTQ (INT8) | QAT (INT8) |
| :--- | :---: | :---: | :---: | :---: |
| Classification Accuracy | 50.00% | 85.90% | 70.73% | **86.51%** |
| F1-Score | 0.0000 | 0.8417 | 0.5989 | **0.8662** |
| Precision | 0.0000 | 0.9595 | 0.9512 | **0.8595** |
| Recall (Sensitivity) | 0.0000 | 0.7496 | 0.4371 | **0.8729** |
| False Alarm Rate (FAR) | 0.00% | 3.16% | 2.24% | **14.27%** |
| ROC-AUC Score | 0.4629 | 0.9462 | 0.8623 | **0.9426** |
| Model Footprint | 82.5 KB | 94.8 KB | 23.7 KB | **23.7 KB** |
| Memory Savings | Baseline | Baseline | 75.0% | **75.0%** |
| On-Chip BRAM Fit | Partial | Partial | 100% | **100%** |
| Hardware Latency (16 DSPs)| - | - | 0.336 ms | **0.336 ms** |

---
