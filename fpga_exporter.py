import os
import json
import numpy as np
import torch
from config import EXPORT_DIR, RESULTS_DIR, FPGA_CONFIG

def analyze_fpga_resources(model, input_shape=(1, 1, 320)):
    total_params = 0
    layer_breakdown = []
    total_macs = 0

    conv1_w = 1 * 16 * 5
    conv1_b = 16
    conv1_params = conv1_w + conv1_b
    conv1_macs = 1 * 16 * 5 * 320
    total_params += conv1_params
    total_macs += conv1_macs
    layer_breakdown.append(("conv1 (Conv1d 1->16, k=5)", conv1_params, conv1_macs, "16x160"))

    conv2_w = 16 * 32 * 3
    conv2_b = 32
    conv2_params = conv2_w + conv2_b
    conv2_macs = 16 * 32 * 3 * 160
    total_params += conv2_params
    total_macs += conv2_macs
    layer_breakdown.append(("conv2 (Conv1d 16->32, k=3)", conv2_params, conv2_macs, "32x80"))

    conv3_w = 32 * 32 * 3
    conv3_b = 32
    conv3_params = conv3_w + conv3_b
    conv3_macs = 32 * 32 * 3 * 80
    total_params += conv3_params
    total_macs += conv3_macs
    layer_breakdown.append(("conv3 (Conv1d 32->32, k=3)", conv3_params, conv3_macs, "32x20"))

    fc1_w = (32 * 20) * 32
    fc1_b = 32
    fc1_params = fc1_w + fc1_b
    fc1_macs = (32 * 20) * 32
    total_params += fc1_params
    total_macs += fc1_macs
    layer_breakdown.append(("fc1 (Linear 640->32)", fc1_params, fc1_macs, "32"))

    fc2_w = 32 * 1
    fc2_b = 1
    fc2_params = fc2_w + fc2_b
    fc2_macs = 32 * 1
    total_params += fc2_params
    total_macs += fc2_macs
    layer_breakdown.append(("fc2 (Linear 32->1)", fc2_params, fc2_macs, "1"))

    fp32_bytes = total_params * 4
    int8_bytes = total_params * 1
    fp32_kb = fp32_bytes / 1024.0
    int8_kb = int8_bytes / 1024.0
    compression_ratio = fp32_bytes / int8_bytes
    bram_needed = int(np.ceil(int8_kb / 4.5))

    clk_period_ns = 1000.0 / FPGA_CONFIG["clock_freq_mhz"]
    latency_1dsp_ms = (total_macs * clk_period_ns) / 1e6
    latency_16dsp_ms = ((total_macs / 16) * clk_period_ns) / 1e6
    latency_32dsp_ms = ((total_macs / 32) * clk_period_ns) / 1e6

    return {
        "total_parameters": total_params,
        "total_macs": total_macs,
        "fp32_size_kb": fp32_kb,
        "int8_size_kb": int8_kb,
        "compression_ratio": f"{compression_ratio:.1f}x (75% savings)",
        "bram_36kb_tiles": bram_needed,
        "bram_utilization_pct": (bram_needed / FPGA_CONFIG["available_brams"]) * 100,
        "latency_1dsp_ms": latency_1dsp_ms,
        "latency_16dsp_ms": latency_16dsp_ms,
        "latency_32dsp_ms": latency_32dsp_ms,
        "layer_breakdown": layer_breakdown
    }

def generate_fpga_hardware_report(summary, output_dir=RESULTS_DIR):
    report_path = os.path.join(output_dir, "fpga_hardware_estimation.txt")

    with open(report_path, "w") as f:
        f.write("========================================================================\n")
        f.write("          FPGA HARDWARE RESOURCE & IMPLEMENTATION ESTIMATION           \n")
        f.write("        Final Year B.Tech Project: Arc Fault Detection with QAT         \n")
        f.write("========================================================================\n\n")
        f.write(f"Target FPGA Chip:          {FPGA_CONFIG['target_chip']}\n")
        f.write(f"System Clock Frequency:    {FPGA_CONFIG['clock_freq_mhz']} MHz (Period = 10.0 ns)\n")
        f.write(f"Arithmetic Precision:      {FPGA_CONFIG['weight_precision']} (Weights)\n")
        f.write(f"Accumulator Width:         32-bit integer\n\n")

        f.write("1. LAYER-BY-LAYER COMPUTATIONAL COMPLEXITY:\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(f"{'Layer Name':<28} | {'Params':<8} | {'MAC Operations':<14} | {'Output Dim'}\n")
        f.write("------------------------------------------------------------------------\n")
        for name, params, macs, out_dim in summary["layer_breakdown"]:
            f.write(f"{name:<28} | {params:<8} | {macs:<14} | {out_dim}\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(f"{'TOTAL':<28} | {summary['total_parameters']:<8} | {summary['total_macs']:<14} |\n\n")

        f.write("2. MEMORY AND STORAGE FOOTPRINT COMPARISON:\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(f"- Baseline FP32 Model Size:      {summary['fp32_size_kb']:.2f} KB\n")
        f.write(f"- Quantized INT8 Model Size:     {summary['int8_size_kb']:.2f} KB\n")
        f.write(f"- On-Chip Memory Reduction:      {summary['compression_ratio']}\n")
        f.write(f"- Required 36Kb BRAM Blocks:     {summary['bram_36kb_tiles']} tiles\n")
        f.write(f"- Target BRAM Utilization:       {summary['bram_utilization_pct']:.2f}% of Artix-7 100T\n")
        f.write("  Note: Model fits entirely in on-chip Block RAM without external DRAM traffic.\n\n")

        f.write("3. ESTIMATED INFERENCE LATENCY & THROUGHPUT:\n")
        f.write("------------------------------------------------------------------------\n")
        f.write(f"- 1 DSP Slice (Sequential):      {summary['latency_1dsp_ms']:.3f} ms / frame  ({1000/summary['latency_1dsp_ms']:.1f} frames/sec)\n")
        f.write(f"- 16 DSP Slices (Parallel):      {summary['latency_16dsp_ms']:.3f} ms / frame ({1000/summary['latency_16dsp_ms']:.1f} frames/sec)\n")
        f.write(f"- 32 DSP Slices (Pipelined):     {summary['latency_32dsp_ms']:.3f} ms / frame ({1000/summary['latency_32dsp_ms']:.1f} frames/sec)\n")
        f.write("========================================================================\n")

    return report_path

def export_model_for_fpga(model, output_dir=EXPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "fpga_weights_int8.h")
    coe_path = os.path.join(output_dir, "fpga_weights_int8.coe")
    json_path = os.path.join(output_dir, "fpga_model_spec.json")

    all_int8_weights = []
    layer_meta = {}

    with open(header_path, "w") as f:
        f.write("/*\n")
        f.write(" * FPGA INT8 Weights and Biases Header File\n")
        f.write(" * Generated for Vivado HLS / C++ Embedded Implementation\n")
        f.write(" */\n\n")
        f.write("#ifndef FPGA_WEIGHTS_INT8_H\n")
        f.write("#define FPGA_WEIGHTS_INT8_H\n\n")
        f.write("#include <stdint.h>\n\n")

        for name, param in model.named_parameters():
            safe_name = name.replace(".", "_")
            data = param.detach().cpu().numpy()

            if "weight" in name:
                max_val = np.max(np.abs(data))
                scale = max_val / 127.0 if max_val > 0 else 1.0
                q_data = np.clip(np.round(data / scale), -128, 127).astype(np.int8)
                flat_q = q_data.flatten()
                all_int8_weights.extend(flat_q.tolist())

                f.write(f"// Layer: {name} | Scale: {scale:.8f}\n")
                f.write(f"const float {safe_name}_scale = {scale:.8f}f;\n")
                f.write(f"const int8_t {safe_name}[{len(flat_q)}] = {{\n")
                for i in range(0, len(flat_q), 16):
                    row = ", ".join(f"{int(v):4d}" for v in flat_q[i:i+16])
                    f.write(f"    {row},\n")
                f.write("};\n\n")
                layer_meta[name] = {"shape": list(data.shape), "scale": float(scale), "count": len(flat_q)}

            elif "bias" in name:
                scale_b = 0.001
                q_bias = np.round(data / scale_b).astype(np.int32)
                flat_b = q_bias.flatten()

                f.write(f"// Layer Bias: {name} | Count: {len(flat_b)}\n")
                f.write(f"const int32_t {safe_name}[{len(flat_b)}] = {{\n")
                row = ", ".join(f"{int(v)}" for v in flat_b)
                f.write(f"    {row}\n")
                f.write("};\n\n")

        f.write("#endif // FPGA_WEIGHTS_INT8_H\n")

    with open(coe_path, "w") as f:
        f.write("; Vivado BRAM Coefficient Initialization File\n")
        f.write("; Radix 16 (Hexadecimal 8-bit signed two's complement)\n")
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        hex_vals = [f"{(int(w) & 0xFF):02X}" for w in all_int8_weights]
        for i in range(0, len(hex_vals), 16):
            chunk = " ".join(hex_vals[i:i+16])
            separator = ";" if (i + 16 >= len(hex_vals)) else ","
            f.write(f"{chunk}{separator}\n")

    with open(json_path, "w") as f:
        json.dump(layer_meta, f, indent=4)

    return header_path, coe_path
