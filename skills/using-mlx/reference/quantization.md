# MLX Quantization Methods

MLX-LM provides multiple quantization approaches optimized for different use cases and quality requirements.

## Table of Contents

- [Overview](#overview)
- [Basic Quantization](#basic-quantization)
- [DWQ (Distilled Weight Quantization)](#dwq-distilled-weight-quantization)
- [AWQ (Activation-aware Weight Quantization)](#awq-activation-aware-weight-quantization)
- [Dynamic Quantization](#dynamic-quantization)
- [GPTQ](#gptq)
- [Comparison and Selection](#comparison-and-selection)

## Overview

Quantization reduces model size and memory usage by representing weights with fewer bits:

| Precision | Bits | Memory vs FP16 | Typical Use |
|-----------|------|----------------|-------------|
| FP16 | 16 | 100% | Maximum quality |
| INT8 | 8 | 50% | Good quality, faster |
| INT4 | 4 | 25% | Best size/speed, acceptable quality |

All MLX quantization methods use group-wise quantization for better accuracy.

## Basic Quantization

Simple round-to-nearest quantization during model conversion.

### CLI

```bash
# 4-bit quantization (default)
mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct -q

# 8-bit quantization
mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct -q --q-bits 8

# Custom group size (smaller = better quality, larger model)
mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct \
    -q --q-bits 4 --q-group-size 32
```

### When to Use

- Quick model conversion
- When quality isn't critical
- Initial testing before advanced methods

## DWQ (Distilled Weight Quantization)

Uses knowledge distillation to preserve quality. A non-quantized teacher model guides the quantization of non-quantized parameters.

### CLI

```bash
mlx_lm.dwq --model meta-llama/Llama-3.2-3B-Instruct \
    --mlx-path ./llama-3.2-3b-dwq \
    --q-bits 4 \
    --iters 100
```

### Python

```python
# DWQ requires running the fine-tuning process
# Use CLI for most cases
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iters` | 100 | Distillation iterations |
| `--batch-size` | 4 | Batch size for calibration |
| `--q-bits` | 4 | Quantization bits |
| `--q-group-size` | 64 | Group size |

### When to Use

- Maximum quality preservation needed
- Fine-tuning is acceptable
- Production deployments requiring quality

## AWQ (Activation-aware Weight Quantization)

Scales and clips weights based on activation patterns before quantization.

### CLI

```bash
mlx_lm.awq --model meta-llama/Llama-3.2-3B-Instruct \
    --mlx-path ./llama-3.2-3b-awq \
    --q-bits 4
```

### How It Works

1. Analyzes activation magnitudes on calibration data
2. Identifies important weights based on activation patterns
3. Scales weights to protect important channels
4. Applies quantization

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-samples` | 128 | Calibration samples |
| `--q-bits` | 4 | Quantization bits |
| `--q-group-size` | 64 | Group size |

### When to Use

- Better quality than basic quantization
- Faster than DWQ
- When activation patterns are important

## Dynamic Quantization

Estimates per-layer sensitivity and applies different precision to different layers.

### CLI

```bash
mlx_lm.dynamic_quant --model meta-llama/Llama-3.2-3B-Instruct \
    --mlx-path ./llama-3.2-3b-dynamic \
    --target-bpw 4.5  # Target bits per weight
```

### How It Works

1. Measures each layer's sensitivity to quantization
2. Allocates more bits to sensitive layers
3. Uses fewer bits for robust layers
4. Achieves target average bits per weight

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-bpw` | 4.0 | Target bits per weight (average) |
| `--n-samples` | 128 | Calibration samples |
| `--min-bits` | 2 | Minimum bits for any layer |
| `--max-bits` | 8 | Maximum bits for any layer |

### When to Use

- Models with varying layer importance
- Want optimal quality at specific size target
- Mixed precision benefits

## GPTQ

Minimizes squared error of each layer's output through optimization.

### CLI

```bash
mlx_lm.gptq --model meta-llama/Llama-3.2-3B-Instruct \
    --mlx-path ./llama-3.2-3b-gptq \
    --q-bits 4
```

### How It Works

1. Processes layers sequentially
2. Uses calibration data to compute Hessian
3. Finds weights that minimize output error
4. Applies quantization with error compensation

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-samples` | 128 | Calibration samples |
| `--q-bits` | 4 | Quantization bits |
| `--q-group-size` | 64 | Group size |
| `--block-size` | 128 | Columns processed together |

### When to Use

- Established, well-tested method
- Good balance of quality and speed
- Compatibility with GPTQ ecosystem

## Comparison and Selection

### Quality Ranking (Best to Worst)

1. **DWQ** - Best quality, uses distillation
2. **AWQ** - Very good, activation-aware
3. **GPTQ** - Good, established method
4. **Dynamic** - Good, adaptive precision
5. **Basic** - Acceptable, fastest

### Speed Ranking (Fastest to Slowest)

1. **Basic** - Seconds
2. **AWQ** - Minutes
3. **Dynamic** - Minutes
4. **GPTQ** - Minutes to hours
5. **DWQ** - Hours (requires training)

### Decision Guide

```
Need quick conversion?
  └─ Yes → Basic quantization

Quality critical for production?
  └─ Yes → DWQ (if time allows) or AWQ

Model has varying layer importance?
  └─ Yes → Dynamic quantization

Want established method?
  └─ Yes → GPTQ

Default choice?
  └─ AWQ (good balance of quality and speed)
```

### Memory During Quantization

All methods except basic require loading the full model for calibration:

| Model Size | Basic | Advanced Methods |
|------------|-------|------------------|
| 3B | ~6 GB | ~8 GB |
| 7B | ~14 GB | ~18 GB |
| 13B | ~26 GB | ~32 GB |

For large models, use `--low-memory` flag when available, or quantize on machines with sufficient RAM.
