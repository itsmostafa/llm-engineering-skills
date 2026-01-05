# MLX Fine-tuning Guide

Complete guide to fine-tuning LLMs with LoRA, QLoRA, and full fine-tuning on Apple Silicon.

## Table of Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [LoRA Training](#lora-training)
- [QLoRA Training](#qlora-training)
- [Full Fine-tuning](#full-fine-tuning)
- [Configuration Options](#configuration-options)
- [Evaluation](#evaluation)
- [Fusing and Exporting](#fusing-and-exporting)
- [Memory Considerations](#memory-considerations)

## Overview

MLX supports three fine-tuning approaches:

| Method | Memory | Quality | Speed |
|--------|--------|---------|-------|
| LoRA | Low | Good | Fast |
| QLoRA | Very Low | Good | Fast |
| Full | High | Best | Slow |

LoRA (Low-Rank Adaptation) trains small adapter matrices while freezing the base model. QLoRA applies LoRA to quantized models.

## Data Preparation

### Supported Formats

MLX-LM accepts JSONL files with either format:

**Completion format:**
```jsonl
{"text": "Complete training text including prompt and response"}
{"text": "Another complete training example"}
```

**Chat format:**
```jsonl
{"messages": [{"role": "user", "content": "What is MLX?"}, {"role": "assistant", "content": "MLX is Apple's machine learning framework."}]}
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

### Directory Structure

```
data/
├── train.jsonl    # Required
├── valid.jsonl    # Optional, for validation
└── test.jsonl     # Optional, for evaluation
```

### Data Preparation Script

```python
import json

def prepare_chat_data(conversations, output_path):
    """Convert conversations to JSONL format."""
    with open(output_path, 'w') as f:
        for conv in conversations:
            messages = []
            for turn in conv:
                messages.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
            f.write(json.dumps({"messages": messages}) + '\n')

def prepare_completion_data(examples, output_path):
    """Convert prompt-response pairs to completion format."""
    with open(output_path, 'w') as f:
        for prompt, response in examples:
            text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
            f.write(json.dumps({"text": text}) + '\n')
```

## LoRA Training

### Basic Training

```bash
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data ./data \
    --iters 1000
```

### With All Options

```bash
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data ./data \
    --iters 1000 \
    --batch-size 4 \
    --lora-rank 8 \
    --lora-layers 16 \
    --learning-rate 1e-5 \
    --adapter-path ./my-adapters \
    --save-every 100 \
    --val-batches 25
```

### Resume Training

```bash
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data ./data \
    --resume-adapter-file ./my-adapters/adapters.safetensors \
    --iters 500  # Additional iterations
```

## QLoRA Training

QLoRA is automatically enabled when using a quantized base model:

```bash
# Using quantized model = QLoRA
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train \
    --data ./data \
    --iters 1000

# Using non-quantized model = regular LoRA
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct \
    --train \
    --data ./data \
    --iters 1000
```

### QLoRA Benefits

- Train 7B models on 8GB Macs
- Train 13B models on 16GB Macs
- Minimal quality loss vs full LoRA

## Full Fine-tuning

Update all model weights (requires more memory):

```bash
mlx_lm.lora --model mlx-community/Llama-3.2-1B-Instruct \
    --train \
    --data ./data \
    --fine-tune-type full \
    --iters 1000 \
    --batch-size 1 \
    --learning-rate 1e-6
```

### When to Use Full Fine-tuning

- Small models (1-3B parameters)
- Sufficient memory available
- Maximum adaptation needed
- Task significantly different from pretraining

## Configuration Options

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-rank` | 8 | Rank of LoRA matrices (higher = more capacity) |
| `--lora-layers` | 16 | Number of layers to adapt |
| `--lora-alpha` | 16 | Scaling factor (typically 2x rank) |
| `--lora-dropout` | 0.0 | Dropout on LoRA layers |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iters` | 1000 | Training iterations |
| `--batch-size` | 4 | Batch size |
| `--learning-rate` | 1e-5 | Learning rate |
| `--warmup` | 100 | Warmup steps |
| `--steps-per-report` | 10 | Log every N steps |
| `--steps-per-eval` | 200 | Evaluate every N steps |
| `--save-every` | 100 | Save checkpoint every N steps |
| `--max-seq-length` | 2048 | Maximum sequence length |
| `--grad-checkpoint` | false | Enable gradient checkpointing |

### Rank Selection Guide

| Rank | Parameters | Use Case |
|------|------------|----------|
| 4 | ~0.1% | Simple adaptations, memory constrained |
| 8 | ~0.2% | Default, good balance |
| 16 | ~0.4% | Complex tasks, more capacity |
| 32 | ~0.8% | Maximum adaptation |
| 64+ | ~1.5%+ | Approaching full fine-tuning |

## Evaluation

### Test Set Evaluation

```bash
mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path ./adapters \
    --test \
    --data ./data
```

### Generate with Adapter

```bash
mlx_lm.generate --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path ./adapters \
    --prompt "Your test prompt"
```

### Python Evaluation

```python
from mlx_lm import load, generate

# Load model with adapter
model, tokenizer = load(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path="./adapters"
)

# Generate
response = generate(model, tokenizer, prompt="Test prompt", max_tokens=256)
print(response)
```

## Fusing and Exporting

### Fuse Adapter into Model

Merge LoRA weights into base model for deployment:

```bash
# Fuse and save as MLX model
mlx_lm.fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path ./adapters \
    --save-path ./fused-model

# Fuse with de-quantization (higher quality)
mlx_lm.fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path ./adapters \
    --save-path ./fused-model \
    --de-quantize
```

### Export to GGUF

For use with llama.cpp and compatible tools:

```bash
mlx_lm.fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path ./adapters \
    --export-gguf \
    --gguf-path ./model.gguf
```

### Upload to Hugging Face

```bash
mlx_lm.fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --adapter-path ./adapters \
    --upload-repo your-username/your-model-name
```

## Memory Considerations

### Estimated Memory Usage

| Model | LoRA (4-bit) | LoRA (16-bit) | Full |
|-------|--------------|---------------|------|
| 1B | ~3 GB | ~4 GB | ~6 GB |
| 3B | ~5 GB | ~8 GB | ~14 GB |
| 7B | ~8 GB | ~16 GB | ~32 GB |
| 13B | ~12 GB | ~28 GB | ~56 GB |

### Memory Optimization

```bash
# Enable gradient checkpointing
mlx_lm.lora --model ... --train --grad-checkpoint

# Reduce batch size
mlx_lm.lora --model ... --train --batch-size 1

# Reduce sequence length
mlx_lm.lora --model ... --train --max-seq-length 1024

# Reduce LoRA rank
mlx_lm.lora --model ... --train --lora-rank 4
```

### Swap Usage

MLX can use swap when memory is insufficient. Monitor with Activity Monitor:
- Green memory pressure: Good
- Yellow: Acceptable, some slowdown
- Red: Significant slowdown, reduce batch size

### Best Practices

1. Start with QLoRA (4-bit base model) for large models
2. Use batch size 1-4 depending on model size
3. Enable gradient checkpointing for 7B+ models
4. Monitor memory pressure during training
5. Save checkpoints frequently (`--save-every 100`)
6. Validate regularly (`--steps-per-eval 100`)
7. Use warmup steps to stabilize training
8. Start with default learning rate (1e-5), adjust if loss doesn't decrease
