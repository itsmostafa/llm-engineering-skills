# Advanced LoRA Techniques

## Table of Contents

- [DoRA (Weight-Decomposed LoRA)](#dora-weight-decomposed-lora)
- [rsLoRA (Rank-Stabilized LoRA)](#rslora-rank-stabilized-lora)
- [Multiple Adapters](#multiple-adapters)
- [Adapter Composition](#adapter-composition)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)
- [Memory Optimization](#memory-optimization)

## DoRA (Weight-Decomposed LoRA)

DoRA decomposes pretrained weights into magnitude and direction components, applying LoRA only to the direction. This often improves performance over standard LoRA.

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True,  # Enable DoRA
    task_type=TaskType.CAUSAL_LM,
)
```

**When to use DoRA:**
- Tasks requiring more precise weight updates
- When standard LoRA underperforms full fine-tuning
- Slightly higher memory than standard LoRA

## rsLoRA (Rank-Stabilized LoRA)

rsLoRA uses a different scaling factor (`lora_alpha / sqrt(r)` instead of `lora_alpha / r`) that stabilizes training across different rank values.

```python
config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_rslora=True,  # Enable rank-stabilized scaling
    task_type=TaskType.CAUSAL_LM,
)
```

**When to use rsLoRA:**
- Experimenting with different rank values
- Using higher ranks (32+) where standard scaling may be unstable
- When hyperparameter tuning rank

## Multiple Adapters

Load and manage multiple adapters for different tasks on the same base model.

### Loading Multiple Adapters

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Load first adapter
model = PeftModel.from_pretrained(base_model, "./adapter-coding", adapter_name="coding")

# Load additional adapters
model.load_adapter("./adapter-writing", adapter_name="writing")
model.load_adapter("./adapter-math", adapter_name="math")

# List loaded adapters
print(model.peft_config.keys())  # ['coding', 'writing', 'math']
```

### Switching Adapters

```python
# Set active adapter
model.set_adapter("coding")
code_output = model.generate(**inputs)

model.set_adapter("writing")
text_output = model.generate(**inputs)

# Use base model without any adapter
with model.disable_adapter():
    base_output = model.generate(**inputs)
```

### Deleting Adapters

```python
model.delete_adapter("math")
```

## Adapter Composition

Combine multiple adapters for multi-task capabilities.

### Weighted Combination

```python
from peft import PeftModel

# Load adapters
model = PeftModel.from_pretrained(base_model, "./adapter-1", adapter_name="adapter1")
model.load_adapter("./adapter-2", adapter_name="adapter2")

# Add weighted combination
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2"],
    weights=[0.7, 0.3],
    adapter_name="combined",
    combination_type="linear",  # or "cat" for concatenation
)

model.set_adapter("combined")
```

### Adapter Concatenation

```python
# Concatenate adapters (increases effective rank)
model.add_weighted_adapter(
    adapters=["adapter1", "adapter2"],
    weights=[1.0, 1.0],
    adapter_name="concat",
    combination_type="cat",
)
```

## Debugging and Troubleshooting

### Common Issues

**1. Loss not decreasing**
```python
# Check trainable parameters
model.print_trainable_parameters()

# Verify LoRA is applied to correct modules
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

**2. NaN/Inf losses**
```python
# Lower learning rate
training_args.learning_rate = 1e-4

# Add gradient clipping
training_args.max_grad_norm = 0.3

# Check for problematic data
for batch in dataloader:
    if torch.isnan(batch["input_ids"]).any():
        print("NaN in inputs!")
```

**3. Out of memory**
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller batch with accumulation
training_args.per_device_train_batch_size = 1
training_args.gradient_accumulation_steps = 16

# Use QLoRA
quantization_config = BitsAndBytesConfig(load_in_4bit=True, ...)
```

**4. Adapter not loading**
```python
# Ensure base model matches
# Check adapter_config.json for base_model_name_or_path

# Force load with different base
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=False,
)
```

### Inspecting Adapter Weights

```python
# Get LoRA weight statistics
for name, param in model.named_parameters():
    if "lora" in name.lower():
        print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
```

### Verifying Adapter Effect

```python
# Compare outputs with and without adapter
inputs = tokenizer("Test prompt", return_tensors="pt").to(model.device)

# With adapter
model.set_adapter("my_adapter")
with torch.no_grad():
    output_with = model(**inputs).logits

# Without adapter
with model.disable_adapter():
    with torch.no_grad():
        output_without = model(**inputs).logits

# Check difference
diff = (output_with - output_without).abs().mean()
print(f"Average logit difference: {diff:.4f}")
```

## Memory Optimization

### Gradient Checkpointing

```python
# Enable before training
model.gradient_checkpointing_enable()

# With input gradients (required for some setups)
model.enable_input_require_grads()
```

### Offloading

```python
# CPU offload for optimizer states
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
)

# Or use DeepSpeed ZeRO Stage 2/3
# In training args or accelerate config
```

### Tracking Memory Usage

```python
def print_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Before training
print_memory()

# After loading model
print_memory()

# After applying LoRA
print_memory()
```

### Reducing Adapter Memory

```python
# Use smaller rank
config = LoraConfig(r=8, ...)  # Instead of r=16

# Target fewer modules
config = LoraConfig(
    target_modules=["q_proj", "v_proj"],  # Skip k_proj, o_proj
    ...
)

# Use 8-bit optimizers
training_args = TrainingArguments(
    optim="adamw_bnb_8bit",
    ...
)
```
