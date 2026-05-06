# Fine-tuning Patterns

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Custom Data Collators](#custom-data-collators)
- [Training Arguments](#training-arguments)
- [Evaluation Metrics](#evaluation-metrics)
- [LoRA and PEFT](#lora-and-peft)
- [Causal LM Fine-tuning](#causal-lm-fine-tuning)
- [Pushing to Hub](#pushing-to-hub)

## Dataset Preparation

### Using the datasets Library

```python
from datasets import load_dataset, Dataset

# Load from Hub
dataset = load_dataset("imdb")
dataset = load_dataset("squad", split="train")

# Load from local files
dataset = load_dataset("json", data_files="data.jsonl")
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Create from Python objects
data = {"text": ["hello", "world"], "label": [0, 1]}
dataset = Dataset.from_dict(data)

# From pandas
import pandas as pd
df = pd.read_csv("data.csv")
dataset = Dataset.from_pandas(df)
```

### Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

# Apply to dataset (batched for efficiency)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove original text column if not needed
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
```

### Train/Test Split

```python
# Split a dataset
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Stratified split for classification
split_dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
```

### Filtering and Preprocessing

```python
# Filter examples
dataset = dataset.filter(lambda x: len(x["text"]) > 100)

# Map with indices
def add_index(example, idx):
    example["idx"] = idx
    return example

dataset = dataset.map(add_index, with_indices=True)

# Shuffle
dataset = dataset.shuffle(seed=42)

# Select subset
small_dataset = dataset.select(range(1000))
```

## Custom Data Collators

### Dynamic Padding

```python
from transformers import DataCollatorWithPadding

# Pads to longest sequence in batch (more efficient than max_length)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
```

### Language Modeling Collator

```python
from transformers import DataCollatorForLanguageModeling

# For masked language modeling (BERT-style)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# For causal language modeling (GPT-style)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)
```

### Sequence-to-Sequence Collator

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,  # Ignore in loss
)
```

### Custom Collator

```python
from dataclasses import dataclass
import torch

@dataclass
class CustomCollator:
    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, features: list[dict]) -> dict:
        texts = [f["text"] for f in features]
        labels = torch.tensor([f["label"] for f in features])

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch
```

## Training Arguments

### Common Configurations

```python
from transformers import TrainingArguments

# Classification fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,  # Mixed precision
    dataloader_num_workers=4,
    report_to="tensorboard",
)

# LLM fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 32
    learning_rate=2e-4,
    num_train_epochs=1,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,  # Save memory
    max_grad_norm=1.0,
)
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `learning_rate` | Peak learning rate (2e-5 for BERT, 2e-4 for LoRA) |
| `warmup_ratio` | Fraction of steps for LR warmup (0.1 typical) |
| `weight_decay` | L2 regularization (0.01 typical) |
| `gradient_accumulation_steps` | Simulate larger batches |
| `gradient_checkpointing` | Trade compute for memory |
| `bf16` / `fp16` | Mixed precision training |
| `optim` | Optimizer (`adamw_torch_fused` fastest) |
| `max_grad_norm` | Gradient clipping threshold |

## Evaluation Metrics

### Using evaluate Library

```python
import evaluate
import numpy as np

# Load metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

### Common Metrics

```python
# Classification
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# Generation / Translation
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Question Answering
squad_metric = evaluate.load("squad")

# Perplexity (for LM)
perplexity = evaluate.load("perplexity")
```

### Custom Metrics

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Custom metric
    correct_per_class = {}
    for pred, label in zip(predictions, labels):
        if label not in correct_per_class:
            correct_per_class[label] = {"correct": 0, "total": 0}
        correct_per_class[label]["total"] += 1
        if pred == label:
            correct_per_class[label]["correct"] += 1

    per_class_accuracy = {
        f"accuracy_class_{k}": v["correct"] / v["total"]
        for k, v in correct_per_class.items()
    }

    return {"accuracy": (predictions == labels).mean(), **per_class_accuracy}
```

## LoRA and PEFT

### Basic LoRA Setup

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=16,                     # Rank
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,214,544,896 || trainable%: 0.13%
```

### Full LoRA Fine-tuning Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Load base model
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Prepare dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_prompt(example):
    if example["input"]:
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

dataset = dataset.map(format_prompt)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Training
from transformers import DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./lora-llama",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    warmup_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Save LoRA weights only
model.save_pretrained("./lora-llama")
```

### Merging LoRA Weights

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./lora-llama")

# Merge into base model
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
```

### QLoRA (Quantized LoRA)

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# Apply LoRA
model = get_peft_model(model, lora_config)
```

## Causal LM Fine-tuning

### Instruction Tuning

```python
def format_instruction(example):
    return {
        "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}<|endoftext|>"
    }

dataset = dataset.map(format_instruction)

# Tokenize with labels
def tokenize_for_clm(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
```

### SFTTrainer (from trl)

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft-model",
    max_length=1024,
    dataset_text_field="text",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,  # Optional: apply LoRA
)

trainer.train()
```

## Pushing to Hub

### Basic Upload

```python
# Login first
from huggingface_hub import login
login()  # Or use: huggingface-cli login

# Push model
model.push_to_hub("username/model-name")
tokenizer.push_to_hub("username/model-name")

# Push with trainer
trainer.push_to_hub()
```

### With Model Card

```python
from huggingface_hub import ModelCard

card = ModelCard.from_template(
    card_data={
        "language": "en",
        "license": "mit",
        "tags": ["text-classification", "sentiment"],
        "datasets": ["imdb"],
        "metrics": ["accuracy", "f1"],
    },
    model_summary="Fine-tuned BERT for sentiment analysis on IMDB dataset.",
)

card.push_to_hub("username/model-name")
```

### Private Models

```python
model.push_to_hub("username/model-name", private=True)
```
