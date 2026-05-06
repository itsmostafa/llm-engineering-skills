---
name: transformers
description: Loading and using pretrained models with Hugging Face Transformers. Use when working with pretrained models from the Hub, running inference with Pipeline API, fine-tuning models with Trainer, or handling text, vision, audio, and multimodal tasks.
---

# Using Hugging Face Transformers

Transformers is the model-definition framework for state-of-the-art machine learning across text, vision, audio, and multimodal domains. It provides unified APIs for loading pretrained models, running inference, and fine-tuning.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Pipeline API](#pipeline-api)
- [Model Loading](#model-loading)
- [Inference Patterns](#inference-patterns)
- [Fine-tuning with Trainer](#fine-tuning-with-trainer)
- [Working with Modalities](#working-with-modalities)
- [Memory and Performance](#memory-and-performance)
- [Best Practices](#best-practices)
- [References](#references)

## Core Concepts

### The Three Core Classes

Every model in Transformers has three core components:

```python
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoProcessor

# Configuration: hyperparameters and architecture settings
config = AutoConfig.from_pretrained("bert-base-uncased")

# Model: the neural network weights
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenizer: converts text inputs to tensors
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Processor: unified preprocessing for vision, audio, and multimodal models
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
```

### The `from_pretrained` Pattern

All loading uses `from_pretrained()` which handles downloading, caching, and device placement:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",  # Automatic device placement
)
```

Transformers v5 examples use `dtype`. On Transformers v4, the equivalent argument is `torch_dtype`.

### Auto Classes

Use task-specific Auto classes for the correct model head:

```python
from transformers import (
    AutoModelForCausalLM,          # Text generation (GPT, Llama)
    AutoModelForSeq2SeqLM,         # Encoder-decoder (T5, BART)
    AutoModelForSequenceClassification,  # Classification
    AutoModelForTokenClassification,     # NER, POS tagging
    AutoModelForQuestionAnswering,       # Extractive QA
    AutoModelForMaskedLM,                # BERT-style masked LM
    AutoModelForImageClassification,     # Vision models
    AutoModelForSpeechSeq2Seq,           # Speech recognition
)
```

## Pipeline API

The `pipeline()` function provides high-level inference with minimal code:

### Text Tasks

```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
output = generator("The secret to success is", max_new_tokens=50)

# Text classification
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Named entity recognition
ner = pipeline("ner", aggregation_strategy="simple")
entities = ner("Hugging Face is based in New York City.")

# Question answering
qa = pipeline("question-answering")
answer = qa(question="What is the capital?", context="Paris is the capital of France.")

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_text, max_length=130, min_length=30)

# Translation
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
result = translator("Hello, how are you?")
```

### Chat/Conversational

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."},
]

response = pipe(messages, max_new_tokens=256)
print(response[0]["generated_text"][-1]["content"])
```

### Vision Tasks

```python
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
```

### Audio Tasks

```python
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
text = transcriber("path/to/audio.mp3")
```

### Multimodal Tasks

```python
vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
```

## Model Loading

### Device Placement

```python
from transformers import AutoModelForCausalLM
import torch

# Automatic placement across available devices
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    device_map="auto",
    dtype=torch.bfloat16,
)

# Specific device
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="cuda:0",
)

# Custom device map for model parallelism
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 1,
    "model.norm": 1,
    "lm_head": 1,
}
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
```

### Loading from Local Path

```python
# Save model locally
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load from local path
model = AutoModelForCausalLM.from_pretrained("./my_model")
tokenizer = AutoTokenizer.from_pretrained("./my_model")
```

### Trust Remote Code

Some models require executing custom code from the Hub:

```python
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,  # Required for custom architectures
)
```

Prefer models with safetensors weights when available. Safetensors avoids pickle execution risks and typically loads faster than legacy `.bin` checkpoints.

## Inference Patterns

### Text Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

# Basic generation
inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# With generation config
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
)
```

### Chat Templates

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Apply chat template
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Getting Embeddings

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts: list[str]) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    attention_mask = inputs["attention_mask"]
    embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = (embeddings * mask_expanded).sum(1)
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

embeddings = get_embeddings(["Hello world", "How are you?"])
```

### Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("I love this movie!", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)

labels = model.config.id2label
for idx, prob in enumerate(predictions[0]):
    print(f"{labels[idx]}: {prob:.4f}")
```

## Fine-tuning with Trainer

### Basic Fine-tuning

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Load data and model
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
)

# Tokenize dataset
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
```

### Pushing to Hub

```python
# Login first: huggingface-cli login

# Push model and tokenizer
model.push_to_hub("my-username/my-fine-tuned-model")
tokenizer.push_to_hub("my-username/my-fine-tuned-model")

# Or use trainer
trainer.push_to_hub()
```

See `reference/fine-tuning.md` for advanced patterns including LoRA, custom data collators, and evaluation metrics.

## Working with Modalities

Use `AutoProcessor` or modality-specific processors for non-text models. Processors handle images, audio, video, and multimodal chat formatting before tensors are sent to the model.

| Modality | Processor | Model Class |
|----------|-----------|-------------|
| Vision | `AutoImageProcessor` | `AutoModelForImageClassification`, `AutoModelForObjectDetection` |
| Audio | `AutoProcessor` | `AutoModelForSpeechSeq2Seq`, `AutoModelForAudioClassification` |
| Vision-language | `AutoProcessor` | `AutoModelForVision2Seq`, task-specific VLM classes |

## Memory and Performance

### Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
    dtype="auto",
)

# 8-bit quantization
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto",
    dtype="auto",
)
```

### Flash Attention

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Requires flash-attn package
    device_map="auto",
)
```

### torch.compile

```python
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
model = torch.compile(model, mode="reduce-overhead")
```

### Batched Inference

```python
texts = ["First prompt", "Second prompt", "Third prompt"]
inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

## Best Practices

1. **Use bfloat16 over float16**: Better numerical stability on modern GPUs
2. **Use `dtype="auto"` when unsure**: It follows model metadata or checkpoint weight dtype without wasting memory
3. **Set pad token for generation**: `tokenizer.pad_token = tokenizer.eos_token`
4. **Use device_map="auto"**: Let Accelerate handle device placement
5. **Prefer safetensors checkpoints**: Avoid pickle-based loading when possible
6. **Enable Flash Attention**: Significant speedup for long sequences
7. **Batch when possible**: Amortize fixed costs across multiple inputs
8. **Use pipeline for quick prototyping**: Switch to manual control for production
9. **Cache models locally**: Set `HF_HOME` environment variable for model cache location
10. **Check model license**: Verify usage rights before deployment

## References

See `reference/` for detailed documentation:
- `fine-tuning.md` - Advanced fine-tuning patterns with LoRA, PEFT, and custom training

External documentation:
- [Transformers documentation](https://huggingface.co/docs/transformers/)
- [Loading models](https://huggingface.co/docs/transformers/models)
- [Bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
