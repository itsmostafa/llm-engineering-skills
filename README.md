# LLM Engineering Skills

A curated collection of Claude Code skills for machine learning engineers. These skills extend Claude's capabilities with deep expertise in PyTorch, Hugging Face Transformers, LoRA fine-tuning, and Apple Silicon optimization with MLX.

## Skills Included

| Skill | Description |
|-------|-------------|
| **pytorch** | Building and training neural networks with PyTorch. Training loops, data pipelines, torch.compile optimization, and distributed training. |
| **transformers** | Loading and using pretrained models with Hugging Face Transformers. Pipeline API, Trainer fine-tuning, and multimodal tasks. |
| **lora** | Parameter-efficient fine-tuning with Low-Rank Adaptation. Train models with ~0.1% of original parameters using QLoRA and adapter merging. |
| **mlx** | Running and fine-tuning LLMs on Apple Silicon with MLX. Model conversion, quantization, LoRA fine-tuning, and local model serving. |
| **rlhf** | Understanding Reinforcement Learning from Human Feedback for aligning language models. Reward modeling, policy optimization, and direct alignment algorithms like DPO. |

## Getting Started

### Prerequisites

- Claude Code CLI (version 1.0.33 or later)

### Installation

#### Claude Code

Add the plugin marketplace and install:

```bash
# Add the marketplace
/plugin marketplace add itsmostafa/llm-engineering-skills

# Install the plugin
/plugin install llm-engineering-skills@itsmostafa-llm-engineering-skills
```

Or install directly using the community CLI:

```bash
npx claude-plugins install @itsmostafa/llm-engineering-skills
```

#### Codex

Install a specific skill using the skill installer:

```bash
$skill-installer install https://github.com/itsmostafa/llm-engineering-skills/tree/main/skills/<skill-name>
```

For example, to install the `rlhf` skill:

```bash
$skill-installer install https://github.com/itsmostafa/llm-engineering-skills/tree/main/skills/rlhf
```

### Usage

Once installed, Claude will automatically use these skills when you work on relevant tasks:

- Ask Claude to help you build a PyTorch training loop
- Request help fine-tuning a model with LoRA
- Get assistance converting models to MLX format
- Work with Hugging Face pipelines and the Trainer API

## License

MIT
