# LLM Engineering Skills

A curated collection of Claude Code skills for machine learning engineers. These skills extend Claude's capabilities with expertise in building LLM agents, prompt engineering, fine-tuning with LoRA/QLoRA, and framework-specific knowledge for PyTorch, Transformers, and MLX.

## Skills Included

| Skill | Description |
|-------|-------------|
| **agents** | Patterns and architectures for building AI agents and workflows. Tool use, multi-step reasoning, and orchestration of LLM-driven tasks. |
| **context-engineering** | Managing LLM context windows in AI agents. Long conversations, multi-step tasks, and maintaining coherence across extended interactions. |
| **lora** | Parameter-efficient fine-tuning with Low-Rank Adaptation. Train models with ~0.1% of original parameters using adapter merging. |
| **mlx** | Running and fine-tuning LLMs on Apple Silicon with MLX. Model conversion, quantization, LoRA fine-tuning, and local model serving. |
| **prompt-engineering** | Crafting effective prompts for LLMs. Designing prompts, improving output quality, and structuring complex instructions. |
| **pytorch** | Building and training neural networks with PyTorch. Training loops, data pipelines, torch.compile optimization, and distributed training. |
| **qlora** | Memory-efficient fine-tuning with 4-bit quantization and LoRA adapters. Fine-tune large models (7B+) on consumer GPUs with limited VRAM. |
| **rlhf** | Reinforcement Learning from Human Feedback for aligning language models. Reward modeling, policy optimization, and DPO. |
| **transformers** | Loading and using pretrained models with Hugging Face Transformers. Pipeline API, Trainer fine-tuning, and multimodal tasks. |

## Getting Started

#### Claude Code

Prerequisites
- Claude Code CLI (version 1.0.33 or later)

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

Prerequisites
- Codex CLI

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
