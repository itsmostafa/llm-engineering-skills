# LLM Engineering Skills

### Turn AI agents into capable LLM engineers.

This repository is a curated collection of practical engineering skills designed to make AI agents like Claude Code and OpenAI Codex dramatically more effective when working on AI and machine learning projects. Instead of generic assistance, these skills give agents concrete knowledge, workflows, and patterns used by real LLM engineers.

The goal is to enable AI agents to reason, plan, and execute AI engineering tasks with confidence and precision.

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

Once installed, AI agents will automatically use these skills when you work on relevant tasks. Here are some examples:

**Fine-tune a 70B model on a single GPU**
> "Help me QLoRA fine-tune Llama 3.1 70B on my custom dataset using a 48GB A6000"

The agent knows NF4 quantization, double quantization for memory savings, paged optimizers to handle memory spikes, and the exact `BitsAndBytesConfig` settings to make it work.

**Build an AI agent with tool use**
> "Create a ReAct agent that can search the web, read files, and execute code to answer research questions"

The agent understands orchestrator-worker patterns, human-in-the-loop checkpoints, and how to design tools that return self-contained, LLM-friendly outputs.

**Align a model with human preferences**
> "Implement DPO training to align my instruction-tuned model using preference data"

The agent knows the Bradley-Terry model, when to use DPO vs PPO, KL regularization to prevent reward hacking, and how to structure preference datasets.

**Run models locally on Apple Silicon**
> "Convert Mistral 7B to MLX format and fine-tune it on my M3 Max with LoRA"

The agent handles model conversion, 4-bit quantization for MLX, and memory-efficient LoRA training optimized for unified memory.

**Manage context in long-running agents**
> "My agent loses track of earlier decisions after 50+ turns. How do I fix this?"

The agent implements hybrid context strategies—trimming old messages while maintaining structured summaries—and designs tools for just-in-time context loading.

**Optimize training performance**
> "My PyTorch training is slow. Help me profile it and add torch.compile with the right backend"

The agent knows `torch.profiler`, when to use `inductor` vs `cudagraphs`, gradient checkpointing trade-offs, and how to identify bottlenecks in data loading.

## Contributing

Contributions are welcome.

If you have practical LLM engineering knowledge, workflows, or patterns that would help AI agents perform better on real projects, feel free to open a pull request.

Please update the `REFERENCES.md` file to include any external references you've used.

## License

MIT
