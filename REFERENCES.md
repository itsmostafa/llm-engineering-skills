# References

External resources and documentation referenced across skills.

## Papers

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT paper, foundational work on RLHF
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - DPO paper, direct alignment without reward models
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - ReAct paper, combining reasoning traces with actions
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - Introduces GRPO for online reasoning model optimization

## Books

- [RLHF Book](https://rlhfbook.com/) - Comprehensive textbook on RLHF by Nathan Lambert

## Engineering Guides

- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - Anthropic guide on managing LLM context windows
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) - Anthropic guide on agent patterns and architectures
- [Session Memory for Agents](https://cookbook.openai.com/examples/agents_sdk/session_memory) - OpenAI cookbook on context trimming and summarization patterns
- [Model Context Protocol](https://modelcontextprotocol.io/docs) - Open standard for connecting agents to tools, resources, and prompts

## Official Documentation

### Frameworks

- [PyTorch 2.11 documentation](https://pytorch.org/docs/stable/) - Deep learning framework
- [MLX](https://ml-explore.github.io/mlx/) - Apple Silicon ML framework
- [MLX-LM](https://github.com/ml-explore/mlx-lm) - LLM generation, quantization, serving, and fine-tuning utilities for MLX
- [LangGraph](https://docs.langchain.com/oss/javascript/langgraph/workflows-agents) - LangChain framework for building agent workflows

### Hugging Face Ecosystem

- [Transformers v5 documentation](https://huggingface.co/docs/transformers/) - Pretrained model library
- [Transformers model loading](https://huggingface.co/docs/transformers/models) - Current `from_pretrained`, dtype, and safetensors guidance
- [Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes) - 8-bit, 4-bit, and QLoRA quantization guidance
- [PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora) - Parameter-efficient fine-tuning, DoRA, rsLoRA, trainable tokens, MoE targeting, and adapter composition
- [TRL v1 documentation](https://huggingface.co/docs/trl/) - Post-training library for SFT, DPO, GRPO, reward modeling, RLOO, and PPO
- [Datasets](https://huggingface.co/docs/datasets/) - Dataset loading and processing
- [Hugging Face Hub](https://huggingface.co/docs/hub/) - Model and dataset hosting

## Model Collections

- [MLX Community](https://huggingface.co/mlx-community) - Pre-converted MLX models on Hugging Face

## Libraries

- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - 8-bit and 4-bit quantization for PyTorch

## Prompt Engineering

- [Claude Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) - Anthropic's official prompting documentation
- [GPT-5.2 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide) - OpenAI cookbook on advanced prompting techniques for GPT-5.2
