# Reward Modeling

This document covers the details of training reward models for RLHF.

## Table of Contents

- [Preference Data Formats](#preference-data-formats)
- [The Bradley-Terry Model](#the-bradley-terry-model)
- [Reward Model Architecture](#reward-model-architecture)
- [Training Procedure](#training-procedure)
- [Scaling and Quality](#scaling-and-quality)
- [Common Issues](#common-issues)

## Preference Data Formats

### Pairwise Comparisons

The standard format is (prompt, chosen, rejected) tuples:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "France is a country in Europe with many cities."
}
```

The chosen response was preferred by annotators over the rejected response.

### Ranked Lists

Some datasets provide rankings over multiple responses:

```
Response A > Response B > Response C > Response D
```

These can be converted to pairwise comparisons by considering all pairs where the ordering is defined.

### Scalar Ratings

Absolute ratings (e.g., 1-5 stars) can be converted to preferences:

- Pairs where ratings differ become (chosen, rejected) tuples
- Ties are typically discarded or downweighted

### Multi-Attribute Preferences

Some datasets collect preferences on multiple dimensions:

- Helpfulness
- Harmlessness
- Honesty
- Factual accuracy

These can be combined into a single preference or used to train separate reward models.

## The Bradley-Terry Model

### Formulation

The Bradley-Terry model assumes preferences are determined by latent "strength" parameters. Given rewards r(A) and r(B) for two responses:

```
P(A preferred over B) = exp(r(A)) / (exp(r(A)) + exp(r(B)))
                      = sigmoid(r(A) - r(B))
```

This says the probability of preferring A depends only on the difference in rewards.

### Loss Function

The negative log-likelihood gives the training loss:

```
L(θ) = -E[log P(chosen > rejected)]
     = -E[log sigmoid(r_θ(chosen) - r_θ(rejected))]
```

Minimizing this loss makes the reward model assign higher scores to chosen responses.

### Margin-Based Variants

Some implementations add a margin m to encourage larger score differences:

```
L = -log sigmoid(r(chosen) - r(rejected) - m)
```

This prevents the model from making arbitrarily small distinctions.

### Limitations

The Bradley-Terry model assumes:

- **Transitivity**: If A > B and B > C, then A > C
- **Independence**: Each comparison is independent
- **No ties**: Every comparison has a clear winner

Human preferences often violate these assumptions. IPO and other methods address some of these limitations.

## Reward Model Architecture

### Standard Approach

Start with the SFT model and replace the language modeling head with a scalar projection:

```
Input: prompt + response
      ↓
[Transformer backbone - shared with SFT model]
      ↓
[Last hidden state of final token]
      ↓
[Linear projection to scalar]
      ↓
Output: reward score
```

The backbone is typically initialized from the SFT model weights.

### Which Token to Use?

Common choices:

- **Last token**: Simple, works well for most cases
- **EOS token**: Ensures the model sees the complete response
- **Average pooling**: More robust but computationally heavier
- **Special reward token**: Added specifically for reward prediction

### Shared vs Separate Backbone

**Shared backbone** (initialize from SFT):
- Faster to train
- Better transfer of language understanding
- Standard approach in practice

**Separate backbone**:
- Can use a different model family
- No risk of forgetting SFT capabilities
- Less common but sometimes used for safety

### Model Size

Reward model size affects:

- **Accuracy**: Larger models generally produce better preference predictions
- **Training cost**: Scales with model parameters
- **Inference cost**: Used repeatedly during RL training
- **Generalization**: Larger models may overfit less to training preferences

Common practice: Use the same size as the SFT model, or slightly smaller for efficiency.

## Training Procedure

### Data Preparation

1. Load preference dataset with (prompt, chosen, rejected) tuples
2. Tokenize prompts and responses
3. Create batches with paired comparisons

### Forward Pass

For each (prompt, chosen, rejected) tuple:

1. Concatenate prompt + chosen → compute r_chosen
2. Concatenate prompt + rejected → compute r_rejected
3. Compute loss = -log sigmoid(r_chosen - r_rejected)

### Optimization

Standard practices:

- **Optimizer**: AdamW with weight decay
- **Learning rate**: 1e-5 to 1e-6 (lower than SFT)
- **Batch size**: As large as memory allows
- **Epochs**: 1-3 epochs typically sufficient
- **Gradient accumulation**: If batch size is limited

### Evaluation

Track during training:

- **Preference accuracy**: How often does r(chosen) > r(rejected)?
- **Calibration**: Do predicted probabilities match observed frequencies?
- **Loss curves**: Training and validation loss

A well-trained reward model typically achieves 65-75% accuracy on held-out preferences (higher suggests overfit, lower suggests the signal is too noisy).

## Scaling and Quality

### Data Quantity

More preference data generally helps, but with diminishing returns:

- Thousands of examples: Minimum viable
- Tens of thousands: Solid performance
- Hundreds of thousands: Approaching saturation

Quality matters more than quantity for smaller datasets.

### Data Quality Factors

**Annotator expertise**: Domain experts produce more consistent preferences

**Clear guidelines**: Explicit criteria reduce annotator disagreement

**Response diversity**: Comparing very different responses provides stronger signal

**Prompt coverage**: Diverse prompts improve generalization

### Annotator Agreement

Inter-annotator agreement indicates preference clarity:

- **High agreement (>80%)**: Clear criteria, reliable signal
- **Medium agreement (60-80%)**: Some subjectivity, typical for helpfulness
- **Low agreement (<60%)**: Highly subjective or unclear criteria

Low agreement suggests the reward model will have a noisy target.

### Ensemble Methods

Using multiple reward models can:

- Reduce variance in reward estimates
- Detect out-of-distribution responses
- Identify reward hacking (disagreement between models)

Common approaches:
- Train models with different random seeds
- Train on different data splits
- Use models of different sizes

## Common Issues

### Length Bias

Reward models often prefer longer responses, even when length isn't informative. Causes:

- Training data may have longer responses as chosen
- Longer responses contain more information by default
- Annotators may equate length with effort

Mitigations:
- Length normalization in reward computation
- Controlling for length in training data
- Explicit length penalties

### Sycophancy

Reward models may prefer responses that agree with the user, even when incorrect. This happens when:

- Annotators prefer agreeable responses
- Training data contains sycophantic patterns
- The model learns to maximize approval rather than correctness

Mitigations:
- Include examples where correct disagreement is preferred
- Train on adversarial examples
- Separate helpfulness from agreement

### Position Bias

The order in which responses are presented can affect preferences:

- Annotators may prefer the first or last response
- This bias can leak into the reward model

Mitigations:
- Randomize presentation order
- Balance chosen/rejected across positions
- Debias during training

### Reward Hacking Susceptibility

Reward models can be exploited by policies that find adversarial inputs:

- Unusual formatting that confuses the model
- Keyword stuffing that triggers high scores
- Length/style exploitation

Mitigations:
- Diverse training data
- Adversarial training
- KL regularization during policy optimization
- Reward model ensembles
