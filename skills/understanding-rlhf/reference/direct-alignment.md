# Direct Alignment Algorithms

Direct alignment methods optimize language models from preference data without training a separate reward model. This document covers DPO and related algorithms.

## Table of Contents

- [Motivation](#motivation)
- [DPO (Direct Preference Optimization)](#dpo-direct-preference-optimization)
- [IPO (Identity Preference Optimization)](#ipo-identity-preference-optimization)
- [KTO (Kahneman-Tversky Optimization)](#kto-kahneman-tversky-optimization)
- [Other Variants](#other-variants)
- [Comparison and Selection](#comparison-and-selection)

## Motivation

### The Standard RLHF Pipeline

Traditional RLHF involves three stages:

1. SFT: Train on demonstrations
2. RM: Train reward model on preferences
3. RL: Optimize policy against reward model

This pipeline is complex:
- Requires training and serving a separate reward model
- RL training is unstable and hyperparameter-sensitive
- Reward hacking can occur during optimization

### The Direct Alignment Insight

Direct alignment algorithms recognize that the RLHF objective has a closed-form optimal policy:

```
π*(y|x) ∝ π_ref(y|x) · exp(R(y|x) / β)
```

This means we can derive loss functions that directly optimize for this relationship, bypassing the need for explicit reward modeling and RL.

### Benefits

- **Simpler pipeline**: One training stage instead of two
- **More stable**: No RL instability
- **Computationally efficient**: No reward model inference during training
- **Fewer hyperparameters**: No PPO-specific tuning

### Trade-offs

- **Less flexible**: Can't reuse the reward model for other purposes
- **Data requirements**: May need more preference data
- **Less interpretable**: No explicit reward scores to inspect

## DPO (Direct Preference Optimization)

### Derivation

Starting from the RLHF objective and its optimal policy, DPO derives that the implicit reward is:

```
R(y|x) = β · log(π(y|x) / π_ref(y|x)) + Z(x)
```

Where Z(x) is a normalization constant that cancels when computing preference probabilities.

Substituting into the Bradley-Terry preference model:

```
P(y_w > y_l | x) = sigmoid(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
```

### Loss Function

The DPO loss maximizes the log-likelihood of observed preferences:

```
L_DPO = -E[log sigmoid(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

Where y_w is the preferred (winning) response and y_l is the dispreferred (losing) response.

### Intuition

DPO directly increases the relative log-probability of preferred responses over dispreferred ones, while implicitly maintaining proximity to the reference policy through the ratio terms.

### Implementation

Training DPO is straightforward:

1. Load the SFT model as both policy and reference
2. For each (prompt, chosen, rejected) tuple:
   - Compute log-probs under current policy
   - Compute log-probs under reference policy (fixed)
   - Compute DPO loss
   - Update policy parameters

The reference model is frozen and used only for computing log-probabilities.

### The β Parameter

β controls the strength of the implicit KL regularization:

- **High β** (>0.5): Strong deviation from reference allowed
- **Medium β** (0.1-0.5): Balanced optimization
- **Low β** (<0.1): Conservative, stays close to reference

Note: This is inverted from PPO where β multiplies the KL penalty. In DPO, higher β means more weight on the preference signal.

### Variants

**DPO with margin**: Add a target margin m:
```
L = -log sigmoid(β · (log ratio_w - log ratio_l) - m)
```

**Label smoothing**: Smooth the preference labels:
```
L = -(1-ε)·log sigmoid(...) - ε·log sigmoid(-...)
```

## IPO (Identity Preference Optimization)

### Motivation

DPO assumes the Bradley-Terry model perfectly describes human preferences. When this assumption is violated, DPO can overfit to spurious patterns.

IPO addresses this by using a different loss formulation that doesn't rely on the Bradley-Terry assumption.

### Loss Function

IPO minimizes:

```
L_IPO = (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x) - 1/(2β))²
```

This is a regression loss that targets a specific margin (1/2β) between preferred and dispreferred responses.

### Key Differences from DPO

| Aspect | DPO | IPO |
|--------|-----|-----|
| Loss type | Cross-entropy | Squared error |
| Target | Maximize sigmoid | Match target margin |
| Assumptions | Bradley-Terry holds | Weaker assumptions |
| Behavior | Can push ratios to ±∞ | Regularizes to target margin |

### When to Prefer IPO

Consider IPO when:
- Preferences are noisy or inconsistent
- You observe over-optimization with DPO
- The Bradley-Terry model seems like a poor fit

## KTO (Kahneman-Tversky Optimization)

### Motivation

Both DPO and IPO require pairwise preferences (chosen vs rejected for the same prompt). KTO works with binary feedback:

- "This response is good" (desirable)
- "This response is bad" (undesirable)

This is easier to collect and more natural for some feedback sources.

### Theoretical Basis

KTO is inspired by prospect theory from behavioral economics, specifically the idea that humans weight losses more heavily than gains (loss aversion).

### Loss Function

For desirable examples:
```
L_desirable = -log sigmoid(β · (log π(y|x)/π_ref(y|x) - KL_ref))
```

For undesirable examples:
```
L_undesirable = -log sigmoid(-β · (log π(y|x)/π_ref(y|x) - KL_ref))
```

Where KL_ref is the expected KL divergence, used to center the optimization.

### Asymmetric Weighting

KTO applies loss aversion by weighting undesirable examples more:

```
L = λ_D · L_desirable + λ_U · L_undesirable
```

With λ_U > λ_D (typical ratio: 1.33).

### Advantages

- **Simpler data**: Binary feedback is easier to collect
- **Unpaired data**: Doesn't require comparing responses to the same prompt
- **Natural fit**: Many feedback sources are binary (thumbs up/down)

### Limitations

- Pairwise comparisons provide more signal per annotation
- Loss aversion parameter requires tuning
- Less studied than DPO

## Other Variants

### ORPO (Odds Ratio Preference Optimization)

ORPO combines SFT and preference optimization into a single objective:

```
L = L_SFT + λ · L_OR
```

Where L_OR is an odds ratio loss on preferences. This eliminates the need for a separate SFT stage.

### SimPO (Simple Preference Optimization)

SimPO simplifies DPO by removing the reference model:

```
L = -log sigmoid(β · (log π(y_w|x) - log π(y_l|x)) / |y_w| - γ)
```

Uses length normalization and a target margin γ instead of reference probabilities.

### RLOO (REINFORCE Leave-One-Out)

Uses REINFORCE with a leave-one-out baseline:

```
baseline = (1/(n-1)) · Σ_{j≠i} R(y_j)
```

Provides low-variance gradient estimates without a learned value function.

### CPO (Contrastive Preference Optimization)

Treats preference optimization as a contrastive learning problem:

```
L = -log(exp(r(y_w)) / (exp(r(y_w)) + exp(r(y_l))))
```

Related to DPO but with different theoretical motivation.

## Comparison and Selection

### Algorithm Properties

| Algorithm | Requires Pairs | Reference Model | Complexity |
|-----------|---------------|-----------------|------------|
| PPO | No (uses RM) | Yes | High |
| DPO | Yes | Yes | Low |
| IPO | Yes | Yes | Low |
| KTO | No | Yes | Low |
| SimPO | Yes | No | Very Low |
| ORPO | Yes | No | Low |

### When to Use Each

**PPO**:
- When you need the reward model for other purposes
- Maximum flexibility in optimization
- Large-scale deployments with established infrastructure

**DPO**:
- Default choice for direct alignment
- When you have clean pairwise preferences
- Simpler pipeline is a priority

**IPO**:
- When preferences are noisy
- Observing over-optimization with DPO
- More conservative optimization desired

**KTO**:
- Binary feedback is more natural for your use case
- Unpaired good/bad examples
- Simpler annotation process needed

**SimPO**:
- Extreme simplicity desired
- No reference model available
- Quick iteration more important than theoretical guarantees

### Empirical Performance

In practice, these algorithms often perform similarly when:
- Data quality is high
- Hyperparameters are tuned appropriately
- The task is well-defined

The choice often comes down to:
1. What data format you have
2. Computational constraints
3. Team expertise and infrastructure

### Hyperparameter Sensitivity

| Algorithm | Key Hyperparameters | Sensitivity |
|-----------|--------------------|----|
| PPO | β, clip, GAE λ, lr | High |
| DPO | β, lr | Medium |
| IPO | β, lr | Medium |
| KTO | β, λ_U/λ_D, lr | Medium |

Direct alignment methods are generally less sensitive than PPO, making them easier to deploy without extensive tuning.

### Combining Approaches

Some practitioners combine approaches:

1. **DPO then PPO**: Use DPO for initial alignment, then refine with PPO
2. **Iterative DPO**: Generate new preferences from the current model, retrain
3. **Multi-stage alignment**: Different algorithms for different objectives

The best approach depends on your specific requirements, data availability, and computational budget.
