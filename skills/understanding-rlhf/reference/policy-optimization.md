# Policy Optimization for RLHF

This document covers reinforcement learning algorithms used to optimize language models against reward signals.

## Table of Contents

- [The RLHF Objective](#the-rlhf-objective)
- [Policy Gradient Foundations](#policy-gradient-foundations)
- [REINFORCE](#reinforce)
- [PPO for RLHF](#ppo-for-rlhf)
- [KL Regularization](#kl-regularization)
- [Practical Considerations](#practical-considerations)
- [Over-Optimization](#over-optimization)

## The RLHF Objective

### Formulation

The RLHF objective balances reward maximization with staying close to a reference policy:

```
J(π) = E_{x~D, y~π(·|x)}[R(x, y)] - β · KL(π || π_ref)
```

Where:
- x is a prompt sampled from the distribution D
- y is a response sampled from the policy π
- R(x, y) is the reward for the response
- π_ref is the reference policy (typically the SFT model)
- β is the KL penalty coefficient

### Intuition

Without the KL term, the policy would collapse to always producing the single highest-reward response. The KL penalty:

- Keeps the policy close to the reference distribution
- Maintains diversity in responses
- Prevents exploitation of reward model weaknesses
- Preserves capabilities learned during pretraining

### Equivalent Formulation

The objective can be rewritten with a per-token KL penalty:

```
R'(x, y) = R(x, y) - β · Σ_t log(π(y_t|x,y_{<t}) / π_ref(y_t|x,y_{<t}))
```

This distributes the KL penalty across tokens, which can improve credit assignment.

## Policy Gradient Foundations

### The Policy Gradient Theorem

For any differentiable policy π_θ, the gradient of expected reward is:

```
∇J(θ) = E[∇log π_θ(a|s) · Q(s, a)]
```

Where Q(s, a) is the action-value function. This allows gradient-based optimization of non-differentiable objectives (like rewards).

### For Language Models

In the RLHF setting:
- State s = (prompt x, tokens generated so far y_{<t})
- Action a = next token y_t
- Policy π_θ = language model
- Q includes both reward and KL penalty

### Variance Reduction

Raw policy gradients have high variance. Common techniques:

**Baseline subtraction**: Subtract a baseline b(s) from the return:
```
∇J(θ) = E[∇log π_θ(a|s) · (Q(s, a) - b(s))]
```

**Advantage estimation**: Use A(s, a) = Q(s, a) - V(s) instead of Q

**Reward normalization**: Normalize rewards across the batch

## REINFORCE

### Algorithm

REINFORCE is the simplest policy gradient method:

1. Sample a response y from π_θ given prompt x
2. Compute the return: R(x, y) - β · KL
3. Compute gradient: ∇log π_θ(y|x) · (return - baseline)
4. Update parameters: θ ← θ + α · gradient

### Advantages

- Simple to implement
- No value function needed
- Works with any reward signal

### Disadvantages

- High variance gradient estimates
- Sample inefficient (one update per trajectory)
- Can be unstable for complex policies

### When to Use

REINFORCE is suitable when:
- Simplicity is prioritized over performance
- Computational resources are limited
- The reward signal is relatively simple

## PPO for RLHF

### Overview

Proximal Policy Optimization (PPO) is the standard algorithm for RLHF. It improves on REINFORCE with:

- Clipped surrogate objective for stable updates
- Value function for variance reduction
- Multiple epochs per batch for sample efficiency

### The Clipped Surrogate Objective

PPO maximizes a clipped objective:

```
L^CLIP(θ) = E[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
```

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) is the probability ratio
- A_t is the advantage estimate
- ε is the clipping parameter (typically 0.1-0.2)

The clipping prevents the policy from moving too far from the previous version.

### Why Clipping Helps

Without clipping, large policy updates can:
- Destabilize training
- Move into poor regions of policy space
- Exploit inaccuracies in advantage estimates

Clipping limits the effective step size, making training more robust.

### Value Function

PPO trains a value function V_φ(s) alongside the policy. This is used for:

- Computing advantage estimates: A = R - V
- Providing a baseline to reduce variance
- Early stopping when value predictions are poor

The value function is typically a separate head on the same transformer backbone.

### Generalized Advantage Estimation (GAE)

GAE balances bias and variance in advantage estimates:

```
A^GAE_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
```

Where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

- λ = 0: Low variance, high bias (just TD error)
- λ = 1: High variance, low bias (Monte Carlo returns)
- Typical: λ = 0.95

### PPO Training Loop

For each iteration:

1. **Rollout**: Generate responses from current policy
2. **Score**: Compute rewards using reward model
3. **Compute advantages**: Using GAE with value function
4. **Update**: Multiple epochs over the batch
   - Compute clipped policy loss
   - Compute value function loss
   - Update both with gradient descent
5. **Update reference**: Optionally update old policy

### Hyperparameters

Key hyperparameters for PPO in RLHF:

| Parameter | Typical Range | Effect |
|-----------|--------------|--------|
| Clip ε | 0.1 - 0.2 | Limits policy update size |
| GAE λ | 0.95 | Bias-variance tradeoff |
| Value loss coef | 0.5 - 1.0 | Weight of value loss |
| Entropy bonus | 0.0 - 0.01 | Encourages exploration |
| Epochs per batch | 1 - 4 | Sample efficiency |

## KL Regularization

### Purpose

The KL penalty β · KL(π || π_ref) serves several crucial functions:

1. **Prevents reward hacking**: The reward model is imperfect; optimizing too hard exploits its weaknesses
2. **Maintains capabilities**: Keeps the model close to the pretrained distribution
3. **Ensures diversity**: Prevents collapse to a single high-reward response
4. **Stabilizes training**: Limits how much the policy can change

### Computing the KL Divergence

For language models, the KL is computed per-token:

```
KL(π || π_ref) = Σ_t E_{y_t~π}[log π(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

In practice, this is approximated using the sampled trajectory:

```
KL ≈ Σ_t [log π(y_t|x,y_{<t}) - log π_ref(y_t|x,y_{<t})]
```

### Choosing β

The KL coefficient β controls the regularization strength:

- **High β** (>0.1): Conservative updates, stays close to reference
- **Medium β** (0.01-0.1): Balanced optimization
- **Low β** (<0.01): Aggressive optimization, higher reward but more drift

### Adaptive KL

Some implementations adjust β dynamically:

- If KL is too high, increase β to slow down
- If KL is too low, decrease β to allow more optimization
- Target a specific KL budget per update

### KL vs Clipping

Both mechanisms limit policy updates, but differently:

- **Clipping**: Hard constraint on probability ratios
- **KL**: Soft penalty on distribution divergence

Using both provides complementary regularization.

## Practical Considerations

### Batch Size

Larger batches reduce variance in gradient estimates:

- Minimum: 64-128 response pairs
- Recommended: 512-2048 for stable training
- Use gradient accumulation if memory-limited

### Learning Rate

Policy learning rates for RLHF are typically lower than SFT:

- Policy: 1e-6 to 5e-6
- Value function: 1e-6 to 1e-5
- Use warmup and potentially decay

### Reward Normalization

Normalizing rewards improves training stability:

- Subtract running mean
- Divide by running standard deviation
- Clip extreme values

### Prompt Distribution

The distribution of prompts affects what the model learns:

- Diverse prompts improve generalization
- Oversampling certain categories can emphasize those behaviors
- Consider curriculum learning (easier prompts first)

### Checkpointing

Save checkpoints frequently because:

- RLHF training can be unstable
- Best checkpoint may not be the final one
- Allows recovery from divergence

## Over-Optimization

### The Problem

As training continues, reward model scores increase but true quality may decrease:

```
Early training: RM score ↑, Human eval ↑
Over-optimization: RM score ↑↑, Human eval ↓
```

This happens because the policy exploits reward model weaknesses rather than improving genuinely.

### Symptoms

Signs of over-optimization:

- **Verbosity**: Responses become unnecessarily long
- **Repetition**: Repeating high-scoring phrases
- **Sycophancy**: Excessive agreement with users
- **Formatting tricks**: Using structures that score well but aren't helpful
- **Divergence**: KL from reference grows unboundedly

### Detection

Monitor during training:

- KL divergence from reference
- Human evaluation on held-out prompts
- Response length and diversity statistics
- Reward model scores on adversarial examples

### Mitigation

Strategies to prevent over-optimization:

1. **Stronger KL penalty**: Increase β to limit divergence
2. **Early stopping**: Stop before rewards plateau
3. **Reward model ensembles**: Use multiple reward models
4. **Reward caps**: Clip maximum reward to prevent exploitation
5. **Periodic human evaluation**: Check actual quality, not just RM scores
6. **Regularization techniques**: Weight decay, dropout on value head
