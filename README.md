# A Research-Grade Framework for a Long-Context, Autonomous Trading Agent

## Overview

This document presents a sophisticated framework for developing an autonomous trading agent capable of leveraging extensive historical market data to make informed trading decisions. It combines the **HyenaDNA** architecture with the **Advantage Actor-Critic (A2C)** reinforcement learning algorithm and employs a meticulous reward design focused on risk-adjusted profitability.

---

## Architectural Foundations

### HyenaDNA for Financial Markets

Financial markets generate long sequences of data requiring models that can handle extensive contexts efficiently. Traditional Transformer-based models suffer from quadratic scaling complexity $O(L^2)$, limiting their practical use. **HyenaDNA** overcomes this limitation through its innovative convolutional architecture, achieving sub-quadratic complexity $O(L \log L)$, enabling the processing of sequences up to one million tokens. This drastically increases computational efficiency and allows for high-fidelity, raw data-driven feature extraction.

**Computational Complexity:**  
$O(L \log L)$

**Efficiency Gain:**
- Up to **160x faster** than Transformers at 1M tokens on RTX 5070 GPU.

HyenaDNA is inherently suited to financial data due to its capacity to maintain fine-grained, single-tick resolution while simultaneously capturing macroeconomic trends, analogous to genomic SNP tracking in biology.

---

### On-Policy Reinforcement Learning: A2C Algorithm

For realistic, sequential decision-making scenarios, we adopt an **on-policy** reinforcement learning strategy to maintain causality and market realism. Specifically, **Advantage Actor-Critic (A2C)** is employed due to its synchronous updates and ability to leverage current policy data exclusively.

**A2C Algorithm Components:**
- **Actor Network:** Decides action probabilities given the current state.
- **Critic Network:** Estimates the expected cumulative future reward from the current state.

**Advantage Estimation:**  
The advantage function quantifies action effectiveness:  
$A(s, a) = Q(s, a) - V(s)$

---

### Generalized Advantage Estimation (GAE)

To stabilize and enhance learning, we utilize **Generalized Advantage Estimation (GAE)**, balancing bias and variance through a parameter $\lambda$:

- $\lambda = 0$: Low variance, high bias.
- $\lambda = 1$: High variance, unbiased Monte Carlo return.
- Optimal balance ($\lambda = 0.95$) ensures stable learning in noisy financial environments.

GAE calculation for each step $t$:

$\delta_t = r_t + \gamma V(s_{t+1})(1 - \text{done}_t) - V(s_t)$

GAE recursive formula:

![GAE recursive formula](https://miro.medium.com/v2/resize:fit:640/format:webp/1*3ZUp_BW-YpCXyL6zJ_ow7A.png)

---

## Environment and System Design

### Observation Space

Structured as a 2D tensor of shape `(sequence_length, num_features)`, containing:

- **Market Features (OHLCV):** Raw data, allowing model-based feature discovery.
- **Technical Indicators:** RSI, MACD, Bollinger Bands, ATR, for accelerated initial convergence.
- **Portfolio State Features:** Equity, position status (-1: short, 0: flat, 1: long), unrealized PnL, steps in position.

| Feature              | Type      | Normalization    |
|----------------------|-----------|------------------|
| Close Price          | Market    | Z-score (%)      |
| Volume               | Market    | Z-score          |
| RSI (14)             | Indicator | Min-Max          |
| MACD (12,26,9)       | Indicator | Z-score          |
| Bollinger Bands      | Indicator | Z-score          |
| ATR (14)             | Indicator | Z-score          |
| Account Equity       | Portfolio | Log transform    |
| Position Status      | Portfolio | One-hot encoding |
| Unrealized PnL       | Portfolio | Scaled by equity |
| Steps in Position    | Portfolio | Min-Max          |

---

### Action Space

Discrete and contextually intelligent action space to simplify agent decisions:

- **Action 0 (HOLD):** No trade executed.
- **Action 1 (GO_LONG):** Opens or maintains a long position.
- **Action 2 (GO_SHORT/CLOSE):** Opens short or closes existing long position.

---

### Reward Function

Composite, dense reward at each timestep guiding towards risk-adjusted profitability:

$Reward_t = \Delta \text{SharpeRatio}_t - \text{TransactionCost}_t + \text{HoldingReward}_t$

**Components:**
- **Differential Sharpe Ratio:** Encourages improvements in risk-adjusted returns per step.
- **Transaction Cost Penalty:** Discourages unnecessary trades.
- **Holding Reward/Penalty:** Encourages cutting losses and running profits based on unrealized PnL.

---

## Training Loop and Optimization

### Sequential A2C Training Loop

**GPU Optimization:**  
- Utilize PyTorch's `torch.compile()` for GPU kernel fusion and efficiency.

**Loss Computation:**
- **Actor Loss:** $-\text{mean}(\log\pi(a|s) \times A)$
- **Critic Loss:** Smooth L1 loss between predicted and actual returns.
- **Entropy Bonus:** Encourage exploration, avoid premature convergence.

Combined Loss:  
$L_\text{total} = L_\text{actor} + c_1 L_\text{critic} - c_2 L_\text{entropy}$

---

## Model Architecture: HyenaActorCritic

**Forward Pass:**  
$\text{logits}, \text{value} = \text{model}(\text{state})$

## References

- Poli et al., *HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution*, NeurIPS 2023.
- Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning*, ICML 2016.
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438.
- Kenyon Research, *Deep Reinforcement Learning in Trading Algorithms*, 2018.
- Alexander Van de Kleut, *Actor-Critic Methods, A2C and GAE*, 2025.
- QuantInsti Blog, *Reinforcement Learning in Trading: Build Smarter Strategies with Q-Learning & Experience Replay*, 2025.

---

## Conclusion

This framework leverages advanced sequence modeling (HyenaDNA) and stable reinforcement learning (A2C with GAE) to create a robust, autonomous trading agent. Through meticulously designed observation and action spaces, along with a nuanced reward function, it achieves realistic an
