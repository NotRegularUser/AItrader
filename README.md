# Mamba-SSM Autonomous Trading Agent

**A state-of-the-art deep reinforcement learning system for cryptocurrency perpetual futures trading**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This project implements an **autonomous BTC/USDT perpetual futures trading agent** combining:

- **Mamba SSM** (State Space Model) backbone for efficient long-sequence modeling
- **Self-supervised pretraining** on market structure before RL fine-tuning  
- **Mixture-of-Experts (MoE)** prediction heads with causal information flow
- **Kelly-optimal position sizing** with uncertainty quantification
- **100x leverage** simulation with realistic fees, funding, and liquidation

The agent learns its own trading strategies through self-supervision rather than hard-coded rules.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│  OHLCV (5m) → Technical Indicators → Order Flow → Sentiment         │
│      ↓              ↓                    ↓            ↓             │
│   50+ Features: RSI, MACD, BB, ADX, CVD, Fibonacci, Funding Rate    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     MAMBA SSM BACKBONE                              │
├─────────────────────────────────────────────────────────────────────┤
│  • 2.8B parameter state-space model (selective scan)                │
│  • O(L) complexity vs O(L²) for Transformers                        │
│  • 128-bar context window (10+ hours at 5m intervals)               │
│  • Cross-horizon attention over [1, 3, 6, 12, 24, 48] bar forecasts │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│              CAUSAL HEAD HIERARCHY (Genius Architecture)            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │  Regime  │ →  │Volatility│ →  │  Return  │ →  │ MAE/MFE  │     │
│   │   MoE    │    │   MoE    │    │ Quantile │    │  Uncert. │     │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│        │               │               │               │           │
│        └───────────────┴───────────────┴───────────────┘           │
│                              ↓                                      │
│                    ┌─────────────────┐                              │
│                    │  Kelly Position │                              │
│                    │     Sizing      │                              │
│                    └────────┬────────┘                              │
│                             ↓                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │Direction │    │  SL/TP   │    │  Policy  │    │Confidence│     │
│   │ Ordinal  │    │  Head    │    │  (A2C)   │    │   Head   │     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Innovations

| Component | Description |
|-----------|-------------|
| **MoE Heads** | 4 experts with top-2 routing + load balancing |
| **Quantile Returns** | Distributional prediction [5%, 25%, 50%, 75%, 95%] |
| **Ordinal Direction** | 5-class: strong_down → weak_down → neutral → weak_up → strong_up |
| **Heteroscedastic MAE/MFE** | Mean + aleatoric uncertainty for risk estimation |
| **Kelly Sizing** | Optimal position sizing with drawdown constraints |
| **ICM Curiosity** | Intrinsic motivation for exploration |
| **Contrastive Learning** | Pattern clustering via InfoNCE loss |
| **Multi-Timeframe (MTF)** | Hierarchical fusion of 5m, 15m, 1h, 4h contexts |

---

## Features

### Multi-Timeframe (MTF) Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL MTF ENCODER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   4h TF ────→ [Encoder] ────┐                                       │
│   (12 bars)                 │                                       │
│                             ↓                                       │
│   1h TF ────→ [Encoder] ───→ Cross-TF ───┐                          │
│   (24 bars)                  Attention    │                         │
│                                          ↓                          │
│   15m TF ───→ [Encoder] ───→ Cross-TF ──→ Fusion ───→ Output        │
│   (48 bars)                  Attention                              │
│                                ↑                                    │
│   5m TF ────→ [Encoder] ──────┘                                     │
│   (128 bars)   + Mamba Backbone                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Information Flow**: Higher timeframes (4h) provide macro context that flows down to lower timeframes (5m) via cross-attention, enabling the model to:
- See the "bigger picture" while trading on short timeframes
- Align short-term trades with longer-term trends
- Avoid counter-trend entries during strong macro moves

### Data Pipeline
- **Multi-timeframe OHLCV**: 5m, 15m, 1h, 4h aligned data from Binance
- **Real-time Binance API** integration for OHLCV, funding rates, long/short ratios
- **50+ engineered features**: RSI, MACD, Bollinger Bands, ADX, ATR, OBV per timeframe
- **Order flow analysis**: CVD (Cumulative Volume Delta), large trade detection
- **Fibonacci retracement** levels with proximity scoring
- **Triple barrier labeling** (Lopez de Prado methodology)

### Trading Environment
- **100x leverage** perpetual futures simulation
- **Realistic costs**: 4 bps taker fees, slippage, 8-hour funding rates
- **Liquidation mechanics** with maintenance margin
- **Dynamic position sizing**: $100-$300 margin ($10K-$30K notional)
- **Risk management**: Adaptive SL/TP within safe bounds (0.15%-0.40% SL)

### Model Architecture
- **Mamba-2.8B** backbone with selective state spaces
- **Hierarchical MTF encoder** with cross-timeframe attention
- **Cross-horizon attention** over 6 forecast horizons
- **LSTM memory** for pattern persistence across episodes
- **Causal head fusion** ensuring logical information flow
- **Uncertainty quantification** for confident decision-making

### Training Pipeline
- **Self-supervised pretraining** (2000+ steps) on market patterns
- **A2C reinforcement learning** with GAE advantage estimation
- **Multi-task auxiliary losses** (direction, volatility, regime, MAE/MFE)
- **Experience replay** with non-lookahead sampling
- **Curiosity-driven exploration** via ICM

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MambaSSMTrader.git
cd MambaSSMTrader

# Create environment
conda create -n mamba-trader python=3.10
conda activate mamba-trader

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers mamba-ssm pandas numpy ta requests

# Optional: CUDA-optimized kernels
pip install causal-conv1d>=1.1.0
```

### Requirements
- Python 3.10+
- PyTorch 2.0+ (CUDA 12.1 recommended)
- 12GB+ VRAM (RTX 3080/4070/5070 or better)
- Binance API keys (optional, for live data)

---

## Quick Start

```python
# Set API keys (optional)
export BINANCE_API_KEY="your_key"
export BINANCE_SECRET="your_secret"

# Run training
python train.py
```

### Training Phases

1. **Data Loading** (~2 min): Downloads OHLCV, computes indicators
2. **Pretraining** (~30 min): Self-supervised learning on market patterns  
3. **RL Training** (~2-4 hours): Policy optimization with environment interaction
4. **Evaluation**: Continuous monitoring of Sharpe ratio, win rate, drawdown

---

## Project Structure

```
MambaSSMTrader/
├── config.py          # All hyperparameters (150+ settings)
├── data_loader.py     # Binance API, feature engineering, labeling
├── environment.py     # Trading simulation with 100x leverage
├── model.py           # Mamba backbone + MoE heads
├── train.py           # Pretraining + RL training loop
└── trading_ai/
    ├── data/          # Cached market data
    ├── saved_models/  # Checkpoints
    └── results/       # Training logs
```

---

## Configuration Highlights

```python
# Leverage & Position Sizing
LEVERAGE = 100                    # 100x perpetual futures
MIN_POSITION_USD = 100.0          # $100 minimum margin
# → Notional: $10,000 - $30,000 per trade

# Multi-Timeframe (MTF)
USE_MTF = True                              # Enable multi-timeframe
MTF_TIMEFRAMES = ["5m", "15m", "1h", "4h"]  # Timeframes to use
MTF_CONTEXT_BARS = {"5m": 128, "15m": 48, "1h": 24, "4h": 12}
MTF_USE_CROSS_TF_ATTENTION = True           # Cross-timeframe attention

# Risk Management
RISK_VOL_MIN_SL = 0.0015          # 0.15% min stop loss
RISK_VOL_MAX_SL = 0.004           # 0.40% max stop loss  
RISK_VOL_MIN_TP = 0.003           # 0.30% min take profit
RISK_VOL_MAX_TP = 0.012           # 1.20% max take profit

# Self-Supervised Learning
USE_ICM_CURIOSITY = True          # Intrinsic motivation
USE_CONTRASTIVE = True            # Pattern clustering
USE_HORIZON_ATTENTION = True      # Multi-timeframe reasoning
USE_ADVANCED_HEADS = True         # MoE + Kelly + Ordinal

# Training
PRETRAIN_STEPS = 2000             # SSL pretraining iterations
GAMMA = 0.95                      # Short horizon for scalping
HORIZONS = [1,3,6,12,24,48]       # 5m to 4h forecast windows
```

---

## Performance Metrics

The model optimizes for:

| Metric | Target | Description |
|--------|--------|-------------|
| **Sharpe Ratio** | > 2.0 | Risk-adjusted returns |
| **Win Rate** | > 55% | Profitable trades / total |
| **Max Drawdown** | < 20% | Worst peak-to-trough |
| **Profit Factor** | > 1.5 | Gross profit / gross loss |
| **Trade Frequency** | 4-8/day | Quality over quantity |

---

## Technical Details

### Mamba State Space Model

```
h_t = Āh_{t-1} + B̄x_t
y_t = Ch_t + Dx_t

Where:
- A, B, C, D are input-dependent (selective scan)
- O(L) complexity vs O(L²) for attention
- Efficient hardware utilization via scan operations
```

### Kelly Criterion Position Sizing

```
f* = (p·b - q) / b

Where:
- f* = optimal fraction of capital
- p = probability of win
- b = win/loss ratio  
- q = probability of loss (1-p)

With drawdown constraint:
f_safe = min(f*, max_dd / expected_loss)
```

### Advantage Estimation (GAE)

```
Â_t = Σ (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

---

## Comparison to Alternatives

| Feature | This Project | QuantConnect | Freqtrade | Academic RL |
|---------|--------------|--------------|-----------|-------------|
| Architecture | Mamba SSM | Traditional | Rule-based | LSTM/Transformer |
| Head Design | MoE + Causal | Linear | N/A | Linear |
| Position Sizing | Kelly-optimal | Fixed | Fixed | Fixed |
| Uncertainty | Distributional | Point estimate | N/A | Point estimate |
| Exploration | ICM + Contrastive | N/A | N/A | ε-greedy |
| Live Trading | Simulation | ✓ | ✓ | Simulation |

---

## Future Roadmap

- [ ] PPO algorithm (more stable than A2C)
- [ ] Multi-timeframe hierarchical input (1m, 5m, 1h, 4h)
- [ ] Walk-forward backtesting module
- [ ] Live trading integration via CCXT
- [ ] Multi-asset portfolio management
- [ ] Distributed training with Ray

---

## References

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023
- Lopez de Prado, *Advances in Financial Machine Learning*, 2018
- Pathak et al., *Curiosity-driven Exploration by Self-Supervised Prediction*, ICML 2017
- Kelly, *A New Interpretation of Information Rate*, Bell System Technical Journal, 1956
- Schulman et al., *High-Dimensional Continuous Control Using GAE*, ICLR 2016

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Author

Developed as a demonstration of advanced ML/RL techniques applied to quantitative finance.

**Skills demonstrated:**
- Deep Learning (PyTorch, Transformers, State Space Models)
- Reinforcement Learning (A2C, GAE, Curiosity-driven Exploration)
- Financial Engineering (Options Greeks, Kelly Criterion, Risk Management)
- Software Architecture (Modular design, Configuration management)
- Data Engineering (API integration, Feature engineering, Time series)
