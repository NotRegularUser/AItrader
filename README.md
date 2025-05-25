# High-Risk/Reward AI Futures Trader — HyenaDNA

**Advanced research framework for autonomous, high-leverage crypto trading AI.**  
Powered exclusively by [HyenaDNA](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf) for full-sequence market modeling.  
> *For research and experimentation. Not for live or production use.*

---

## 🚦 Project Overview

- **Single-Model, Zero-RL:** No PPO, A2C, LoRA, or quantization. Pure supervised sequence modeling on stepwise simulated futures market.
- **Full Market Simulation:** 100x leverage, real margin/capital/fee logic, all positions and trades executed realistically, with forced resets on low balance.
- **Live-Accurate Data Feeding:** Rolling window of 500+ bars per step. Each timestep, the AI gets all technicals, price history, and full account state.
- **No Batching, No Shuffling:** Strict 1-bar-at-a-time simulation. All training and inference happen exactly as a real trader would experience the market.

---

## 🧩 Code Structure

- **`config.py`:** Central hub for all hyperparameters, thresholds, technicals, and trading rules.
- **`data_utils.py`:** Loads, processes, and windows Binance data. Generates all technicals (EMA, RSI, MACD, Bollinger, ATR, etc.). Strict walk-forward splits.
- **`env.py`:** High-precision simulation—margin, fees, PnL, liquidation, resets, slippage, forced closure, position/account context.
- **`model.py`:** Direct HyenaDNA integration, multi-head outputs (actions, multi-horizon forecasts, indicators, risk metrics, etc.), no quantization.
- **`losses.py`:** Handles full multi-head supervised loss. Every head and auxiliary output gets a real loss target.
- **`train.py`:** Market simulation and training loop. Feeds the environment 1 bar at a time, stepwise logging, walk-forward CV, all device/GPU optimizations.
- **`metrics.py`, `test.py`:** Evaluation, traceability, sanity checks.

---

## 🏎️ HyenaDNA Utilization & Optimization

- **Full Sequence Input:** HyenaDNA gets 500+ time steps (candles) of full-featured market/account context per action—true long-term memory.
- **No RL Shortcuts:** No quantization, no LoRA, no reward shaping—just direct supervision on true market simulation.
- **GPU-Optimized:** All operations run on GPU. Data never leaves device during inference/train. BF16 precision for speed and expressiveness.
- **Strict Sequentialism:** All "batches" are single-bar; no random sampling or shuffling. True stepwise training, *not* random forecasting.

---

## 📊 Key Features

- **Realistic Trade Simulation:** 100x leverage, position sizing, margin, liquidation, TP/SL, fees, and forced resets at low equity.
- **Multi-Head Output:** Model predicts actions, multi-horizon returns, risks, quality/confidence, position metrics, and all major technicals.
- **Robust Logging:** Per-step logs include actions, confidence, trade PnL, resets, reasonings, and all environment triggers.
- **CV and Validation:** Walk-forward validation splits—train/test like a real trading bot would experience time, no data leaks.
- **Immediate Live-Data Swap:** Change one line in `data_utils.py` to switch from historical to live Binance data.

---

## ⚡ Performance

- **Simulation Speed:** ~0.02s per bar (RTX 5070), 40–60% GPU utilization, 4–6GB VRAM with full context/features.
- **Scalable:** Expand window/features as GPU allows. All logic adapts to larger context or more complex simulations.
- **No CPU/IO Bottleneck:** Data stays on GPU, all computations are device-first.

---

## 🔬 Latest Observations

- **AI still in learning phase:** Early runs show difficulty with multi-step trade reasoning and reward assignment; model tends to open/close positions prematurely.
- **Model coverage:** All output heads are now covered with supervised losses; all targets must be defined for proper gradient flow.
- **Environment logic:** Precision in trade/position, liquidation, margin resets—no unrealistic infinite equity or skipping.

---

## 🛣️ Roadmap & Next Steps

- **To do:**  
    - Reinforcement learning integration (PPO/A2C) for explicit reward handling.
    - Trajectory-aware modeling (Decision Transformers) for multi-step optimization.
    - Multi-agent support, more robust risk/position management.
    - Web dashboard, visualizations, and live streaming.
- **Current scope:**  
    - Pure HyenaDNA supervised learning, no RL, quant, or LoRA.
    - Single-agent, direct simulation.

---

## 🏁 Quickstart

1. **Install:** Python 3.11+, PyTorch 2.7+, `transformers`, `deepspeed`.
2. **Configure:** Edit `config.py` for all model/environment parameters.
3. **Run:** `python train.py` to launch full simulation and training.
4. **Review:** Check `/checkpoints` for logs, metrics, and traceability.
5. **Switch to live:** Swap to live Binance feed in `data_utils.py` for real-time testing.

---

## ⚠️ Disclaimer

**For AI/ML research only.  
Not financial advice, not for real-money trading.  
All results are simulated.**

---
