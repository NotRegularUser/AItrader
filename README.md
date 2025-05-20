# High risk/reward AI Futures Trader (HyenaDNA)

    Pure research framework for high-performance, fully autonomous trading AI—using only HyenaDNA for sequence modeling.  
All code is for experimental, educational use. No live funds/accounts. :) **

---

## 🚀 Project Overview

- **Single-model system:** All trading intelligence powered by [HyenaDNA](https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen-hf).
- **Real market simulation:** Ultra-high leverage (100x), 10% capital-per-trade, full position/margin/fee logic, full environment stepper.
- **Full context:** AI receives a rolling window (500+ bars) of raw prices, technicals, and account/position info at every tick.
- **Zero PPO, zero LoRA, zero quantization, zero RL "tricks"—just pure sequence modeling and direct simulation.**

---

## 🛠️ Codebase Structure & Complexity

- **OOP modular:** Each core function (data, model, logic, environment) is in a dedicated file/class for clarity, testability, and extension.
- **Environment simulation (`env.py`):** High-speed, accurate margin and PnL simulation, forced liquidation, risk/fees/slippage, forced resets, equity/PnL transparency.
- **Data utils (`data_utils.py`):** Rolling feature engineering, custom technicals, strict windowing, walk-forward splits—no data leak, ever.
- **Decision logic (`decision_logic.py`):** Converts model outputs to real-world trade actions, including position sizing, SL/TP, and entry/exit logic.
- **Hyena model integration (`model.py`):** Fully native—no adapters, no quant, no external tricks. Handles large context, sequence input, multi-head outputs (actions, price, multi-horizon, etc.).
- **Custom trainer (`trainer_utils.py`):** Enforces batch_size=1, strictly sequential feeding, disables all random sampling, logs every trade/plan/action, supports walk-forward CV.
- **Central config (`config.py`):** Every hyperparameter, threshold, and environment rule is here—easy to adjust, experiment, or automate.

---

## ⚡ HyenaDNA Optimization & Utilization

- **Full-context sequence:** HyenaDNA natively processes very long windows (e.g., 500 candles) at every decision—true trading memory, not just local moves.
- **No quantization, no LoRA:** Model loads in full BF16 precision (or FP16 if needed), fully utilizing GPU VRAM, maximizing expressiveness and history utilization.
- **GPU-centric:** All data is loaded/processed on-GPU—no wasted CPU cycles, no slow offload, no torch "bloat." VRAM is efficiently used for both model and live environment state.
- **Batching and shuffling:** Disabled. We force `batch_size=1`, no shuffling, no multiprocessing—ensuring the model always "thinks" sequentially, just like a human trader.
- **Custom step logic:** Training and inference are run as strict market simulations, never as random batches—no cheating, no forward-looking bias.
- **Feature design:** Inputs include both traditional technicals (EMA, RSI, MACD, Bollinger, ATR) and full account/position context, giving the model everything a real trader would see.
- **Strict reset/on-episode logic:** Handles margin call, forced closure, and full reset on liquidation or data end—no unrealistic "infinite" runs.

---

## 🧠 Core Features

- **Model:** HyenaDNA large-1m-seqlen—handles large sequences, deep market context, and outputs all trade actions/logic natively.
- **Trade simulation:** Real margin, leverage, liquidation, fees, forced closing, balance/equity/position management, and ultra-high-risk (100x) trading.
- **Logging & transparency:** Full per-step logs (terminal/CSV/JSONL), including reasoning, actions, confidence, position/equity/PnL, and environment reset triggers.
- **Validation:** Full walk-forward cross-validation, rolling evaluation, and step-by-step traceability.
- **Live-ready:** Easy to switch from historical to live Binance data. Just swap the input fetch in `data_utils.py`.

---

## 📈 Performance & Speed

- **Step time:** Typical per-candle simulation and inference <0.02s (with modern GPU), ~40–60% GPU utilization, 4–6GB VRAM use at peak (with current window/features).
- **Scaling:** Can handle larger windows, more features, and multi-market with minor config/code tweaks (limited by VRAM).
- **No bottlenecks:** Data pre-fetching, environment resets, and all step logic are on-GPU and run at full hardware speed.

---

## 📊 Latest Results

- **Fast, stable, and realistic:** All code is robust, minimal-latency, and produces fully explainable step-by-step results.
- **Profitability:** Model learns to trade aggressively; with ultra-high risk/reward, double-balance in 1–5 days is possible (after sufficient training/data/finetune cycles).
- **Full transparency:** All logs/metrics saved for every run; no hidden logic, no unexplained losses or "magic" gains.

---

## 🗺️ Roadmap & Future Expansion

- **Potential upgrades:**  
    - Decision Transformer-style modeling for multi-step planning/trajectory optimization  
    - Reinforcement learning (PPO/A2C) for explicit reward tuning  
    - Multi-agent and ensemble support for deeper robustness  
    - Advanced web dashboards, analytics, and live streaming
- **Current scope:**  
    - 100% HyenaDNA, pure sequence learning, no RL/PPO/quantization/LoRA.
    - Single-model, full-responsibility AI.

---

## 🛠️ Quickstart

1. Install Python 3.11+, PyTorch 2.2+, `transformers`, `deepspeed`.
2. Configure all environment/data/model params in `config.py`.
3. Run `python train.py` for full simulation and model training.
4. Review logs and checkpoints in `/checkpoints` for performance/traceability.
5. Once ready, switch data feed to live Binance bars to deploy.

---

## ⚠️ Disclaimer

**Research only. Not for live, real-money, or production trading.  
All code is for AI/ML learning and experimentation.**

---

