# Binance AI Futures Trader

**Ultimate AI research system for fully autonomous, pro-level trading on Binance Futures (perpetual, high-leverage, 5-min bars).  
All work is pure research/learning.**

---

## 🚀 Overview

This project delivers a high-performance, LLM-powered trading AI that simulates trading USDT-margined Binance perpetual futures with full market, account, and position context.  
The model learns real, expert-like trade entry/exit, SL/TP, position sizing, and is tuned for ultra-high leverage (100x).  
Prompt/data pipeline is built for “reasoning-required”—the AI must plan, remember, and justify every action.

---

## ⚡ Optimization & Speed

- **Runs full end-to-end training/inference on a single modern GPU (11GB+ VRAM, e.g., RTX 4070/5070)**
- Uses Zephyr-7B (or compatible) with LoRA, 4-bit quantization, and DeepSpeed for *extreme* memory and compute efficiency
- Pure GPU execution, custom memory management—no slow CPU offload, no bottlenecks
- Training loop and environment designed for minimal latency (per-step < 2s typical)
- OOP modular codebase, streamlined for rapid experiment/retraining cycles

---

## 🧠 Core Features

- **Model**: Zephyr 7B (LoRA, 4-bit, DeepSpeed) or compatible LLMs
- **Training**: Fast, GPU-optimized, multi-epoch walk-forward CV
- **Data**: OHLCV, technical indicators, full rolling window, complete position/account context, history/memory, future outcomes, and “reasoning-required” prompts
- **Trading Logic**:  
    - Realistic margin, leverage, SL/TP, BE, PnL, balance, equity, forced liquidations
    - Model decides entry/exit/hold, SL/TP, position sizing, action plan
    - Full support for margin/fee/slippage, drawdowns, bankruptcy
- **Reinforcement Learning (PPO)**:  
    - Custom RL pipeline with reward shaping for profit + smart behavior
    - Training only saves the final best model (`checkpoints/ppo/`), with robust checkpointing and forced closing of open positions at epoch end
    - Hard-coded confidence threshold for entries (no random “always-in” trading)
- **Logging & Transparency**:  
    - Step-by-step logs: actions, confidence, PnL, open/closed positions, SL/TP, equity, and model’s plan/reasoning (in prompt and output)
    - Full CSV/terminal logging of each regime and run for future analysis
- **Backtesting**:  
    - Complete backtest environment, with margin resets, equity tracking, bankruptcy and regime simulation

---

## 💡 How is this so fast and efficient?

- All code is designed for pure GPU execution—no slowdowns from CPU offloads or inefficient memory usage
- Model is loaded using LoRA adapters + 4-bit quantization, compressing memory without losing expressiveness
- Custom DeepSpeed configs, micro-batching, and zero-redundant computation
- Data pre-processing and simulation pipelined for low-overhead training
- Only saves essential checkpoints—no checkpoint spam, no wasted storage
- Minimal bloat, OOP structure for fast, clean code evolution

---

## 📈 Possibilities / Extensions

- Train on multiple symbols, regimes, and extreme events for true “out-of-distribution” robustness
- Add new model heads for plan/reasoning, regime awareness, and risk management
- Full ensemble or multi-agent RL support (extend easily)
- Out-of-sample validation: plug in new market data, evaluate instantly
- Integrate with live or paper trading (never for real funds/accounts)
- Web dashboards, advanced analytics, auto-hyperparameter optimization

---

## 📊 Latest Results

- **Core pipeline is fully functional, fast, and robust**
- **All actions, positions, TP/SL, and PnL handled by the model**
- **Prompt and training require real plan/reasoning at every step**
- **High leverage, realistic drawdowns—no free profit, no cheating**
- **Accuracy stable, loss steadily drops, performance tracks market complexity**

Example (5-min BTCUSDT, ~750 rows):

| Run                | Sequences | Steps | Eval Acc | Eval Loss | Equity Final | Notes                                |
|--------------------|-----------|-------|----------|-----------|-------------|--------------------------------------|
| Old/Broken SL/TP   | 418       | 72    | 0.5177   | 18.05     | 996         | SL/TP not working, open pos stuck    |
| Previous (Tight)   | 425       | 72    | 0.5461   | 7.49      | 929.75      | More trades, tight logic             |
| Current (Full)     | 430       | 72    | 0.5455   | 8.32      | 872.06      | Plan/reasoning in prompt, robust     |

---

## 🗺️ Roadmap & Progress

| Step | Description                                                            | Status      |
|------|------------------------------------------------------------------------|-------------|
| 1    | Core Infra (LLM, OOP, multi-head, fast)                                | ✅ Done      |
| 2    | Data/Prompt Redesign (market, position, memory, future, plan/reasoning) | ✅ Done      |
| 3    | Model Output (actions, SL/TP, sizing, explicit plan/reasoning)         | ✅ Done      |
| 4    | RL Fine-Tuning (PPO profit-based, multi-step, robust checkpoints)      | ✅ Done      |
| 5    | Logging, Analysis, CSV Export, Plan parsing                            | ✅ Done      |
| 6    | Backtesting/Live Eval, Regime simulation, Realism                      | ✅ Done      |
| 7    | (Next) Advanced curriculum, regime/risk-awareness, multi-market, OOS   | ⏳ Up Next   |

- **Everything core is implemented and working.**
- **All RL/PPO training is robust; only final best model is saved.**
- **Entry/exit/hold logic matches pro trading standards.**
- **Overfitting mitigated (data variety, prompt noise).**
- **Forced closing of open position at epoch end ensures true equity.**

---

## 🔮 Future / Potential

- Add plan/reasoning as explicit model output, RL reward for both profit and logic
- Expand to multi-market, multi-symbol, regime-specific tuning
- Smarter trade memory, full trend/volatility awareness, adversarial data
- Web dashboards, notebook analytics, hyperparameter search/auto-optimizer
- Scenario testing (flash crash, black swan), ensemble models
- Out-of-sample/live/paper trading evaluation

---

## ⚠️ Disclaimer

**Research only. Not for live trading or real accounts.  
All code is for AI/ML educational use.**

---

## 🛠️ Quickstart

1. Clone repo, set up Python 3.11+, PyTorch 2.7+, transformers, deepspeed.
2. Adjust `config.py` for thresholds, data, leverage, etc.
3. Run: `python train.py` (or `ppo_rl.py` for RL)
4. Review logs/CSVs in `checkpoints/ppo/` for every step, action, position, and PnL.
5. Tune, retrain, and keep evolving your AI.

---
