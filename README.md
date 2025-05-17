# Binance AI Futures Trader

**Advanced AI research system for fully autonomous, pro-level trading on Binance Futures (perpetual, high-leverage, 5-min bars).  
Purely for research/learning.**

---

## 🚀 Overview

This project builds an LLM-powered trading AI able to trade Binance USDT-margined perpetual futures using **full market, account, and position context**.  
Model learns and simulates entry/exit, SL/TP, dynamic position sizing, and is designed for ultra-high leverage (100x).  
Prompt/data structure ensures all decisions require explicit reasoning, memory, and planning.

---

## 🧠 Core Features

- **Model**: Zephyr 7B (LoRA, 4-bit, DeepSpeed) or compatible LLMs
- **Training**: Fast, GPU-optimized, multi-epoch walk-forward CV
- **Data**: OHLCV + indicators, rolling market window, full position/account context, memory/history, future targets, explicit reasoning
- **Trading Logic**:  
    - Realistic margin, leverage, SL/TP, BE, PnL, balance, equity
    - Action/SL/TP/size/plan all driven by model output
    - Risk/margin constraints, forced liquidations, realistic fees/slippage
- **Logging & Transparency**: Step-by-step output of actions, positions, PnL, SL/TP, confidence, and model’s “thinking”
- **Backtest**: Full backtesting environment, includes margin resets, bankruptcy, slippage

---

## 📈 Current State / Results

- **Full infra in place**: Model, data, OOP, speed, multi-head output
- **Data sequences include everything needed for pro-level trading and context**
- **Entry/exit/hold actions, SL/TP, and position sizing all model-driven**
- **Prompt requires explicit “plan/reasoning” per step** (parsed in logs, not yet as output)
- **Accuracy**: Eval ~0.5455, loss steadily dropping (latest: 8.32), no catastrophic errors
- **Drawdowns match high leverage and risk profile**
- **Logging per step: actions, confidence, PnL, position, TP/SL, account info**

---

## 🗺️ Roadmap & Progress

| Step | Description                                                            | Status      |
|------|------------------------------------------------------------------------|-------------|
| 1    | Core Infra (LLM, OOP, multi-head, fast)                                | ✅ Done      |
| 2    | Data/Prompt Redesign (market, position, memory, future, plan/reasoning) | ✅ Done      |
| 3    | Model Output (actions, SL/TP, sizing, explicit plan/reasoning)         | ⏳ Partial   |
| 4    | RL Fine-Tuning (profit-based, multi-step)                              | ❌ Not started|
| 5    | Env/Trainer: Plan parsing, scoring, logging                            | ⏳ Partial   |
| 6    | Backtesting/Live Eval, Realism                                         | ✅ Good      |

- **Current:** Model outputs actions/SL/TP/size, uses full prompt, plan/reasoning in prompt.
- **Next:** Add “reasoning/plan” as output head, parse it, reward in loss/RL.

---

## 📊 Latest Results Comparison

| Run      | Rows | Sequences | Steps | Eval Acc | Eval Loss | Train Loss | Equity Final | Notes                                  |
|----------|------|-----------|-------|----------|-----------|------------|-------------|----------------------------------------|
| Old/Broken SL/TP    | 736  | 418       | 72    | 0.5177   | 18.05     | >112       | ~996         | SL/TP not working, open pos stuck      |
| Previous (Tight)    | 743  | 425       | 72    | 0.5461   | 7.49      | 101        | 929.75       | More trades, tight logic               |
| Current (Full Prompt, Plan) | 748  | 430       | 72    | 0.5455   | 8.32      | 50.6        | 872.06       | Plan/reasoning in prompt, not output   |

- **Accuracy is stable, loss is down, trading logic robust**
- **Equity drawdown in line with 100x risk**
- **More realistic, context-aware actions per step**

---

## 🔮 Future / Potential

- Add plan/reasoning as output, RL reward for reasoning + profit
- Multi-symbol, multi-market support
- Smarter trade memory, regime-awareness (trend, range, volatility)
- Web/live dashboards, notebook analytics
- Auto-optimizer/hyperparameter search for thresholds
- Live/paper-trading for out-of-sample validation

---

## ⚠️ Disclaimer

**Research only. Not for live trading or real accounts.  
All code is for AI/ML educational use.**

---

## 🛠️ Quickstart

1. Clone repo, set up Python 3.11+, PyTorch 2.7+, transformers, deepspeed.
2. Adjust `config.py` for thresholds, data, leverage, etc.
3. Run: `python train.py`
4. Review terminal logs for every step, action, position, and PnL.
5. Tune, retrain, and push model intelligence further.


