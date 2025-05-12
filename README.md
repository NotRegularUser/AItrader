# 📈 AI Trading Bot Project (Binance Futures)

An advanced AI-driven trading bot designed to trade Binance Futures using 5-minute candle data with high leverage (100×). The bot leverages state-of-the-art deep learning (Transformer-based models with LoRA optimization) and comprehensive technical indicators for intelligent decision-making.

---

## 🚀 Project Overview

### 📂 Project Structure
```plaintext
ai_trading_bot/
├── configs/
│   ├── settings.py       # Configurations & hyperparameters
│   └── ds_config.json    # DeepSpeed optimization settings
├── data/
│   ├── binance_client.py # Fetch historical data from Binance
│   ├── indicators.py     # Compute technical indicators
│   └── dataset.py        # Prepare datasets and sequences
├── environment/
│   └── trading_env.py    # Trading simulation environment
├── models/
│   ├── multitask_model.py # Multi-task prediction model
│   └── trainer.py         # Customized trainer logic
├── utils/
│   ├── logging_utils.py   # Logging utilities
│   └── helpers.py         # General helper functions
├── train.py               # Main entry point for training
└── README.md              # Project details and instructions
```


## 🧠 AI Model Details
- **Base Model:** [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **Quantization:** BitsAndBytes (4-bit NF4 quantization)
- **LoRA Optimization:** Low-rank adaptation to fine-tune specific layers efficiently
- **Tasks:**
  - Direction prediction (long or short)
  - Expected return estimation
  - Adaptive stop-loss and take-profit calculation

---

## 📊 Technical Indicators
The bot uses the following indicators calculated on-the-fly:
- EMA (Exponential Moving Average)
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- ATR (Average True Range)
- ADX (Average Directional Index)

---

## 🎯 Trading Strategy
The trading logic dynamically manages:
- **Risk Management:** ATR-based stop-loss & take-profit levels, adaptive sizing, and volatility regime adjustments.
- **Entry Conditions:** High-confidence trade signals (minimum 95% probability, clear directional margin).
- **Exit Logic:** Automatic exits on hitting SL/TP levels, plus dynamic trailing stops for profit locking.

---

## 🧩 Project Optimization & Scalability
- **Modular Design:** Easy addition/removal of indicators, models, or strategies.
- **Optimized Data Handling:** JIT-compiled numeric calculations (Numba), efficient batching, and memory management.
- **Training Speed & Efficiency:** Utilizes DeepSpeed stage 2 optimizations, FP16/BF16 precision training, and optimized batching strategies.

---

## 🔧 Installation & Setup
1. **Install Dependencies:**
```bash
pip install -r requirements.txt
