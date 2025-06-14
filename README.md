# TradeLens

**See Your Trading in a New Light: Your Personal AI-Powered Trading Journal.**

> "The goal is not to be right all the time, but to be wrong for shorter periods." — TradeLens helps you understand *how* and *when* you are wrong, so you can build on what works.


---

## 📖 About The Project

For most individual investors, trading decisions often rely on intuition, news, and a vague "feel" for the market. We rarely perform a systematic review to answer critical questions:

* Am I actually beating the market?
* Am I better at short-term scalps or long-term holds?
* Which sectors are my "Alpha zone," and which are my blind spots?
* Does the size of my trade or the day of the week affect my win rate?

**TradeLens** is engineered to answer these questions. It is not a trading bot or a stock-picking service. It is a powerful **personal trading analysis engine**. By ingesting your (or mock) trading history, it combines real-world market data and technical indicators, then leverages data visualization, machine learning, and deep learning models to give you a deep, objective "check-up" on your own trading behavior.

This project serves as a complete, end-to-end case study in building a Python data science pipeline—from data generation and environment setup to advanced modeling and professional backtesting.

## ✨ Key Features

* **📈 Multi-Dimensional Performance Analysis (`comprehensive_analyzer.py`)**
    * **Holding Period:** Compares the win rate of your short-term vs. long-term trades.
    * **Sector Breakdown:** Identifies which market sectors you excel or struggle in.
    * **Trade Size-Category:** Analyzes how the financial size of your trades impacts performance.
    * **Timing Analysis:** Reveals performance patterns based on the day of the week you enter trades.
    * **Progress Visualization:** Plots your rolling win-rate over time to track skill development.

* **🧠 AI-Powered Prediction (`run_lstm_analysis.py`)**
    * **Sequence Generation:** Transforms market state (price, volume, and dozens of technical indicators) into time-series sequences for deep learning.
    * **LSTM Model:** Builds and trains a Long Short-Term Memory (LSTM) network, a model with "memory" designed to predict trade outcomes based on recent market patterns.
    * **Establishes a Baseline:** Provides a measurable deep learning benchmark that can be systematically improved.

* **📊 Professional Backtesting vs. Market (`backtester.py`)**
    * **Alpha and Beta:** Calculates your strategy's true Alpha (excess return vs. benchmark) and Beta (market volatility).
    * **Benchmark Comparison:** Visually plots your equity curve against a benchmark (e.g., S&P 500) to see exactly when and where you outperform or underperform.
    * **Professional Metrics:** Automatically generates dozens of institutional-grade metrics like Sharpe Ratio, Sortino Ratio, Max Drawdown, and more.

## 🛠️ Tech Stack

* **Core Language**: Python 3.11
* **Data Handling**: Pandas, NumPy
* **Market Data**: yfinance
* **Technical Analysis**: pandas-ta
* **Machine Learning**: Scikit-learn
* **Deep Learning**: TensorFlow (Keras)
* **Strategy Backtesting**: Backtesting.py
* **Visualization**: Matplotlib

## 🚀 Getting Started

This project was developed and validated in a clean Python 3.11 virtual environment. Following these steps is highly recommended for a smooth experience.

### Prerequisites

* **Python 3.11** must be installed on your system. If you don't have it, you can install it via [Homebrew](https://brew.sh/) (`brew install python@3.11`) or from the official [Python website](https://www.python.org/downloads/).
* **Git** for cloning the repository.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Shuai-fa/TradeLens.git](https://github.com/Shuai-fa/TradeLens.git)
    cd TradeLens
    ```

2.  **Create and activate a Python 3.11 virtual environment:**
    *(This is the most critical step to ensure dependency compatibility)*
    ```bash
    # Create the virtual environment using your python3.11 installation
    python3.11 -m venv venv

    # Activate the environment (on macOS/Linux)
    source venv/bin/activate
    ```
    Upon successful activation, your terminal prompt will be prefixed with `(venv)`.

3.  **Install all dependencies:**
    Inside the activated `(venv)` environment, run the following command to install all required libraries.
    ```bash
    # Upgrade pip first
    python -m pip install --upgrade pip

    # Install all required libraries
    python -m pip install pandas numpy yfinance pandas_ta scikit-learn tensorflow matplotlib backtesting
    ```

### Usage

You can now run any of the three main analysis modules. Make sure your `(venv)` is active.

* **To run the comprehensive performance analysis:**
    ```bash
    python comprehensive_analyzer.py
    ```
    *This will print 4 analytical reports to the console and display a pop-up chart showing your performance trend.*

* **To train the LSTM deep learning model:**
    ```bash
    python run_lstm_analysis.py
    ```
    *This will print the model's architecture and begin the training process, which may take several minutes.*

* **To run the professional backtest against the S&P 500:**
    ```bash
    python backtester.py
    ```
    *This will print professional quantitative metrics to the console and display an interactive plot comparing your strategy's equity curve to the benchmark.*

## 🗺️ Roadmap

* [ ] **Enhance Feature Library:** Integrate fundamental data (e.g., P/E, P/S ratios) or alternative data (e.g., sentiment analysis) as model features.
* [ ] **Advanced Model Tuning:** Implement hyperparameter optimization for the LSTM model using tools like KerasTuner or Optuna.
* [ ] **Broker API Integration:** Allow users to automatically import their real trading history from popular brokerage APIs.
* [ ] **Web Interface:** Develop a user-friendly web dashboard using Flask or Django to display analysis results interactively.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
