# run_analysis.py (The Ultimate Sanitization Version)
# This version adds a forceful index reset before the final merge.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Import libraries with helpful error messages ---
try:
    import yfinance as yf
    import pandas_ta as ta
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except ImportError as e:
    print(f"❌ A necessary library is missing: {e.name}")
    print("Please run this command: pip3 install pandas numpy yfinance pandas_ta scikit-learn")
    exit()

# ==============================================================================
# STEP 1: DATA GENERATION
# ==============================================================================
def generate_mock_trades(num_trades=200):
    """Generates a DataFrame of simulated trade records."""
    print("STEP 1: Generating mock trade data...")
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'GS', 'BAC', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    trade_list = []
    for _ in range(num_trades):
        ticker = random.choice(TICKERS)
        entry_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        exit_date = entry_date + timedelta(days=random.randint(5, 120))
        entry_price = round(random.uniform(50.0, 500.0), 2)
        is_win = np.random.rand() < 0.55
        price_change_percent = random.uniform(0.05, 0.40)
        exit_price = round(entry_price * (1 + price_change_percent if is_win else 1 - price_change_percent), 2)
        trade_list.append({'Ticker': ticker, 'EntryDate': entry_date, 'ExitDate': exit_date, 'EntryPrice': entry_price, 'ExitPrice': exit_price})
    df = pd.DataFrame(trade_list)
    df['IsWin'] = (df['ExitPrice'] > df['EntryPrice'])
    print(f"✅ Generated {len(df)} mock trades.\n")
    return df

# ==============================================================================
# STEP 2: FEATURE ENGINEERING
# ==============================================================================
def add_features_to_trades(trades_df):
    """Enriches trades by processing each ticker individually."""
    print("STEP 2: Enriching trades with market data...")
    
    unique_tickers = trades_df['Ticker'].unique().tolist()
    start_date = trades_df['EntryDate'].min() - timedelta(days=250)
    end_date = trades_df['EntryDate'].max() + timedelta(days=1)
    
    all_market_data_with_ta = []

    print(f"  -> Downloading and processing data for {len(unique_tickers)} tickers...")
    for i, ticker in enumerate(unique_tickers):
        market_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        if market_data.empty: continue
        
        market_data.rename(columns=str.lower, inplace=True)
        if 'close' not in market_data.columns: continue

        market_data.ta.sma(length=50, append=True)
        market_data.ta.sma(length=200, append=True)
        market_data.ta.rsi(length=14, append=True)
        market_data.ta.macd(append=True)
        
        market_data['Ticker'] = ticker
        all_market_data_with_ta.append(market_data.reset_index())

    if not all_market_data_with_ta:
        print("❌ Failed to process any market data.")
        return None

    full_market_data = pd.concat(all_market_data_with_ta)
    
    # --- THE ULTIMATE FIX: Forcefully sanitize the index before merging. ---
    full_market_data.reset_index(drop=True, inplace=True)
    # --------------------------------------------------------------------

    print("  -> Merging trade data with features...")
    trades_df['merge_date'] = pd.to_datetime(trades_df['EntryDate']).dt.date
    full_market_data['merge_date'] = pd.to_datetime(full_market_data['Date']).dt.date
    
    enriched_trades = pd.merge(
        trades_df,
        full_market_data,
        on=['Ticker', 'merge_date'],
        how='left'
    ).drop(columns=['merge_date'])
    
    print("✅ Feature engineering complete.\n")
    return enriched_trades

# ==============================================================================
# STEP 3: MODEL TRAINING & EVALUATION
# ==============================================================================
def train_and_evaluate_model(featured_df):
    """Trains a model and prints its performance."""
    print("STEP 3: Training and evaluating predictive model...")
    
    feature_columns = ['sma_50', 'sma_200', 'rsi_14', 'macdh_12_26_9']
    df_cleaned = featured_df.dropna(subset=feature_columns)

    if df_cleaned.empty:
        print("❌ No data available for training. This could be due to trades on non-trading days or failed TA calculations.")
        return

    y = df_cleaned['IsWin']
    X = df_cleaned[feature_columns]

    print(f"  -> Preparing to train on {len(X)} trades.")