# comprehensive_analyzer.py
# An all-in-one script for deep, multi-faceted trade performance analysis.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Import necessary libraries ---
try:
    import yfinance as yf
    import pandas_ta as ta
    # We need matplotlib for plotting our progress
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"❌ A necessary library is missing: {e.name}")
    print("Please ensure you are in your virtual environment and have run the pip install command.")
    exit()

# ==============================================================================
# DATA PREPARATION (Reusing functions from our previous script)
# ==============================================================================
def generate_mock_trades(num_trades=200):
    """Generates a DataFrame of simulated trade records."""
    print("STEP 1: Generating mock trade data...")
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'GS', 'BAC', 'XOM', 'CVX', 'JNJ', 'PFE', 'UNH']
    trade_list = []
    for _ in range(num_trades):
        ticker = random.choice(TICKERS)
        sector_map = {'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'], 'Finance': ['JPM', 'GS', 'BAC'], 'Energy': ['XOM', 'CVX'], 'Healthcare': ['JNJ', 'PFE', 'UNH']}
        sector = [s for s, t in sector_map.items() if ticker in t][0]
        
        entry_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365*2-60))
        holding_days = int(np.random.choice([random.randint(5, 30), random.randint(31, 500)], p=[0.7, 0.3]))
        exit_date = entry_date + timedelta(days=holding_days)
        
        entry_price = round(random.uniform(50.0, 500.0), 2)
        quantity = random.randint(10, 200)

        is_win = np.random.rand() < 0.55
        price_change_percent = random.uniform(0.05, 0.40)
        exit_price = round(entry_price * (1 + price_change_percent if is_win else 1 - price_change_percent), 2)
        
        trade_list.append({'Ticker': ticker, 'Sector': sector, 'EntryDate': entry_date, 'ExitDate': exit_date, 
                           'EntryPrice': entry_price, 'ExitPrice': exit_price, 'Quantity': quantity})
                           
    df = pd.DataFrame(trade_list)
    df['IsWin'] = (df['ExitPrice'] > df['EntryPrice'])
    df['HoldingDays'] = (df['ExitDate'] - df['EntryDate']).dt.days
    print(f"✅ Generated {len(df)} mock trades.\n")
    return df

# ==============================================================================
# NEW ANALYSIS FUNCTIONS
# ==============================================================================

def analyze_by_holding_period(df):
    """Analyzes win rate by long-term vs. short-term trades."""
    print("\n--- 📊 1. Analysis by Holding Period ---")
    threshold = 90  # Days
    df['TradeType'] = np.where(df['HoldingDays'] > threshold, 'Long-Term', 'Short-Term')
    
    analysis = df.groupby('TradeType')['IsWin'].agg(['count', lambda x: x.mean() * 100])
    analysis.rename(columns={'<lambda_0>': 'WinRate (%)'}, inplace=True)
    
    print(analysis)

def analyze_by_sector(df):
    """Analyzes win rate by stock sector."""
    print("\n--- 📊 2. Analysis by Sector ---")
    
    analysis = df.groupby('Sector')['IsWin'].agg(['count', lambda x: x.mean() * 100])
    analysis.rename(columns={'<lambda_0>': 'WinRate (%)'}, inplace=True)
    
    print(analysis.sort_values(by='WinRate (%)', ascending=False))

def analyze_by_trade_size(df):
    """Analyzes win rate by the financial size of the trade."""
    print("\n--- 📊 3. Analysis by Trade Size ---")
    df['TradeValue'] = df['EntryPrice'] * df['Quantity']
    
    bins = [0, 5000, 20000, np.inf]
    labels = ['Small (< $5k)', 'Medium ($5k - $20k)', 'Large (> $20k)']
    df['ValueCategory'] = pd.cut(df['TradeValue'], bins=bins, labels=labels, right=False)
    
    analysis = df.groupby('ValueCategory', observed=True)['IsWin'].agg(['count', lambda x: x.mean() * 100])
    analysis.rename(columns={'<lambda_0>': 'WinRate (%)'}, inplace=True)
    
    print(analysis)

def analyze_by_trading_time(df):
    """Analyzes win rate by the day of the week."""
    print("\n--- 📊 4. Analysis by Trading Time ---")
    df['DayOfWeek'] = df['EntryDate'].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    analysis = df.groupby('DayOfWeek')['IsWin'].agg(['count', lambda x: x.mean() * 100])
    analysis.rename(columns={'<lambda_0>': 'WinRate (%)'}, inplace=True)
    
    # Sort by the day of the week
    analysis = analysis.reindex(day_order).dropna()
    
    print(analysis)
    print("\n* Note: Time-of-day analysis (e.g., morning vs afternoon) requires trade timestamp data, not just the date.")

def analyze_progress_over_time(df):
    """Analyzes and plots rolling win rate to track progress."""
    print("\n--- 📊 5. Analysis of Progress Over Time ---")
    if len(df) < 20:
        print("Not enough trades to perform a rolling analysis (minimum 20 required).")
        return

    df_sorted = df.sort_values('EntryDate').reset_index(drop=True)
    
    # Calculate the win rate over the last 20 trades
    df_sorted['RollingWinRate'] = df_sorted['IsWin'].rolling(window=20).mean() * 100
    
    print("Plotting your 20-trade rolling win rate. Close the plot window to continue...")

    # Plotting the result
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_sorted['EntryDate'], df_sorted['RollingWinRate'], label='20-Trade Rolling Win Rate', color='royalblue')
    ax.axhline(y=50, color='red', linestyle='--', label='50% Level')
    
    ax.set_title('Performance Trend Over Time', fontsize=16)
    ax.set_xlabel('Date of Trade', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    # This will pop up a window with the chart.
    plt.show()


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # Generate the data. This now includes IsWin and HoldingDays.
    my_trades_df = generate_mock_trades()
    
    # --- Run all analyses on the generated data ---
    analyze_by_holding_period(my_trades_df)
    analyze_by_sector(my_trades_df)
    analyze_by_trade_size(my_trades_df)
    analyze_by_trading_time(my_trades_df)
    analyze_progress_over_time(my_trades_df)
    
    print("\n🎉🎉🎉 Comprehensive Analysis Finished! 🎉🎉🎉")