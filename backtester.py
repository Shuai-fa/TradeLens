# backtester.py (The Final, Perfected Version)

import pandas as pd
import random
from datetime import datetime, timedelta

try:
    import yfinance as yf
    from backtesting import Backtest, Strategy
except ImportError:
    print("❌ Critical library not found.")
    print("Please ensure you are in your virtual environment and have run: pip install backtesting")
    exit()

# ==============================================================================
# STEP 1: DATA GENERATION
# ==============================================================================
def generate_mock_trades(num_trades=100):
    """Generates a DataFrame of simulated trade records."""
    print("STEP 1: Generating mock trade data...")
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM', 'XOM']
    trade_list = []
    for _ in range(num_trades):
        ticker = random.choice(TICKERS)
        entry_date = datetime(2022, 2, 1) + timedelta(days=random.randint(0, 365*2))
        exit_date = entry_date + timedelta(days=random.randint(5, 120))
        trade_list.append({'Ticker': ticker, 'EntryDate': pd.to_datetime(entry_date).normalize(), 
                           'ExitDate': pd.to_datetime(exit_date).normalize()})
    df = pd.DataFrame(trade_list)
    print(f"✅ Generated {len(df)} mock trades.\n")
    return df

# ==============================================================================
# STEP 2: DEFINE THE TRADING STRATEGY
# ==============================================================================
class FromLogStrategy(Strategy):
    trades_log = None 
    
    def init(self):
        # A quick lookup for dates that have any trading activity
        self.active_dates = set(self.trades_log['EntryDate']).union(set(self.trades_log['ExitDate']))

    def next(self):
        current_date = self.data.index[-1].normalize()
        
        # Only do something if today is an active trading day to improve performance
        if current_date not in self.active_dates:
            return

        # Iterate through each stock's data feed the strategy is aware of
        for stock_data in self.datas:
            ticker = stock_data._name
            
            # Check for SELL signals for this specific stock
            sells_today = self.trades_log[(self.trades_log['ExitDate'] == current_date) & (self.trades_log['Ticker'] == ticker)]
            if not sells_today.empty and stock_data.position:
                self.position.close(data=stock_data)
                print(f"{current_date.date()}: SELL {ticker}")

            # Check for BUY signals
            buys_today = self.trades_log[(self.trades_log['EntryDate'] == current_date) & (self.trades_log['Ticker'] == ticker)]
            if not buys_today.empty:
                if stock_data.position:
                    self.position.close(data=stock_data)
                self.buy(data=stock_data, size=0.1)
                print(f"{current_date.date()}: BUY {ticker}")

# ==============================================================================
# STEP 3: PREPARE DATA & RUN BACKTEST
# ==============================================================================
def run_backtest(trades_df):
    """Initializes and runs the backtest with the most robust data-handling."""
    print("STEP 2: Preparing data and running backtest...")
    
    benchmark_ticker = '^GSPC'
    all_tickers = trades_df['Ticker'].unique().tolist()
    
    print(f"  -> Downloading data for {len(all_tickers)} stocks + Benchmark ({benchmark_ticker})...")
    
    # Download all data into one single, multi-indexed dataframe
    # This is efficient and ensures all data shares the same date index.
    all_data_raw = yf.download(
        all_tickers + [benchmark_ticker], 
        start='2022-01-01', 
        end='2025-12-31', 
        auto_adjust=True, 
        group_by='ticker' # Use group_by='ticker' for a predictable MultiIndex structure
    )
    
    # --- The Final, Correct Way to Prepare Data Feeds ---
    data_feeds_kwargs = {}
    for ticker in all_tickers:
        try:
            # Extract columns for a single ticker
            df = all_data_raw[ticker].copy()
            df.rename(columns=str.capitalize, inplace=True)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(col in df.columns for col in required_cols):
                 # Important: The key must be a valid Python variable name
                 valid_key = ticker.replace('-', '_').replace('.', '_')
                 data_feeds_kwargs[valid_key] = df[required_cols]
            else:
                 print(f"Warning: Skipping {ticker} due to missing required columns.")
        except KeyError:
            print(f"Warning: Could not find data for ticker '{ticker}' in downloaded data. Skipping.")
    
    # Prepare benchmark data separately
    try:
        benchmark_data = all_data_raw[benchmark_ticker].copy()
        benchmark_data.rename(columns=str.capitalize, inplace=True)
    except KeyError:
        print(f"❌ Could not find benchmark data for {benchmark_ticker}. Cannot run backtest.")
        return

    if not data_feeds_kwargs:
        print("❌ Could not prepare any stock data for backtesting.")
        return

    # Pass the generated trades to our strategy class
    FromLogStrategy.trades_log = trades_df

    # Initialize the backtest by unpacking the dictionary as keyword arguments
    # e.g., Backtest(AAPL=df_aapl, MSFT=df_msft, ..., strategy=...)
    bt = Backtest(
        strategy=FromLogStrategy, 
        cash=100_000,
        commission=.002,
        **data_feeds_kwargs
    )
    
    stats = bt.run(benchmark=benchmark_data)
    
    print("\n--- 📈 PROFESSIONAL BACKTEST REPORT 📈 ---")
    print(stats)
    
    print("\nDisplaying performance plot. Close the plot window to exit.")
    bt.plot()

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    my_trades = generate_mock_trades()
    run_backtest(my_trades)
    print("\n🎉🎉🎉 Backtest Finished! 🎉🎉🎉")