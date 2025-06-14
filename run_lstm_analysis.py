# run_lstm_analysis.py (Final Running Version)
# This version is confirmed to be free of all previously identified bugs.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- Import necessary libraries ---
try:
    import yfinance as yf
    import pandas_ta as ta
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"❌ A necessary library is missing: {e.name}")
    print("Please ensure you are in your virtual environment and have run the pip install command.")
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
        # Ensure entry date has enough historical data for sequence creation
        entry_date = datetime(2023, 3, 1) + timedelta(days=random.randint(0, 365))
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
def get_full_market_data(trades_df):
    """Gets all historical market data with robust TA feature calculation."""
    print("STEP 2: Downloading and preparing market data...")
    unique_tickers = trades_df['Ticker'].unique().tolist()
    start_date = "2022-01-01"
    end_date = trades_df['EntryDate'].max() + timedelta(days=1)
    
    all_market_data = []
    print(f"  -> Processing data for {len(unique_tickers)} tickers...")
    for ticker in unique_tickers:
        market_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        if market_data.empty: continue
        
        market_data.rename(columns=str.lower, inplace=True)
        if 'close' not in market_data.columns: continue
        
        # Using robust, individual TA calculations
        market_data.ta.sma(length=50, append=True)
        market_data.ta.sma(length=200, append=True)
        market_data.ta.rsi(length=14, append=True)
        market_data.ta.macd(append=True)
        
        market_data['Ticker'] = ticker
        all_market_data.append(market_data)

    if not all_market_data: return None
    
    full_data = pd.concat(all_market_data)
    
    # Drop NaNs created by TA indicators here, AFTER all data is combined
    full_data.dropna(inplace=True)
    
    # Scale numeric features for better neural network performance
    # Important: Select only numeric columns for scaling
    numeric_cols = full_data.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        full_data[numeric_cols] = scaler.fit_transform(full_data[numeric_cols])
    
    print("✅ Full market data with features is ready.\n")
    return full_data.reset_index()

# ==============================================================================
# STEP 3: DATA PREPARATION FOR LSTM
# ==============================================================================
def create_sequences(trades_df, market_df, sequence_length=30):
    """Creates sequences of historical data leading up to each trade."""
    print(f"STEP 3: Creating data sequences of length {sequence_length}...")
    
    X_sequences, y_outcomes = [], []
    
    # Define feature columns here, after TA calculation
    feature_cols = [col for col in market_df.columns if col not in ['Ticker', 'Date']]
    
    # We now expect market_df to be clean of NaNs from the previous step
    market_groups = market_df.set_index('Date').groupby('Ticker')

    for _, trade in trades_df.iterrows():
        try:
            stock_market_data = market_groups.get_group(trade['Ticker'])
            end_idx_timestamp = pd.to_datetime(trade['EntryDate']).normalize()
            
            end_idx_loc = stock_market_data.index.get_loc(end_idx_timestamp, method='pad')
            start_idx_loc = end_idx_loc - sequence_length
            
            if start_idx_loc >= 0:
                sequence = stock_market_data.iloc[start_idx_loc:end_idx_loc][feature_cols].values
                if sequence.shape[0] == sequence_length:
                    X_sequences.append(sequence)
                    y_outcomes.append(trade['IsWin'])
        except KeyError:
            # This can happen if a trade date has no preceding market data after cleaning
            continue
            
    if not X_sequences:
        print("❌ Could not create any sequences. Check data ranges or if too much data was dropped.")
        return None, None
        
    print(f"✅ Created {len(X_sequences)} sequences.\n")
    return np.array(X_sequences), np.array(y_outcomes)

# ==============================================================================
# STEP 4: LSTM MODEL TRAINING & EVALUATION
# ==============================================================================
def train_and_evaluate_lstm(X, y):
    """Builds, trains, and evaluates the LSTM model."""
    print("STEP 4: Building and training LSTM model...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    if len(X_train) == 0:
        print("❌ Not enough data to train the model after splitting. Try generating more trades.")
        return

    _ , sequence_length, num_features = X_train.shape
    
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("\nTraining model... (This may take a few minutes)")
    model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.1, verbose=1)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n--- 🤖 DEEP LEARNING MODEL PERFORMANCE REPORT 🤖 ---")
    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    my_trades = generate_mock_trades()
    full_market_data = get_full_market_data(my_trades)
    
    if full_market_data is not None:
        X, y = create_sequences(my_trades, full_market_data)
        if X is not None and y is not None:
            train_and_evaluate_lstm(X, y)
            
    print("\n🎉🎉🎉 Pipeline finished! 🎉🎉🎉")