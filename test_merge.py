# test_merge.py
# A minimal script to test the core pandas merge functionality.

import pandas as pd
from datetime import date

print("--- Minimal Merge Test Initialized ---\n")

# 1. Create a simple "left" DataFrame (simulating your trades)
# It has a normal RangeIndex (0, 1) and regular columns.
trades = pd.DataFrame({
    'Ticker': ['AAPL', 'MSFT'],
    'EntryDate': [pd.Timestamp('2024-01-05'), pd.Timestamp('2024-01-08')]
})
# Create the merge key column
trades['merge_date'] = trades['EntryDate'].dt.date

print("--- Left DataFrame (Trades) ---")
print(trades)
print(f"Index type: {type(trades.index)}")
print("\n")


# 2. Create a simple "right" DataFrame (simulating market data)
# It also has a normal RangeIndex and regular columns.
market_data = pd.DataFrame({
    'Date': [pd.Timestamp('2024-01-05'), pd.Timestamp('2024-01-08')],
    'Ticker': ['AAPL', 'MSFT'],
    'close': [180.0, 400.0]
})
# Create the same merge key column
market_data['merge_date'] = market_data['Date'].dt.date

print("--- Right DataFrame (Market Data) ---")
print(market_data)
print(f"Index type: {type(market_data.index)}")
print("\n")


# 3. Perform the merge using the robust method
# We are merging on two simple columns: 'Ticker' and 'merge_date'
try:
    print("--- Attempting Merge... ---")
    merged_df = pd.merge(
        trades,
        market_data,
        on=['Ticker', 'merge_date'],
        how='left'
    )

    print("✅ Merge successful!")
    print("\n--- Final Merged DataFrame ---")
    print(merged_df)

except Exception as e:
    print(f"❌ Merge Failed!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")