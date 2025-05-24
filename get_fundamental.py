import pandas as pd
import yfinance as yf
import time

INPUT_CSV = 'ten_x_stocks_from_file_alpaca_only.csv'
OUTPUT_CSV = 'ten_x_stocks_with_fundamentals.csv'
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2  # seconds

# Read the tickers from the CSV
original_df = pd.read_csv(INPUT_CSV)
tickers = original_df['symbol'].dropna().unique().tolist()

fundamental_data = []

for i in range(0, len(tickers), BATCH_SIZE):
    batch = tickers[i:i+BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1} of {(len(tickers) + BATCH_SIZE - 1)//BATCH_SIZE}")
    for ticker in batch:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamental_data.append({
                'symbol': ticker,
                'marketCap': info.get('marketCap'),
                'industry': info.get('industry')
            })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            fundamental_data.append({
                'symbol': ticker,
                'marketCap': None,
                'industry': None
            })
    time.sleep(SLEEP_BETWEEN_BATCHES)

fundamental_df = pd.DataFrame(fundamental_data)

# Merge with the original CSV on 'symbol'
merged_df = pd.merge(original_df, fundamental_df, on='symbol', how='left')

# Save to new CSV
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved merged data to {OUTPUT_CSV}")
