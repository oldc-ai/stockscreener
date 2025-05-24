import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
# import yfinance as yf # Removed yfinance
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
# from alpaca.trading.enums import AssetClass # Not strictly needed if not listing assets from Alpaca
import concurrent.futures
from functools import lru_cache
import time

# Initialize Alpaca clients
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False) # Unused in this script, can be removed if not planned for future use
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Constants for batch processing
BATCH_SIZE = 100  # Number of symbols to process in each batch
MAX_WORKERS = 4   # Number of parallel workers
RATE_LIMIT_DELAY = 0.1  # Delay between API calls in seconds

# Removed get_stock_details_from_yf function

def load_tickers_from_file(filepath="all_tickers.txt"):
    """Loads tickers from a specified file, one ticker per line."""
    tickers = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                ticker = line.strip().upper()
                # Keep isalpha for basic symbol validation before sending to Alpaca
                if ticker and ticker.isalpha(): 
                    tickers.append(ticker)
                elif ticker: 
                    print(f"Skipping ticker '{ticker}' from file: contains non-alphabetical characters or is invalid.")
        print(f"Loaded {len(tickers)} valid tickers from {filepath}")
        if not tickers and os.path.exists(filepath):
             print(f"Warning: No valid (alphabetic) tickers found in {filepath}.")
        return tickers
    except FileNotFoundError:
        print(f"Error: Ticker file {filepath} not found.")
        return []
    except Exception as e:
        print(f"Error reading ticker file {filepath}: {e}")
        return []

def get_symbols_to_analyze(ticker_filepath="all_tickers.txt"):
    """Loads tickers from file. No external enrichment."""
    
    symbols = load_tickers_from_file(ticker_filepath)
    if not symbols:
        print("No tickers to process from file.")
        return []
            
    print(f"\nPrepared {len(symbols)} symbols for analysis directly from {ticker_filepath}.")
    return symbols

@lru_cache(maxsize=1000)
def get_historical_data_batch(symbols, start_date, end_date):
    """Get historical data for a batch of symbols"""
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment='all'  # Use adjusted prices to account for splits and dividends
    )
    try:
        bars = data_client.get_stock_bars(request_params)
        time.sleep(RATE_LIMIT_DELAY)  # Respect rate limits
        return bars.df
    except Exception as e:
        # More specific error logging for Alpaca API calls
        if "subscription" in str(e).lower() or "forbidden" in str(e).lower() :
            print(f"Error for batch {symbols}: Data access issue (e.g. subscription/permissions). {e}")
        elif "not found" in str(e).lower():
            print(f"Error for batch {symbols}: Symbols likely not found on Alpaca or no data in range. {e}")
        else:
            print(f"Error getting data for batch {symbols} from Alpaca: {e}")
        return None

def find_max_return(df):
    """Find the maximum return between any two points in the period using O(n) algorithm"""
    if df is None or df.empty:
        return 1.0, None, None, None, None
        
    # Sort the dataframe by date to ensure chronological order
    df = df.sort_index()
    
    prices = df['close'].values
    dates = df.index.values
    
    min_price = float('inf')
    min_price_date = None
    actual_max_return_ratio = 1.0
    start_date_for_max_return = None
    end_date_for_max_return = None
    start_price = None
    end_price = None
    
    for i, (price, date) in enumerate(zip(prices, dates)):
        if isinstance(date, tuple):
            date = date[1]
            
        if price < min_price and price > 0:
            min_price = price
            min_price_date = date
        
        if min_price != float('inf') and min_price > 0:
            if date > min_price_date:
                current_return_ratio = price / min_price
                if current_return_ratio > actual_max_return_ratio:
                    actual_max_return_ratio = current_return_ratio
                    start_date_for_max_return = min_price_date
                    end_date_for_max_return = date
                    start_price = min_price
                    end_price = price
    
    return actual_max_return_ratio, start_date_for_max_return, end_date_for_max_return, start_price, end_price

def process_symbol_batch(symbols, start_date, end_date):
    """Process a batch of symbols and return results"""
    results = []
    df = get_historical_data_batch(tuple(symbols), start_date, end_date)
    
    if df is not None and not df.empty:
        for symbol in symbols:
            symbol_data = df[df.index.get_level_values('symbol') == symbol]
            if not symbol_data.empty:
                max_return_ratio, return_start_date, return_end_date, start_price, end_price = find_max_return(symbol_data)
                if max_return_ratio >= 10:
                    results.append({
                        'symbol': symbol,
                        'max_return_factor': round(max_return_ratio, 2),
                        'start_price': round(start_price, 2) if start_price is not None else 'N/A',
                        'end_price': round(end_price, 2) if end_price is not None else 'N/A',
                        'return_start_date': return_start_date.strftime('%Y-%m-%d') if return_start_date is not None else 'N/A',
                        'return_end_date': return_end_date.strftime('%Y-%m-%d') if return_end_date is not None else 'N/A',
                        'return_period_days': (return_end_date - return_start_date).days if return_start_date is not None and return_end_date is not None else 'N/A'
                    })
    
    return results

def main():
    # Set date range (10 years ago to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)
    
    print("Loading symbols from file...")
    symbols_to_process = get_symbols_to_analyze("all_tickers.txt") 
    
    if not symbols_to_process:
        print("No symbols to analyze from file. Exiting...")
        return
    
    # Split symbols into batches
    symbol_batches = [symbols_to_process[i:i + BATCH_SIZE] 
                     for i in range(0, len(symbols_to_process), BATCH_SIZE)]
    
    ten_x_stocks = []
    total_batches = len(symbol_batches)
    
    print(f"\nProcessing {len(symbols_to_process)} symbols in {total_batches} batches...")
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(process_symbol_batch, batch, start_date, end_date): i 
            for i, batch in enumerate(symbol_batches)
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_batch)):
            batch_num = future_to_batch[future]
            try:
                batch_results = future.result()
                ten_x_stocks.extend(batch_results)
                print(f"Completed batch {batch_num + 1}/{total_batches} - Found {len(batch_results)} 10x stocks")
            except Exception as e:
                print(f"Error processing batch {batch_num + 1}: {e}")
    
    # Convert to DataFrame and sort by max return
    if ten_x_stocks:
        df = pd.DataFrame(ten_x_stocks)
        # Update expected columns
        expected_cols = ['symbol', 'max_return_factor', 'start_price', 'end_price',
                        'return_start_date', 'return_end_date', 'return_period_days']
        
        # Ensure all expected columns are present
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None 
        df = df[expected_cols] 
        df = df.sort_values('max_return_factor', ascending=False)
        
        print("\nStocks from file that achieved 10x return (or more) in the past 10 years:")
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('ten_x_stocks_from_file_alpaca_only.csv', index=False)
        print("\nResults have been saved to 'ten_x_stocks_from_file_alpaca_only.csv'")
    else:
        print("No stocks from the provided list met the 10x return criteria.")

if __name__ == "__main__":
    main() 