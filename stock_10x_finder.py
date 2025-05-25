import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
# from alpaca.trading.enums import AssetClass # Not strictly needed if not listing assets from Alpaca
import concurrent.futures
from functools import lru_cache
import time
from ticker_utils import get_tickers
import argparse

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

def get_symbols_to_analyze(use_cache: bool = True) -> list:
    """
    Get list of symbols to analyze from GitHub repository.
    Returns a list of ticker symbols.
    """
    return get_tickers(use_cache=use_cache)

@lru_cache(maxsize=1000)
def get_historical_data_batch(symbols, start_date, end_date):
    """
    Get historical data for a batch of symbols.
    Uses caching to avoid repeated API calls.
    """
    try:
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
        return data
    except Exception as e:
        print(f"Error downloading data for batch: {e}")
        return None

def find_max_return(df):
    """
    Find the maximum return for a given stock.
    Returns a tuple of (max_return, start_date, end_date)
    """
    if df is None or df.empty:
        return None, None, None
    
    # Calculate daily returns
    returns = df['Adj Close'].pct_change()
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Find the maximum return
    max_return = cum_returns.max()
    end_date = cum_returns.idxmax()
    
    # Find the start date (first date with non-zero return)
    start_date = cum_returns[cum_returns > 1].index[0] if len(cum_returns[cum_returns > 1]) > 0 else None
    
    return max_return, start_date, end_date

def process_symbol_batch(symbols, start_date, end_date):
    """
    Process a batch of symbols and find their maximum returns.
    """
    results = []
    data = get_historical_data_batch(tuple(symbols), start_date, end_date)
    
    if data is None:
        return results
    
    for symbol in symbols:
        try:
            if symbol in data.columns.levels[0]:
                symbol_data = data[symbol]
                max_return, start_date, end_date = find_max_return(symbol_data)
                
                if max_return is not None and max_return > 1:
                    results.append({
                        'symbol': symbol,
                        'max_return': max_return,
                        'start_date': start_date,
                        'end_date': end_date
                    })
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find stocks that have had 10x returns')
    parser.add_argument('--years', type=int, default=10, help='Number of years to look back')
    parser.add_argument('--min-return', type=float, default=10.0, help='Minimum return to consider (e.g., 10.0 for 10x)')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of symbols to process in each batch')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--no-cache', action='store_true', help='Disable ticker cache and fetch fresh from GitHub')
    args = parser.parse_args()
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.years * 365)
    
    # Get symbols to analyze
    symbols = get_symbols_to_analyze(use_cache=not args.no_cache)
    print(f"Analyzing {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
    
    # Split symbols into batches
    batches = [symbols[i:i + args.batch_size] for i in range(0, len(symbols), args.batch_size)]
    
    all_results = []
    processed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_symbol_batch, batch, start_date, end_date): batch for batch in batches}
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            processed_count += len(batch)
            
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                
                # Progress reporting
                progress_pct = (processed_count / len(symbols)) * 100
                print(f"Progress: {processed_count:4}/{len(symbols)} ({progress_pct:5.1f}%) | "
                      f"Found {len(batch_results)} candidates in this batch | "
                      f"Total candidates: {len(all_results)}")
                
            except Exception as e:
                print(f"Batch failed: {str(e)[:50]}")
    
    # Filter and sort results
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df[results_df['max_return'] >= args.min_return]
        results_df = results_df.sort_values('max_return', ascending=False)
        
        # Save results
        output_file = f"stock_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        print(f"\nFound {len(results_df)} stocks with {args.min_return}x+ returns")
        print("\nTop 10 stocks by return:")
        for _, row in results_df.head(10).iterrows():
            print(f"{row['symbol']:6} - {row['max_return']:.1f}x from {row['start_date'].date()} to {row['end_date'].date()}")
    else:
        print("\nNo stocks found meeting the criteria.")

if __name__ == "__main__":
    main() 