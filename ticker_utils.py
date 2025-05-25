import requests
from typing import List
import os
from datetime import datetime, timedelta

def fetch_tickers_from_github() -> List[str]:
    """
    Fetch the latest ticker list from GitHub repository.
    Returns a list of ticker symbols.
    """
    url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Split the content into lines and clean up
        tickers = [line.strip().upper() for line in response.text.splitlines() if line.strip()]
        
        print(f"Successfully fetched {len(tickers)} tickers from GitHub")
        return tickers
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tickers from GitHub: {e}")
        # Fallback to local file if GitHub fetch fails
        return load_tickers_from_local()

def load_tickers_from_local(filename: str = 'all_tickers.txt') -> List[str]:
    """
    Load tickers from local file as fallback.
    """
    try:
        with open(filename, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        print(f"Loaded {len(tickers)} tickers from local file {filename}")
        return tickers
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return []

def get_tickers(use_cache: bool = True, cache_duration_hours: int = 24) -> List[str]:
    """
    Get tickers with optional caching.
    
    Args:
        use_cache: Whether to use cached tickers
        cache_duration_hours: How long to keep the cache (in hours)
    
    Returns:
        List of ticker symbols
    """
    cache_file = 'tickers_cache.txt'
    cache_timestamp_file = 'tickers_cache_timestamp.txt'
    
    if use_cache:
        # Check if cache exists and is fresh
        if os.path.exists(cache_file) and os.path.exists(cache_timestamp_file):
            try:
                with open(cache_timestamp_file, 'r') as f:
                    cache_time = datetime.fromisoformat(f.read().strip())
                
                if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
                    # Cache is fresh, use it
                    with open(cache_file, 'r') as f:
                        tickers = [line.strip() for line in f if line.strip()]
                    print(f"Using cached tickers ({len(tickers)} symbols)")
                    return tickers
            except Exception as e:
                print(f"Error reading cache: {e}")
    
    # Fetch fresh tickers
    tickers = fetch_tickers_from_github()
    
    # Update cache
    try:
        with open(cache_file, 'w') as f:
            f.write('\n'.join(tickers))
        with open(cache_timestamp_file, 'w') as f:
            f.write(datetime.now().isoformat())
        print("Updated ticker cache")
    except Exception as e:
        print(f"Error updating cache: {e}")
    
    return tickers 