#!/usr/bin/env python3
"""
Episodic Pivot (EP) Detector - Technical Analysis Focus
Based on Qullamaggie's setup criteria from: https://qullamaggie.com/how-to-master-a-setup-episodic-pivots/

EP Criteria (Technical Only):
1. Gap up of 10% or more
2. Massive volume surge (3x+ average)
3. Stocks that have been sideways for 3-6+ months
4. Price action confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from typing import List, Dict, Tuple, Optional
import os
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf
from ticker_utils import get_tickers

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='EP Detector with industry and market cap filtering')
    parser.add_argument('--exclude-industry', nargs='+', help='List of industries to exclude (e.g., "Technology" "Healthcare")')
    parser.add_argument('--exclude-market-cap-below', type=float, help='Exclude stocks with market cap below this value (in millions)')
    parser.add_argument('--no-cache', action='store_true', help='Disable ticker cache and fetch fresh from GitHub')
    return parser.parse_args()

class EPDetector:
    def __init__(self, api_key: str = None, secret_key: str = None, exclude_industries: List[str] = None, min_market_cap: float = None):
        """
        Initialize EP Detector with Alpaca credentials
        If no credentials provided, will look for environment variables
        """
        # Get Alpaca credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            print("Warning: Alpaca credentials not found in environment variables.")
            print("Please create a .env file with your credentials:")
            print("ALPACA_API_KEY=your_api_key_here")
            print("ALPACA_SECRET_KEY=your_secret_key_here")
            raise ValueError("Alpaca credentials are required. Please set them in .env file or pass them to the constructor.")
        
        # Initialize Alpaca client
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.results = []
        self.failed_tickers = []
        
        # Store filtering parameters
        self.exclude_industries = [ind.lower() for ind in (exclude_industries or [])]
        self.min_market_cap = min_market_cap  # in millions

    def check_fundamentals(self, ticker: str) -> Tuple[bool, str]:
        """
        Check if stock meets fundamental criteria (industry and market cap)
        Returns: (meets_criteria, reason)
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check industry
            if self.exclude_industries and 'industry' in info:
                industry = info['industry'].lower()
                if any(excluded.lower() in industry for excluded in self.exclude_industries):
                    return False, f"Industry {industry} is excluded"
            
            # Check market cap
            if self.min_market_cap and 'marketCap' in info:
                market_cap_millions = info['marketCap'] / 1_000_000  # Convert to millions
                if market_cap_millions < self.min_market_cap:
                    return False, f"Market cap {market_cap_millions:.2f}M below threshold {self.min_market_cap}M"
            
            return True, "Meets fundamental criteria"
            
        except Exception as e:
            print(f"Error checking fundamentals for {ticker}: {e}")
            return True, "Error checking fundamentals, proceeding with technical analysis"

    def load_tickers(self, filename: str = 'all_tickers.txt') -> List[str]:
        """Load ticker symbols from file"""
        try:
            with open(filename, 'r') as f:
                tickers = [line.strip().upper() for line in f if line.strip()]
            print(f"Loaded {len(tickers)} tickers from {filename}")
            return tickers
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return []
    
    def get_stock_data(self, ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Get stock data for a single ticker using Alpaca, including pre/post market data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get regular market hours data
            request_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                adjustment='raw'
            )
            
            bars = self.client.get_stock_bars(request_params)
            
            if bars.df.empty:
                return None
                
            # Get pre/post market data for the last day
            extended_hours_params = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=end_date - timedelta(days=1),
                end=end_date,
                adjustment='raw'
            )
            
            extended_bars = self.client.get_stock_bars(extended_hours_params)
            
            # Process regular market data
            df = bars.df.reset_index()
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            df['Date'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('Date')
            df = df.drop(['timestamp', 'symbol'], axis=1, errors='ignore')
            
            # Process extended hours data if available
            pre_market_data = None
            post_market_data = None
            
            if not extended_bars.df.empty:
                ext_df = extended_bars.df.reset_index()
                ext_df['Date'] = pd.to_datetime(ext_df['timestamp'])
                
                # Filter pre-market (4:00 AM - 9:30 AM EST)
                pre_market = ext_df[
                    (ext_df['Date'].dt.hour >= 4) & 
                    (ext_df['Date'].dt.hour < 9) |
                    ((ext_df['Date'].dt.hour == 9) & (ext_df['Date'].dt.minute < 30))
                ]
                
                # Filter post-market (4:00 PM - 8:00 PM EST)
                post_market = ext_df[
                    (ext_df['Date'].dt.hour >= 16) & 
                    (ext_df['Date'].dt.hour < 20)
                ]
                
                if not pre_market.empty:
                    pre_market_data = pd.DataFrame({
                        'Open': [pre_market['open'].iloc[0]],
                        'High': [pre_market['high'].max()],
                        'Low': [pre_market['low'].min()],
                        'Close': [pre_market['close'].iloc[-1]],
                        'Volume': [pre_market['volume'].sum()]
                    })
                
                if not post_market.empty:
                    post_market_data = pd.DataFrame({
                        'Open': [post_market['open'].iloc[0]],
                        'High': [post_market['high'].max()],
                        'Low': [post_market['low'].min()],
                        'Close': [post_market['close'].iloc[-1]],
                        'Volume': [post_market['volume'].sum()]
                    })
            
            return df, pre_market_data, post_market_data
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_batch_stock_data(self, tickers: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
        """Get stock data for multiple tickers in a single API call"""
        try:
            end_date = datetime.now() - timedelta(minutes=16)
            start_date = end_date - timedelta(days=days)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.client.get_stock_bars(request_params)
            
            result = {}
            if not bars.df.empty:
                # Group by symbol
                for symbol in tickers:
                    try:
                        symbol_data = bars.df[bars.df.index.get_level_values('symbol') == symbol]
                        if not symbol_data.empty:
                            df = symbol_data.reset_index()
                            # Rename columns to match our expected format
                            df = df.rename(columns={
                                'open': 'Open',
                                'high': 'High', 
                                'low': 'Low',
                                'close': 'Close',
                                'volume': 'Volume'
                            })
                            
                            # Set timestamp as index
                            df['Date'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('Date')
                            df = df.drop(['timestamp', 'symbol'], axis=1, errors='ignore')
                            
                            result[symbol] = df
                    except Exception as e:
                        # Skip individual symbol errors
                        continue
            
            return result
                
        except Exception as e:
            print(f"Error fetching batch data: {e}")
            return {}
    
    def process_batch(self, batch_tickers: List[str]) -> List[Dict]:
        """Process a batch of tickers for EP setups"""
        results = []
        
        # Get data for entire batch in one API call
        batch_data = self.get_batch_stock_data(batch_tickers)
        
        for ticker in batch_tickers:
            try:
                data = batch_data.get(ticker)
                if data is not None and len(data) >= 30:
                    result = self.screen_for_ep_with_data(ticker, data)
                    if result:
                        results.append(result)
            except Exception as e:
                # Skip individual ticker errors
                continue
        
        return results
    
    def screen_for_ep_with_data(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        """Screen a single ticker for EP setup with pre-loaded data"""
        # Calculate key technical metrics
        gap_percent = self.calculate_gap_up(data)
        volume_surge = self.calculate_volume_surge(data)
        sideways_info = self.check_sideways_consolidation(data)
        price_strength = self.calculate_price_strength(data)
        volatility_metrics = self.calculate_volatility_metrics(data)
        
        # Current price info
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume_20d = data['Volume'].iloc[-21:-1].mean() if len(data) > 20 else 0
        
        # Filter out penny stocks (price < $1)
        if current_price < 1.0:
            return None
        
        # Price change metrics
        price_1d = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
        price_5d = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]) * 100 if len(data) > 5 else 0
        
        # HARD REQUIREMENT: EP must have 10%+ gap up (Qullamaggie's definition)
        if gap_percent < 10.0:
            return None
        
        # EP Technical Scoring System (100 points max)
        ep_score = 0
        criteria_met = []
        
        # 1. Gap up criteria (most important - 40 points max)
        # Note: All candidates already have 10%+ gap due to hard requirement above
        if gap_percent >= 15:
            ep_score += 40
            criteria_met.append(f"Big gap: {gap_percent:.1f}%")
        elif gap_percent >= 10:
            ep_score += 30
            criteria_met.append(f"Gap up: {gap_percent:.1f}%")
        
        # 2. Volume surge criteria (30 points max)
        if volume_surge >= 5:
            ep_score += 30
            criteria_met.append(f"Huge volume: {volume_surge:.1f}x")
        elif volume_surge >= 3:
            ep_score += 25
            criteria_met.append(f"Volume surge: {volume_surge:.1f}x")
        elif volume_surge >= 2:
            ep_score += 10
            criteria_met.append(f"Good volume: {volume_surge:.1f}x")
        
        # 3. Sideways consolidation (20 points max)
        if sideways_info["is_sideways"] and sideways_info["consolidation_days"] >= 60:
            ep_score += 20
            criteria_met.append(f"Sideways {sideways_info['consolidation_days']} days")
        elif sideways_info["consolidation_days"] >= 30:
            ep_score += 10
            criteria_met.append(f"Some consolidation")
        
        # 4. Price strength (10 points max)
        if price_strength["recent_high"]:
            ep_score += 5
            criteria_met.append("New highs")
        if price_strength["price_above_50ma"]:
            ep_score += 5
            criteria_met.append("Above 50MA")
        
        # Since we already require 10%+ gap, lower the minimum score threshold
        min_score_threshold = 30  # At least gap (30) + some additional criteria
        
        if ep_score >= min_score_threshold:
            # Only check fundamentals if it's a valid EP setup
            meets_fundamentals, fundamental_reason = self.check_fundamentals(ticker)
            if not meets_fundamentals:
                return None
                
            return {
                "ticker": ticker,
                "ep_score": ep_score,
                
                # Price metrics
                "current_price": current_price,
                "gap_percent": gap_percent,
                "price_1d": price_1d,
                "price_5d": price_5d,
                
                # Volume metrics  
                "volume_surge": volume_surge,
                "current_volume": current_volume,
                "avg_volume_20d": avg_volume_20d,
                
                # Technical indicators
                "rsi": price_strength["rsi"],
                "above_50ma": price_strength["price_above_50ma"],
                "recent_high": price_strength["recent_high"],
                "atr": volatility_metrics["atr"],
                "volatility": volatility_metrics["volatility"],
                
                # Consolidation analysis
                "sideways_consolidation": sideways_info,
                
                # Summary
                "criteria_met": criteria_met,
                "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return None
    
    def calculate_gap_up(self, data: pd.DataFrame, pre_market_data: Optional[pd.DataFrame] = None, post_market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate gap up percentage considering pre/post market data
        
        Args:
            data: Regular market hours data
            pre_market_data: Pre-market data (4:00 AM - 9:30 AM EST)
            post_market_data: Post-market data (4:00 PM - 8:00 PM EST)
        """
        if len(data) < 2:
            return 0.0
        
        previous_close = data['Close'].iloc[-2]
        if previous_close == 0:
            return 0.0
            
        # Check pre-market gap
        if pre_market_data is not None and not pre_market_data.empty:
            pre_market_price = pre_market_data['Close'].iloc[-1]
            pre_market_gap = ((pre_market_price - previous_close) / previous_close) * 100
            if pre_market_gap >= 10.0:  # If pre-market gap is significant
                return pre_market_gap
        
        # Check regular market gap
        current_open = data['Open'].iloc[-1]
        current_close = data['Close'].iloc[-1]
        if current_close < current_open:
            return 0.0
            
        regular_gap = ((current_open - previous_close) / previous_close) * 100
        
        # Check post-market gap
        if post_market_data is not None and not post_market_data.empty:
            post_market_price = post_market_data['Close'].iloc[-1]
            post_market_gap = ((post_market_price - current_close) / current_close) * 100
            if post_market_gap >= 10.0:  # If post-market gap is significant
                return post_market_gap
        
        return regular_gap
    
    def calculate_volume_surge(self, data: pd.DataFrame, lookback_days: int = 20) -> float:
        """Calculate volume surge as multiple of average volume"""
        if len(data) < lookback_days + 1:
            return 0.0
        
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].iloc[-(lookback_days+1):-1].mean()
        
        if avg_volume < 100000:
            return 0.0
            
        volume_multiple = current_volume / avg_volume
        return volume_multiple
    
    def check_sideways_consolidation(self, data: pd.DataFrame, lookback_days: int = 120) -> Dict:
        """Check if stock has been consolidating sideways for 3-6+ months"""
        if len(data) < lookback_days:
            return {"is_sideways": False, "consolidation_days": 0, "range_percent": 0}
        
        # Look at the period before the latest move (exclude last 5 days)
        consolidation_data = data.iloc[-(lookback_days+5):-5]
        
        if len(consolidation_data) < 60:  # Need at least 60 days for meaningful analysis
            return {"is_sideways": False, "consolidation_days": 0, "range_percent": 0}
        
        high = consolidation_data['High'].max()
        low = consolidation_data['Low'].min()
        range_percent = ((high - low) / low) * 100
        
        # Consider sideways if trading range is less than 40% over the period
        is_sideways = range_percent < 40
        
        return {
            "is_sideways": is_sideways,
            "consolidation_days": len(consolidation_data),
            "range_percent": range_percent,
            "consolidation_high": high,
            "consolidation_low": low
        }
    
    def calculate_price_strength(self, data: pd.DataFrame) -> Dict:
        """Calculate various price strength metrics"""
        if len(data) < 50:
            return {"price_above_50ma": False, "rsi": 50, "recent_high": False}
        
        # Simple moving averages
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        
        current_price = data['Close'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        
        # Check if making new highs recently
        recent_high = current_price >= data['High'].iloc[-20:].max()
        
        # Simple RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        return {
            "price_above_50ma": current_price > sma_50 if not pd.isna(sma_50) else False,
            "rsi": current_rsi,
            "recent_high": recent_high,
            "sma_20": data['SMA_20'].iloc[-1] if not pd.isna(data['SMA_20'].iloc[-1]) else current_price,
            "sma_50": sma_50 if not pd.isna(sma_50) else current_price
        }
    
    def calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate volatility and trading range metrics"""
        if len(data) < 20:
            return {"atr": 0, "volatility": 0}
        
        # Average True Range
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # 20-day volatility (standard deviation of returns)
        returns = data['Close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * 100
        
        return {
            "atr": atr if not pd.isna(atr) else 0,
            "volatility": volatility if not pd.isna(volatility) else 0
        }
    
    def screen_for_ep(self, ticker: str) -> Optional[Dict]:
        """Screen a single ticker for EP setup - Technical analysis only"""
        data_result = self.get_stock_data(ticker)
        if data_result is None:
            return None
            
        data, pre_market_data, post_market_data = data_result
        if data is None or len(data) < 30:
            return None
        
        # Calculate key technical metrics
        gap_percent = self.calculate_gap_up(data, pre_market_data, post_market_data)
        volume_surge = self.calculate_volume_surge(data)
        sideways_info = self.check_sideways_consolidation(data)
        price_strength = self.calculate_price_strength(data)
        volatility_metrics = self.calculate_volatility_metrics(data)
        
        # Current price info - use post-market price if available
        current_price = post_market_data['Close'].iloc[-1] if post_market_data is not None else data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume_20d = data['Volume'].iloc[-21:-1].mean() if len(data) > 20 else 0
        
        # Filter out penny stocks (price < $1)
        if current_price < 1.0:
            return None
        
        # Price change metrics
        price_1d = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
        price_5d = ((current_price - data['Close'].iloc[-6]) / data['Close'].iloc[-6]) * 100 if len(data) > 5 else 0
        
        # HARD REQUIREMENT: EP must have 10%+ gap up (Qullamaggie's definition)
        if gap_percent < 10.0:
            return None
        
        # EP Technical Scoring System (100 points max)
        ep_score = 0
        criteria_met = []
        
        # 1. Gap up criteria (most important - 40 points max)
        # Note: All candidates already have 10%+ gap due to hard requirement above
        if gap_percent >= 15:
            ep_score += 40
            criteria_met.append(f"Big gap: {gap_percent:.1f}%")
        elif gap_percent >= 10:
            ep_score += 30
            criteria_met.append(f"Gap up: {gap_percent:.1f}%")
        
        # 2. Volume surge criteria (30 points max)
        if volume_surge >= 5:
            ep_score += 30
            criteria_met.append(f"Huge volume: {volume_surge:.1f}x")
        elif volume_surge >= 3:
            ep_score += 25
            criteria_met.append(f"Volume surge: {volume_surge:.1f}x")
        elif volume_surge >= 2:
            ep_score += 10
            criteria_met.append(f"Good volume: {volume_surge:.1f}x")
        
        # 3. Sideways consolidation (20 points max)
        if sideways_info["is_sideways"] and sideways_info["consolidation_days"] >= 60:
            ep_score += 20
            criteria_met.append(f"Sideways {sideways_info['consolidation_days']} days")
        elif sideways_info["consolidation_days"] >= 30:
            ep_score += 10
            criteria_met.append(f"Some consolidation")
        
        # 4. Price strength (10 points max)
        if price_strength["recent_high"]:
            ep_score += 5
            criteria_met.append("New highs")
        if price_strength["price_above_50ma"]:
            ep_score += 5
            criteria_met.append("Above 50MA")
        
        # Since we already require 10%+ gap, lower the minimum score threshold
        min_score_threshold = 30  # At least gap (30) + some additional criteria
        
        if ep_score >= min_score_threshold:
            return {
                "ticker": ticker,
                "ep_score": ep_score,
                
                # Price metrics
                "current_price": current_price,
                "gap_percent": gap_percent,
                "price_1d": price_1d,
                "price_5d": price_5d,
                
                # Volume metrics  
                "volume_surge": volume_surge,
                "current_volume": current_volume,
                "avg_volume_20d": avg_volume_20d,
                
                # Technical indicators
                "rsi": price_strength["rsi"],
                "above_50ma": price_strength["price_above_50ma"],
                "recent_high": price_strength["recent_high"],
                "atr": volatility_metrics["atr"],
                "volatility": volatility_metrics["volatility"],
                
                # Consolidation analysis
                "sideways_consolidation": sideways_info,
                
                # Summary
                "criteria_met": criteria_met,
                "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return None
    
    def scan_all_tickers(self, max_tickers: Optional[int] = None, 
                        batch_size: int = 20, max_workers: int = 4,
                        use_cache: bool = True) -> List[Dict]:
        """Scan all tickers for EP setups using batching and parallel processing"""
        tickers = get_tickers(use_cache=use_cache)
        
        if max_tickers:
            tickers = tickers[:max_tickers]
            print(f"Limiting scan to first {max_tickers} tickers")
        
        print(f"Scanning {len(tickers)} tickers for Episodic Pivot setups (Technical Analysis)...")
        print(f"Using batched processing: {batch_size} tickers per batch, {max_workers} parallel workers")
        print("Alpaca data source with 10%+ gap requirement enforced\n")
        
        # Split tickers into batches
        batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
        
        all_results = []
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(self.process_batch, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                processed_count += len(batch)
                
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Progress reporting
                    progress_pct = (processed_count / len(tickers)) * 100
                    print(f"Progress: {processed_count:4}/{len(tickers)} ({progress_pct:5.1f}%) | "
                          f"Found {len(batch_results)} EPs in this batch | "
                          f"Total EPs: {len(all_results)}")
                    
                    # Show any EPs found in this batch
                    for result in batch_results:
                        print(f"  ✓ {result['ticker']:6} | Score: {result['ep_score']:2.0f} | "
                              f"Gap: {result['gap_percent']:5.1f}% | Vol: {result['volume_surge']:4.1f}x")
                    
                except Exception as e:
                    print(f"Batch failed: {str(e)[:50]}")
                
                # Rate limiting between batches
                time.sleep(0.5)  # Small delay between batch completions
        
        print(f"\nScan complete! Found {len(all_results)} potential Episodic Pivot setups")
        print(f"Processed {processed_count} tickers total")
        return all_results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to CSV file"""
        if not results:
            print("No results to save")
            return
        
        if filename is None:
            filename = f"ep_scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Flatten the results for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                "ticker": result["ticker"],
                "ep_score": result["ep_score"],
                "current_price": result["current_price"],
                "gap_percent": result["gap_percent"],
                "price_1d": result["price_1d"],
                "price_5d": result["price_5d"],
                "volume_surge": result["volume_surge"],
                "current_volume": result["current_volume"],
                "avg_volume_20d": result["avg_volume_20d"],
                "rsi": result["rsi"],
                "above_50ma": result["above_50ma"],
                "recent_high": result["recent_high"],
                "atr": result["atr"],
                "volatility": result["volatility"],
                "is_sideways": result["sideways_consolidation"]["is_sideways"],
                "consolidation_days": result["sideways_consolidation"]["consolidation_days"],
                "range_percent": result["sideways_consolidation"]["range_percent"],
                "criteria_met": " | ".join(result["criteria_met"]),
                "scan_date": result["scan_date"]
            }
            flattened_results.append(flat_result)
        
        df = pd.DataFrame(flattened_results)
        df = df.sort_values("ep_score", ascending=False)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        # Print summary
        print(f"\n=== EPISODIC PIVOT SCAN SUMMARY ===")
        print(f"Total EPs found: {len(results)}")
        print(f"Average EP score: {df['ep_score'].mean():.1f}")
        print(f"Average gap: {df['gap_percent'].mean():.1f}%")
        print(f"Average volume surge: {df['volume_surge'].mean():.1f}x")
        print(f"\nTop 5 candidates:")
        for i, row in df.head().iterrows():
            print(f"  {row['ticker']:6} - Score: {row['ep_score']:3.0f} - Gap: {row['gap_percent']:5.1f}% - Vol: {row['volume_surge']:4.1f}x - ${row['current_price']:6.2f}")

def main():
    """Main function to run the EP detector"""
    args = parse_args()
    
    # Initialize EP detector with filtering options
    detector = EPDetector(
        exclude_industries=args.exclude_industry,
        min_market_cap=args.exclude_market_cap_below
    )
    
    print(f"Starting EP scan...")
    if args.exclude_industry:
        print(f"Excluding industries: {', '.join(args.exclude_industry)}")
    if args.exclude_market_cap_below:
        print(f"Excluding stocks with market cap below ${args.exclude_market_cap_below}M")
    
    results = detector.scan_all_tickers(use_cache=not args.no_cache)
    
    if results:
        print(f"\nFound {len(results)} potential EP setups!")
        detector.save_results(results)
    else:
        print("\nNo EP setups found.")

if __name__ == "__main__":
    main()
