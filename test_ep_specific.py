#!/usr/bin/env python3
"""
Quick test of EP detector with specific popular tickers
"""

from ep_detector import EPDetector
import pandas as pd
import traceback
import argparse

def analyze_ticker(ticker: str, require_sideways: bool = True):
    print(f"\nAnalyzing {ticker} for EP setup... (sideways required: {require_sideways})")
    try:
        detector = EPDetector()
        
        # Get data
        print("Fetching data...")
        data_result = detector.get_stock_data(ticker)
        if data_result is None:
            print(f"Could not get data for {ticker}")
            return
            
        data, pre_market_data, post_market_data = data_result
        print(f"Got {len(data)} days of data")
        
        # Calculate all metrics
        print("Calculating metrics...")
        gap_percent = detector.calculate_gap_up(data, pre_market_data, post_market_data)
        volume_surge = detector.calculate_volume_surge(data)
        sideways_info = detector.check_sideways_consolidation(data)
        price_strength = detector.calculate_price_strength(data)
        volatility_metrics = detector.calculate_volatility_metrics(data)
        
        # Print detailed analysis
        print("\nDetailed Analysis:")
        print(f"1. Gap Analysis:")
        print(f"   - Gap Percentage: {gap_percent:.2f}%")
        if pre_market_data is not None:
            print(f"   - Pre-market data available")
        if post_market_data is not None:
            print(f"   - Post-market data available")
        
        print(f"\n2. Volume Analysis:")
        print(f"   - Volume Surge: {volume_surge:.2f}x average")
        print(f"   - Current Volume: {data['Volume'].iloc[-1]:,.0f}")
        print(f"   - 20-day Avg Volume: {data['Volume'].iloc[-21:-1].mean():,.0f}")
        
        print(f"\n3. Price Analysis:")
        print(f"   - Current Price: ${data['Close'].iloc[-1]:.2f}")
        print(f"   - RSI: {price_strength['rsi']:.1f}")
        print(f"   - Above 50MA: {price_strength['price_above_50ma']}")
        print(f"   - Recent High: {price_strength['recent_high']}")
        
        print(f"\n4. Consolidation Analysis:")
        print(f"   - Sideways: {sideways_info['is_sideways']}")
        print(f"   - Consolidation Days: {sideways_info['consolidation_days']}")
        print(f"   - Range Percent: {sideways_info['range_percent']:.1f}%")
        print(f"   - Price Trend: {sideways_info['price_trend']:.1f}%")
        print(f"   - Above 50MA: {sideways_info['above_50ma_percent']:.1f}%")
        print(f"   - Significant Runs: {sideways_info['significant_runs']}")
        
        print(f"\n5. Volatility Metrics:")
        print(f"   - ATR: {volatility_metrics['atr']:.2f}")
        print(f"   - Volatility: {volatility_metrics['volatility']:.1f}%")
        
        # Check if it meets EP criteria
        print("\nEP Criteria Check:")
        print(f"1. Gap up 10%+: {'✓' if gap_percent >= 10.0 else '✗'} ({gap_percent:.1f}%)")
        print(f"2. Volume surge 3x+: {'✓' if volume_surge >= 3.0 else '✗'} ({volume_surge:.1f}x)")
        print(f"3. Sideways 3-6+ months: {'✓' if sideways_info['is_sideways'] and sideways_info['consolidation_days'] >= 60 else '✗'} ({sideways_info['consolidation_days']} days)")
        print(f"4. Price strength: {'✓' if price_strength['recent_high'] or price_strength['price_above_50ma'] else '✗'}")
        
        # Debug: Check minimum score threshold
        ep_score = 0
        criteria_met = []
        
        # 1. Gap up criteria (40 points max)
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
        
        print(f"\nScore Calculation:")
        print(f"Total Score: {ep_score}/100")
        print(f"Criteria Met: {', '.join(criteria_met)}")
        print(f"Minimum Required Score: 30")
        
        # Use the detector's screen_for_ep with the flag
        result = detector.screen_for_ep(ticker, require_sideways=require_sideways)
        if result:
            print("\nThis ticker qualifies as an EP!")
        else:
            print("\nThis ticker does NOT qualify as an EP.")
            if gap_percent < 10.0:
                print("Reason: Gap up less than 10%")
            elif ep_score < 30:
                print("Reason: Total score below minimum threshold of 30")
            elif require_sideways and (not sideways_info["is_sideways"] or sideways_info["consolidation_days"] < 60):
                print("Reason: Does not meet sideways consolidation requirement")
        
    except Exception as e:
        print(f"Error analyzing {ticker}:")
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EP detector for a specific ticker.")
    parser.add_argument("--no-sideways", action="store_true", help="Disable sideways consolidation requirement.")
    parser.add_argument("--ticker", type=str, default="NNE", help="Ticker symbol to analyze.")
    args = parser.parse_args()
    
    analyze_ticker(args.ticker, require_sideways=not args.no_sideways) 