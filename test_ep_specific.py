#!/usr/bin/env python3
"""
Quick test of EP detector with specific popular tickers
"""

from ep_detector import EPDetector

def test_specific_tickers():
    """Test EP detector with specific well-known tickers"""
    
    # Popular tickers that might have EP patterns
    test_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'AMD', 'COIN', 'PLTR', 'SOFI', 'RBLX', 'HOOD', 'RIVN', 'LCID',
        'SNOW', 'CRWD', 'ZM', 'SHOP', 'SQ', 'ROKU', 'UBER', 'LYFT',
        'TDOC', 'PTON', 'BYND', 'SPCE', 'GME', 'AMC', 'BB', 'NOK'
    ]
    
    print("=== EP DETECTOR TEST - SPECIFIC TICKERS ===")
    print(f"Testing {len(test_tickers)} popular tickers for EP setups...\n")
    
    detector = EPDetector()
    results = []
    
    for i, ticker in enumerate(test_tickers):
        print(f"Testing {i+1:2}/{len(test_tickers)}: {ticker:6}", end=" ")
        
        try:
            result = detector.screen_for_ep(ticker)
            if result:
                results.append(result)
                print(f"✓ EP! Score: {result['ep_score']:2.0f} | Gap: {result['gap_percent']:5.1f}% | Vol: {result['volume_surge']:4.1f}x")
            else:
                print("-")
        except Exception as e:
            print(f"✗ Error: {str(e)[:30]}")
    
    print(f"\n=== RESULTS ===")
    if results:
        print(f"Found {len(results)} EP candidates:")
        
        # Sort by score
        results.sort(key=lambda x: x['ep_score'], reverse=True)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['ticker']}")
            print(f"   Score: {result['ep_score']}/100")
            print(f"   Gap: {result['gap_percent']:.1f}% | Volume: {result['volume_surge']:.1f}x")
            print(f"   Price: ${result['current_price']:.2f} | RSI: {result['rsi']:.1f}")
            print(f"   1D: {result['price_1d']:+.1f}% | 5D: {result['price_5d']:+.1f}%")
            print(f"   Consolidation: {result['sideways_consolidation']['consolidation_days']} days")
            print(f"   Criteria: {', '.join(result['criteria_met'])}")
        
        # Save results
        detector.save_results(results, 'ep_test_specific_results.csv')
        
    else:
        print("No EP setups found in these specific tickers.")
        print("This is normal - EPs are rare and typically occur around earnings or major news.")
        print("Try running during earnings season or after market-moving events.")

if __name__ == "__main__":
    test_specific_tickers() 