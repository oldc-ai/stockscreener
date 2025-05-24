# Episodic Pivot (EP) Detector - Technical Analysis Focus

A Python script that scans stocks for **Episodic Pivot** setups using **Alpaca market data** and focusing on **technical analysis only**. Based on [Qullamaggie's methodology](https://qullamaggie.com/how-to-master-a-setup-episodic-pivots/).

## What is an Episodic Pivot?

An Episodic Pivot is a gap up of **10% or more** with **massive volume** caused by news that forces a revaluation of the stock. According to Qullamaggie, these setups can be extremely profitable when traded correctly.

### Key EP Criteria (Technical Focus):1. **Gap up of 10% or more** ⚠️ **MANDATORY REQUIREMENT**2. **Massive volume surge** (3x+ average daily volume)3. **Stocks that have been sideways** for 3-6+ months4. **Price action confirmation** (technical strength)

## Features

This script scans for EP setups by analyzing **technical indicators only**:

- ✅ **Gap percentage** (overnight/premarket gaps)- ✅ **Volume surge analysis** (current vs 20-day average)  - ✅ **Sideways consolidation detection** (3-6 month base building)- ✅ **Price strength metrics** (RSI, moving averages, recent highs)- ✅ **Volatility analysis** (ATR, price volatility)- ✅ **Penny stock filter** (automatically excludes stocks < $1)- ✅ **Technical scoring system** (0-100 points)

## Data Source

- **Alpaca Markets API** - Professional-grade market data
- **Real-time and historical** price/volume data
- **Higher reliability** than free data sources
- **Better rate limits** for scanning

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set up Alpaca API credentials:
```bash
# Set environment variables (recommended)
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_SECRET_KEY="your_secret_key_here"
```

3. Make sure you have `all_tickers.txt` with ticker symbols (one per line)

**Note:** The script works without Alpaca credentials using free market data, but having an account provides better reliability.

## Usage

### Basic Usage (Test Run)
```bash
python ep_detector.py
```
*This scans the first 50 tickers by default for testing*

### Scan All Tickers
```python
# Edit ep_detector.py and change this line:
results = detector.scan_all_tickers(max_tickers=None)  # Remove max_tickers limit
```

### Custom Usage with API Keys
```python
from ep_detector import EPDetector

# With API credentials
detector = EPDetector(api_key='your_key', secret_key='your_secret')

# Or use environment variables
detector = EPDetector()

# Scan specific tickers
custom_tickers = ['AAPL', 'TSLA', 'NVDA']
results = []
for ticker in custom_tickers:
    result = detector.screen_for_ep(ticker)
    if result:
        results.append(result)

# Save results
detector.save_results(results, 'my_custom_scan.csv')
```

## Output

The script generates:

1. **Real-time console output** with scanning progress
2. **CSV file** with detailed technical analysis (`ep_scan_results_YYYYMMDD_HHMMSS.csv`)
3. **Summary report** with top technical candidates

### Sample Output:
```
=== EPISODIC PIVOT DETECTOR ===
Technical Analysis Focus - Using Alpaca Data

Scanning 1/50: A      -
Scanning 2/50: AA     -
Scanning 3/50: AAPL   ✓ EP FOUND! Score: 75 | Gap: 12.3% | Vol: 4.2x
...

=== TOP EP CANDIDATES ===

1. AAPL
   Score: 75/100
   Gap: 12.3% | Volume: 4.2x avg
   Price: $187.45 | RSI: 68.5
   1D: +11.2% | 5D: +8.7%
   Consolidation: 89 days, 15.3% range
   Criteria: Gap up: 12.3%, Volume surge: 4.2x, Sideways 89 days
```

## Technical Scoring System

The script uses a **100-point technical scoring system**:

### Gap Analysis (40 points max)
- **15%+ gap** = 40 points  
- **10-15% gap** = 30 points
- **7-10% gap** = 15 points

### Volume Analysis (30 points max)
- **5x+ avg volume** = 30 points
- **3-5x avg volume** = 25 points  
- **2-3x avg volume** = 10 points

### Consolidation Analysis (20 points max)
- **60+ days sideways** = 20 points
- **30+ days consolidation** = 10 points
- *Sideways = <40% trading range*

### Price Strength (10 points max)
- **Recent new highs** = 5 points
- **Above 50-day MA** = 5 points

**Minimum Score**: 30 points (after 10%+ gap requirement is met)⚠️ **IMPORTANT**: All candidates must have a 10%+ gap up before any scoring begins. This is a hard requirement, not optional.

## Key Files

- `ep_detector.py` - Main scanner script (Alpaca-based)
- `all_tickers.txt` - List of ticker symbols to scan  
- `requirements.txt` - Python dependencies
- `ep_scan_results_*.csv` - Output files with technical analysis

## Technical Indicators Analyzed

### Price Metrics
- Current price, gap percentage
- 1-day and 5-day price changes
- 20-day and 50-day moving averages

### Volume Analysis  
- Current volume vs 20-day average
- Volume surge multiplier
- Volume consistency

### Volatility Metrics
- Average True Range (ATR)
- 20-day price volatility  
- Trading range analysis

### Momentum Indicators
- RSI (14-period)
- Recent high detection
- Moving average position

## Rate Limiting & Performance

- **Alpaca-optimized** rate limiting (1 second per 5 requests)
- **Faster scanning** than free APIs
- **Error handling** for network issues
- **Progress tracking** every 50 tickers

## Customization

### Adjust Technical Criteria
```python
# In screen_for_ep() method, modify thresholds:

# Increase minimum gap requirement
if gap_percent >= 15:  # Changed from 10
    ep_score += 40

# Increase volume requirement  
if volume_surge >= 5:  # Changed from 3
    ep_score += 30
```

### Add Custom Technical Filters```python# Default: Filters out penny stocks (< $1) automatically# You can adjust this threshold:if current_price < 5:  # Filter stocks under $5    return None# Filter by volatilityif volatility_metrics["volatility"] > 10:  # Very volatile    return None# Filter by volumeif avg_volume_20d < 100000:  # Low volume stocks    return None```

### Modify Consolidation Detection
```python
# Tighter consolidation requirement
is_sideways = range_percent < 25  # Changed from 40

# Longer consolidation period
lookback_days = 180  # Changed from 120 (6 months)
```

## Advanced Usage

### Real-time Market Scanning
```bash
# Run during market hours for live data
python ep_detector.py

# Automated scanning (Linux/Mac)
while true; do
    python ep_detector.py
    sleep 3600  # Run every hour
done
```

### Integration Examples
```python
# Export to JSON for other tools
import json
with open('ep_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Filter by specific criteria
high_score_eps = [r for r in results if r['ep_score'] >= 60]
big_gap_eps = [r for r in results if r['gap_percent'] >= 15]
```

### Backtesting Setup
```python
# Scan historical dates
from datetime import datetime, timedelta

# Scan for EPs from specific date range
historical_detector = EPDetector()
# Modify get_stock_data() to use custom date ranges
```

## Troubleshooting

### Common Issues:

1. **"No data" errors**: Some tickers may be delisted or have insufficient history
2. **Rate limiting**: Script automatically handles Alpaca rate limits  
3. **API credentials**: Script works without credentials but may have limitations
4. **Network timeouts**: Script continues scanning other tickers

### Optimization Tips:

1. **Use API credentials** for better reliability
2. **Scan during market hours** for most current data
3. **Filter ticker list** to focus on liquid stocks
4. **Adjust scoring thresholds** based on market conditions

## Market Context

### Best Times to Scan:
- **Earnings season** (quarterly)
- **After market close** for gap analysis  
- **Pre-market hours** for overnight gaps
- **During high news periods**

### Market Conditions:
- **Bull markets**: Lower thresholds may work
- **Bear markets**: Raise minimum criteria
- **Low volatility**: Focus on bigger gaps
- **High volatility**: Add volume confirmation

## References

- [Qullamaggie EP Setup Guide](https://qullamaggie.com/how-to-master-a-setup-episodic-pivots/)
- [Alpaca Markets API Documentation](https://alpaca.markets/docs/)
- Original methodology by Pradeep Bonde (Stockbee)

---

**Disclaimer**: This tool is for educational and research purposes. Technical analysis does not guarantee future results. Always do your own research and risk management before making any trading decisions. 