# Stock Scanner

An automated stock scanner that detects Episodic Pivot patterns and analyzes them using AI. This tool helps identify potential breakout stocks based on technical analysis and news sentiment.

## Features

- Detects Episodic Pivot (EP) patterns in stocks
- Analyzes technical indicators and price action
- Fetches and analyzes relevant news
- Uses AI to generate comprehensive analysis
- Sends reports to Discord
- Runs automatically twice daily on weekdays

## GitHub Actions Setup

1. Fork this repository to your GitHub account

2. Set up GitHub Secrets:
   - Go to your repository settings
   - Click on "Secrets and variables" → "Actions"
   - Add the following secrets:
     ```
     GOOGLE_API_KEY=your_key_here
     ALPACA_API_KEY=your_key_here
     ALPACA_SECRET_KEY=your_key_here
     DISCORD_WEBHOOK_URL=your_webhook_here
     ```

3. The workflow will automatically run:
   - Every weekday at 9:00 AM ET
   - Every weekday at 10:00 PM ET
   - You can also manually trigger it from the "Actions" tab

4. To modify the schedule:
   - Edit the cron expressions in `.github/workflows/stock_scanner.yml`
   - Current schedule:
     - `0 13 * * 1-5` (9:00 AM ET on weekdays)
     - `0 2 * * 2-6` (10:00 PM ET on weekdays)

## Required API Keys

1. **Google API Key**
   - Used for AI analysis
   - Get it from [Google Cloud Console](https://console.cloud.google.com)
   - Enable the Gemini API

2. **Alpaca API Keys**
   - Used for stock data
   - Get them from [Alpaca Markets](https://alpaca.markets)
   - Paper trading keys are sufficient

3. **Discord Webhook URL**
   - Used for sending reports
   - Create a webhook in your Discord server settings

## Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a .env file with your API keys:
   ```
   GOOGLE_API_KEY=your_key_here
   ALPACA_API_KEY=your_key_here
   ALPACA_SECRET_KEY=your_key_here
   DISCORD_WEBHOOK_URL=your_webhook_here
   ```

4. Run the scripts:
   ```bash
   python ep_detector.py
   python ai_analyzer.py
   ```

## Bull Run Research

Use `bull_run_system.py` to re-check the cached 10-bagger list against fresh
adjusted price history, discard low-quality names, and export an initial
add/caution/exit framework for sustained bull runs.

1. Install the dependencies listed in `requirements.txt`.
2. Run the research script:
   ```bash
   PYTHONPATH=.deps python3 bull_run_system.py
   ```
3. Review the outputs in `analysis_outputs/`:
   - `bull_run_report.md`
   - `bull_run_candidates_clean.csv`
   - `bull_run_signal_summary.csv`
   - `bull_run_signal_events.csv`
   - `bull_run_latest_actions.csv`

Current filter defaults keep only names that still show a fresh adjusted-data
10x run, started above $2, lasted at least 252 trading days, and had workable
liquidity. The exit rule is intentionally conservative and should be treated as
the weakest part of the first-pass system.

## Daily Trading System

Use `daily_trading_system.py` for the daily workflow:

1. Detect new ideas using:
   - a CANSLIM-style proxy score
   - Qullamaggie-style EP detection
2. Read `watchlist.csv` as the list of current holdings.
3. Produce position guidance for owned names:
   - `ADD_ON_BREAKOUT`
   - `ADD_ON_PULLBACK`
   - `PREPARE_HEDGE`
   - `HEDGE`
   - `EXIT`
4. Send the daily report to Discord.

The CANSLIM logic is a proxy implementation because the repo does not have IBD's
proprietary ratings. It uses available earnings growth, revenue growth,
institutional ownership, relative strength, liquidity, and market regime data.

### Watchlist

Local use: fill in `watchlist.csv` with your holdings:

```csv
symbol,shares,cost_basis,notes
NVDA,50,118.25,Core AI leader
APP,30,85.10,Tighter risk after parabolic move
```

GitHub Actions use: set a `WATCHLIST_CSV` repository secret with the same CSV
content. That lets the workflow use your holdings without committing them to the
repo. A sample format is in `watchlist.example.csv`.

### Local Dry Run

```bash
PYTHONPATH=.deps python3 daily_trading_system.py --max-tickers 50 --use-yfinance-only --dry-run --no-discord
```

### Daily Automation

The workflow file `.github/workflows/daily_trading_system.yml` runs the system on
weekdays after the US cash close and sends the report to `DISCORD_WEBHOOK_URL`.
Required secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `DISCORD_WEBHOOK_URL`
- `WATCHLIST_CSV`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational purposes only. Do not use it for actual trading without proper risk management and understanding of the markets. 
