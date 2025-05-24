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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This tool is for educational purposes only. Do not use it for actual trading without proper risk management and understanding of the markets. 