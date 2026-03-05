import os
import glob
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import json
import yfinance as yf

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# Configure Alpaca API
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")

# Configure Discord Webhook
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
if not DISCORD_WEBHOOK_URL:
    raise ValueError("Please set DISCORD_WEBHOOK_URL in your .env file")

def list_available_models():
    """List all available Gemini models."""
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model: {m.name}")
            print(f"Display name: {m.display_name}")
            print(f"Description: {m.description}")
            print("-" * 50)

def get_latest_scan_file():
    """Get the most recent ep_scan_results CSV file."""
    files = glob.glob('ep_scan_results_*.csv')
    if not files:
        raise FileNotFoundError("No ep_scan_results CSV files found")
    return max(files, key=os.path.getctime)

def fetch_news_for_ticker(ticker, limit=3):
    """Fetch latest news for a ticker from Finviz."""
    try:
        # Construct the Finviz URL
        url = f"https://finviz.com/quote.ashx?t={ticker}&ty=c&ta=1&p=d"
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all news rows
        news_rows = soup.find_all('tr', {'class': 'cursor-pointer'})
        if not news_rows:
            return [f"No news found for {ticker}"]
        
        news_list = []
        for row in news_rows[:limit]:  # Limit the number of news items
            try:
                # Get the date and time from the first td
                date_td = row.find('td', {'width': '130'})
                if not date_td:
                    continue
                date_text = date_td.text.strip()
                
                # Get the news link container
                news_container = row.find('div', {'class': 'news-link-container'})
                if not news_container:
                    continue
                
                # Get the news text from the left div
                news_left = news_container.find('div', {'class': 'news-link-left'})
                if not news_left:
                    continue
                news_link = news_left.find('a', {'class': 'tab-link-news'})
                if not news_link:
                    continue
                news_text = news_link.text.strip()
                
                # Get the source from the right div
                news_right = news_container.find('div', {'class': 'news-link-right'})
                if not news_right:
                    continue
                source = news_right.find('span').text.strip('()')
                
                # Format the news item
                news_list.append(f"[{date_text}] {news_text} (Source: {source})")
                
            except Exception as e:
                print(f"Error parsing news row: {e}")
                continue
        
        # Add a small delay to be respectful to the website
        time.sleep(1)
        
        return news_list if news_list else [f"No recent news found for {ticker}"]
        
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return [f"Error fetching news: {e}"]

def get_market_cap(ticker):
    """Get market cap for a ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', 0)
        return market_cap
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {e}")
        return 0

def format_market_cap(market_cap):
    """Format market cap in human readable format (e.g., 1.2B, 500M)."""
    if market_cap >= 1_000_000_000:  # Billion
        return f"{market_cap/1_000_000_000:.1f}B"
    elif market_cap >= 1_000_000:  # Million
        return f"{market_cap/1_000_000:.1f}M"
    else:
        return f"{market_cap:,.0f}"

def analyze_tickers_with_gemini(tickers_data, news_dict):
    """Analyze tickers using Gemini AI, including news and market cap."""
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    # Get market cap for each ticker
    market_caps = {ticker: get_market_cap(ticker) for ticker in tickers_data}
    
    # Sort tickers by market cap (descending)
    sorted_tickers = sorted(tickers_data.keys(), key=lambda x: market_caps[x], reverse=True)
    
    # Prepare the prompt
    prompt = """Analyze these stock tickers that showed gap up patterns. For each ticker, use both the technical data and the latest news headlines to explain:
1. What this stock does and why the gap up likely occurred (within 5 sentences, referencing news if relevant)
2. Technical strength assessment (1 sentence)

Data to analyze:
"""
    for ticker in sorted_tickers:
        data = tickers_data[ticker]
        market_cap = format_market_cap(market_caps[ticker])
        prompt += f"\nTicker: {ticker} (Market Cap: {market_cap})\nTechnical Data: {data}\nNews Headlines:\n"
        for news in news_dict.get(ticker, []):
            prompt += f"  {news}\n"
    prompt += "\nFormat the response as a brief report with bullet points for each ticker, maintaining the order by market cap (largest to smallest), also include the market cap right after the ticker name."
    
    # Generate analysis
    response = model.generate_content(prompt)
    return response.text

def send_to_discord(content, title="Stock Analysis Report"):
    """Send a message to Discord channel using webhook."""
    try:
        # Split content if it's too long (Discord has a 4096 character limit for embeds)
        if len(content) > 4000:  # Leave some room for title and other formatting
            chunks = [content[i:i+4000] for i in range(0, len(content), 4000)]
        else:
            chunks = [content]

        for i, chunk in enumerate(chunks):
            # Create the embed message
            embed = {
                "title": f"{title} (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else title,
                "description": chunk,
                "color": 0x00ff00,  # Green color
                "timestamp": datetime.now().isoformat()
            }
            
            # Create the payload
            payload = {
                "embeds": [embed]
            }
            
            # Print the payload for debugging
            print(f"Sending payload to Discord: {json.dumps(payload, indent=2)}")
            
            # Send the webhook
            response = requests.post(
                DISCORD_WEBHOOK_URL,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            # Print response details for debugging
            print(f"Discord API Response Status: {response.status_code}")
            print(f"Discord API Response: {response.text}")
            
            response.raise_for_status()
            
            # Add a small delay between chunks
            if i < len(chunks) - 1:
                time.sleep(1)
        
        print("Successfully sent report to Discord")
        
    except requests.exceptions.RequestException as e:
        print(f"Error sending to Discord: {e}")
        if hasattr(e.response, 'text'):
            print(f"Discord API Error Response: {e.response.text}")
    except Exception as e:
        print(f"Unexpected error sending to Discord: {e}")

def main():
    try:
        # First, list available models
        print("Available Gemini Models:")
        print("=" * 50)
        list_available_models()
        print("\n")
        
        # Get the latest scan file
        latest_file = get_latest_scan_file()
        print(f"Analyzing data from: {latest_file}")
        
        # Read the CSV file
        df = pd.read_csv(latest_file)
        
        # Prepare data for analysis
        tickers_data = {row['ticker']: row.to_dict() for _, row in df.iterrows()}
        
        # Fetch news for each ticker
        print("Fetching news for tickers...")
        news_dict = {ticker: fetch_news_for_ticker(ticker) for ticker in tickers_data}
        
        # Get AI analysis
        analysis = analyze_tickers_with_gemini(tickers_data, news_dict)
        
        # Print the analysis
        print("\nAI Analysis Report:")
        print("=" * 50)
        print(analysis)
        print("=" * 50)
        
        # Send the analysis to Discord
        send_to_discord(analysis)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Send error to Discord if something goes wrong
        send_to_discord(f"Error occurred during analysis: {str(e)}", "Error Report")

if __name__ == "__main__":
    main()
