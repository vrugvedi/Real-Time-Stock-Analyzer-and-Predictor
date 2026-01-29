import duckdb
import yfinance as yf
import time
import os
import datetime
import pandas as pd
import pytz
import random
import numpy as np
import json

# --- SETTINGS ---
DEMO_MODE = True  
DEFAULT_STOCKS = ['ADANIENT.NS', 'IDEA.NS', 'YESBANK.NS', 'TCS.NS', 'INFY.NS']
# ----------------

DB_PATH = "../data/stock_data.db"
CONFIG_PATH = "../data/stocks.json"

# Ensure directories exist
if not os.path.exists("../data"):
    os.makedirs("../data")

# Initialize Config if missing
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(DEFAULT_STOCKS, f)

def get_watched_stocks():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except:
        return DEFAULT_STOCKS

def fetch_live_data():
    # Initialize DB
    try:
        con = duckdb.connect(DB_PATH)
        con.execute("CREATE TABLE IF NOT EXISTS realtime_stocks (timestamp TIMESTAMP, ticker VARCHAR, price DOUBLE, open DOUBLE, high DOUBLE, low DOUBLE, volume DOUBLE)")
        con.close()
    except Exception as e:
        print(f"❌ DB Init Error: {e}")

    IST = pytz.timezone('Asia/Kolkata')
    print(f"📡 Producer Active... (Demo Mode: {DEMO_MODE})")
    print(f"📂 Database: {DB_PATH}")

    last_prices = {}

    while True:
        try:
            current_stocks = get_watched_stocks()
            if not current_stocks:
                print("⚠️ Stock list is empty. Check stocks.json"); time.sleep(5); continue

            # Connect to DB
            con = duckdb.connect(DB_PATH)
            now = datetime.datetime.now(IST).replace(microsecond=0, tzinfo=None)
            
            if not DEMO_MODE:
                # --- REAL MODE ---
                # print(f"📥 Downloading live data for {len(current_stocks)} stocks...")
                try:
                    data = yf.download(current_stocks, period='1d', interval='1m', progress=False, timeout=5)
                    
                    for ticker in current_stocks:
                        try:
                            # Handle structure diff between 1 stock and N stocks
                            if len(current_stocks) == 1:
                                price = float(data['Close'].iloc[-1])
                                open_p = float(data['Open'].iloc[-1])
                                high_p = float(data['High'].iloc[-1])
                                low_p = float(data['Low'].iloc[-1])
                                vol = float(data['Volume'].iloc[-1])
                            else:
                                price = float(data['Close'][ticker].iloc[-1])
                                open_p = float(data['Open'][ticker].iloc[-1])
                                high_p = float(data['High'][ticker].iloc[-1])
                                low_p = float(data['Low'][ticker].iloc[-1])
                                vol = float(data['Volume'][ticker].iloc[-1])
                            
                            if pd.isna(price): continue
                            con.execute(f"INSERT INTO realtime_stocks VALUES ('{now}', '{ticker}', {price}, {open_p}, {high_p}, {low_p}, {vol})")
                        except: continue
                except Exception as e:
                    print(f"⚠️ Feed Error: {e}")
            
            else:
                # --- DEMO SIMULATION MODE ---
                # print("🎲 Generating simulation data...")
                for ticker in current_stocks:
                    # Initialize Price (First Run Only)
                    if ticker not in last_prices:
                        try:
                            # Try fetching real price once
                            # print(f"   🔎 Fetching start price for {ticker}...")
                            hist = yf.Ticker(ticker).history(period='1d')
                            if not hist.empty:
                                last_prices[ticker] = hist['Close'].iloc[-1]
                            else:
                                raise ValueError("Empty")
                        except:
                            # FALLBACK: If internet fails, use random price
                            # print(f"   ⚠️ Network failed. Using fallback price for {ticker}")
                            last_prices[ticker] = random.uniform(500, 3000)
                    
                    # Random Walk Math
                    change_pct = random.uniform(-0.002, 0.002) # +/- 0.2%
                    price = last_prices[ticker] * (1 + change_pct)
                    last_prices[ticker] = price
                    
                    # Fake OHLC
                    open_p = price * (1 + random.uniform(-0.0005, 0.0005))
                    high_p = max(price, open_p) * (1 + random.uniform(0, 0.001))
                    low_p = min(price, open_p) * (1 - random.uniform(0, 0.001))
                    vol = random.randint(1000, 50000)
                    
                    con.execute(f"INSERT INTO realtime_stocks VALUES ('{now}', '{ticker}', {price}, {open_p}, {high_p}, {low_p}, {vol})")

            con.close()
            print(f"✅ Tick: {now.strftime('%H:%M:%S')} | {len(current_stocks)} Stocks Updated")
            time.sleep(60)
            
        except KeyboardInterrupt:
            print("\n🛑 Producer Stopped.")
            break
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    fetch_live_data()