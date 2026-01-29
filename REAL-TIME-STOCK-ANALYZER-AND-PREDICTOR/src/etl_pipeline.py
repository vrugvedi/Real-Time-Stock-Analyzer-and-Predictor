import pandas as pd
import numpy as np
import yfinance as yf
import os

# CONFIG
STOCKS = ['ADANIENT.NS', 'IDEA.NS', 'YESBANK.NS', 'TCS.NS', 'INFY.NS']
PROCESSED_DIR = "../data/processed"

if not os.path.exists(PROCESSED_DIR): os.makedirs(PROCESSED_DIR)

def calculate_features(df):
    df = df.copy()
    
    # 1. Standard Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['Std_20'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['Std_20'])
    
    # RSI (14)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 2. Advanced: Momentum Slopes (Is RSI rising or falling?)
    df['RSI_Slope'] = df['RSI'].diff(3) # Change in RSI over 3 days
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 3. Target: 1 if Tomorrow > Today, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df

def run_pipeline():
    print("🚀 Starting Classification Pipeline...")
    for ticker in STOCKS:
        print(f"📥 Processing {ticker}...")
        try:
            df = yf.download(ticker, period="max", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if df.empty: continue
            
            df_clean = calculate_features(df)
            
            save_path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
            df_clean.to_parquet(save_path)
            print(f"✅ Saved {len(df_clean)} rows.")
            
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")

if __name__ == "__main__":
    run_pipeline()