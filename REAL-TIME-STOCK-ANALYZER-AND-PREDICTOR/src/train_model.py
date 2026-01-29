import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../data/models"
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

def train_system(ticker):
    print(f"\n🧠 Training AI Suite for {ticker}...")
    
    path = os.path.join(PROCESSED_DIR, f"{ticker}.parquet")
    if not os.path.exists(path): return
    
    df = pd.read_parquet(path)
    
    # Features
    features = ['RSI', 'RSI_Slope', 'MACD_Hist', 'BB_Pos', 'Dist_SMA']
    
    # Normalize features for Logistic Regression
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    
    X = df[features].values
    y = df['Target'].values
    
    # Split
    split = int(len(X) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # --- MODEL 1: RANDOM FOREST (The Balanced Choice) ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    joblib.dump(rf, os.path.join(MODELS_DIR, f"{ticker}_RF.pkl"))
    print(f"   🌲 Random Forest Accuracy: {acc_rf:.2%}")

    # --- MODEL 2: GRADIENT BOOSTING (The Sniper) ---
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    acc_gb = accuracy_score(y_test, gb.predict(X_test))
    joblib.dump(gb, os.path.join(MODELS_DIR, f"{ticker}_GBM.pkl"))
    print(f"   🚀 Gradient Boost Accuracy: {acc_gb:.2%}")

    # --- MODEL 3: LOGISTIC REGRESSION (The Skeptic) ---
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test))
    joblib.dump(lr, os.path.join(MODELS_DIR, f"{ticker}_LR.pkl"))
    print(f"   📐 Logistic Reg Accuracy:  {acc_lr:.2%}")

if __name__ == "__main__":
    STOCKS = ['ADANIENT.NS', 'IDEA.NS', 'YESBANK.NS', 'TCS.NS', 'INFY.NS']
    for s in STOCKS: train_system(s)