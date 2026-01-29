import streamlit as st
import duckdb
import pandas as pd
import plotly.graph_objects as go
import time
import os
import joblib
import numpy as np
import json
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# 1. PAGE CONFIG
st.set_page_config(
    page_title="CDAC GROUP-6", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded" 
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(145deg, #1E1E1E, #252525);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .big-font { font-size: 32px !important; font-weight: 700; margin: 5px 0; }
    .label-font { color: #AAAAAA; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    .sub-label { font-size: 12px; color: #666; margin-top: 5px; }
    .js-plotly-plot .plotly .modebar { display: none; }
</style>
""", unsafe_allow_html=True)

# PATHS
try: current_dir = os.path.dirname(os.path.abspath(__file__))
except: current_dir = os.getcwd()

DATA_DIR = os.path.join(current_dir, '..', 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
CONFIG_PATH = os.path.join(DATA_DIR, 'stocks.json')

for d in [DATA_DIR, PROCESSED_DIR, MODELS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# ---------------------------------------------------------
# UTILS: MANAGE STOCKS
# ---------------------------------------------------------
def get_stock_list():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f: return json.load(f)
        except: pass
    return ['ADANIENT.NS', 'IDEA.NS', 'YESBANK.NS', 'TCS.NS', 'INFY.NS']

def save_stock_list(stocks):
    with open(CONFIG_PATH, 'w') as f: json.dump(stocks, f)

def train_new_stock(ticker):
    """Mini Pipeline to Download Data & Train Models on the fly"""
    try:
        df = yf.download(ticker, period="max", progress=False)
        if df.empty: return False, "No data found on Yahoo Finance"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Features
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['Std_20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA_20'] + 2*df['Std_20']
        df['BB_Lower'] = df['SMA_20'] - 2*df['Std_20']
        
        delta = df['Close'].diff()
        up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df['RSI'] = 100 - (100/(1+rs))
        
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        df['RSI_Slope'] = df['RSI'].diff(3)
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        
        df.to_parquet(os.path.join(PROCESSED_DIR, f"{ticker}.parquet"))
        
        X = df[['RSI', 'RSI_Slope', 'MACD_Hist', 'BB_Pos', 'Dist_SMA']].values
        y = df['Target'].values
        
        joblib.dump(RandomForestClassifier(n_estimators=100, max_depth=7).fit(X, y), os.path.join(MODELS_DIR, f"{ticker}_RF.pkl"))
        joblib.dump(GradientBoostingClassifier(n_estimators=50).fit(X, y), os.path.join(MODELS_DIR, f"{ticker}_GBM.pkl"))
        joblib.dump(LogisticRegression(max_iter=500).fit(X, y), os.path.join(MODELS_DIR, f"{ticker}_LR.pkl"))
        
        return True, "Success"
    except Exception as e: return False, str(e)

def delete_stock_data(ticker):
    """Removes stock from JSON, Models, Parquet and DB"""
    try:
        # 1. Update JSON
        current_list = get_stock_list()
        if ticker in current_list:
            current_list.remove(ticker)
            save_stock_list(current_list)
        
        # 2. Delete Files
        files = [
            os.path.join(PROCESSED_DIR, f"{ticker}.parquet"),
            os.path.join(MODELS_DIR, f"{ticker}_RF.pkl"),
            os.path.join(MODELS_DIR, f"{ticker}_GBM.pkl"),
            os.path.join(MODELS_DIR, f"{ticker}_LR.pkl")
        ]
        for f in files:
            if os.path.exists(f): os.remove(f)
            
        # 3. Clean DB
        if os.path.exists(DB_PATH):
            con = duckdb.connect(DB_PATH)
            con.execute(f"DELETE FROM realtime_stocks WHERE ticker = '{ticker}'")
            con.close()
            
        return True, f"Deleted {ticker}"
    except Exception as e:
        return False, str(e)

# ---------------------------------------------------------
# INFERENCE ENGINE
# ---------------------------------------------------------
def get_ai_signal(ticker, current_price, history_df, model_type="RF"):
    try:
        file_map = {"Random Forest": "RF", "Gradient Boost": "GBM", "Logistic Reg": "LR"}
        model_path = os.path.join(MODELS_DIR, f"{ticker}_{file_map.get(model_type, 'RF')}.pkl")
        if not os.path.exists(model_path): return 0.5 
        model = joblib.load(model_path)
        
        if history_df.empty: return 0.5 
        df = history_df.copy()
        new_row = pd.DataFrame({'Close': [current_price]})
        df = pd.concat([df, new_row], ignore_index=True)
        
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['Std_20'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['SMA_20'] + 2*df['Std_20']
        df['BB_Lower'] = df['SMA_20'] - 2*df['Std_20']
        
        delta = df['Close'].diff()
        up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        df['RSI'] = 100 - (100/(1+rs))
        
        exp12 = df['Close'].ewm(span=12).mean()
        exp26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        df['RSI_Slope'] = df['RSI'].diff(3)
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['Dist_SMA'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        
        last_row = df[['RSI', 'RSI_Slope', 'MACD_Hist', 'BB_Pos', 'Dist_SMA']].iloc[-1].values.reshape(1, -1)
        if np.isnan(last_row).any(): return 0.5
        
        return model.predict_proba(last_row)[0][1]
    except: return 0.5

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("🎛 Control Panel")
stock_list = get_stock_list()

# --- MANAGE STOCKS SECTION ---
with st.sidebar.expander("⚙️ Manage Stocks", expanded=False):
    tab1, tab2 = st.tabs(["➕ Add", "🗑️ Delete"])
    
    with tab1:
        new_ticker = st.text_input("Ticker (e.g. RELIANCE.NS)")
        if st.button("Add Stock"):
            if new_ticker and new_ticker not in stock_list:
                with st.spinner(f"Training AI for {new_ticker}..."):
                    success, msg = train_new_stock(new_ticker)
                    if success:
                        stock_list.append(new_ticker)
                        save_stock_list(stock_list)
                        st.success("Added!")
                        time.sleep(1); st.rerun()
                    else: st.error(msg)
            elif new_ticker in stock_list:
                st.warning("Exists!")

    with tab2:
        del_ticker = st.selectbox("Select to Delete", stock_list, key="del_box")
        if st.button("Confirm Delete"):
            if del_ticker:
                success, msg = delete_stock_data(del_ticker)
                if success:
                    st.success(f"Deleted {del_ticker}")
                    time.sleep(1); st.rerun()
                else: st.error(msg)

# MAIN CONTROLS
selected_ticker = st.sidebar.selectbox("Select Asset", stock_list) if stock_list else None
selected_model = st.sidebar.radio("Select AI Model", ["Random Forest", "Gradient Boost", "Logistic Reg"])
view_mode = st.sidebar.radio("View Mode", ["Live (Intraday)", "Daily (3 Months)"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Active Model: **{selected_model}**")

# Handle Empty List Case
if not selected_ticker:
    st.warning("No stocks found. Please add a stock in the sidebar.")
    st.stop()

# Load History
context_path = os.path.join(PROCESSED_DIR, f"{selected_ticker}.parquet")
df_history = pd.read_parquet(context_path).tail(100) if os.path.exists(context_path) else pd.DataFrame()

# ---------------------------------------------------------
# DASHBOARD LOOP
# ---------------------------------------------------------
@st.fragment(run_every=1)
def live_dashboard_loop():
    try:
        if not os.path.exists(DB_PATH):
            st.error("Database missing. Run `producer.py`."); return

        con = duckdb.connect(DB_PATH, read_only=True)
        try: con.execute("SELECT 1 FROM realtime_stocks LIMIT 1")
        except: st.warning("⏳ Waiting for Producer..."); con.close(); return

        # 1. FETCH METRICS DATA (Always Live)
        df_live = con.execute(f"SELECT * FROM realtime_stocks WHERE ticker='{selected_ticker}' ORDER BY timestamp DESC LIMIT 100").df()
        con.close()
        
        if not df_live.empty:
            current_price = df_live.iloc[0]['price']
            
            # AI Inference
            prob = get_ai_signal(selected_ticker, current_price, df_history, selected_model)
            prob_pct = prob * 100
            
            st.markdown(f"<h2 style='text-align: center; margin-top: 10px; margin-bottom: 20px;'>{selected_ticker} Analysis</h2>", unsafe_allow_html=True)
            
            # --- METRICS (Always Live) ---
            c1, c2, c3 = st.columns([1, 1.2, 1])
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label-font">Last Price</div>
                    <div class="big-font" style="color: #00F0FF;">₹{current_price:.2f}</div>
                    <div class="sub-label">Updated Real-time</div>
                </div>""", unsafe_allow_html=True)

            with c2:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = prob_pct,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    number = {'suffix': "%", 'font': {'size': 26, 'color': 'white'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#333"},
                        'bar': {'color': "rgba(0,0,0,0)"}, 'bgcolor': "#1E1E1E", 'borderwidth': 0,
                        'steps': [{'range': [0, 45], 'color': "#D32F2F"}, {'range': [45, 55], 'color': "#424242"}, {'range': [55, 100], 'color': "#388E3C"}],
                        'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': prob_pct}
                    }
                ))
                fig_gauge.update_layout(height=140, margin=dict(l=20,r=20,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.markdown(f"""<div style="text-align: center; color: #AAA; font-size: 14px; margin-top: -10px;">BULLISH PROBABILITY</div>""", unsafe_allow_html=True)

            with c3:
                if prob > 0.60:   decision = "STRONG BUY"; color = "#4CAF50"
                elif prob > 0.52: decision = "BUY"; color = "#81C784"
                elif prob < 0.40: decision = "STRONG SELL"; color = "#F44336"
                elif prob < 0.48: decision = "SELL"; color = "#E57373"
                else:             decision = "NEUTRAL"; color = "#9E9E9E"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label-font">Recommendation</div>
                    <div class="big-font" style="color:{color};">{decision}</div>
                    <div class="sub-label">Confidence: {abs(prob-0.5)*200:.0f}%</div>
                </div>""", unsafe_allow_html=True)

            # --- CHART LOGIC (Live vs Daily) ---
            if view_mode == "Live (Intraday)":
                chart_data = df_live.sort_values('timestamp')
                x_col = 'timestamp'
                y_col = 'price'
                title_text = "📉 Intraday Price Action (IST)"
            else:
                # Daily Mode
                if not df_history.empty:
                    chart_data = df_history.copy().reset_index() # Ensure index is available
                    # Parquet usually has Date as index or column. Handle both.
                    if 'Date' not in chart_data.columns and isinstance(df_history.index, pd.DatetimeIndex):
                        chart_data['Date'] = df_history.index
                    x_col = 'Date' # Or index
                    y_col = 'Close'
                    title_text = "📅 Daily Price History (Last 3 Months)"
                else:
                    st.warning("History data not available.")
                    chart_data = pd.DataFrame()

            if not chart_data.empty:
                # --- AUTO-SCALE Y-AXIS ---
                y_min = chart_data[y_col].min()
                y_max = chart_data[y_col].max()
                padding = (y_max - y_min) * 0.05 # 5% padding
                if padding == 0: padding = y_max * 0.01 # Handle flat line
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_data[x_col], y=chart_data[y_col],
                    mode='lines',
                    fill='none', # Removed 'tozeroy' to allow correct auto-scaling
                    line=dict(color='#00F0FF', width=2),
                    name='Price'
                ))
                
                fig.update_layout(
                    template="plotly_dark", height=400, margin=dict(l=0,r=0,t=30,b=0),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, title="Time"), 
                    # FORCE Y-AXIS RANGE
                    yaxis=dict(showgrid=True, gridcolor="#222", title="Price (INR)", range=[y_min - padding, y_max + padding]),
                    hovermode="x unified", title={'text': title_text, 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(f"Waiting for live data for {selected_ticker}...")
    except Exception as e:
        st.error(f"Error: {e}")

live_dashboard_loop()