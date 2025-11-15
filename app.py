import streamlit as st
import ccxt
import pandas as pd
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import plotly.graph_objects as go

# --- Configuration ---
SYMBOL = 'BNB/USDT'
TIMEFRAME = '1m'
LOOKBACK = 300  # use last N candles
APPLY_RSI_FILTER = True

# Initialize exchange
try:
    ex = ccxt.kucoin()
except Exception as e:
    st.error(f"Error initializing exchange: {e}")
    st.stop()

# --- Page Setup ---
st.set_page_config(
    page_title=f"{SYMBOL} Trading Signal",
    page_icon="📈",
    layout="wide"
)

st.title(f"📈 {SYMBOL} Real-Time Signal Dashboard")
st.caption(f"Tracking {SYMBOL} on the {TIMEFRAME} timeframe. Data refreshes every 60 seconds.")

# --- Initialize Session State for History ---
# This creates a persistent list for signal history and a way to track the last signal
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = pd.DataFrame(columns=['timestamp', 'signal', 'price', 'reason'])
if 'last_signal_key' not in st.session_state:
    st.session_state.last_signal_key = None


# --- Core Logic Functions ---

@st.cache_data(ttl=60)  # Cache data for 60 seconds
def fetch_df():
    """Fetches OHLCV data and calculates indicators."""
    try:
        ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LOOKBACK)
        d = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms', utc=True)
        
        # Calculate indicators
        d['ema9'] = EMAIndicator(d['close'], window=9).ema_indicator()
        d['ema21'] = EMAIndicator(d['close'], window=21).ema_indicator()
        d['rsi14'] = RSIIndicator(d['close'], window=14).rsi()
        
        return d.dropna().reset_index(drop=True)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame() # Return empty dataframe on error

def get_signal(d):
    """Determines the trading signal and updates history."""
    i = len(d) - 1
    if i < 1:  # Need at least 2 rows to compare
        return {'timestamp': pd.NaT, 'price': 0, 'rsi': 0, 'signal': 'WAITING', 'reason': 'Initializing data...'}

    # Get current values
    price = d.loc[i, 'close']
    rsi = d.loc[i, 'rsi14']
    t = d.loc[i, 'timestamp']
    
    # Crossover logic
    bull = (d.loc[i, 'ema9'] > d.loc[i, 'ema21']) and (d.loc[i-1, 'ema9'] <= d.loc[i-1, 'ema21'])
    bear = (d.loc[i, 'ema9'] < d.loc[i, 'ema21']) and (d.loc[i-1, 'ema9'] >= d.loc[i-1, 'ema21'])

    # Apply RSI filter if enabled
    if APPLY_RSI_FILTER:
        bull = bull and (rsi < 70)
        bear = bear and (rsi > 30)
    
    # Determine signal
    signal = 'HOLD'
    reason = 'No crossover signal'
    
    if bull:
        signal = 'BUY'
        reason = 'EMA-9 crossed above EMA-21 (RSI < 70)'
    elif bear:
        signal = 'SELL'
        reason = 'EMA-9 crossed below EMA-21 (RSI > 30)'
        
    # --- Update History Logic ---
    # A unique key for this specific signal event (the timestamp)
    current_signal_key = t
    
    # Check if this is a new signal time we haven't logged yet
    # This logs EVERY signal (BUY, SELL, or HOLD) once per minute
    if current_signal_key != st.session_state.last_signal_key:
        new_signal = pd.DataFrame([{
            'timestamp': t, 
            'signal': signal, 
            'price': price, 
            'reason': reason
        }])
        
        # Add the new signal to the top of the history DataFrame
        st.session_state.signal_history = pd.concat(
            [new_signal, st.session_state.signal_history], 
            ignore_index=True
        )
        st.session_state.last_signal_key = current_signal_key

    return {'timestamp': t, 'price': price, 'rsi': rsi, 'signal': signal, 'reason': reason}

def create_chart(d):
    """Creates a Plotly candlestick chart with EMA overlays."""
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=d['timestamp'],
        open=d['open'],
        high=d['high'],
        low=d['low'],
        close=d['close'],
        name='Price'
    ))
    
    # EMAs
    fig.add_trace(go.Scatter(
        x=d['timestamp'], 
        y=d['ema9'], 
        mode='lines', 
        name='EMA 9 (Fast)', 
        line=dict(color='cyan', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=d['timestamp'], 
        y=d['ema21'], 
        mode='lines', 
        name='EMA 21 (Slow)', 
        line=dict(color='orange', width=2)
    ))
    
    # Layout
    fig.update_layout(
        title=f'{SYMBOL} Price and Indicators',
        yaxis_title='Price (USDT)',
        xaxis_title='Timestamp (UTC)',
        xaxis_rangeslider_visible=False,
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Main App ---

# Fetch data
df = fetch_df()

if df.empty:
    st.warning("Could not fetch data. Please check connection or symbol and try again.")
    st.stop()

# Get current signal and update history
signal_info = get_signal(df)

# Display Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Last Price", f"${signal_info['price']:,.3f}")

# Color-code the signal
if signal_info['signal'] == 'BUY':
    signal_color = "success"
    signal_icon = "⬆️"
elif signal_info['signal'] == 'SELL':
    signal_color = "error"
    signal_icon = "⬇️"
else:
    signal_color = "normal"
    signal_icon = "➡️"

col2.metric("Current Signal", f"{signal_info['signal']} {signal_icon}", delta_color=signal_color)
col3.metric("RSI (14)", f"{signal_info['rsi']:.2f}")

st.info(f"**Signal Justification:** {signal_info['reason']}")
st.caption(f"Last update: {signal_info['timestamp']} UTC")


# Display Chart
fig = create_chart(df)
st.plotly_chart(fig, use_container_width=True)

# --- MODIFIED: Signal Log Table ---
st.subheader("📊 Signal Log")
if st.session_state.signal_history.empty:
    st.info("Waiting for the first signal to be logged...")
else:
    # Format the dataframe for better display
    display_df = st.session_state.signal_history.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['price'] = display_df['price'].apply(lambda x: f"${x:,.3f}")
    
    # Add a cap to prevent the app from slowing down (shows latest 1000 signals)
    display_df = display_df.head(1000) 
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Display Raw Data (optional)
with st.expander("Show Recent Raw Data"):
    st.dataframe(df.tail(10).sort_values('timestamp', ascending=False), hide_index=True)

# --- Auto-Refresh ---
st.caption("Page will refresh automatically in 60 seconds.")
time.sleep(60)
st.rerun()