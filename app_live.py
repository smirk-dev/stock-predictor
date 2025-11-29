"""
Stock Predictor Pro - Live Market Edition
Real-time stock prediction with daily updates
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data import DataService

# Page configuration
st.set_page_config(
    page_title="Stock Predictor Pro - Live",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .stock-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-live { background: #10b981; color: white; }
    .status-cache { background: #f59e0b; color: white; }
    .status-error { background: #ef4444; color: white; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_service' not in st.session_state:
    st.session_state.data_service = DataService()
    
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
    
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Predictor Pro - Live Market Edition</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Stock+Predictor", 
                use_column_width=True)
        st.markdown("### 🌍 Real-Time Global Markets")
        st.markdown("Track and predict stocks from NYSE, NASDAQ, NSE, BSE and more!")
        
        # Market overview
        with st.expander("📊 Market Overview", expanded=True):
            show_market_overview()
        
        # Cache stats
        with st.expander("💾 Cache Statistics"):
            show_cache_stats()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Stock Search", 
        "📊 Analysis", 
        "🤖 Train Model", 
        "🔮 Predictions"
    ])
    
    with tab1:
        show_stock_search()
    
    with tab2:
        show_analysis()
    
    with tab3:
        show_training()
    
    with tab4:
        show_predictions()


def show_market_overview():
    """Display market indices overview."""
    try:
        overview = st.session_state.data_service.get_market_overview()
        
        if not overview.empty:
            for _, row in overview.iterrows():
                change_color = "🟢" if row['change_pct'] >= 0 else "🔴"
                st.markdown(f"**{row['name']}**: {row['price']:.2f} {change_color} {row['change_pct']:+.2f}%")
        else:
            st.info("Market data unavailable")
    except Exception as e:
        st.error(f"Error loading market data: {e}")


def show_cache_stats():
    """Display cache statistics."""
    stats = st.session_state.data_service.get_cache_stats()
    
    if stats:
        st.metric("Cached Stocks", stats.get('unique_tickers', 0))
        st.metric("Total Records", f"{stats.get('total_records', 0):,}")
        st.metric("Database Size", f"{stats.get('db_size_mb', 0):.1f} MB")
        
        if st.button("🗑️ Clear Old Cache"):
            st.session_state.data_service.clear_old_cache()
            st.success("Cleared cache older than 2 years")
            st.rerun()


def show_stock_search():
    """Stock search interface."""
    st.markdown("## 🔍 Search & Load Stock Data")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search input
        search_query = st.text_input(
            "Search stocks by ticker or name",
            placeholder="e.g., AAPL, Tesla, RELIANCE.NS, TCS.BO",
            help="For Indian stocks, add .NS (NSE) or .BO (BSE) suffix"
        )
    
    with col2:
        market = st.selectbox("Market", ["US", "India", "Both"])
    
    # Popular stocks section
    st.markdown("### 🌟 Popular Stocks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🇺🇸 United States**")
        us_stocks = st.session_state.data_service.get_popular_stocks('us')
        selected_us = st.selectbox("Select US Stock", [""] + us_stocks, key="us_select")
        
    with col2:
        st.markdown("**🇮🇳 India**")
        india_stocks = st.session_state.data_service.get_popular_stocks('india')
        selected_india = st.selectbox("Select Indian Stock", [""] + india_stocks, key="india_select")
    
    # Cached stocks
    st.markdown("### 💾 Recently Cached Stocks")
    cached = st.session_state.data_service.get_cached_tickers()
    if cached:
        cached_display = st.multiselect("Quick load from cache", cached, key="cached_select")
    
    # Load button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        ticker_input = st.text_input("Or enter ticker directly:", 
                                     value=selected_us or selected_india or "")
    
    with col2:
        period = st.selectbox("Data Period", ["1y", "2y", "5y", "max"], index=1)
    
    with col3:
        force_refresh = st.checkbox("Force Refresh", value=False)
    
    if st.button("📥 Load Stock Data", type="primary", use_container_width=True):
        load_stock_data(ticker_input, period, force_refresh)


def load_stock_data(ticker: str, period: str, force_refresh: bool):
    """Load stock data with progress indication."""
    if not ticker:
        st.error("Please enter a ticker symbol")
        return
    
    with st.spinner(f"Loading {ticker}..."):
        try:
            df, source = st.session_state.data_service.get_stock_data(
                ticker, period, force_refresh
            )
            
            if df.empty:
                st.error(f"❌ No data found for {ticker}")
                return
            
            # Save to session state
            st.session_state.current_ticker = ticker
            st.session_state.stock_data = df
            
            # Show success with source badge
            source_badges = {
                'api': '<span class="status-badge status-live">🔴 LIVE</span>',
                'cache': '<span class="status-badge status-cache">💾 CACHED</span>',
                'cache_fallback': '<span class="status-badge status-error">⚠️ FALLBACK</span>'
            }
            
            st.markdown(f"""
                <div class="stock-card">
                    <h2>✅ {ticker} Loaded Successfully</h2>
                    <p>{source_badges.get(source, '')}</p>
                    <p><strong>{len(df):,}</strong> records from 
                       <strong>{df['date'].min().strftime('%Y-%m-%d')}</strong> to 
                       <strong>{df['date'].max().strftime('%Y-%m-%d')}</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Show preview
            st.markdown("### 📊 Data Preview")
            st.dataframe(df.tail(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading {ticker}: {e}")


def show_analysis():
    """Stock analysis tab."""
    st.markdown("## 📊 Technical Analysis")
    
    if st.session_state.stock_data is None:
        st.info("👈 Please load a stock from the Search tab first")
        return
    
    df = st.session_state.stock_data
    ticker = st.session_state.current_ticker
    
    # Price chart
    st.markdown(f"### 📈 {ticker} Price History")
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=ticker
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    st.markdown("### 📊 Trading Volume")
    
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=df['date'],
        y=df['volume'],
        name="Volume",
        marker_color='lightblue'
    ))
    
    fig_vol.update_layout(
        title=f"{ticker} Trading Volume",
        yaxis_title="Volume",
        xaxis_title="Date",
        template="plotly_white",
        height=300
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${df.iloc[-1]['close']:.2f}")
    
    with col2:
        change = df.iloc[-1]['close'] - df.iloc[-2]['close']
        change_pct = (change / df.iloc[-2]['close']) * 100
        st.metric("Daily Change", f"${change:.2f}", f"{change_pct:+.2f}%")
    
    with col3:
        st.metric("Volume", f"{df.iloc[-1]['volume']:,.0f}")
    
    with col4:
        avg_vol = df['volume'].tail(20).mean()
        st.metric("Avg Volume (20d)", f"{avg_vol:,.0f}")


def show_training():
    """Model training tab."""
    st.markdown("## 🤖 Train Prediction Model")
    
    if st.session_state.stock_data is None:
        st.info("👈 Please load a stock first")
        return
    
    st.info("🚧 Training interface coming soon! This will include:")
    st.markdown("""
    - **Model Selection**: LSTM, GRU, Random Forest, Ensemble
    - **Hyperparameter Tuning**: Sequence length, layers, learning rate
    - **Real-time Training Progress**: Live loss curves and metrics
    - **Model Comparison**: Side-by-side performance metrics
    - **Auto-save Best Model**: Automatic caching of best performers
    """)


def show_predictions():
    """Predictions tab."""
    st.markdown("## 🔮 Stock Predictions")
    
    if st.session_state.stock_data is None:
        st.info("👈 Please load a stock and train a model first")
        return
    
    st.info("🚧 Predictions interface coming soon! This will include:")
    st.markdown("""
    - **Next Day Prediction**: Tomorrow's price forecast
    - **Multi-day Forecasting**: 7-day and 30-day predictions
    - **Confidence Intervals**: Prediction uncertainty ranges
    - **Historical Accuracy**: Past prediction performance
    - **Buy/Sell Signals**: AI-powered trading recommendations
    """)


if __name__ == "__main__":
    main()
