"""
Streamlit GUI for Stock Prediction System
Interactive web application for stock analysis and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import get_config
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.gru_model import GRUPredictor
from src.models.baseline_models import BaselinePredictor
from src.models.ensemble_model import EnsemblePredictor
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.backtesting import Backtester
from src.utils.logging_config import setup_logging

# Page configuration
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        color: #ffffff;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = get_config()
    setup_logging(log_level='INFO')

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = StockDataLoader(st.session_state.config.project_root)

if 'available_tickers' not in st.session_state:
    st.session_state.available_tickers = st.session_state.data_loader.get_available_tickers(
        st.session_state.config.get_path('data.individual_stocks_dir')
    )

if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False


def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">📈 Stock Predictor Pro</p>', unsafe_allow_html=True)
    st.markdown("### Advanced Stock Market Analysis & Prediction Platform")
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Explorer", "🔧 Feature Engineering", 
         "🤖 Train Model", "🔮 Make Predictions", "📈 Backtesting", "ℹ️ About"]
    )
    
    # Display selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🔧 Feature Engineering":
        show_feature_engineering()
    elif page == "🤖 Train Model":
        show_training_page()
    elif page == "🔮 Make Predictions":
        show_prediction_page()
    elif page == "📈 Backtesting":
        show_backtesting_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display home page."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Data Analysis</h3>
            <p>Explore 5 years of S&P 500 stock data with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Models</h3>
            <p>Train LSTM, GRU, Random Forest, and Ensemble models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Predictions</h3>
            <p>Make accurate price forecasts with trained models</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📌 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Stocks", len(st.session_state.available_tickers))
    
    with col2:
        saved_models = list(Path('models/saved_models').glob('*.keras')) + list(Path('models/saved_models').glob('*.pkl'))
        st.metric("Saved Models", len(saved_models))
    
    with col3:
        st.metric("Technical Indicators", "20+")
    
    with col4:
        status = "✅ Ready" if st.session_state.available_tickers else "⚠️ No Data"
        st.metric("Status", status)
    
    st.markdown("---")
    
    # Getting started guide
    st.subheader("🚀 Quick Start Guide")
    
    with st.expander("1️⃣ Explore Your Data", expanded=True):
        st.write("""
        Navigate to **📊 Data Explorer** to:
        - View historical price data
        - Analyze trends and patterns
        - Compare multiple stocks
        - Check data quality
        """)
    
    with st.expander("2️⃣ Train a Model"):
        st.write("""
        Go to **🤖 Train Model** to:
        - Select a stock ticker
        - Choose model type (LSTM, GRU, RF, Ensemble)
        - Configure hyperparameters
        - Monitor training progress
        """)
    
    with st.expander("3️⃣ Make Predictions"):
        st.write("""
        Visit **🔮 Make Predictions** to:
        - Load a trained model
        - Predict future prices
        - Visualize forecasts
        - Export results
        """)
    
    st.markdown("---")
    
    # Sample stocks showcase
    st.subheader("💼 Popular Stocks")
    
    popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM']
    available_popular = [s for s in popular_stocks if s in st.session_state.available_tickers]
    
    if available_popular:
        cols = st.columns(len(available_popular[:4]))
        for i, ticker in enumerate(available_popular[:4]):
            with cols[i]:
                if st.button(f"📊 {ticker}", key=f"quick_{ticker}"):
                    st.session_state.selected_ticker = ticker
                    st.info(f"Selected {ticker}! Go to Data Explorer to view details.")


def show_data_explorer():
    """Display data exploration page."""
    
    st.title("📊 Data Explorer")
    
    # Ticker selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_ticker = st.selectbox(
            "Select Stock Ticker",
            st.session_state.available_tickers,
            index=0 if st.session_state.available_tickers else None
        )
    
    with col2:
        st.metric("Total Stocks Available", len(st.session_state.available_tickers))
    
    if selected_ticker:
        try:
            # Load data
            with st.spinner(f"Loading {selected_ticker} data..."):
                df = st.session_state.data_loader.load_individual_stock(
                    selected_ticker,
                    st.session_state.config.get_path('data.individual_stocks_dir')
                )
            
            # Data summary
            st.success(f"✅ Loaded {len(df)} records for {selected_ticker}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Start Date", df['date'].min().strftime('%Y-%m-%d'))
            with col2:
                st.metric("End Date", df['date'].max().strftime('%Y-%m-%d'))
            with col3:
                st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
            with col4:
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
                st.metric("Total Return", f"{price_change:+.2f}%")
            
            st.markdown("---")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "📊 Statistics", "📉 Volume Analysis", "🔍 Raw Data"])
            
            with tab1:
                # Interactive price chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df['date'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='OHLC'
                ))
                
                fig.update_layout(
                    title=f'{selected_ticker} Stock Price History',
                    yaxis_title='Price ($)',
                    xaxis_title='Date',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                
                fig_volume.update_layout(
                    title='Trading Volume',
                    yaxis_title='Volume',
                    xaxis_title='Date',
                    height=300,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_volume, use_container_width=True)
            
            with tab2:
                # Statistical analysis
                st.subheader("📊 Statistical Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Price Statistics**")
                    price_stats = df[['open', 'high', 'low', 'close']].describe()
                    st.dataframe(price_stats, use_container_width=True)
                
                with col2:
                    st.write("**Volume Statistics**")
                    volume_stats = df['volume'].describe()
                    st.dataframe(pd.DataFrame(volume_stats), use_container_width=True)
                
                # Distribution plots
                st.subheader("📈 Price Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = px.histogram(
                        df, x='close',
                        title='Close Price Distribution',
                        nbins=50
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    returns = df['close'].pct_change()
                    fig_returns = px.histogram(
                        returns.dropna(),
                        title='Daily Returns Distribution',
                        nbins=50
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
            
            with tab3:
                # Volume analysis
                st.subheader("📉 Volume Analysis")
                
                # Volume moving average
                df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume', opacity=0.3))
                fig.add_trace(go.Scatter(x=df['date'], y=df['volume_ma_20'], 
                                        name='20-day MA', line=dict(color='red', width=2)))
                
                fig.update_layout(
                    title='Volume with 20-day Moving Average',
                    yaxis_title='Volume',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                # Raw data table
                st.subheader("🔍 Raw Data")
                
                # Date range filter
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", df['date'].min())
                with col2:
                    end_date = st.date_input("End Date", df['date'].max())
                
                # Filter data
                mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
                filtered_df = df[mask]
                
                st.dataframe(filtered_df, use_container_width=True, height=400)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"{selected_ticker}_data.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")


def show_feature_engineering():
    """Display feature engineering page."""
    
    st.title("🔧 Feature Engineering")
    st.write("Calculate and visualize technical indicators")
    
    # Ticker selection
    selected_ticker = st.selectbox(
        "Select Stock Ticker",
        st.session_state.available_tickers
    )
    
    if selected_ticker and st.button("🔄 Calculate Technical Indicators"):
        try:
            # Load data
            with st.spinner(f"Loading {selected_ticker} data..."):
                df = st.session_state.data_loader.load_individual_stock(
                    selected_ticker,
                    st.session_state.config.get_path('data.individual_stocks_dir')
                )
            
            # Add features
            with st.spinner("Calculating technical indicators..."):
                engineer = FeatureEngineer()
                df_features = engineer.add_all_features(df, st.session_state.config.feature_config)
            
            st.success(f"✅ Added {len(df_features.columns) - len(df.columns)} technical indicators!")
            
            # Display indicators
            st.subheader("📊 Technical Indicators")
            
            # Tabs for different indicator categories
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend", "💪 Momentum", "📊 Volatility", "📉 Volume"])
            
            with tab1:
                st.write("**Moving Averages**")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['close'], 
                                        name='Close', line=dict(width=2)))
                
                if 'sma_20' in df_features.columns:
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['sma_20'], 
                                            name='SMA 20', line=dict(dash='dash')))
                if 'sma_50' in df_features.columns:
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['sma_50'], 
                                            name='SMA 50', line=dict(dash='dash')))
                if 'sma_200' in df_features.columns:
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['sma_200'], 
                                            name='SMA 200', line=dict(dash='dash')))
                
                fig.update_layout(title='Price with Moving Averages', height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'rsi_14' in df_features.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['rsi_14'], 
                                                name='RSI', line=dict(color='purple')))
                        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig.update_layout(title='RSI (Relative Strength Index)', height=400, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'stoch_k' in df_features.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['stoch_k'], 
                                                name='%K', line=dict(color='blue')))
                        if 'stoch_d' in df_features.columns:
                            fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['stoch_d'], 
                                                    name='%D', line=dict(color='orange')))
                        fig.update_layout(title='Stochastic Oscillator', height=400, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if all(col in df_features.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['close'], 
                                            name='Close', line=dict(color='black')))
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['bb_upper'], 
                                            name='Upper Band', line=dict(dash='dash', color='red')))
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['bb_middle'], 
                                            name='Middle Band', line=dict(dash='dash', color='blue')))
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['bb_lower'], 
                                            name='Lower Band', line=dict(dash='dash', color='green')))
                    
                    fig.update_layout(title='Bollinger Bands', height=500, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                if 'obv' in df_features.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_features['date'], y=df_features['obv'], 
                                            name='OBV', line=dict(color='teal')))
                    fig.update_layout(title='On-Balance Volume (OBV)', height=500, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature list
            st.subheader("📋 All Calculated Features")
            feature_cols = [col for col in df_features.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'name']]
            st.write(f"Total features: **{len(feature_cols)}**")
            
            with st.expander("View feature list"):
                st.write(feature_cols)
        
        except Exception as e:
            st.error(f"Error calculating features: {str(e)}")


def show_training_page():
    """Display model training page."""
    
    st.title("🤖 Train Model")
    
    # Add clear model button if model exists
    if st.session_state.trained_model is not None:
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            st.info(f"📦 Current model: {st.session_state.trained_model['model_type']} for {st.session_state.trained_model['ticker']}")
        with col_btn:
            if st.button("🗑️ Clear Model", type="secondary"):
                st.session_state.trained_model = None
                st.session_state.training_complete = False
                st.success("✅ Model cleared! Ready for fresh training.")
                st.rerun()
        st.markdown("---")
    
    st.write("Train machine learning models for stock price prediction")
    
    # Training configuration
    col1, col2 = st.columns(2)
    
    with col1:
        selected_ticker = st.selectbox(
            "Select Stock Ticker",
            st.session_state.available_tickers,
            key="train_ticker"
        )
        
        model_type = st.selectbox(
            "Select Model Type",
            ["LSTM", "GRU", "Random Forest", "Ensemble"],
            help="LSTM: Best accuracy, GRU: Faster training, RF: Feature importance, Ensemble: Most robust"
        )
    
    with col2:
        sequence_length = st.slider("Sequence Length (days)", 20, 100, 60)
        prediction_horizon = st.selectbox("Prediction Horizon (days ahead)", [1, 5, 10, 30])
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_split = st.slider("Train Split", 0.5, 0.9, 0.7)
            val_split = st.slider("Validation Split", 0.05, 0.3, 0.15)
        
        with col2:
            if model_type in ["LSTM", "GRU"]:
                epochs = st.slider("Epochs", 10, 200, 50)
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            else:
                epochs = 50
                batch_size = 32
        
        with col3:
            if model_type in ["LSTM", "GRU"]:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                    value=0.001
                )
            else:
                learning_rate = 0.001
    
    st.markdown("---")
    
    # Training button
    if st.button("🚀 Start Training", type="primary"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load data
            status_text.text("📥 Loading data...")
            progress_bar.progress(10)
            
            df = st.session_state.data_loader.load_individual_stock(
                selected_ticker,
                st.session_state.config.get_path('data.individual_stocks_dir')
            )
            
            # Feature engineering
            status_text.text("🔧 Engineering features...")
            progress_bar.progress(20)
            
            engineer = FeatureEngineer()
            df = engineer.add_all_features(df, st.session_state.config.feature_config)
            
            # Preprocessing
            status_text.text("⚙️ Preprocessing data...")
            progress_bar.progress(30)
            
            preprocessor = DataPreprocessor()
            df = preprocessor.handle_missing_values(df)
            df = preprocessor.add_target_variable(df, horizon=prediction_horizon)
            df = df.dropna(subset=['target'])
            
            # Split data
            test_split = 1 - train_split - val_split
            splits = preprocessor.split_data(df, train_split, val_split, test_split)
            train_df, val_df, test_df = splits['train'], splits['val'], splits['test']
            
            st.info(f"📊 Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Scale features (NOT target!)
            feature_cols = preprocessor.get_feature_importance_columns(train_df)
            
            st.info(f"🔢 Using {len(feature_cols)} features: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
            
            # Scale features only
            train_scaled, val_scaled, test_scaled = preprocessor.scale_features(
                train_df, val_df, test_df, feature_cols
            )
            
            # Keep unscaled target for training (CRITICAL!)
            train_scaled['target'] = train_df['target'].values
            val_scaled['target'] = val_df['target'].values
            test_scaled['target'] = test_df['target'].values
            
            # Validate target values
            st.info(f"🎯 Target range: ${train_df['target'].min():.2f} to ${train_df['target'].max():.2f}")
            st.info(f"📊 Current price range: ${train_df['close'].min():.2f} to ${train_df['close'].max():.2f}")
            
            progress_bar.progress(40)
            
            # Initialize variables
            model = None
            history = None
            
            # Create sequences
            if model_type in ["LSTM", "GRU", "Ensemble"]:
                status_text.text("📦 Creating sequences...")
                
                X_train, y_train = preprocessor.create_sequences(
                    train_scaled, sequence_length, target_col='target', feature_cols=feature_cols
                )
                X_val, y_val = preprocessor.create_sequences(
                    val_scaled, sequence_length, target_col='target', feature_cols=feature_cols
                )
                X_test, y_test = preprocessor.create_sequences(
                    test_scaled, sequence_length, target_col='target', feature_cols=feature_cols
                )
                
                st.info(f"📊 Sequence shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
                st.info(f"📊 Target shapes - y_train: {y_train.shape}, y_test: {y_test.shape}")
                st.info(f"✅ Model will use {X_train.shape[2]} features (excluding target column)")
                
                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    st.error("❌ Not enough data to create sequences! Reduce sequence length or use more data.")
                    return
                    
            else:
                X_train = train_scaled[feature_cols].values
                y_train = train_scaled['target'].values
                X_val = val_scaled[feature_cols].values
                y_val = val_scaled['target'].values
                X_test = test_scaled[feature_cols].values
                y_test = test_scaled['target'].values
            
            progress_bar.progress(50)
            
            # Validate data before training
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                st.error("❌ Target values contain NaN or Inf! Data preprocessing failed.")
                return
            
            # Train model
            status_text.text(f"🏋️ Training {model_type} model...")
            
            if model_type == "LSTM":
                model = LSTMPredictor({
                    'units': [128, 64, 32],
                    'dropout': 0.2,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': 15
                })
                with st.spinner("Training LSTM... This may take a few minutes."):
                    history = model.train(X_train, y_train, X_val, y_val)
                
            elif model_type == "GRU":
                model = GRUPredictor({
                    'units': [128, 64],
                    'dropout': 0.2,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': 15
                })
                with st.spinner("Training GRU... This may take a few minutes."):
                    history = model.train(X_train, y_train, X_val, y_val)
            
            elif model_type == "Random Forest":
                model = BaselinePredictor('random_forest')
                with st.spinner("Training Random Forest..."):
                    model.train(X_train, y_train)
                history = None
            
            progress_bar.progress(80)
            
            # Evaluate
            status_text.text("📊 Evaluating model...")
            
            if model is None:
                st.error("Model training failed - model is None")
                return
            
            metrics = model.evaluate(X_test, y_test)
            
            progress_bar.progress(100)
            status_text.text("✅ Training complete!")
            
            # Display results
            st.success("🎉 Model trained successfully!")
            
            st.subheader("📈 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
            with col4:
                st.metric("Directional Accuracy", f"{metrics.get('directional_accuracy', 0)*100:.2f}%")
            
            # Training history
            if history and model_type in ["LSTM", "GRU"]:
                st.subheader("📊 Training History")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history['loss'], name='Training Loss', mode='lines'))
                if 'val_loss' in history:
                    fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss', mode='lines'))
                
                fig.update_layout(
                    title='Model Loss Over Epochs',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Predictions plot
            st.subheader("🔮 Predictions vs Actual")
            
            predictions = model.predict(X_test)
            test_dates = test_scaled['date'].iloc[-len(y_test):].values
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(x=test_dates, y=predictions, name='Predicted', 
                                    mode='lines', line=dict(dash='dash')))
            
            fig.update_layout(
                title=f'{selected_ticker} - Actual vs Predicted Prices',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Save model with all necessary info
            st.session_state.trained_model = {
                'model': model,
                'ticker': selected_ticker,
                'model_type': model_type,
                'metrics': metrics,
                'preprocessor': preprocessor,
                'feature_cols': feature_cols,
                'sequence_length': sequence_length,
                'scaler': preprocessor.feature_scaler,
                'n_features': len(feature_cols),
                'prediction_horizon': prediction_horizon,
                'last_close_price': df['close'].iloc[-1]
            }
            st.session_state.training_complete = True
            
            st.info("💾 Model saved to session. You can now make predictions!")
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_prediction_page():
    """Display prediction page."""
    
    st.title("🔮 Make Predictions")
    
    if not st.session_state.training_complete or st.session_state.trained_model is None:
        st.warning("⚠️ No trained model available. Please train a model first!")
        
        if st.button("Go to Training Page"):
            st.session_state.page = "🤖 Train Model"
            st.rerun()
        
        return
    
    trained_info = st.session_state.trained_model
    
    st.success(f"✅ Using trained {trained_info['model_type']} model for {trained_info['ticker']}")
    
    # Model info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", trained_info['model_type'])
    with col2:
        st.metric("Stock", trained_info['ticker'])
    with col3:
        st.metric("RMSE", f"{trained_info['metrics']['rmse']:.4f}")
    with col4:
        st.metric("R² Score", f"{trained_info['metrics'].get('r2', 0):.4f}")
    
    st.markdown("---")
    
    # Prediction configuration
    st.subheader("🎯 Prediction Configuration")
    
    days_ahead = st.slider("Days Ahead to Predict", 1, 30, 1)
    
    if st.button("🔮 Generate Prediction", type="primary"):
        try:
            with st.spinner("Generating prediction..."):
                # Load recent data
                df = st.session_state.data_loader.load_individual_stock(
                    trained_info['ticker'],
                    st.session_state.config.get_path('data.individual_stocks_dir')
                )
                
                # Add features
                engineer = FeatureEngineer()
                df = engineer.add_all_features(df, st.session_state.config.feature_config)
                df = trained_info['preprocessor'].handle_missing_values(df)
                
                # Get recent data
                sequence_length = trained_info['sequence_length']
                feature_cols = trained_info['feature_cols']
                
                # Take enough data for sequence
                recent_data = df.tail(sequence_length + 50).copy()
                
                # Verify all feature columns exist
                missing_cols = [col for col in feature_cols if col not in recent_data.columns]
                if missing_cols:
                    st.error(f"❌ Missing features in data: {missing_cols}")
                    st.info("Please retrain the model with the current data.")
                    return
                
                # Scale using the SAME scaler from training - only the exact features
                recent_scaled = recent_data.copy()
                recent_scaled[feature_cols] = trained_info['scaler'].transform(
                    recent_data[feature_cols]
                )
                
                # For LSTM/GRU: Create sequence with the last sequence_length rows
                if trained_info['model_type'] in ['LSTM', 'GRU']:
                    # Get the last sequence_length rows
                    last_sequence = recent_scaled[feature_cols].tail(sequence_length).values
                    # Reshape to (1, sequence_length, n_features)
                    X_pred = last_sequence.reshape(1, sequence_length, len(feature_cols))
                else:
                    # For Random Forest: just use the last row
                    X_pred = recent_scaled[feature_cols].iloc[-1:].values
                
                # Verify shape
                expected_features = trained_info['n_features']
                if trained_info['model_type'] in ['LSTM', 'GRU']:
                    if X_pred.shape[2] != expected_features:
                        st.error(f"Feature mismatch: Model expects {expected_features} features, got {X_pred.shape[2]}")
                        return
                
                # Predict
                prediction = trained_info['model'].predict(X_pred)[0]
                
                # Current price
                current_price = df['close'].iloc[-1]
                current_date = df['date'].iloc[-1]
                
                # Calculate metrics
                predicted_change = prediction - current_price
                predicted_change_pct = (predicted_change / current_price) * 100
                
                # Display results
                st.success("✅ Prediction generated!")
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Current Price</h4>
                        <h2>${:.2f}</h2>
                        <p>{}</p>
                    </div>
                    """.format(current_price, current_date.strftime('%Y-%m-%d')), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h4>Predicted Price</h4>
                        <h2>${:.2f}</h2>
                        <p>in {} day(s)</p>
                    </div>
                    """.format(prediction, days_ahead), unsafe_allow_html=True)
                
                with col3:
                    direction_emoji = "📈" if predicted_change > 0 else "📉"
                    color = "green" if predicted_change > 0 else "red"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Expected Change {direction_emoji}</h4>
                        <h2 style="color: {color};">${predicted_change:+.2f}</h2>
                        <p style="color: {color};">({predicted_change_pct:+.2f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction chart
                st.subheader("📈 Price Forecast")
                
                # Create forecast visualization
                dates = [current_date + timedelta(days=i) for i in range(days_ahead + 1)]
                prices = [current_price] + [current_price + (prediction - current_price) * (i / days_ahead) 
                                            for i in range(1, days_ahead + 1)]
                
                fig = go.Figure()
                
                # Historical prices (last 30 days)
                hist_data = df.tail(30)
                fig.add_trace(go.Scatter(
                    x=hist_data['date'],
                    y=hist_data['close'],
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{trained_info["ticker"]} Price Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Export results
                results_df = pd.DataFrame({
                    'Ticker': [trained_info['ticker']],
                    'Current Date': [current_date.strftime('%Y-%m-%d')],
                    'Current Price': [f'${current_price:.2f}'],
                    'Predicted Price': [f'${prediction:.2f}'],
                    'Days Ahead': [days_ahead],
                    'Change': [f'${predicted_change:.2f}'],
                    'Change %': [f'{predicted_change_pct:+.2f}%'],
                    'Direction': ['UP' if predicted_change > 0 else 'DOWN']
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name=f"prediction_{trained_info['ticker']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def show_backtesting_page():
    """Display backtesting page."""
    
    st.title("📈 Backtesting")
    st.write("Test trading strategies with historical data")
    
    st.info("🚧 Backtesting functionality coming soon! Train a model first to enable backtesting.")


def show_about_page():
    """Display about page."""
    
    st.title("ℹ️ About Stock Predictor Pro")
    
    st.markdown("""
    ### 🎯 Overview
    
    **Stock Predictor Pro** is an advanced machine learning platform for stock market analysis and prediction.
    Built with state-of-the-art deep learning models and comprehensive technical analysis tools.
    
    ### 🌟 Features
    
    - **📊 Data Analysis**: Explore 5 years of S&P 500 historical data
    - **🔧 Feature Engineering**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - **🤖 AI Models**: LSTM, GRU, Random Forest, and Ensemble predictions
    - **🔮 Predictions**: Accurate price forecasting with confidence metrics
    - **📈 Backtesting**: Test trading strategies with historical performance
    - **🎨 Visualizations**: Interactive charts and detailed analytics
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **ML/DL**: TensorFlow, Keras, scikit-learn
    - **Data**: pandas, numpy
    - **Visualization**: Plotly, matplotlib, seaborn
    - **Technical Analysis**: TA-Lib, pandas-ta
    
    ### 📊 Models
    
    **LSTM (Long Short-Term Memory)**
    - 3-layer architecture with dropout
    - Best for capturing long-term dependencies
    - Highest accuracy for time series
    
    **GRU (Gated Recurrent Unit)**
    - 2-layer architecture
    - Faster training than LSTM
    - Good balance of speed and accuracy
    
    **Random Forest**
    - 200 decision trees
    - Provides feature importance
    - Great for baseline comparisons
    
    **Ensemble**
    - Combines all models
    - Weighted average predictions
    - Most robust and reliable
    
    ### ⚠️ Disclaimer
    
    This platform is for **educational purposes only**. Stock market prediction involves substantial risk.
    - Not financial advice
    - Past performance ≠ future results
    - Always do your own research
    - Never invest more than you can afford to lose
    
    ### 📚 Resources
    
    - [TensorFlow Documentation](https://www.tensorflow.org/)
    - [Technical Analysis Guide](https://www.investopedia.com/technical-analysis-4689657)
    - [Machine Learning for Trading](https://www.ml4trading.io/)
    
    ### 👨‍💻 Version
    
    **Version 1.0.0** - November 2025
    
    ---
    
    Built with ❤️ for stock market analysis and ML education
    """)


if __name__ == "__main__":
    main()
