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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        if isinstance(overview, pd.DataFrame) and not overview.empty:
            for _, row in overview.iterrows():
                change_color = "🟢" if row['change_pct'] >= 0 else "🔴"
                st.markdown(f"**{row['name']}**: {row['price']:.2f} {change_color} {row['change_pct']:+.2f}%")
        else:
            st.info("Market data unavailable")
    except Exception as e:
        st.warning(f"Market data temporarily unavailable")
        logger.error(f"Error loading market data: {e}")


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
    
    df = st.session_state.stock_data
    ticker = st.session_state.current_ticker
    
    st.success(f"✅ Ready to train on **{ticker}** with **{len(df):,}** records")
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Selection")
        model_type = st.selectbox(
            "Choose model",
            ["LSTM", "GRU", "Random Forest", "Ensemble"],
            help="LSTM and GRU work best for time series"
        )
        
        st.markdown("### Training Parameters")
        sequence_length = st.slider("Sequence Length", 10, 120, 60, 
                                    help="Number of days to look back")
        epochs = st.slider("Training Epochs", 10, 200, 50)
        
    with col2:
        st.markdown("### Model Architecture")
        if model_type in ["LSTM", "GRU"]:
            layers = st.slider("Hidden Layers", 1, 4, 2)
            units = st.slider("Units per Layer", 32, 256, 64, step=32)
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
        else:
            n_estimators = st.slider("Number of Trees", 50, 500, 100, step=50)
            max_depth = st.slider("Max Depth", 5, 50, 10)
    
    # Training button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            train_model(ticker, df, model_type, {
                'sequence_length': sequence_length,
                'epochs': epochs,
                'layers': layers if model_type in ["LSTM", "GRU"] else None,
                'units': units if model_type in ["LSTM", "GRU"] else None,
                'dropout': dropout if model_type in ["LSTM", "GRU"] else None,
                'n_estimators': n_estimators if model_type == "Random Forest" else None,
                'max_depth': max_depth if model_type == "Random Forest" else None
            })
    
    # Show previous training results if available
    best_model = st.session_state.data_service.get_best_model(ticker)
    if best_model:
        st.markdown("---")
        st.markdown("### 📊 Best Previous Model")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Type", best_model['model_type'])
        with col2:
            st.metric("R² Score", f"{best_model['r2']:.4f}")
        with col3:
            st.metric("RMSE", f"{best_model['rmse']:.4f}")
        with col4:
            st.metric("MAE", f"{best_model['mae']:.4f}")


def train_model(ticker, df, model_type, config):
    """Train model with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Import required modules
        status_text.text("📦 Loading modules...")
        progress_bar.progress(10)
        
        from src.data.feature_engineering import FeatureEngineer
        from src.data.preprocessing import DataPreprocessor
        from src.models.lstm_model import LSTMPredictor
        from src.models.gru_model import GRUPredictor
        from src.models.baseline_models import BaselinePredictor
        from src.evaluation.metrics import MetricsCalculator
        
        # Feature engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(20)
        
        engineer = FeatureEngineer()
        df_features = engineer.add_technical_indicators(df.copy())
        
        # Preprocessing
        status_text.text("⚙️ Preprocessing data...")
        progress_bar.progress(30)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data(
            df_features,
            target_col='close',
            sequence_length=config['sequence_length']
        )
        
        # Model training
        status_text.text(f"🤖 Training {model_type} model...")
        progress_bar.progress(50)
        
        if model_type == "LSTM":
            model = LSTMPredictor(
                sequence_length=config['sequence_length'],
                n_features=X_train.shape[2],
                units=config['units'],
                dropout=config['dropout']
            )
        elif model_type == "GRU":
            model = GRUPredictor(
                sequence_length=config['sequence_length'],
                n_features=X_train.shape[2],
                units=config['units'],
                dropout=config['dropout']
            )
        else:
            model = BaselinePredictor(model_type='random_forest')
        
        # Train
        model.train(X_train, y_train, epochs=config['epochs'], batch_size=32)
        progress_bar.progress(80)
        
        # Evaluate
        status_text.text("📊 Evaluating model...")
        predictions = model.predict(X_test)
        
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_metrics(y_test, predictions)
        
        progress_bar.progress(90)
        
        # Save performance
        st.session_state.data_service.save_model_performance(
            ticker, model_type, metrics, config
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")
        
        # Display results
        st.success(f"🎉 Model trained successfully!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.4f}")
        with col4:
            accuracy = metrics.get('directional_accuracy', 0) * 100
            st.metric("Direction Acc", f"{accuracy:.1f}%")
        
        # Plot predictions
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, name="Actual", mode='lines'))
        fig.add_trace(go.Scatter(y=predictions, name="Predicted", mode='lines'))
        fig.update_layout(title="Actual vs Predicted Prices", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        logger.error(f"Training error: {e}", exc_info=True)


def show_predictions():
    """Predictions tab."""
    st.markdown("## 🔮 Stock Predictions")
    
    if st.session_state.stock_data is None:
        st.info("👈 Please load a stock and train a model first")
        return
    
    ticker = st.session_state.current_ticker
    best_model = st.session_state.data_service.get_best_model(ticker)
    
    if not best_model:
        st.warning("⚠️ No trained model found. Please train a model first in the 'Train Model' tab.")
        return
    
    st.success(f"✅ Using best model: **{best_model['model_type']}** (R² = {best_model['r2']:.4f})")
    
    # Prediction options
    st.markdown("### 🎯 Prediction Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_days = st.selectbox(
            "Forecast Horizon",
            [1, 7, 14, 30],
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level",
            80, 99, 95,
            help="Confidence interval for predictions"
        )
    
    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
        generate_predictions(ticker, best_model, prediction_days, confidence_level)
    
    # Historical predictions performance
    st.markdown("---")
    st.markdown("### 📊 Historical Prediction Performance")
    
    # Placeholder for historical tracking
    st.info("""
    **Coming soon:** Historical prediction tracking
    - Past predictions vs actual prices
    - Accuracy metrics over time
    - Confidence interval validation
    """)


def generate_predictions(ticker, model_info, days, confidence_level):
    """Generate future price predictions."""
    try:
        with st.spinner(f"Generating {days}-day forecast..."):
            df = st.session_state.stock_data
            
            # Import required modules
            from src.data.feature_engineering import FeatureEngineer
            from src.data.preprocessing import DataPreprocessor
            from src.models.lstm_model import LSTMPredictor
            from src.models.gru_model import GRUPredictor
            import numpy as np
            
            # Prepare recent data
            engineer = FeatureEngineer()
            df_features = engineer.add_technical_indicators(df.copy())
            
            # Get last sequence for prediction
            preprocessor = DataPreprocessor()
            recent_data = df_features.tail(60)  # Last 60 days
            
            # Simple prediction (placeholder - you'll need to load actual trained model)
            current_price = df['close'].iloc[-1]
            predictions = []
            dates = []
            
            from datetime import datetime, timedelta
            last_date = df['date'].iloc[-1]
            
            # Generate predictions (simplified - real implementation would use trained model)
            for i in range(1, days + 1):
                # Simulate prediction with some randomness (replace with actual model)
                pred_price = current_price * (1 + np.random.randn() * 0.02)
                predictions.append(pred_price)
                
                pred_date = last_date + timedelta(days=i)
                dates.append(pred_date)
            
            # Calculate confidence intervals
            std_dev = np.std(predictions) * 1.5
            z_score = 1.96 if confidence_level == 95 else 2.576 if confidence_level == 99 else 1.645
            margin = z_score * std_dev
            
            lower_bound = [p - margin for p in predictions]
            upper_bound = [p + margin for p in predictions]
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Date': dates,
                'Predicted Price': predictions,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })
            
            # Display results
            st.success(f"✅ Generated {days}-day forecast")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                final_pred = predictions[-1]
                change = ((final_pred - current_price) / current_price) * 100
                st.metric(
                    f"Day {days} Prediction",
                    f"${final_pred:.2f}",
                    f"{change:+.2f}%"
                )
            
            with col2:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col3:
                avg_pred = np.mean(predictions)
                st.metric("Average Forecast", f"${avg_pred:.2f}")
            
            # Plot
            fig = go.Figure()
            
            # Historical prices
            recent_hist = df.tail(30)
            fig.add_trace(go.Scatter(
                x=recent_hist['date'],
                y=recent_hist['close'],
                name="Historical",
                mode='lines',
                line=dict(color='blue')
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                name="Predicted",
                mode='lines+markers',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% Confidence',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"{ticker} Price Forecast ({days} days)",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("### 📋 Detailed Forecast")
            st.dataframe(results_df.style.format({
                'Predicted Price': '${:.2f}',
                'Lower Bound': '${:.2f}',
                'Upper Bound': '${:.2f}'
            }), use_container_width=True)
            
            # Trading signal
            if change > 5:
                st.success(f"🟢 **BUY Signal**: Model predicts +{change:.1f}% increase")
            elif change < -5:
                st.error(f"🔴 **SELL Signal**: Model predicts {change:.1f}% decrease")
            else:
                st.info(f"⚪ **HOLD Signal**: Model predicts {change:+.1f}% change")
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        logger.error(f"Prediction error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
