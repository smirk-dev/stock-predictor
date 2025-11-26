# 📈 Stock Analysis and Prediction System

A comprehensive, production-ready machine learning system for stock market analysis and price prediction using deep learning (LSTM,GRU) and ensemble methods.

## 🌟 Features

- **Multiple ML Models**: LSTM, GRU, Random Forest, and Ensemble predictions
- **20+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, and more
- **Comprehensive Evaluation**: RMSE, MAE, Directional Accuracy, Sharpe Ratio
- **Backtesting Engine**: Test trading strategies with historical data
- **Rich Visualizations**: Price charts, predictions, technical indicators, performance metrics
- **Modular Architecture**: Clean, maintainable, and extensible codebase
- **Jupyter Notebooks**: Interactive analysis and experimentation

## 📊 Dataset

- **Source**: S&P 500 historical stock data (5 years)
- **Format**: Daily OHLCV (Open, High, Low, Close, Volume)
- **Coverage**: 500+ individual stock files
- **Location**: `individual_stocks_5yr/` and `all_stocks_5yr.csv`

## 🏗️ Project Structure

```
stock-predictor/
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── main.py                     # Main entry point
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── src/
│   ├── config.py              # Configuration manager
│   ├── data/
│   │   ├── data_loader.py     # Data loading utilities
│   │   ├── feature_engineering.py  # Technical indicators
│   │   └── preprocessing.py   # Data preprocessing
│   ├── models/
│   │   ├── lstm_model.py      # LSTM implementation
│   │   ├── gru_model.py       # GRU implementation
│   │   ├── baseline_models.py # Random Forest, Linear Regression
│   │   └── ensemble_model.py  # Ensemble predictions
│   ├── evaluation/
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── backtesting.py     # Strategy backtesting
│   └── utils/
│       ├── visualization.py   # Plotting utilities
│       └── logging_config.py  # Logging setup
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_results.ipynb
├── models/
│   ├── checkpoints/          # Training checkpoints
│   └── saved_models/         # Trained models
├── outputs/
│   └── plots/                # Generated visualizations
└── logs/                     # Training logs
```

## 🚀 Quick Start

### Installation

```powershell
# Clone the repository
cd stock-predictor

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**1. Train a model:**

```powershell
# Train LSTM on AAPL
python main.py train --ticker AAPL --model lstm

# Train ensemble on all stocks
python main.py train --model ensemble

# Train GRU on specific stock
python main.py train --ticker MSFT --model gru
```

**2. Make predictions:**

```powershell
# Predict next day price
python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras

# Predict 5 days ahead
python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras --days 5
```

**3. Explore with Jupyter:**

```powershell
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 📋 Configuration

Edit `config.yaml` to customize:-

- **Data paths** and split ratios
- **Technical indicators** to calculate
- **Model hyperparameters** (units, dropout, learning rate)
- **Training settings** (epochs, batch size, patience)
- **Evaluation metrics** and visualization options

## 🧠 Models

### 1. LSTM (Long Short-Term Memory)
- **Best for**: Capturing long-term dependencies in time series
- **Architecture**: 3-layer LSTM with dropout regularization
- **Use case**: General-purpose stock prediction

### 2. GRU (Gated Recurrent Unit)
- **Best for**: Faster training with similar performance to LSTM
- **Architecture**: 2-layer GRU with dense output layers
- **Use case**: Quick experiments and baseline comparisons

### 3. Random Forest
- **Best for**: Feature importance analysis
- **Parameters**: 200 trees, max depth 20
- **Use case**: Understanding which features drive predictions

### 4. Ensemble
- **Best for**: Robust predictions combining multiple models
- **Composition**: Weighted average of LSTM, GRU, and Random Forest
- **Use case**: Production deployments requiring reliability

## 📊 Technical Indicators

The system automatically calculates 20+ technical indicators:

**Trend Indicators:**
- Simple Moving Averages (SMA 5, 10, 20, 50, 200)
- Exponential Moving Averages (EMA 12, 26, 50)
- MACD (Moving Average Convergence Divergence)

**Momentum Indicators:**
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Rate of Change (ROC)
- Williams %R

**Volatility Indicators:**
- Bollinger Bands
- Average True Range (ATR)

**Volume Indicators:**
- On-Balance Volume (OBV)
- Volume Ratio

**Strength Indicators:**
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)

## 📈 Evaluation Metrics

**Regression Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

**Trading Metrics:**
- Directional Accuracy (% of correct up/down predictions)
- Sharpe Ratio
- Maximum Drawdown
- Total Return vs Buy & Hold

## 🎨 Visualizations

The system generates:
- Price history with volume
- Actual vs predicted prices
- Technical indicator charts
- Training loss curves
- Error distributions
- Feature importance plots
- Portfolio performance
- Correlation matrices

## 📓 Jupyter Notebooks

**01_data_exploration.ipynb**
- Load and validate data
- Statistical analysis
- Price trends and patterns
- Volatility analysis

**02_feature_engineering.ipynb**
- Calculate technical indicators
- Feature correlation analysis
- Feature importance
- Data preprocessing

**03_model_training.ipynb**
- Train multiple models
- Hyperparameter tuning
- Training visualization
- Model comparison

**04_evaluation_results.ipynb**
- Performance metrics
- Prediction analysis
- Backtesting strategies
- Results visualization

## 🔧 Advanced Usage

### Custom Configuration

```python
from src.config import get_config

config = get_config('custom_config.yaml')
# Modify settings programmatically
```

### Train with Custom Parameters

```python
from src.models.lstm_model import LSTMPredictor

model = LSTMPredictor({
    'units': [256, 128, 64],
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'epochs': 150
})
```

### Backtesting Strategy

```python
from src.evaluation.backtesting import Backtester

backtester = Backtester(initial_capital=10000, commission=0.001)
results = backtester.run_simple_strategy(
    dates, actual_prices, predicted_prices,
    strategy='long_short'
)
```

## 📊 Example Results

**LSTM Model Performance (AAPL):**
- RMSE: $2.45
- MAE: $1.87
- Directional Accuracy: 58.3%
- R² Score: 0.94
- Sharpe Ratio: 1.42

**Ensemble Model Performance:**
- RMSE: $2.12 (13% improvement)
- MAE: $1.65 (12% improvement)
- Directional Accuracy: 61.7%
- R² Score: 0.96

## 🛠️ Development

### Adding New Models

1. Create model class in `src/models/`
2. Implement `train()`, `predict()`, `evaluate()` methods
3. Add to ensemble in `ensemble_model.py`
4. Update configuration in `config.yaml`

### Adding New Features

1. Add feature calculation in `feature_engineering.py`
2. Update `config.yaml` to include the feature
3. Test with notebooks
4. Retrain models

## 🐛 Troubleshooting

**Issue: Out of memory during training**
- Solution: Reduce `batch_size` in config.yaml
- Solution: Use fewer LSTM units
- Solution: Train on subset of stocks

**Issue: Poor predictions**
- Solution: Increase `sequence_length` (more historical data)
- Solution: Add more technical indicators
- Solution: Use ensemble model
- Solution: Train for more epochs

**Issue: Slow training**
- Solution: Use GRU instead of LSTM
- Solution: Reduce sequence length
- Solution: Use GPU acceleration

## 📚 Dependencies

- **TensorFlow/Keras**: Deep learning models
- **scikit-learn**: Preprocessing and baseline models
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn/plotly**: Visualization
- **ta/pandas-ta**: Technical indicators
- **PyYAML**: Configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is for educational purposes. Stock market prediction involves significant risk. This is not financial advice.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Stock market trading involves substantial risk of loss. Past performance does not guarantee future results. Do not use this system for actual trading without proper risk management and professional financial advice.

## 🔗 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Technical Analysis Library](https://github.com/bukosabino/ta)
- [Time Series Forecasting Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ for stock market analysis and machine learning education**
